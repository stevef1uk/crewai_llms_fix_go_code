import os
import sys
import logging
import subprocess
import time
import signal
import yaml
from dataclasses import dataclass
from typing import Optional, Union, List, Dict
import argparse
import traceback
import shutil
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
import re
import requests
from langchain_community.llms import LlamaCpp
import json
from groq import Groq  # Add this import at the top


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GoConfig:
    working_directory: str
    command: str
    timeout_seconds: int
    max_iterations: int
    fix_file: Optional[str] = None
    environment: Optional[dict] = None


class GoRunner:
    def __init__(self, config_path: str, llm):
        self.config = self._load_config(config_path)
        self.process = None
        self.llm = llm
        self.original_code = None

    def _get_script_dir(self):
        """Get the directory of the current script."""
        return os.path.dirname(os.path.abspath(__file__))
        
    def _load_config(self, config_path: str) -> GoConfig:
        """Loads the configuration from the specified YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            return GoConfig(
                working_directory=config_data['working_directory'],
                command=config_data['command'],
                timeout_seconds=config_data.get('timeout_seconds', 30),
                max_iterations=config_data.get('max_iterations', 3),
                fix_file=config_data.get('fix_file'),
                environment=config_data.get('environment', {})
            )
    
    def _load_agent_config(self, config_path: str) -> dict:
        """Load agent configuration from a YAML file."""
        script_dir = self._get_script_dir()
        config_path = os.path.join(script_dir, config_path)
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Check for required keys
            required_keys = ['name', 'role', 'goal', 'backstory', 'description', 'expected_output']
            for key in required_keys:
                 if key not in config:
                    raise ValueError(f"Agent config missing required key: {key} in {config_path}")
            
            # Check that all values are not empty
            for key, value in config.items():
                if not value:
                    raise ValueError(f"Agent config value is empty: {key} in {config_path}")

            return config
        except Exception as e:
           logger.error(f"Failed to load agent config from {config_path}: {str(e)}")
           raise

    def _create_analyzer_task(self, filename: str) -> Task:
        """Create a task for the analyzer agent"""
        agent_config = self._load_agent_config('config/analyzer_agent.yaml')
        try:
            with open(filename, 'r') as f:
                code = f.read()
            
            description = agent_config['description'].format(code=code)
            
            return Task(
                description=description,
                expected_output=agent_config['expected_output'],
                agent=Agent(
                    name=agent_config['name'],
                    role=agent_config['role'],
                    goal=agent_config['goal'],
                    backstory=agent_config['backstory'],
                    verbose=True,
                    llm = self.llm
                )
            )
        except Exception as e:
            logger.error(f"Failed to create analyzer task: {str(e)}")
            raise


    def _create_fixer_task(self, filename: str, analysis: str) -> Task:
        """Create a task for the fixer agent"""
        agent_config = self._load_agent_config('config/fixer_agent.yaml')
        try:
            with open(filename, 'r') as f:
                original_code = f.read()
                
            description = agent_config['description'].format(original_code=original_code, analysis=analysis)
            
            return Task(
                description=description,
                expected_output=agent_config['expected_output'],
                 agent=Agent(
                    name=agent_config['name'],
                    role=agent_config['role'],
                    goal=agent_config['goal'],
                    backstory=agent_config['backstory'],
                    verbose=True,
                    llm=self.llm,
                )
            )
        except Exception as e:
            logger.error(f"Failed to create fixer task: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to create fixer task: {str(e)}")
            raise
    
    def _clean_code_output(self, output) -> str:
        """Clean the code output from the LLM response"""
        logger.debug(f"Input to clean_code_output:\n{output}")
        if hasattr(output, 'content'):
            # Handle CrewOutput object
            code = output.content
        else:
            # Handle string output
            code = str(output)
        
        # Remove any markdown code block markers
        code = code.replace('```go', '').replace('```', '')
        
        # Remove leading and trailing non code lines
        # Do nothing, return the code
        return code
    
    def _validate_code(self, code: str) -> bool:
        """Simple validation to check the code is valid go, used before we write to disk"""
        try:
            # Check if vendor directory exists
            vendor_path = os.path.join(self.config.working_directory, "vendor")
            has_vendor = os.path.exists(vendor_path)

            # Initialize the module if it doesn't exist
            if not os.path.exists(os.path.join(self.config.working_directory, "go.mod")):
                process = subprocess.Popen(
                    ['go', 'mod', 'init', 'temp'],
                    stderr=subprocess.PIPE,
                    cwd=self.config.working_directory
                )
                _, stderr = process.communicate()
                if process.returncode != 0:
                    stderr_str = stderr.decode()
                    # Ignore GOSUMDB warnings
                    if not "verifying module: checksum database disabled by GOSUMDB=off" in stderr_str:
                        logger.error(f"Error initializing go module:\n{stderr_str}")
                        return False

            # Build the code with appropriate vendor flag if needed
            build_cmd = ['go', 'build']
            if has_vendor:
                build_cmd.append('-mod=vendor')

            process = subprocess.Popen(
                build_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.config.working_directory
            )
            stdout, stderr = process.communicate(input=code.encode())

            if process.returncode != 0:
                stderr_str = stderr.decode()
                # Ignore GOSUMDB warnings
                if not "verifying module: checksum database disabled by GOSUMDB=off" in stderr_str:
                    logger.error(f"Invalid Go code returned:\n{stderr_str}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating Go code: {str(e)}\n{traceback.format_exc()}")
            return False
        
    def _prepare_go_environment(self, env):
        """Prepares the Go environment by running go mod tidy and handling vendor dependencies."""
        try:
            # Run go mod tidy
            print("Running 'go mod tidy'...")
            result = subprocess.run(
                ['go', 'mod', 'tidy'],
                cwd=self.config.working_directory,
                check=True,
                env=env,
                capture_output=True,
                text=True
            )
            logger.info(f"'go mod tidy' output:\n{result.stdout}")
            if result.stderr and "verifying module: checksum database disabled by GOSUMDB=off" not in result.stderr:
                logger.error(f"'go mod tidy' errors:\n{result.stderr}")

            # Check if vendor directory exists
            vendor_path = os.path.join(self.config.working_directory, "vendor")
            has_vendor = os.path.exists(vendor_path)

            if has_vendor:
                print("Vendor directory detected, syncing dependencies...")
                # First, try to download all dependencies
                download_result = subprocess.run(
                    ['go', 'mod', 'download', 'all'],
                    cwd=self.config.working_directory,
                    check=False,  # Don't fail on download issues
                    env=env,
                    capture_output=True,
                    text=True
                )
                if download_result.returncode != 0:
                    logger.warning(f"Warning during dependency download:\n{download_result.stderr}")

                # Then vendor them
                vendor_result = subprocess.run(
                    ['go', 'mod', 'vendor', '-v'],
                    cwd=self.config.working_directory,
                    check=False,  # Don't fail on vendor issues
                    env=env,
                    capture_output=True,
                    text=True
                )
                logger.info(f"'go mod vendor' output:\n{vendor_result.stdout}")
                if vendor_result.returncode != 0:
                    logger.error(f"Error during vendoring:\n{vendor_result.stderr}")
                    # If vendoring fails, try to continue without vendor mode
                    return True  # Continue anyway, will try non-vendor mode

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Error during Go environment preparation:\n{e.stderr}")
            return False
 
    def _build_go_program(self, go_file, env):
        """Build the go program using go build, supporting both vendored and non-vendored projects"""
        temp_executable = "tmp_proj_ignore_me"
        temp_executable_path = os.path.join(self.config.working_directory, temp_executable)
        
        try:
            # Check if vendor directory exists
            vendor_path = os.path.join(self.config.working_directory, "vendor")
            has_vendor = os.path.exists(vendor_path)

            # Ensure the go_file path is in the correct format
            if not go_file.startswith('./'):
                go_file = f"./{go_file}"

            # Prepare build command based on vendor status
            build_cmd = ['go', 'build']
            if has_vendor:
                # First try with strict vendor mode
                build_cmd.append('-mod=vendor')
            build_cmd.extend(['-o', temp_executable, go_file])

            print(f"Building go program... with file {go_file} in {self.config.working_directory}")
            print(f"Using build command: {' '.join(build_cmd)}")
            
            result = subprocess.run(
                build_cmd,
                cwd=self.config.working_directory,
                check=False,
                env=env,
                capture_output=True,
                text=True
            )
            
            # If build fails with vendor mode, try with readonly mode
            if result.returncode != 0 and has_vendor:
                logger.warning("Build failed with strict vendor mode, attempting with readonly mode")
                build_cmd[build_cmd.index('-mod=vendor')] = '-mod=readonly'
                result = subprocess.run(
                    build_cmd,
                    cwd=self.config.working_directory,
                    check=False,
                    env=env,
                    capture_output=True,
                    text=True
                )

            logger.info(f"'go build' output:\n{result.stdout}")
            if result.returncode != 0:
                logger.error(f"Error during 'go build':\n{result.stderr}")
                logger.info("Set build ok flag false")
                return False, temp_executable_path
            
            logger.info("Go build was ok")
            return True, temp_executable_path
            
        except Exception as e:
            logger.error(f"Error during build: {str(e)}")
            logger.info("Set build ok flag false")
            return False, temp_executable_path
    

    def _run_go_program(self, temp_executable_path, env) -> bool:
      """Runs the go program and checks its output for success or failure"""
      try:
            # Start the Go program
            print(f"Starting Go program: {temp_executable_path}")
            kwargs = {'preexec_fn': os.setsid}
            self.process = subprocess.Popen(
                    [temp_executable_path], # Run the compiled binary with full path
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                **kwargs
            )
            
            start_time = time.time()
            while time.time() - start_time < self.config.timeout_seconds:
                if self.process.poll() is not None:
                    # Process has ended
                    stdout, stderr = self.process.communicate()
                    if self.process.returncode == 0:
                        logger.info(f"Process ended with return code: {self.process.returncode}, output:\n{stdout}")
                        return True
                    else:
                        logger.error(f"Process ended with return code: {self.process.returncode}, output:\n{stdout}, errors: \n{stderr}")
                    break
                time.sleep(0.1)
            else:
                # Timeout hit, terminate the process
                 logger.warning(f"Timeout of {self.config.timeout_seconds} seconds reached, terminating process.")
                 self.terminate()
                 return False 
      except subprocess.CalledProcessError as e:
        logger.error(f"Error during running program: {e}")
        logger.error(f"Output: {e.output}")
        return False
      except Exception as e:
          logger.error(f"Error running Go program: {e}")
          logger.error(f"Error: {e}")
          return False

    def _cleanup_temp_executable(self, temp_executable_path):
       """Removes the temporary executable after a run"""
       if os.path.exists(temp_executable_path):
            os.remove(temp_executable_path)
            logger.info(f"Removed temporary executable: {temp_executable_path}")

    def _fix_code_with_llm(self, full_fix_path, code, file_valid) -> str:
        """Uses the LLM agents to fix broken code."""
        if file_valid:
            logger.info("Code is already valid, skipping LLM agents.")
            return code
            
        for iteration in range(self.config.max_iterations):
            logger.info(f"\nDebug Iteration {iteration + 1}:")
            logger.error("Code is not valid Go syntax, attempting to fix with LLM")

            # Step 1: Analyze code
            analyzer_task = self._create_analyzer_task(full_fix_path)
            crew = Crew(
                agents=[analyzer_task.agent],
                tasks=[analyzer_task],
                process=Process.sequential
            )
            analysis = crew.kickoff()

            # Step 2: Generate fixed code
            fixer_task = self._create_fixer_task(full_fix_path, analysis)
            crew = Crew(
                agents=[fixer_task.agent],
                tasks=[fixer_task],
                process=Process.sequential
            )
            fixed_output = crew.kickoff()
            fixed_code = self._clean_code_output(fixed_output)

            if not fixed_code:
                logger.error("Failed to generate valid fixed code")
                return None

            # Write the fixed code to the file
            try:
                with open(full_fix_path, 'w') as f:
                    f.write(fixed_code)
                logger.info("Wrote fixed code to original file")
                
                # Verify the file was written correctly
                with open(full_fix_path, 'r') as f:
                    actual_content = f.read()
                if actual_content != fixed_code:
                    logger.error("File content doesn't match fixed code after writing")
                    return None
                    
                return fixed_code
                
            except Exception as e:
                logger.error(f"Error writing fixed code to file: {str(e)}")
                return None

        logger.error(f"Failed to fix code after {self.config.max_iterations} iterations")
        return None


    def run(self) -> bool:
        # Split the command and add the go file name as the last argument
        command_parts = self.config.command.split()
        if not command_parts[-1].lower().endswith(".go"):
            logger.error(f"The command does not end in a `.go` file, please check your config")
            return False
    
        source_file = command_parts[-1]
        runner_command = "go run " + source_file
        
        # Extract just the go file
        #go_file = os.path.basename(source_file)
        # We want to keep the relative path structure, not just the basename
        go_file = source_file  # Remove the os.path.basename call
        
        # Construct the full path for the source file
        full_source_path = os.path.join(self.config.working_directory, source_file)

        # Check that file exists, if not exit
        if not os.path.exists(full_source_path):
            logger.error(f"Source file does not exist: {full_source_path}")
            return False
            
        # Get the fix file and construct full path
        fix_file = self.config.fix_file
        if not fix_file:
             full_fix_path = full_source_path
             logger.info(f"No fix file specified, using source file {full_source_path}")
        else:
            full_fix_path = os.path.join(self.config.working_directory, fix_file)
            logger.info(f"Fix file specified, will be fixing: {full_fix_path}")
         
         # Check that fix file exists, if not exit
        if not os.path.exists(full_fix_path):
             logger.error(f"Fix file does not exist: {full_fix_path}")
             return False
            
        # Create merged environment variables
        env = os.environ.copy()
        if self.config.environment:
            env.update(self.config.environment)
        
        # Force go to use a local sum db by setting it to off in the environment
        original_gosumdb = os.environ.get('GOSUMDB')
        if not original_gosumdb:
           os.environ['GOSUMDB'] = 'off'
        
        if not self._prepare_go_environment(env):
            return False
        
        build_ok, temp_executable_path = self._build_go_program(go_file, env)
        
        # Read the code we want to fix
        try:
          with open(full_fix_path, 'r') as f:
                self.original_code = f.read()
        except Exception as e:
            logger.error(f"Failed to read source file: {str(e)}")
            return False
        
        code = self.original_code
        
         # Create a backup of the file
        backup_path = full_fix_path + ".bak"
        try:
            shutil.copy2(full_fix_path, backup_path)
            logger.info(f"Created backup of '{full_fix_path}' at '{backup_path}'")
        except Exception as e:
            logger.error(f"Failed to create backup of '{full_fix_path}': {e}")
    
        if build_ok:
            logger.info("Code is already valid, skipping LLM agents.")
            file_valid = True
        else:
            file_valid = False
            code = self._fix_code_with_llm(full_fix_path, code, file_valid)
            if not code:
                logger.error("Failed to fix code with LLM.")
                self._cleanup_temp_executable(temp_executable_path)
                return False
            
            # Re-build after fix
            build_ok, temp_executable_path = self._build_go_program(go_file, env)
            if not build_ok:
               logger.error("Code is still invalid after LLM fix.")
               self._cleanup_temp_executable(temp_executable_path)
               return False
            else:
                file_valid = True # Set the flag to ensure we exit the fix loop

        # Run the program
        success = self._run_go_program(temp_executable_path, env)

        # Clean up the temporary executable after a run
        self._cleanup_temp_executable(temp_executable_path)

        if self.process and self.process.poll() is None:
            self.terminate()
            
        return success

    def terminate(self):
        if self.process:
            print("Terminating process group...")
            try:
                # Kill the entire process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("Process didn't terminate gracefully, forcing kill...")
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)

                stdout, stderr = self.process.communicate()
                print("Final output:", stdout)
                if stderr:
                    print("Final errors:", stderr)
            except ProcessLookupError:
                print("Process already terminated")

    def _extract_code(self, text: str) -> str:
        """Extract code from the response, handling both markdown and plain text."""
        if "</think>" in text:
            parts = text.split("</think>")
            text = parts[-1].strip()

        if "```" in text:
            code_blocks = text.split("```")
            for i in range(len(code_blocks)-2, -1, -2):
                if i % 2 == 1:
                    block = code_blocks[i]
                    if " " in block.split("\n")[0]:
                        block = "\n".join(block.split("\n")[1:])
                    return block.strip()
        return text.strip()


def main():
    parser = argparse.ArgumentParser(description='Fix Go code issues')
    parser.add_argument('config', nargs='?', help='Path to YAML configuration file')
    parser.add_argument('-llm', '--llm', choices=['gemini', 'ollama', 'openai', 'deepseek', 'groq'], 
                       default='gemini',
                       help='LLM provider to use')
    parser.add_argument('--groq-model', 
                       default='llama-3.3-70b-versatile', 
                       choices=['mixtral-8x7b-32768', 
                               'llama-3.3-70b-versatile',
                               'deepseek-r1-distill-llama-70b'],
                       help='Groq model name')
    parser.add_argument('--ollama-host', default='http://localhost:11434', help='Ollama host URL')
    parser.add_argument('--ollama-model', default='mistral', help='Ollama model name')
    parser.add_argument('--openai-model', default='gpt-3.5-turbo', help='OpenAI model name')
    parser.add_argument('--deepseek-url', help='DeepSeek API URL')
    parser.add_argument('--deepseek-key', help='DeepSeek API key')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                       help='Increase verbosity level (use -v, -vv, or -vvv)')
    
    args = parser.parse_args()
    
    # Check if config is provided after parsing other args
    if not args.config:
        print("Usage: python script.py [-llm <gemini|ollama|openai|deepseek|groq>]"
              " [--groq-model <model>] [-v|-vv|-vvv] <config.yaml>")
        sys.exit(1)
    
    print(f"Current working directory: {os.getcwd()}")
    print("Loading .env file...")
    load_dotenv()
    print(f"Environment variables after loading:")
    print(f"GOOGLE_API_KEY exists: {bool(os.getenv('GOOGLE_API_KEY'))}")
    print(f"OPENAI_API_KEY exists: {bool(os.getenv('OPENAI_API_KEY'))}")
    print(f"DEEPSEEK_KEY exists: {bool(os.getenv('DEEPSEEK_KEY'))}")
    print(f"GROQ_API_KEY exists: {bool(os.getenv('GROQ_API_KEY'))}")
    
    from crewai.llm import LLM
    
    class GeminiLLM(LLM):
        def __init__(self, model, temperature, google_api_key, verbose=False):
            super().__init__(
                model=model,
                api_key=google_api_key,
                temperature=temperature,
            )
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                verbose=verbose,
                temperature=temperature,
                google_api_key=google_api_key
            )
            
            self.prompt = ChatPromptTemplate.from_messages([
                ("user", "{prompt}")
            ])

        def call(self, prompt: str, **kwargs) -> str:
            # Handle CrewAI's message format
            if isinstance(prompt, dict) and 'messages' in prompt:
                messages = prompt['messages']
                # Extract only user messages, ignore system/assistant messages
                user_messages = []
                for msg in messages:
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        user_messages.append(msg['content'])
                prompt = "\n".join(user_messages)
            elif isinstance(prompt, list):
                # Handle list of messages similarly
                user_messages = []
                for msg in prompt:
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        user_messages.append(msg['content'])
                prompt = "\n".join(user_messages)
            
            return self.llm.invoke(prompt).content

    class OllamaLLM(LLM):
        def __init__(self, model, base_url, temperature=0.7, verbose=False):
            super().__init__(
                model=model,
                temperature=temperature,
            )
            self.base_url = base_url.rstrip('/')  # Remove trailing slash if present
            self.model = model
            self.temperature = temperature
            self.verbose = verbose
            
            if self.verbose:
                print(f"Initializing OllamaLLM with model: {model}")

        def _extract_code(self, text: str) -> str:
            """Extract code from the response, handling both markdown and plain text."""
            if "</think>" in text:
                parts = text.split("</think>")
                text = parts[-1].strip()

            if "```" in text:
                code_blocks = text.split("```")
                for i in range(len(code_blocks)-2, -1, -2):
                    if i % 2 == 1:
                        block = code_blocks[i]
                        if " " in block.split("\n")[0]:
                            block = "\n".join(block.split("\n")[1:])
                        return block.strip()
            return text.strip()

        def call(self, prompt: str, **kwargs) -> str:
            # Handle CrewAI's message format
            if isinstance(prompt, dict) and 'messages' in prompt:
                messages = prompt['messages']
            elif isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{'role': 'user', 'content': str(prompt)}]

            # Convert messages to a single string
            full_prompt = ""
            for msg in messages:
                if isinstance(msg, dict) and 'content' in msg:
                    role = msg.get('role', 'user')
                    content = msg['content']
                    full_prompt += f"{role}: {content}\n\n"

            if self.verbose:
                print(f"Sending request to: {self.base_url}/api/generate")
                
            response = requests.post(
                f"{self.base_url}/api/generate",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "temperature": self.temperature,
                    "stream": False
                }
            )
            
            if self.verbose:
                print(f"Response status: {response.status_code}")
                if response.status_code != 200:
                    print(f"Response text: {response.text}")
            
            response.raise_for_status()
            result = response.json()
            
            if 'response' in result:
                return self._extract_code(result['response'])
            return ""

    class OpenAILLM(LLM):
        def __init__(self, model, temperature=0.7, verbose=False, openai_api_key=None):
             super().__init__(
                model=model,
                temperature=temperature
            )
             self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                verbose=verbose,
                openai_api_key=openai_api_key,
            )
             self.prompt = ChatPromptTemplate.from_messages([
                ("user", "{prompt}")
            ])

        def call(self, prompt: str, **kwargs) -> str:
            # Handle CrewAI's message format
            if isinstance(prompt, dict) and 'messages' in prompt:
                messages = prompt['messages']
                # Extract only user messages, ignore system/assistant messages
                user_messages = []
                for msg in messages:
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        user_messages.append(msg['content'])
                prompt = "\n".join(user_messages)
            elif isinstance(prompt, list):
                # Handle list of messages similarly
                user_messages = []
                for msg in prompt:
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        user_messages.append(msg['content'])
                prompt = "\n".join(user_messages)
            
            # Pass the string directly to invoke
            return self.llm.invoke(prompt).content

    class LlamaCppServerLLM(LLM):
        def __init__(self, url="http://localhost:8080", temperature=0.7, verbose=False):
            super().__init__(
                model="llama-cpp",
                temperature=temperature
            )
            self.url = url
            self.temperature = temperature
            self.verbose = verbose
            self.api_key = "ARun-YvW8Q_V5Q2l6rjMp8WpZM6Ic-y-wp0BUtWOJM0"
            self._message_buffer = []  # Buffer to store message chunks

        def _send_request(self, prompt_text: str):
            """Send a single request to the API."""
            data = {
                "prompt": prompt_text,
                "max_tokens": 10000,
                "temperature": self.temperature,
                "stream": False
            }

            if self.verbose:
                print("\nSending complete request to server:")
                print(json.dumps(data, indent=2))

            try:
                response = requests.post(
                    self.url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    },
                    json=data
                )
                
                if self.verbose:
                    print(f"Response status: {response.status_code}")
                
                response.raise_for_status()
                return response.json()
            except Exception as e:
                print(f"Error in API call: {str(e)}")
                raise

        def invoke(self, prompt, **kwargs):
            """Handle chunked messages from CrewAI."""
            try:
                # Add the new chunk to our buffer
                if isinstance(prompt, dict) and 'messages' in prompt:
                    self._message_buffer.extend(prompt['messages'])
                elif isinstance(prompt, list):
                    self._message_buffer.extend(prompt)
                else:
                    self._message_buffer.append({
                        'role': 'user',
                        'content': str(prompt)
                    })

                # Check if this is the last chunk (contains main function)
                is_last_chunk = False
                for msg in (prompt['messages'] if isinstance(prompt, dict) and 'messages' in prompt else [prompt]):
                    if isinstance(msg, dict) and 'content' in msg:
                        if 'func main()' in msg['content']:
                            is_last_chunk = True
                            break

                if not is_last_chunk:
                    # If not the last chunk, just store it and return empty string
                    return ""

                # If this is the last chunk, process all accumulated messages
                all_content = []
                current_role = None
                current_content = []

                for msg in self._message_buffer:
                    if isinstance(msg, dict) and 'content' in msg:
                        if msg.get('role') != current_role:
                            if current_content:
                                all_content.append(f"{current_role}: {' '.join(current_content)}")
                                current_content = []
                            current_role = msg.get('role')
                        current_content.append(msg['content'])

                # Add the last role's content
                if current_content:
                    all_content.append(f"{current_role}: {' '.join(current_content)}")

                # Create the final prompt
                final_prompt = "\n\n".join(all_content)

                # Clear the buffer
                self._message_buffer = []

                # Send the complete prompt
                response = self._send_request(final_prompt)
                
                if 'choices' in response and len(response['choices']) > 0:
                    return response['choices'][0]['text'].strip()
                else:
                    raise ValueError(f"Unexpected response format: {response}")

            except Exception as e:
                print(f"Error during API call: {str(e)}")
                self._message_buffer = []  # Clear buffer on error
                return str(e)

    class DeepSeekLLM(LLM):
        def __init__(self, model, base_url, temperature=0.7, verbose=False, api_key=None):
            super().__init__(
                model=model,
                temperature=temperature,
            )
            self.base_url = base_url
            self.api_key = api_key
            self.temperature = temperature
            self.verbose = verbose
            
            print(f"Initializing DeepSeekLLM with API key present: {bool(api_key)}")

        def _extract_code(self, text: str) -> str:
            """Extract code from the response, handling both markdown and plain text."""
            if "</think>" in text:
                parts = text.split("</think>")
                text = parts[-1].strip()

            if "```" in text:
                code_blocks = text.split("```")
                for i in range(len(code_blocks)-2, -1, -2):
                    if i % 2 == 1:
                        block = code_blocks[i]
                        if " " in block.split("\n")[0]:
                            block = "\n".join(block.split("\n")[1:])
                        return block.strip()
            return text.strip()

        def call(self, prompt: str, **kwargs) -> str:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Handle CrewAI's message format
            if isinstance(prompt, dict) and 'messages' in prompt:
                messages = prompt['messages']
            elif isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{'role': 'user', 'content': str(prompt)}]

            # Convert messages to a single string
            full_prompt = ""
            for msg in messages:
                if isinstance(msg, dict) and 'content' in msg:
                    role = msg.get('role', 'user')
                    content = msg['content']
                    full_prompt += f"{role}: {content}\n\n"

            if self.verbose:
                print(f"Sending request to: {self.base_url}")
                
            response = requests.post(
                self.base_url,
                headers=headers,
                json={
                    "model": "llama2",
                    "prompt": full_prompt,
                    "max_tokens": 10000,
                    "temperature": self.temperature,
                    "stream": False
                }
            )
            
            if self.verbose:
                print(f"Response status: {response.status_code}")
                if response.status_code != 200:
                    print(f"Response text: {response.text}")
            
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                return self._extract_code(result['choices'][0]['text'].strip())
            return ""

    class GroqLLM(LLM):
        def __init__(self, model, temperature=0.7, verbose=False, groq_api_key=None):
            super().__init__(
                model=model,
                temperature=temperature
            )
            if verbose:
                print(f"Initializing GroqLLM with model: {model}")
            
            self.client = Groq(api_key=groq_api_key)
            self.model = model
            self.temperature = temperature
            self.verbose = verbose

        def call(self, prompt: str, **kwargs) -> str:
            if self.verbose:
                print(f"Calling Groq API with model: {self.model}")
            
            # Handle CrewAI's message format
            if isinstance(prompt, dict) and 'messages' in prompt:
                messages = prompt['messages']
            elif isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{"role": "user", "content": str(prompt)}]

            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=self.temperature
            )
            
            return chat_completion.choices[0].message.content

    if args.llm == 'gemini':
        llm = GeminiLLM(
            model='models/gemini-pro',
            verbose=True,
            temperature=0.99,
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )
    elif args.llm == 'ollama':
        ollama_host = args.ollama_host or "http://localhost:11434"
        print(f"Connecting to Ollama at: {ollama_host}")
        
        # Verify Ollama is running
        try:
            response = requests.get(f"{ollama_host}/api/version")
            if response.status_code != 200:
                print(f"Error: Cannot connect to Ollama at {ollama_host}")
                print("Please ensure Ollama is running and the host URL is correct")
                sys.exit(1)
        except requests.exceptions.ConnectionError:
            print(f"Error: Cannot connect to Ollama at {ollama_host}")
            print("Please ensure Ollama is running and the host URL is correct")
            sys.exit(1)
            
        llm = OllamaLLM(
            model=args.ollama_model,
            base_url=ollama_host,
            temperature=0.99,
            verbose=True
        )
    elif args.llm == 'openai':
        llm = OpenAILLM(
            model=args.openai_model,
            temperature=0.99,
            verbose=True,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
    elif args.llm == 'deepseek':
        deepseek_key = os.getenv('DEEPSEEK_KEY')  # Get the key directly from environment
        if not deepseek_key:  # Check the actual key value, not args.deepseek_key
            print("Error: DEEPSEEK_KEY environment variable is not set")
            sys.exit(1)
            
        llm = DeepSeekLLM(
            model='llama2',
            base_url=args.deepseek_url,
            temperature=0.99,
            verbose=True,
            api_key=deepseek_key  # Use the key we just got from environment
        )
    elif args.llm == 'groq':
        groq_key = os.getenv('GROQ_API_KEY')
        if not groq_key:
            print("Error: GROQ_API_KEY environment variable is not set")
            sys.exit(1)
            
        llm = GroqLLM(
            model=args.groq_model,
            temperature=0.99,
            verbose=args.verbose >= 1,
            groq_api_key=groq_key
        )
    else:
        raise ValueError(f"Invalid LLM provider: {args.llm}")
    
    config_path = args.config
    runner = GoRunner(config_path, llm)
    
    # Set up signal handlers
    def signal_handler(signum, frame):
        print("\nReceived interrupt signal, cleaning up...")
        runner.terminate()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    success = runner.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
