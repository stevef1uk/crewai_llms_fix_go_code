import os
import sys
import logging
import subprocess
import time
import signal
import yaml
from dataclasses import dataclass
from typing import Optional
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
            # Initialize the module if it doesn't exist
            if not os.path.exists(os.path.join(self.config.working_directory,"go.mod")):
                process = subprocess.Popen(['go', 'mod', 'init', 'temp'],  stderr=subprocess.PIPE, cwd=self.config.working_directory)
                _, stderr = process.communicate()
                if process.returncode != 0:
                    stderr_str = stderr.decode()
                    # Ignore GOSUMDB warnings
                    if not "verifying module: checksum database disabled by GOSUMDB=off" in stderr_str:
                        logger.error(f"Error initializing go module:\n{stderr_str}")
                        return False

            # Build the code
            process = subprocess.Popen(['go', 'build'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.config.working_directory)
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
        """Prepares the Go environment by running go mod tidy and go get."""
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
        
           # Run go get
            print("Running 'go get'...")
            result = subprocess.run(
                ['go', 'get',  './...'],
                cwd=self.config.working_directory,
                check=True,
                env=env,
                capture_output=True,
                text=True
            )
            logger.info(f"'go get' output:\n{result.stdout}")
            if result.stderr:
                logger.error(f"'go get' errors:\n{result.stderr}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during 'go mod tidy' or 'go get':\n{e.stderr}")
            return False
            
    def _build_go_program(self, go_file, env):
        """Build the go program using go build"""
        temp_executable = "tmp_proj_ignore_me"
        temp_executable_path = os.path.join(self.config.working_directory, temp_executable)
        try:
            print(f"Building go program... with file {go_file} in {self.config.working_directory}")
            result = subprocess.run(
                ['go', 'build', '-o', temp_executable, go_file],
                cwd=self.config.working_directory,
                check=False,  # Don't raise exception, we'll handle the error
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
                        logger.info(f"Process ended with return code: {self.process.returncode}, output:\n {stdout}")
                        return True
                    else:
                        logger.error(f"Process ended with return code: {self.process.returncode}, output:\n {stdout} \n, errors: \n {stderr}")
                    break
                time.sleep(0.1)
            else:
                # Timeout hit, terminate the process
                 logger.warning(f"Timeout of {self.config.timeout_seconds} seconds reached, terminating process.")
                 self.terminate()
                 return False 
      except subprocess.CalledProcessError as e:
        logger.error(f"Error during running program: ")
        logger.error(f"Output: {e.output}")
        return False
      except Exception as e:
          logger.error(f"Error running Go program: ")
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
        go_file = os.path.basename(source_file)
        
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



def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <config.yaml> [--llm <gemini|ollama|openai>] [--ollama-host <host_url>] [--ollama-model <model_name>] [--openai-model <model_name>]")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description='Debug Go code using AI agents')
    parser.add_argument('config', help='Path to YAML configuration file')
    parser.add_argument('--llm', choices=['gemini', 'ollama', 'openai'], default='gemini', help='LLM provider to use')
    parser.add_argument('--ollama-host', default='http://localhost:11434', help='Ollama host URL')
    parser.add_argument('--ollama-model', default='mistral', help='Ollama model name')
    parser.add_argument('--openai-model', default='gpt-3.5-turbo', help='OpenAI model name')
    
    args = parser.parse_args()
    
    load_dotenv()
    
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
            return self.llm.invoke(self.prompt.format_messages(prompt=prompt)).content

    class OllamaLLM(LLM):
        def __init__(self, model, base_url, temperature=0.7, verbose=False):
            super().__init__(
                model=model,
                temperature=temperature,
            )
            self.llm = Ollama(
                model=model,
                base_url=base_url,
                temperature=temperature,
                verbose=verbose
            )
            
            self.prompt = ChatPromptTemplate.from_messages([
                ("user", "{prompt}")
            ])

        def call(self, prompt: str, **kwargs) -> str:
            return self.llm.invoke(prompt)

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
           return self.llm.invoke(self.prompt.format_messages(prompt=prompt)).content

    if args.llm == 'gemini':
        llm = GeminiLLM(
            model='models/gemini-pro',
            verbose=True,
            temperature=0.99,
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )
    elif args.llm == 'ollama':
        llm = OllamaLLM(
            model=args.ollama_model,
            base_url=args.ollama_host,
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
