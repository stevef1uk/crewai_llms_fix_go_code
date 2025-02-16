import os
import sys
import logging
import subprocess
import yaml
from dataclasses import dataclass
from typing import Optional
import argparse
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import shutil
import requests
from groq import Groq
from abc import ABC, abstractmethod
import litellm
from litellm.litellm_core_utils import get_llm_provider_logic
from crewai.llm import LLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    working_directory: str
    source_file: str
    test_file: Optional[str] = None
    timeout_seconds: int = 30
    max_iterations: int = 3
    environment: Optional[dict] = None
    test_working_directory: Optional[str] = None  # Will default to working_directory if not set

    def __post_init__(self):
        # If test_working_directory not specified, use working_directory
        if self.test_working_directory is None:
            self.test_working_directory = self.working_directory

class GroqLLM(LLM):
    def __init__(self, model, temperature=0.7, verbose=False, groq_api_key=None):
        super().__init__(
            model=model,
            temperature=temperature
        )
        if verbose:
            print(f"Initializing GroqLLM with model: {model}")
            
        self.client = Groq(api_key=groq_api_key)
        self.model = model.replace('groq/', '')  # Remove any 'groq/' prefix
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
            model=self.model,  # Use the clean model name without prefix
            temperature=self.temperature
        )
        
        return chat_completion.choices[0].message.content

class GeminiLLM(LLM):
    def __init__(self, model, temperature, google_api_key, verbose=False):
        super().__init__(
            model=model,
            temperature=temperature
        )
        if verbose:
            print(f"Initializing GeminiLLM with model: {model}")
            
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            verbose=verbose,
            temperature=temperature,
            google_api_key=google_api_key
        )
        self.prompt = ChatPromptTemplate.from_messages([("user", "{prompt}")])
        self.verbose = verbose

    def call(self, prompt: str, **kwargs) -> str:
        if self.verbose:
            print(f"Calling Gemini LLM with prompt: {prompt}")
            
        # Handle CrewAI's message format
        if isinstance(prompt, dict) and 'messages' in prompt:
            messages = prompt['messages']
            # Convert messages to a single string
            prompt = "\n".join([m['content'] for m in messages])
        elif isinstance(prompt, list):
            # Convert list of messages to a single string
            prompt = "\n".join([m['content'] for m in prompt])
            
        response = self.llm.invoke(self.prompt.format_messages(prompt=prompt)).content
        
        if self.verbose:
            print(f"Gemini LLM response: {response}")
            
        return response

class OpenAILLM(LLM):
    def __init__(self, model, temperature=0.7, verbose=False, openai_api_key=None):
        super().__init__(
            model=model,
            temperature=temperature
        )
        if verbose:
            print(f"Initializing OpenAI LLM with model: {model}")
            
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            verbose=verbose,
            openai_api_key=openai_api_key,
        )
        self.prompt = ChatPromptTemplate.from_messages([("user", "{prompt}")])
        self.verbose = verbose

    def call(self, prompt: str, **kwargs) -> str:
        if self.verbose:
            print(f"Calling OpenAI LLM with prompt: {prompt}")
            
        # Handle CrewAI's message format
        if isinstance(prompt, dict) and 'messages' in prompt:
            messages = prompt['messages']
            # Convert messages to a single string
            prompt = "\n".join([m['content'] for m in messages])
        elif isinstance(prompt, list):
            # Convert list of messages to a single string
            prompt = "\n".join([m['content'] for m in prompt])
            
        response = self.llm.invoke(self.prompt.format_messages(prompt=prompt)).content
        
        if self.verbose:
            print(f"OpenAI LLM response: {response}")
            
        return response

class TestGenerator:
    def __init__(self, config_path: str, llm):
        self.config = self._load_config(config_path)
        self.llm = llm

    def _load_config(self, config_path: str) -> TestConfig:
        """Loads the configuration from the specified YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            source_file = config_data.get('source_file') or config_data.get('fix_file')
            if not source_file:
                raise ValueError("No source_file specified in config")
                
            return TestConfig(
                working_directory=config_data['working_directory'],
                source_file=source_file,
                test_file=config_data.get('test_file'),
                timeout_seconds=config_data.get('timeout_seconds', 30),
                max_iterations=config_data.get('max_iterations', 3),
                environment=config_data.get('environment', {})
            )

    def _get_script_dir(self):
        """Get the directory of the current script."""
        return os.path.dirname(os.path.abspath(__file__))

    def _load_agent_config(self, config_path: str) -> dict:
        """Load agent configuration from a YAML file."""
        script_dir = self._get_script_dir()
        config_path = os.path.join(script_dir, config_path)
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load agent config from {config_path}: {str(e)}")
            raise

    def _create_test_generator_task(self, source_code: str) -> Task:
        agent_config = self._load_agent_config('config/test_generator_agent.yaml')
        return Task(
            description=agent_config['description'].format(source_code=source_code),
            expected_output=agent_config['expected_output'],
            agent=Agent(
                name=agent_config['name'],
                role=agent_config['role'],
                goal=agent_config['goal'],
                backstory=agent_config['backstory'],
                verbose=True,
                llm=self.llm
            )
        )

    def _create_analyzer_task(self, filename: str) -> Task:
        """Create a task for analyzing code issues"""
        agent_config = self._load_agent_config('config/analyzer_agent.yaml')
        try:
            with open(filename, 'r') as f:
                code = f.read()
            
            return Task(
                description=agent_config['description'].format(code=code),
                expected_output=agent_config['expected_output'],
                agent=Agent(
                    name=agent_config['name'],
                    role=agent_config['role'],
                    goal=agent_config['goal'],
                    backstory=agent_config['backstory'],
                    verbose=True,
                    llm=self.llm
                )
            )
        except Exception as e:
            logger.error(f"Failed to create analyzer task: {str(e)}")
            raise

    def _create_fixer_task(self, filename: str, analysis: str) -> Task:
        """Create a task for fixing code issues"""
        agent_config = self._load_agent_config('config/fixer_agent.yaml')
        try:
            with open(filename, 'r') as f:
                original_code = f.read()
                
            return Task(
                description=agent_config['description'].format(original_code=original_code, analysis=analysis),
                expected_output=agent_config['expected_output'],
                agent=Agent(
                    name=agent_config['name'],
                    role=agent_config['role'],
                    goal=agent_config['goal'],
                    backstory=agent_config['backstory'],
                    verbose=True,
                    llm=self.llm
                )
            )
        except Exception as e:
            logger.error(f"Failed to create fixer task: {str(e)}")
            raise

    def _create_test_analyzer_task(self, test_output: str) -> Task:
        """Create a task for analyzing test results"""
        agent_config = self._load_agent_config('config/test_analyzer_agent.yaml')
        
        return Task(
            description=agent_config['description'].format(test_output=test_output),
            expected_output=agent_config['expected_output'],
            agent=Agent(
                name=agent_config['name'],
                role=agent_config['role'],
                goal=agent_config['goal'],
                backstory=agent_config['backstory'],
                verbose=True,
                llm=self.llm
            )
        )

    def _clean_code_output(self, output) -> str:
        """Clean the code output from the LLM response"""
        if hasattr(output, 'content'):
            # Handle CrewOutput object
            code = output.content
        else:
            # Handle string output
            code = str(output)
        
        # Extract only the Go code block if multiple blocks exist
        if "```go" in code:
            blocks = code.split("```go")
            if len(blocks) > 1:
                code = blocks[1].split("```")[0]
        
        # Remove any remaining markdown code block markers
        code = code.replace('```go', '').replace('```bash', '').replace('```plaintext', '').replace('```', '')
        
        # Remove leading and trailing whitespace
        return code.strip()

    def _find_test_binaries(self, test_dir: str) -> list[str]:
        """Find all potential test binaries in the directory"""
        binaries = []
        try:
            for file in os.listdir(test_dir):
                # Look for any file ending in .test and executable test files
                if file.endswith('.test') or (file.startswith('test') and os.access(os.path.join(test_dir, file), os.X_OK)):
                    binaries.append(os.path.join(test_dir, file))
                    
            if binaries:
                logger.debug(f"Found test binaries: {binaries}")
        except Exception as e:
            logger.warning(f"Error scanning for test binaries: {e}")
        return binaries

    def _cleanup_test_binaries(self, test_dir: str) -> None:
            """Clean up any test binaries found in the directory"""
            binaries = self._find_test_binaries(test_dir)
            for binary in binaries:
                try:
                    os.remove(binary)
                    logger.debug(f"Removed test binary: {binary}")
                except Exception as e:
                    logger.warning(f"Failed to remove test binary {binary}: {e}")

    def _get_package_name(self, file_path: str) -> str:
        """Extract package name from Go source file"""
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('package '):
                        return line.split()[1].strip()
        except Exception as e:
            logger.error(f"Failed to read package name from {file_path}: {e}")
        return "main"  
    
    def _find_module_root(self, start_dir: str) -> Optional[str]:
        """Find the root directory containing go.mod"""
        current = os.path.abspath(start_dir)
        while current != os.path.dirname(current):  # Stop at root directory
            if os.path.exists(os.path.join(current, "go.mod")):
                return current
            current = os.path.dirname(current)
        return None


    def _validate_test_code(self, test_path: str) -> bool:
        """Validate the test code and handle dependencies"""
        test_dir = os.path.dirname(test_path)
        
        try:
            # Find module root if it exists
            module_root = self._find_module_root(test_dir)
            if module_root:
                logger.debug(f"Found module root at: {module_root}")
                
                # Check for vendor directory
                has_vendor = os.path.exists(os.path.join(module_root, "vendor"))
                logger.debug(f"Vendor directory exists: {has_vendor}")
            
            # Set up environment
            env = os.environ.copy()
            if module_root:
                env['GO111MODULE'] = 'on'
                env['GOWORK'] = 'off'
                if has_vendor:
                    env['GOFLAGS'] = '-mod=vendor'
            else:
                env['GO111MODULE'] = 'off'
                
            # Run in the test directory, not module root
            logger.debug(f"Running test compilation in {test_dir}")
            cmd = ['go', 'test', '-c']
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=test_dir,
                env=env
            )
            
            if process.returncode != 0:
                logger.debug(f"Test compilation failed: {process.stderr}")
                return False
                
            return True
                
        except Exception as e:
            logger.error(f"Error validating test code: {str(e)}")
            return False
    
    def _run_tests(self) -> tuple[bool, str]:
        """Run the test code and capture both success/failure and output"""
        try:
            test_dir = os.path.abspath(self.config.test_working_directory)
            logger.debug(f"Test directory: {test_dir}")
            
            # Get module information
            module_path, module_root = self._get_module_info(test_dir)
            if not module_path or not module_root:
                return False, "No go.mod found or invalid module configuration"
            
            logger.debug(f"Module path: {module_path}")
            logger.debug(f"Module root: {module_root}")
            
            # Get relative package path from module root to test directory
            rel_path = os.path.relpath(test_dir, module_root)
            package_path = f"{module_path}/{rel_path}"
            logger.debug(f"Package path: {package_path}")
            
            # Environment setup
            env = os.environ.copy()
            env['GO111MODULE'] = 'on'
            env['GOWORK'] = 'off'  # Disable workspace mode
            
            # Build tests first
            build_cmd = ['go', 'test', '-c']
            if os.path.exists(os.path.join(module_root, "vendor")):
                build_cmd.append('-mod=vendor')
            
            logger.debug(f"Running build command: {' '.join(build_cmd)} in {module_root}")
            build_process = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                cwd=module_root,
                env=env
            )
            
            # If build fails, return the errors
            if build_process.returncode != 0:
                build_errors = build_process.stderr
                logger.debug(f"Build failed with errors:\n{build_errors}")
                return False, build_errors
            
            # Run the tests
            test_cmd = ['go', 'test', '-v']
            if os.path.exists(os.path.join(module_root, "vendor")):
                test_cmd.append('-mod=vendor')
            test_cmd.append(f"./{rel_path}")
            
            logger.debug(f"Running test command: {' '.join(test_cmd)} in {module_root}")
            process = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                cwd=module_root,
                timeout=self.config.timeout_seconds,
                env=env
            )
            
            output = process.stdout + process.stderr
            success = process.returncode == 0
            
            # Clean up test binary
            self._cleanup_test_binaries(test_dir)
            
            return success, output
            
        except subprocess.TimeoutExpired:
            self._cleanup_test_binaries(test_dir)
            return False, "Test execution timed out"
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}")
            self._cleanup_test_binaries(test_dir)
            return False, f"Error running tests: {str(e)}"
    

    def _get_module_info(self, dir_path: str) -> tuple[Optional[str], Optional[str]]:
        """Get module path and root directory from go.mod"""
        current = os.path.abspath(dir_path)
        while current != os.path.dirname(current):  # Stop at root
            go_mod = os.path.join(current, "go.mod")
            if os.path.exists(go_mod):
                try:
                    with open(go_mod, 'r') as f:
                        for line in f:
                            if line.startswith('module '):
                                return line.split()[1].strip(), current
                except Exception as e:
                    logger.error(f"Error reading go.mod: {e}")
                    break
            current = os.path.dirname(current)
        return None, None

    def _create_test_fixer_task(self, source_code: str, test_code: str, build_errors: str) -> Task:
        """Create a task for fixing test code"""
        agent_config = self._load_agent_config('config/test_fixer_agent.yaml')
        
        return Task(
            description=agent_config['description'].format(
                source_code=source_code,
                test_code=test_code,
                build_errors=build_errors
            ),
            expected_output=agent_config['expected_output'],
            agent=Agent(
                name=agent_config['name'],
                role=agent_config['role'],
                goal=agent_config['goal'],
                backstory=agent_config['backstory'],
                verbose=True,
                llm=self.llm
            )
        )
 
    def _get_test_errors(self, test_dir: str) -> str:
        """Run tests and get both compilation and test failures"""
        try:
            # Debug directory structure
            logger.debug(f"Current test directory: {test_dir}")
            logger.debug(f"Directory contents: {os.listdir(test_dir)}")
            
            # Run go env to see Go environment
            env_process = subprocess.run(
                ['go', 'env'],
                capture_output=True,
                text=True,
                cwd=test_dir
            )
            logger.debug(f"Go environment:\n{env_process.stdout}")
            
            # Basic test command without any module flags first
            cmd = ['go', 'test', '-v']
            logger.debug(f"Running test command: {' '.join(cmd)} in {test_dir}")
            
            # Run the tests
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=test_dir,
                timeout=30
            )
            
            # Capture all output
            all_output = process.stdout + process.stderr
            
            if process.returncode != 0:
                return all_output
            return ""
            
        except subprocess.TimeoutExpired:
            return "Test execution timed out"
        except Exception as e:
            return f"Error running tests: {str(e)}"
    

    def _fix_code_with_llm(self, test_path: str, source_path: str) -> bool:
        """Uses the LLM agents to fix broken test code."""
        try:
            # Read source code
            with open(source_path, 'r') as f:
                source_code = f.read()
            
            # Read current test code
            with open(test_path, 'r') as f:
                test_code = f.read()
                
        except Exception as e:
            logger.error(f"Error reading source or test files: {str(e)}")
            return False

        for iteration in range(self.config.max_iterations):
            logger.info(f"\nFix Iteration {iteration + 1}:")
            
            # Get both compilation and test failures
            test_dir = os.path.dirname(test_path)
            build_errors = self._get_test_errors(test_dir)
            
            if not build_errors:
                logger.info("No errors found, tests pass")
                return True
                
            logger.debug(f"Test errors to fix:\n{build_errors}")
            
            # Generate fixed code using the test fixer agent
            fixer_task = self._create_test_fixer_task(source_code, test_code, build_errors)
            crew = Crew(
                agents=[fixer_task.agent],
                tasks=[fixer_task],
                process=Process.sequential
            )
            fixed_output = crew.kickoff()
            fixed_code = self._clean_code_output(fixed_output)

            if not fixed_code:
                logger.error("Failed to generate valid fixed code")
                return False

            # Write the fixed code to the file
            try:
                with open(test_path, 'w') as f:
                    f.write(fixed_code)
                logger.info("Wrote fixed test code to file")
                
                # Verify if the code now compiles and tests pass
                if self._validate_test_code(test_path):
                    logger.info("Fixed test code compiles successfully")
                    return True
                    
            except Exception as e:
                logger.error(f"Error writing fixed code to file: {str(e)}")
                return False

        logger.error(f"Failed to fix test code after {self.config.max_iterations} iterations")
        return False

  
    def _check_directory_structure(self):
        """Debug helper to check directory structure"""
        test_dir = self.config.test_working_directory
        logger.debug(f"Checking directory structure in {test_dir}")
        
        try:
            # List directory contents
            contents = os.listdir(test_dir)
            logger.debug(f"Directory contents: {contents}")
            
            # Check go.mod
            go_mod_path = os.path.join(test_dir, "go.mod")
            if os.path.exists(go_mod_path):
                with open(go_mod_path, 'r') as f:
                    logger.debug(f"go.mod contents:\n{f.read()}")
                    
            # Check test file
            test_file = os.path.join(test_dir, os.path.basename(self.config.test_file))
            if os.path.exists(test_file):
                with open(test_file, 'r') as f:
                    logger.debug(f"Test file contents:\n{f.read()}")
                    
        except Exception as e:
            logger.error(f"Error checking directory structure: {e}")

   
    def generate_and_run_tests(self) -> bool:
        """Main method to generate and run tests"""
        # Get full source file path
        source_path = os.path.join(self.config.working_directory, self.config.source_file)
        source_dir = os.path.dirname(source_path)
        
        try:
            with open(source_path, 'r') as f:
                source_code = f.read()
        except Exception as e:
            logger.error(f"Failed to read source file: {str(e)}")
            return False

        # Generate test file path - keep it in same directory as source
        if self.config.test_file:
            test_path = os.path.join(source_dir, os.path.basename(self.config.test_file))
        else:
            # Generate test file name from source file
            base_name = os.path.splitext(os.path.basename(self.config.source_file))[0]
            test_path = os.path.join(source_dir, f"{base_name}_test.go")
        
        logger.debug(f"Source path: {source_path}")
        logger.debug(f"Test path: {test_path}")
        logger.debug(f"Working in directory: {source_dir}")

        # Create backup of any existing test file
        if os.path.exists(test_path):
            backup_path = test_path + ".bak"
            try:
                shutil.copy2(test_path, backup_path)
                logger.info(f"Created backup of '{test_path}' at '{backup_path}'")
            except Exception as e:
                logger.error(f"Failed to create backup of '{test_path}': {e}")

        # Generate tests
        generator_task = self._create_test_generator_task(source_code)
        crew = Crew(
            agents=[generator_task.agent],
            tasks=[generator_task],
            process=Process.sequential
        )
        test_code = self._clean_code_output(crew.kickoff())

        # Write test file
        try:
            os.makedirs(os.path.dirname(test_path), exist_ok=True)
            with open(test_path, 'w') as f:
                f.write(test_code)
            logger.info(f"Wrote test code to {test_path}")
        except Exception as e:
            logger.error(f"Failed to write test file: {str(e)}")
            return False

        # Handle dependencies before validation
        if not self._handle_dependencies(source_dir):
            logger.error("Failed to handle dependencies")
            return False

        # Set working directory for validation and tests
        self.config.test_working_directory = source_dir

        # Validate and fix test code if needed
        while True:
            if self._validate_test_code(test_path):
                break
                
            logger.info("Generated tests failed to compile, attempting to fix...")
            if not self._fix_code_with_llm(test_path, source_path):
                logger.error("Failed to fix test code")
                return False
                
        # Run tests from the source directory
        logger.debug(f"Running tests in: {source_dir}")
        success, test_output = self._run_tests()
        logger.info(f"Test execution output:\n{test_output}")

        # Analyze test results
        analyzer_task = self._create_test_analyzer_task(test_output)
        crew = Crew(
            agents=[analyzer_task.agent],
            tasks=[analyzer_task],
            process=Process.sequential
        )
        analysis = crew.kickoff()
        logger.info(f"\nTest Analysis:\n{analysis}")

        return success
    
    def _handle_dependencies(self, test_dir: str) -> bool:
        """Handle external package dependencies considering vendor directory"""
        try:
            logger.info("Handling test dependencies...")
            
            # Check if vendor directory exists
            vendor_path = os.path.join(test_dir, "vendor")
            has_vendor = os.path.exists(vendor_path)
            logger.debug(f"Vendor directory exists: {has_vendor}")
            
            # Set up environment
            env = os.environ.copy()
            if has_vendor:
                env['GOFLAGS'] = '-mod=vendor'
                logger.info("Using vendored dependencies")
            else:
                env['GOFLAGS'] = '-mod=mod'
                logger.info("Using module dependencies")

            # First run go mod tidy
            tidy_cmd = ['go', 'mod', 'tidy']
            logger.debug(f"Running: {' '.join(tidy_cmd)}")
            tidy_result = subprocess.run(
                tidy_cmd,
                cwd=test_dir,
                env=env,
                capture_output=True,
                text=True
            )
            if tidy_result.returncode != 0:
                logger.error(f"go mod tidy failed:\n{tidy_result.stderr}")
                return False

            if has_vendor:
                # If using vendor, sync the vendor directory
                vendor_cmd = ['go', 'mod', 'vendor']
                logger.debug(f"Running: {' '.join(vendor_cmd)}")
                vendor_result = subprocess.run(
                    vendor_cmd,
                    cwd=test_dir,
                    env=env,
                    capture_output=True,
                    text=True
                )
                if vendor_result.returncode != 0:
                    logger.error(f"go mod vendor failed:\n{vendor_result.stderr}")
                    return False
            else:
                # If not using vendor, download dependencies
                get_cmd = ['go', 'get', 'github.com/stretchr/testify/assert']
                logger.debug(f"Running: {' '.join(get_cmd)}")
                get_result = subprocess.run(
                    get_cmd,
                    cwd=test_dir,
                    env=env,
                    capture_output=True,
                    text=True
                )
                if get_result.returncode != 0:
                    logger.error(f"go get failed:\n{get_result.stderr}")
                    return False

            logger.info("Dependencies handled successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error handling dependencies: {str(e)}")
            return False

def force_groq_provider(model: str):
    return "groq", None, None, None

# Patch litellm.get_llm_provider
litellm.get_llm_provider = force_groq_provider

def main():
    parser = argparse.ArgumentParser(description='Generate and run Go unit tests')
    parser.add_argument('config', help='Path to YAML configuration file')
    parser.add_argument('--llm', choices=['gemini', 'ollama', 'openai', 'deepseek', 'groq'], 
                       default='gemini',
                       help='LLM provider to use')
    parser.add_argument('--groq-model', 
                       default='llama-3.3-70b-versatile', 
                       choices=['mixtral-8x7b-32768', 
                               'llama-3.3-70b-versatile',
                               'deepseek-r1-distill-llama-70b'],
                       help='Groq model name')
    parser.add_argument('--ollama-host', default='http://localhost:11434',
                      help='Ollama host URL')
    parser.add_argument('--ollama-model', default='mistral',
                      help='Ollama model name')
    parser.add_argument('--openai-model', default='gpt-3.5-turbo',
                      help='OpenAI model name')
    parser.add_argument('--deepseek-url', 
                      default='http://localhost:8080/v1/completions',
                      help='DeepSeek API URL')
    parser.add_argument('--deepseek-key', 
                      default=os.getenv('DEEPSEEK_KEY'),  # Get from environment variable
                      help='DeepSeek API Key')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                      help='Increase verbosity (can be used multiple times, e.g., -vv)')
    
    args = parser.parse_args()
    load_dotenv()

    # Configure logging based on verbosity level
    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    else:  # args.verbose >= 2
        log_level = logging.DEBUG

    # Update root logger and our logger
    logging.getLogger().setLevel(log_level)
    logger.setLevel(log_level)

    # Add more detailed formatting for debug level
    if args.verbose >= 2:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
        )
        for handler in logging.getLogger().handlers:
            handler.setFormatter(formatter)

    # Add DeepSeek LLM class
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
            
            if verbose:
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
                logger.debug(f"Sending request to: {self.base_url}")
                
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
                logger.debug(f"Response status: {response.status_code}")
                if response.status_code != 200:
                    logger.debug(f"Response text: {response.text}")
            
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                return self._extract_code(result['choices'][0]['text'].strip())
            return ""

    # Initialize appropriate LLM based on args
    if args.llm == 'gemini':
        google_key = os.getenv('GOOGLE_API_KEY')
        if not google_key:
            print("Error: GOOGLE_API_KEY environment variable is not set")
            sys.exit(1)
            
        llm = GeminiLLM(
            model='gemini-pro',
            verbose=args.verbose >= 1,
            temperature=0.99,
            google_api_key=google_key
        )
    elif args.llm == 'ollama':
        llm = OllamaLLM(
            model=args.ollama_model,
            base_url=args.ollama_host,
            temperature=0.99,
            verbose=args.verbose >= 1
        )
    elif args.llm == 'openai':
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            print("Error: OPENAI_API_KEY environment variable is not set")
            sys.exit(1)
            
        llm = OpenAILLM(
            model=args.openai_model,
            temperature=0.99,
            verbose=args.verbose >= 1,
            openai_api_key=openai_key
        )
    elif args.llm == 'deepseek':
        deepseek_key = os.getenv('DEEPSEEK_KEY')
        if not deepseek_key:
            print("Error: DEEPSEEK_KEY environment variable is not set")
            sys.exit(1)
            
        llm = DeepSeekLLM(
            model='llama2',
            base_url=args.deepseek_url,
            temperature=0.99,
            verbose=args.verbose >= 1,
            api_key=deepseek_key
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

    logger.debug(f"Initialized {args.llm} LLM")
    generator = TestGenerator(args.config, llm)
    
    logger.info("Starting test generation and execution...")
    success = generator.generate_and_run_tests()
    
    if success:
        logger.info("Test generation and execution completed successfully")
    else:
        logger.error("Test generation and execution failed")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
