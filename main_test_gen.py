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
        """Create a task for generating unit tests"""
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
            code = output.content
        else:
            code = str(output)
        
        code = code.replace('```go', '').replace('```', '')
        return code.strip()

    def _validate_test_code(self, test_path: str) -> bool:
        """Validate the generated test code compiles"""
        test_binary = os.path.join(self.config.working_directory, "main.test")
        try:
            # Ensure we're in a module, but don't try to tidy it
            if not os.path.exists(os.path.join(self.config.working_directory, "go.mod")):
                logger.info("Initializing go module...")
                module_path = "test/local"  # Use a simple local module path
                init_process = subprocess.run(
                    ['go', 'mod', 'init', module_path],
                    capture_output=True,
                    text=True,
                    cwd=self.config.working_directory
                )
                if init_process.returncode != 0:
                    logger.error(f"Failed to initialize go module:\n{init_process.stderr}")
                    return False

            # Skip go mod tidy - we don't need external dependencies for basic tests
            logger.debug("Skipping go mod tidy for local package tests")

            # Try to compile the tests
            process = subprocess.run(
                ['go', 'test', '-c'],
                capture_output=True,
                text=True,
                cwd=self.config.working_directory
            )

            # Clean up test binary regardless of compilation result
            if os.path.exists(test_binary):
                try:
                    os.remove(test_binary)
                    logger.debug(f"Removed test binary: {test_binary}")
                except Exception as e:
                    logger.warning(f"Failed to remove test binary {test_binary}: {str(e)}")

            if process.returncode == 0:
                return True
            logger.error(f"Test compilation failed:\n{process.stderr}")
            return False

        except Exception as e:
            logger.error(f"Error validating test code: {str(e)}")
            if os.path.exists(test_binary):
                try:
                    os.remove(test_binary)
                    logger.debug(f"Removed test binary after error: {test_binary}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to remove test binary {test_binary}: {str(cleanup_error)}")
            return False

            # Skip go mod tidy - we don't need external dependencies for basic tests
            logger.debug("Skipping go mod tidy for local package tests")
            
            try:
                # Try to compile the tests
                process = subprocess.run(
                    ['go', 'test', '-c'],
                    capture_output=True,
                    text=True,
                    cwd=self.config.working_directory
                )
                
                # Always attempt to remove the test binary
                if os.path.exists(test_binary):
                    try:
                        os.remove(test_binary)
                        logger.debug(f"Removed test binary: {test_binary}")
                    except Exception as e:
                        logger.warning(f"Failed to remove test binary {test_binary}: {str(e)}")
                
                if process.returncode == 0:
                    return True
                logger.error(f"Test compilation failed:\n{process.stderr}")
                return False
                
            except Exception as e:
                logger.error(f"Error validating test code: {str(e)}")
                return False

            # Try to compile the tests
            process = subprocess.run(
                ['go', 'test', '-c'],
                capture_output=True,
                text=True,
                cwd=self.config.working_directory
            )
            if process.returncode == 0:
                return True
            logger.error(f"Test compilation failed:\n{process.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error validating test code: {str(e)}")
            return False

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
            
            # Get build errors by attempting to compile
            process = subprocess.run(
                ['go', 'test', '-c'],
                capture_output=True,
                text=True,
                cwd=self.config.working_directory
            )
            build_errors = process.stderr if process.returncode != 0 else ""
            
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
                
                # Verify if the code now compiles
                if self._validate_test_code(test_path):
                    logger.info("Fixed test code compiles successfully")
                    return True
                    
            except Exception as e:
                logger.error(f"Error writing fixed code to file: {str(e)}")
                return False

        logger.error(f"Failed to fix test code after {self.config.max_iterations} iterations")
        return False

    def _run_tests(self) -> tuple[bool, str]:
        """Run the unit tests and capture output"""
        try:
            process = subprocess.run(
                ['go', 'test', '-v', './...'],
                capture_output=True,
                text=True,
                cwd=self.config.working_directory,
                timeout=self.config.timeout_seconds
            )
            return process.returncode == 0, process.stdout + process.stderr
        except subprocess.TimeoutExpired:
            return False, "Test execution timed out"
        except Exception as e:
            return False, f"Error running tests: {str(e)}"

    def generate_and_run_tests(self) -> bool:
        """Main method to generate and run tests"""
        # Read source code
        source_path = os.path.join(self.config.working_directory, self.config.source_file)
        try:
            with open(source_path, 'r') as f:
                source_code = f.read()
        except Exception as e:
            logger.error(f"Failed to read source file: {str(e)}")
            return False

        # Generate test file name if not specified
        if not self.config.test_file:
            base_name = os.path.splitext(self.config.source_file)[0]
            self.config.test_file = f"{base_name}_test.go"

        test_path = os.path.join(self.config.working_directory, self.config.test_file)

        # Generate tests
        generator_task = self._create_test_generator_task(source_code)
        crew = Crew(
            agents=[generator_task.agent],
            tasks=[generator_task],
            process=Process.sequential
        )
        test_code = self._clean_code_output(crew.kickoff())

        # Create a backup of any existing test file
        if os.path.exists(test_path):
            backup_path = test_path + ".bak"
            try:
                shutil.copy2(test_path, backup_path)
                logger.info(f"Created backup of '{test_path}' at '{backup_path}'")
            except Exception as e:
                logger.error(f"Failed to create backup of '{test_path}': {e}")

        # Write test file
        try:
            with open(test_path, 'w') as f:
                f.write(test_code)
            logger.info(f"Wrote test code to {test_path}")
        except Exception as e:
            logger.error(f"Failed to write test file: {str(e)}")
            return False

        # Validate and fix test code if needed
        if not self._validate_test_code(test_path):
            logger.info("Generated tests failed to compile, attempting to fix...")
            if not self._fix_code_with_llm(test_path, source_path):
                logger.error("Failed to fix test code")
                return False

        # Run tests
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

def main():
    parser = argparse.ArgumentParser(description='Generate and run Go unit tests')
    parser.add_argument('config', help='Path to YAML configuration file')
    parser.add_argument('--llm', choices=['gemini', 'ollama', 'openai'], default='gemini',
                      help='LLM provider to use')
    parser.add_argument('--ollama-host', default='http://localhost:11434',
                      help='Ollama host URL')
    parser.add_argument('--ollama-model', default='mistral',
                      help='Ollama model name')
    parser.add_argument('--openai-model', default='gpt-3.5-turbo',
                      help='OpenAI model name')
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
            self.prompt = ChatPromptTemplate.from_messages([("user", "{prompt}")])

        def call(self, prompt: str, **kwargs) -> str:
            logger.debug(f"Calling Gemini LLM with prompt: {prompt}")
            response = self.llm.invoke(self.prompt.format_messages(prompt=prompt)).content
            logger.debug(f"Gemini LLM response: {response}")
            return response

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
            self.prompt = ChatPromptTemplate.from_messages([("user", "{prompt}")])

        def call(self, prompt: str, **kwargs) -> str:
            logger.debug(f"Calling Ollama LLM with prompt: {prompt}")
            response = self.llm.invoke(prompt)
            logger.debug(f"Ollama LLM response: {response}")
            return response

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
            self.prompt = ChatPromptTemplate.from_messages([("user", "{prompt}")])

        def call(self, prompt: str, **kwargs) -> str:
            logger.debug(f"Calling OpenAI LLM with prompt: {prompt}")
            response = self.llm.invoke(self.prompt.format_messages(prompt=prompt)).content
            logger.debug(f"OpenAI LLM response: {response}")
            return response

    # Initialize appropriate LLM based on args
    if args.llm == 'gemini':
        llm = GeminiLLM(
            model='models/gemini-pro',
            verbose=args.verbose >= 1,
            temperature=0.99,
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )
    elif args.llm == 'ollama':
        llm = OllamaLLM(
            model=args.ollama_model,
            base_url=args.ollama_host,
            temperature=0.99,
            verbose=args.verbose >= 1
        )
    elif args.llm == 'openai':
        llm = OpenAILLM(
            model=args.openai_model,
            temperature=0.99,
            verbose=args.verbose >= 1,
            openai_api_key=os.getenv('OPENAI_API_KEY')
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
