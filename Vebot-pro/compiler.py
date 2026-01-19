import subprocess
import tempfile
import os

class Compiler:
    def __init__(self, config: dict):
        """
        Compiler reads all required options from config dict.

        This makes the compiler self-contained, and adding new 
        config fields no longer requires modifications in Controller.
        """
        self.config = config

        # Load config fields
        self.use_rcl = config.get("use_rcl", True)

        # Only required in RCL mode
        self.compiler_path = config.get("compiler_path", None)
        self.code_path = config.get("code_path", None)

        if self.use_rcl:
            if not self.compiler_path or not self.code_path:
                raise ValueError("RCL mode requires 'compiler_path' and 'code_path'.")
            
    def compile_code(self, program: str):
        """
        Compile the given program.
        """
        if self.use_rcl:
            return self.compile_rcl_program(program)
        else:
            return self.compile_python_program(program)

    # ---------------------------------------------------
    # RCL compilation
    # ---------------------------------------------------
    def compile_rcl_program(self, program: str):
        with open(self.code_path, 'w', encoding='utf-8') as file:
            print(program, file=file)

        command = ['java', '-jar', self.compiler_path, self.code_path]
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return result.stdout, result.stderr

    # ---------------------------------------------------
    # Python syntax checking
    # ---------------------------------------------------
    def compile_python_program(self, program: str):
        """
        Check Python syntax using compile(), without executing the code.
        After capturing the raw error message, it will be passed
        through a post-processing hook for further enhancement.
        """
        try:
            compile(program, "<generated_code>", "exec")
            return program, ""      
        except SyntaxError as e:
            raw_err = (
                f"SyntaxError: {e.msg} at line {e.lineno}, column {e.offset}\n"
                f">>> {e.text.strip() if e.text else ''}"
            )

            processed_err = self.postprocess_python_error(raw_err)
            return "", processed_err
        except Exception as e:
            raw_err = f"Unexpected error: {str(e)}"
            processed_err = self.postprocess_python_error(raw_err)
            return "", processed_err

    # ---------------------------------------------------
    # Alternative compilation method (to be implemented)
    # ---------------------------------------------------
    def compile_code_alternative(self, program: str):
        """
        Alternative compilation method for special phases.
        TODO: Implement custom compilation logic here.

        Returns:
            tuple: (stdout, stderr) similar to compile_code()
        """
        # Placeholder implementation
        # You can implement custom compilation logic here
        pass

    # ---------------------------------------------------
    # Error post-processing hook (currently empty)
    # ---------------------------------------------------
    def postprocess_python_error(self, err_msg: str) -> str:
        """
        Post-process Python syntax/compile errors.
        This function is intentionally left simple and can be
        overridden or expanded for custom formatting, AST-based
        suggestions, or additional metadata extraction.
        """
        # TODO: future enhancement of error processing
        return err_msg
