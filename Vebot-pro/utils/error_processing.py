# utils/error_processing.py

class CompileErrorProcessor:
    """
    Placeholder for future compile error post-processing.
    You can extend this class to parse, clean, categorize,
    or summarize compiler output.
    """

    def __init__(self, config=None):
        self.config = config or {}

    def process_compile_error(self, stderr: str) -> str:
        """
        Main entry for compiler error post-processing.
        Currently returns stderr unchanged.
        Later you may:
          - clean formatting
          - extract meaningful lines
          - categorize error types
          - trim stack traces
          - etc.
        """
        # TODO: implement actual logic
        return stderr or ""
