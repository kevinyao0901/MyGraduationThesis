"""
Phase execution logger for tracking inputs, outputs, and execution time.
Uses decorator pattern to minimize code intrusion.
"""

import time
import json
import os
from datetime import datetime
from pathlib import Path


class PhaseExecutionLog:
    """Records execution details for a single phase."""

    def __init__(self, phase_name, task):
        self.phase_name = phase_name
        self.task = task
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.input_data = {}
        self.output_data = {}
        self.next_state = None
        self.error = None
        self.timestamp = datetime.now().isoformat()

    def start(self, context_snapshot):
        """Mark the start of phase execution."""
        self.start_time = time.time()
        self.input_data = self._serialize_context(context_snapshot)

    def end(self, context_snapshot, next_state):
        """Mark the end of phase execution."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.output_data = self._serialize_context(context_snapshot)
        self.next_state = next_state

    def set_error(self, error):
        """Record an error that occurred during execution."""
        self.error = str(error)
        self.end_time = time.time()
        if self.start_time:
            self.duration = self.end_time - self.start_time

    def _serialize_context(self, ctx):
        """Serialize context data without truncation."""
        serialized = {}
        for key, value in ctx.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                serialized[key] = value
            elif isinstance(value, (list, dict)):
                # Keep lists and dicts as-is for JSON serialization
                serialized[key] = value
            else:
                # For other types, convert to string representation
                serialized[key] = str(value)
        return serialized

    def to_dict(self):
        """Convert log to dictionary format."""
        return {
            "phase_name": self.phase_name,
            "task": self.task,
            "timestamp": self.timestamp,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": round(self.duration, 3) if self.duration else None,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "next_state": self.next_state,
            "error": self.error
        }


class PhaseLogger:
    """
    Centralized logger for phase executions.
    Singleton pattern to ensure single instance across the system.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.logs = []
        self.output_dir = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.enabled = False  # Default to disabled

    def configure(self, output_dir="results"):
        """Configure logger output directory and enable logging."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = True

    def is_enabled(self):
        """Check if logging is enabled."""
        return self.enabled

    def create_log(self, phase_name, task):
        """Create a new log entry for a phase."""
        return PhaseExecutionLog(phase_name, task)

    def add_log(self, log):
        """Add a completed log entry."""
        self.logs.append(log)

    def save(self, filename=None):
        """Save all logs to JSON file."""
        if not self.output_dir:
            self.configure()

        if filename is None:
            filename = f"phase_execution_log_{self.session_id}.json"

        output_path = self.output_dir / filename

        log_data = {
            "session_id": self.session_id,
            "total_phases": len(self.logs),
            "total_duration": sum(log.duration for log in self.logs if log.duration),
            "phases": [log.to_dict() for log in self.logs]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        print(f"\n[PhaseLogger] Execution log saved to: {output_path}")
        return output_path

    def print_summary(self):
        """Print a summary of all phase executions."""
        if not self.logs:
            print("\n[PhaseLogger] No phase executions recorded.")
            return

        print("\n" + "="*80)
        print("PHASE EXECUTION SUMMARY")
        print("="*80)

        for i, log in enumerate(self.logs, 1):
            status = "ERROR" if log.error else "SUCCESS"
            duration = f"{log.duration:.3f}s" if log.duration else "N/A"

            print(f"\n{i}. {log.phase_name}")
            print(f"   Status: {status}")
            print(f"   Duration: {duration}")
            print(f"   Next State: {log.next_state}")

            if log.error:
                print(f"   Error: {log.error}")

        total_duration = sum(log.duration for log in self.logs if log.duration)
        print(f"\nTotal Execution Time: {total_duration:.3f}s")
        print("="*80 + "\n")

    def reset(self):
        """Reset logger for a new session."""
        self.logs = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")


def log_phase_execution(phase_method):
    """
    Decorator to automatically log phase execution.
    Only logs if PhaseLogger is enabled.

    Usage:
        @log_phase_execution
        def run(self, ctx):
            # phase implementation
            return next_state
    """
    def wrapper(self, ctx):
        logger = PhaseLogger()

        # Skip logging if not enabled
        if not logger.is_enabled():
            return phase_method(self, ctx)

        phase_name = self.__class__.__name__
        task = ctx.get("task", "")

        # Create log entry
        log = logger.create_log(phase_name, task)

        # Take snapshot of input context
        input_snapshot = dict(ctx)
        log.start(input_snapshot)

        try:
            # Execute the phase
            next_state = phase_method(self, ctx)

            # Take snapshot of output context
            output_snapshot = dict(ctx)
            log.end(output_snapshot, next_state)

            # Add log to logger
            logger.add_log(log)

            return next_state

        except Exception as e:
            # Record error
            log.set_error(e)
            logger.add_log(log)
            raise

    return wrapper
