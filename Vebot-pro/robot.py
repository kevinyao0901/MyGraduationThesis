# robot.py

from abc import ABC, abstractmethod


# ============================================================
# Base Robot Backend (abstract interface)
# ============================================================

class RobotBackend(ABC):
    """Abstract base class for all robot backends."""

    def __init__(self, config):
        # 保存完整 config（即使子类不使用）
        self.config = config

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def send_code(self, program):
        pass

    @abstractmethod
    def run_program(self, program):
        """Return object with `.error` attribute."""
        pass

    @abstractmethod
    def shutdown(self):
        pass



# ============================================================
# Simulation backend
# ============================================================

class SimulationRobotBackend(RobotBackend):
    def __init__(self, config):
        super().__init__(config)  # 保存完整 config

        sim_cfg = config.get("simulation_robot", {})
        self.sim_speed = sim_cfg.get("speed", 1.0)
        self.verbose = sim_cfg.get("verbose", True)

    def initialize(self):
        print(f"[Robot] Initializing simulation (speed={self.sim_speed})")

    def send_code(self, program):
        if self.verbose:
            print("[Robot] (Simulation) Code received:")
            print(program)

    def run_program(self, program):
        print("[Robot] Running program in simulation...")
        class Result:
            error = None
        return Result()

    def shutdown(self):
        print("[Robot] Simulation shutdown.")



# ============================================================
# Real hardware backend
# ============================================================

class RealRobotBackend(RobotBackend):
    def __init__(self, config):
        super().__init__(config)

        real_cfg = config.get("real_robot", {})
        self.ip = real_cfg.get("ip", "127.0.0.1")
        self.port = real_cfg.get("port", 30002)
        self.protocol = real_cfg.get("protocol", "tcp")

    def initialize(self):
        print(f"[Robot] Connecting to real robot {self.ip}:{self.port} ({self.protocol})")

    def send_code(self, program):
        print("[Robot] Sending compiled program to real robot...")
        # TODO: implement real communication (TCP/serial etc.)

    def run_program(self, program):
        print("[Robot] Executing program on real hardware...")
        class Result:
            error = None
        return Result()

    def shutdown(self):
        print("[Robot] Disconnecting from real robot...")


# ============================================================
# Test backend (used for debugging and Phases development)
# ============================================================

class TestRobotBackend(RobotBackend):
    def __init__(self, config):
        super().__init__(config)

        test_cfg = config.get("test_robot", {})
        self.error_pattern = test_cfg.get("error_pattern", None)
        self.counter = 0

    def initialize(self):
        print("[Robot] Test mode initialized (no-op).")

    def send_code(self, program):
        print("[Robot] (Test) Code accepted.")

    def run_program(self, program):
        print("[Robot] (Test) Running mock program...")

        class Result:
            error = None

        # Example: depending on counter, return fake error
        if self.error_pattern:
            self.counter += 1
            if self.counter % 2 == 1:
                Result.error = self.error_pattern

        return Result()

    def shutdown(self):
        print("[Robot] Test mode shutdown.")


# ============================================================
# Factory: choose backend based on config["robot_mode"]
# ============================================================

class RobotFactory:
    BACKENDS = {
        "simulation": SimulationRobotBackend,
        "real": RealRobotBackend,
        "test": TestRobotBackend,
    }

    @staticmethod
    def create_robot(config):
        mode = config.get("robot_mode", "simulation")

        backend_cls = RobotFactory.BACKENDS.get(mode)
        if backend_cls is None:
            raise ValueError(f"Unsupported robot_mode: {mode}")

        return backend_cls(config)


# ============================================================
# Public Robot interface
# ============================================================

class Robot:
    """
    Robot wrapper used by SystemController.
    It selects the correct backend using RobotFactory.
    """

    def __init__(self, config):
        self.config = config
        self.backend = RobotFactory.create_robot(config)

    # Standard interface: just forward
    def initialize(self):
        self.backend.initialize()

    def send_code(self, program):
        self.backend.send_code(program)

    def run_program(self, program):
        return self.backend.run_program(program)

    def shutdown(self):
        self.backend.shutdown()
