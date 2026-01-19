# system_controller.py

from phases import _PHASE_REGISTRY, PhaseBase
from LLM import LLMManager
from robot import Robot
from compiler import Compiler
from utils.phase_logger import PhaseLogger
from pathlib import Path
from datetime import datetime


# system_controller.py

class SystemController:

    def __init__(self, config):
        self.config = config

        # ---- Build manager for multiple LLMs ----
        llm_manager = LLMManager(config)

        compiler = Compiler(config)
        robot = Robot(config)
        self.context = Context(llm_manager, compiler, robot)

        # ---- Phase registry ----
        self.phases = {
            name: cls(config, self.context)
            for name, cls in _PHASE_REGISTRY.items()
        }

        # ---- Configure phase logger ----
        self.enable_logging = config.get("enable_phase_logging", False)
        self.session_dir = None  # 为本次测试创建的目录
        if self.enable_logging:
            self.logger = PhaseLogger()
            self._setup_session_directory(config)

    def _setup_session_directory(self, config):
        """为本次测试运行创建单独的会话目录"""
        base_dir = Path(config.get("log_output_dir", "results"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = base_dir / f"session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # 配置 logger 使用会话目录
        self.logger.configure(str(self.session_dir))

        print(f"\n[Session] Results will be saved to: {self.session_dir}\n")

    # ======================================================
    # Main finite state machine
    # ======================================================
    def launch_system(self, tasks):
        """
        执行一个或多个任务。

        Args:
            tasks: 单个任务字符串或任务列表
        """
        # 统一处理为列表格式
        if isinstance(tasks, str):
            tasks = [tasks]

        total_tasks = len(tasks)
        results = []

        for task_idx, task in enumerate(tasks, 1):
            print(f"\n{'='*80}")
            print(f"Processing Task {task_idx}/{total_tasks}")
            print(f"{'='*80}")
            print(f"Task: {task}\n")

            # 重置 logger 以为每个任务单独记录
            if self.enable_logging:
                if task_idx > 1:
                    self.logger.reset()

            # 重置所有 LLM 实例的消息历史，确保每个 task 独立
            if task_idx > 1:
                for llm_name in self.context.llm_manager.names():
                    llm = self.context.llm_manager.get(llm_name)
                    llm.reset_messages()

            ctx = {"task": task, "task_index": task_idx}
            state = self.config.get("initial_state", "GENERATION")

            try:
                while True:
                    phase = self.phases[state]
                    next_state = phase.run(ctx)

                    if next_state == "DONE":
                        print(f"✅ Task {task_idx} finished successfully.\n")

                        # 保存此任务的日志
                        if self.enable_logging:
                            self.logger.print_summary()
                            log_filename = f"task_{task_idx}.json"
                            self.logger.save(filename=log_filename)

                        results.append({
                            "task_index": task_idx,
                            "task": task,
                            "status": "success",
                            "result": ctx.get("compiled_code")
                        })
                        break

                    state = next_state

            except Exception as e:
                print(f"❌ Task {task_idx} failed: {str(e)}\n")

                # 保存失败任务的日志
                if self.enable_logging:
                    self.logger.print_summary()
                    log_filename = f"task_{task_idx}.json"
                    self.logger.save(filename=log_filename)

                results.append({
                    "task_index": task_idx,
                    "task": task,
                    "status": "failed",
                    "error": str(e)
                })

        # 打印总结
        print(f"\n{'='*80}")
        print(f"All Tasks Completed: {total_tasks} tasks")
        print(f"{'='*80}")
        success_count = sum(1 for r in results if r["status"] == "success")
        failed_count = sum(1 for r in results if r["status"] == "failed")
        print(f"✅ Success: {success_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"{'='*80}\n")

        return results


# ============================================================
# Context object shared among all phases
# ============================================================

class Context:
    def __init__(self, llm_manager, compiler, robot):
        """
        Shared resources for all phases.
        Now uses llm_manager instead of a single llm!
        """
        self.llm_manager = llm_manager   # 多 LLM 管理器
        self.compiler = compiler
        self.robot = robot