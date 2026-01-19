from LLM import *
from compiler import *
from robot import *
import yaml
from datetime import datetime

MAX_ITERATION = 6
ERROR_PROMPT = "Please revise the code as it contains the following errors:"
RUN_LOG_BASE = "final_result"
RUNTIME_LOG_BASE = "final_result"
ERROR_LOG_PATH = "/home/winter/robot/evaluaiton/monitor/error.txt"
PROPA = 'tasks/interaction/task5/RSL_casual.yaml'

def _ensure_dir(p: str):
    """Ensure that the target directory exists."""
    os.makedirs(p, exist_ok=True)

def _new_run_dir(base: str) -> str:
    """Create a new run directory with a timestamp."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, f"rsl_compile{ts}")
    _ensure_dir(run_dir)
    return run_dir

def _save_attempt(run_dir: str, idx: int, code: str, stdout: str, stderr: str, meta: dict):
    """Save all data from one iteration: code, stdout, stderr, and metadata."""
    _ensure_dir(run_dir)
    with open(os.path.join(run_dir, f"attempt_{idx:02d}.py"), "w", encoding="utf-8") as f:
        f.write(code)
    with open(os.path.join(run_dir, f"attempt_{idx:02d}_stdout.txt"), "w", encoding="utf-8") as f:
        f.write(stdout or "")
    with open(os.path.join(run_dir, f"attempt_{idx:02d}_stderr.txt"), "w", encoding="utf-8") as f:
        f.write(stderr or "")
    with open(os.path.join(run_dir, f"attempt_{idx:02d}_meta.txt"), "w", encoding="utf-8") as f:
        for k, v in (meta or {}).items():
            f.write(f"{k}={v}\n")

def _build_feedback_task(base_task: str, stderr: str) -> str:
    """
    Build a feedback task for LLM with the previous compile error appended.
    The error is truncated to avoid extremely long prompts.
    """
    err_snip = stderr[-4000:] if stderr and len(stderr) > 4000 else (stderr or "")
    return (
        base_task.strip()
        + "\n\n### compile error context\n"
        + ERROR_PROMPT + "\n"
        + "```\n" + err_snip + "\n```\n"
        + "Please output ONLY the final program in the custom robot control language."
        + "\"cube\" notin get_operable_objs is wrong, you must write list = get_operable_objs;\n  if  \"cube\" notin list {......}"
        + "patrol_positions[0*2+1] is wrong, you can't use [0*2+1] and you can't use patrol_positions[1]"
    )

def _new_runtime_dir(base: str = RUNTIME_LOG_BASE) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, f"rsl_runtime_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def _read_error_log(path: str = ERROR_LOG_PATH) -> str:
    try:
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

def _save_runtime_attempt(run_dir: str, idx: int, code: str, stdout: str, stderr: str,
                          runtime_err_before: str, meta: dict):
    """
    仅用于“运行时修复”阶段的记录：本轮开始前的 runtime 错误、生成的代码、编译输出。
    不记录编译阶段的反馈尝试。
    """
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, f"attempt_{idx:02d}_runtime_error.txt"), "w", encoding="utf-8") as f:
        f.write(runtime_err_before or "")
    with open(os.path.join(run_dir, f"repaired_code{idx:02d}.py"), "w", encoding="utf-8") as f:
        f.write(code or "")
    with open(os.path.join(run_dir, f"attempt_{idx:02d}_stdout.txt"), "w", encoding="utf-8") as f:
        f.write(stdout or "")
    with open(os.path.join(run_dir, f"attempt_{idx:02d}_stderr.txt"), "w", encoding="utf-8") as f:
        f.write(stderr or "")
    with open(os.path.join(run_dir, f"attempt_{idx:02d}_meta.txt"), "w", encoding="utf-8") as f:
        for k, v in (meta or {}).items():
            f.write(f"{k}={v}\n")

def _build_runtime_feedback_task(base_task: str, runtime_err: str) -> str:
    err_snip = runtime_err[-6000:] if runtime_err and len(runtime_err) > 6000 else (runtime_err or "")
    return (
        # base_task.strip() + 先不加上，不然上下文太长了
        "\n\n### runtime error context\n"
        + "Runtime error occurred after deployment. Please fix the program following these errors:\n"
        + "```\n" + err_snip + "\n```\n"
        + "Please output ONLY the final program in the custom robot control language."
    )

def _save_success_summary(run_dir: str, iterations: int, runtime_attempts: int):
    """
    在 run_dir 中记录成功总结：编译修改次数和运行时修复次数。
    """
    _ensure_dir(run_dir)
    summary_path = os.path.join(run_dir, "success_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== SUCCESS SUMMARY ===\n")
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
        f.write(f"compile_iterations: {iterations}\n")
        f.write(f"runtime_repair_attempts: {runtime_attempts}\n")


class SystemController:
    def __init__(self, model, prompt_path,use_shots,shot_path):
        # 初始化各个模块
        self.robot = Robot()
        self.llm = LLM(
            model_type=model,
            shots_path=shot_path,
            prompt_path=prompt_path,
            use_shots=use_shots,
        )
        self.compiler = Compiler(
            compiler_path="./language/RSL_Compiler.jar",
            code_path="./language/demo.txt"
        )

    def initialize_system(self):
        """初始化系统和各模块"""
        # 需要根据具体机器人定义好修改好robot.py后才能执行
        # self.robot.initialize()

    def handle_task(self, task):
        """
        Try to generate and compile code up to MAX_ITERATION times.
        - Each attempt is logged in a dedicated run folder.
        - If compilation fails, stderr is appended to the next LLM prompt for refinement.
        - Stops immediately if compilation succeeds.
        """
        run_dir = _new_run_dir(RUN_LOG_BASE)
        base_task = task
        iterations = 0
        last_stdout = ""
        last_stderr = ""
        generated_code = ""

        for i in range(1, MAX_ITERATION + 1):
            iterations = i
            if i == 1:
                generated_code = self.llm.generate_code(base_task, error='')
            else:
                feedback_task = _build_feedback_task(base_task, last_stderr) 
                generated_code = self.llm.generate_code(feedback_task, error=last_stderr)

            print(f"\n===== ITERATION {i}/{MAX_ITERATION} =====")
            print("Generated code preview:")
            print((generated_code[:800] + "...\n") if len(generated_code) > 800 else generated_code)

            # Compile the generated code
            stdout, stderr = self.compiler.compile_code(generated_code)
            last_stdout, last_stderr = stdout, stderr

            # Save this attempt to log directory
            _save_attempt(
                run_dir, i, generated_code, stdout, stderr,
                meta={
                    "iteration": i,
                    "timestamp": datetime.now().isoformat(),
                    "has_error": "1" if stderr else "0",
                }
            )

            # If compilation succeeded
            if not stderr:
                print(f"✅ Compilation succeeded (iteration {i}). Log dir: {run_dir}")
                print("Compiler output:", stdout)
                self.robot.send_code(stdout)

                # —— runtime repair loop (UPDATED) ——
                runtime_dir = _new_runtime_dir()
                attempt = 0


                while True:
                    rt_err = _read_error_log(ERROR_LOG_PATH)

                    # 没有运行时错误 → 修复完成
                    if rt_err is None or len(rt_err) < 5:
                        # 记录成功总结到编译日志目录 run_dir
                        _save_success_summary(run_dir, iterations=iterations, runtime_attempts=attempt)
                        print(f"✅ Runtime OK (no errors). Total iterations until success: "
                            f"{iterations} (compile), runtime repairs: {attempt}")
                        return stdout

                    # 有运行时错误才进入修复；超过上限则退出
                    attempt += 1
                    if attempt > MAX_ITERATION:
                        print("🧯 Max runtime repair iterations reached; runtime errors still present.")
                        return stdout

                    print("Runtime error (snippet):")
                    print((rt_err[:800] + "...") if len(rt_err) > 800 else rt_err)

                    # 构造仅用于“运行时修复”的反馈任务；不记录编译阶段反馈
                    feedback_task = _build_runtime_feedback_task(base_task, rt_err)
                    repaired_code = self.llm.generate_code(feedback_task, error=rt_err)

                    # 编译修复后的代码；如果编译失败，则“直接结束整个程序”，不再继续迭代
                    r_stdout, r_stderr = self.compiler.compile_code(repaired_code)

                    # 记录本次运行时修复尝试（重点记录 rt_err）
                    _save_runtime_attempt(
                        runtime_dir, attempt, repaired_code, r_stdout, r_stderr, rt_err,
                        meta={
                            "iteration": attempt,
                            "timestamp": datetime.now().isoformat(),
                            "stage": "runtime_repair",
                            "has_compile_error": "1" if r_stderr else "0",
                        }
                    )

                    if r_stderr:
                        print("❌ Compilation failed during runtime repair. Stopping without further iterations.")
                        return stdout  # 按你的要求：编译错误直接结束整个程序

                    # 编译通过 → 下发修复后的程序，然后回到循环起点再次检查 error.txt
                    self.robot.send_code(r_stdout)


            # If failed, continue to the next iteration
            print("❌ Compilation error:")
            print(stderr)

        # If reached the maximum number of iterations without success
        print(f"🧯 Max iterations ({MAX_ITERATION}) reached with errors. Log dir: {run_dir}")
        print("Last compilation error:")
        print(last_stderr)
        print(f"Total iterations: {iterations}")
        return last_stdout

    
    def launch_system(self,user_task):
        """负责一直运行，直到机器人正常完成任务"""

        return self.handle_task(user_task)

    def shutdown_system(self):
        """关闭系统和所有连接"""
        # self.robot.shutdown()
        print("系统已关闭")

def load_first_two_keys_as_task(yaml_path: str) -> str:
    """
    从 yaml 文件中读取**按文件顺序的前两个键**的值，拼接成单个字符串返回。
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # 取前两个键（PyYAML 在 Python 3.7+ 默认保序）
    items = list(data.items())[:2]
    parts = []
    for k, v in items:
        # 将值转成字符串；如果是列表/字典，按需要转成紧凑文本
        if isinstance(v, (dict, list)):
            parts.append(yaml.safe_dump(v, allow_unicode=True).strip())
        else:
            parts.append(str(v).strip())

    return " ".join(parts).strip()


# 示例用法
if __name__ == "__main__":
    '''
    "llama": LlamaLLM,
    "api":APILLM,
    '''
    controller = SystemController(model='8b',shot_path = "tasks/navigation/task3/framework_shots.yaml", prompt_path=PROPA,use_shots=True)  # 启动模拟模式

    # 初始化系统
    controller.initialize_system()

    # 指定任务描述 YAML 路径（你可以按需修改这个路径）
    task_yaml_path = 'tasks/navigation/task3/task.yaml'

    # 从 YAML 读取前两个键并拼成 task
    task = load_first_two_keys_as_task(task_yaml_path)

    # 处理任务
    controller.launch_system(task)
