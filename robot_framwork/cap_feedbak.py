import os
import re
import yaml
import time
import subprocess
from datetime import datetime
from LLM import *

DST_PATH = "/home/winter/robot/evaluaiton/monitor/test.py"
RUN_LOG_DIR = "final_result"
PROPATH = "tasks/interaction/task5/CAP.yaml"

def load_first_two_keys_as_task(yaml_path: str) -> str:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    items = list(data.items())[:2]
    parts = []
    for _, v in items:
        if isinstance(v, (dict, list)):
            parts.append(yaml.safe_dump(v, allow_unicode=True).strip())
        else:
            parts.append(str(v).strip())
    return " ".join(parts).strip()

def write_code_to_file(code: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 去掉所有 '''...''' 包裹的内容
    code = re.sub(r"'''(.*?)'''", r"\1", code, flags=re.DOTALL)
    if not code.startswith("#!/usr/bin/env python"):
        code = "#!/usr/bin/env python\n# coding: utf-8\n" + "from ros_cmd_utils_quick import query_user,set_end_position, set_gripper_angles, get_obj_position, grasp, approach, get_operable_objs, send_nav_goal, move_forward, move_backward, turn_left, turn_right\n" + code
        # code = "#!/usr/bin/env python\n# coding: utf-8\n" + code
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        os.chmod(path, 0o755)
    except Exception:
        pass

def run_script_in_conda(path: str, timeout: int = 300):
    """Run script inside conda env `env_isaaclab` and return (rc, stdout, stderr)."""
    try:
        proc = subprocess.run(
            [
                "bash", "-c",
                f"source /home/winter/anaconda3/etc/profile.d/conda.sh && "
                f"conda activate env_isaaclab && "
                f"source /opt/ros/noetic/setup.bash && "
                f"source /home/winter/robot_ws/devel/setup.bash && "
                f"source /home/winter/catkin_ws/devel/setup.bash && "
                f"export PYTHONPATH=/home/winter/robot_ws/devel/lib/python3/dist-packages:"
                f"/home/winter/catkin_ws/devel/lib/python3/dist-packages:$PYTHONPATH && "
                f"python3 {path}"
            ],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        so = e.stdout or ""
        se = (e.stderr or "") + f"\n[runner] Timeout after {timeout}s."
        return -1, so, se
    except Exception as e:
        return -2, "", f"[runner] Failed to run script: {e}"

def save_attempt(run_dir: str, attempt_idx: int, code: str, stdout: str, stderr: str, rc: int):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, f"attempt_{attempt_idx:02d}.py"), "w", encoding="utf-8") as f:
        f.write(code)
    with open(os.path.join(run_dir, f"attempt_{attempt_idx:02d}_stdout.txt"), "w", encoding="utf-8") as f:
        f.write(stdout)
    with open(os.path.join(run_dir, f"attempt_{attempt_idx:02d}_stderr.txt"), "w", encoding="utf-8") as f:
        f.write(stderr)
    with open(os.path.join(run_dir, f"attempt_{attempt_idx:02d}_meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"return_code={rc}\n")

def build_feedback_task(original_task: str, stderr: str, prev_code: str, max_len: int = 6000) -> str:
    """
    Append concise error context to the task for the next LLM iteration.
    We cap the payload to avoid over-long prompts.
    """
    # Trim very long stderr and code
    err_snip = stderr[-3000:] if len(stderr) > 3000 else stderr
    code_snip = prev_code[-2500:] if len(prev_code) > 2500 else prev_code

    feedback = (
        original_task.strip()
        + "\n\n"
        + "### runtime error context\n"
        + "The previous attempt failed at runtime. Here is the error output:\n"
        + "```\n" + err_snip + "\n```\n"
        + "Please fix the code and try again.\n"
        + "Only output the final program in the custom robot control language (no explanations).\n"
        + "Remember to import the libary before the scripts!!"
    )
    # Keep within max_len if necessary
    if len(feedback) > max_len:
        feedback = feedback[:max_len]
    return feedback

def iterative_run(llm: LLM, base_task: str, max_iters: int = 5, timeout: int = 300):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUN_LOG_DIR, f"CAP_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    task = base_task
    for i in range(1, max_iters + 1):
        print(f"\n===== ITERATION {i}/{max_iters} =====")
        code = llm.generate_code(task)
        # code = ""
        print("----- Generated code (preview) -----")
        print(code[:1000] + ("...\n" if len(code) > 1000 else ""))

        write_code_to_file(code, DST_PATH)
        rc, out, err = run_script_in_conda(DST_PATH, timeout=timeout)
        print("===== STDOUT =====")
        print(out)
        print("===== STDERR =====")
        print(err)
        print(f"Return code: {rc}")

        save_attempt(run_dir, i, code, out, err, rc)

        if rc == 0:
            print(f"✅ Success at iteration {i}. Logs saved in: {run_dir}")
            return {"success": True, "iterations": i, "run_dir": run_dir}

        # Build new task with error feedback
        task = build_feedback_task(base_task, err, code)
        time.sleep(1.0)  # small pause between attempts

    print(f"❌ Reached max iterations ({max_iters}) without success. Logs saved in: {run_dir}")
    return {"success": False, "iterations": max_iters, "run_dir": run_dir}

if __name__ == "__main__":
    llm = LLM(
        model_type="oss",
        shots_path="tasks/uncertain/task6/CAP_shots.yaml",
        prompt_path=PROPATH,
        use_shots=True
    )

    base_task = load_first_two_keys_as_task('tasks/uncertain/task6/task.yaml')
    result = iterative_run(llm, base_task, max_iters=6, timeout=300)

    if result["success"]:
        print(f"🎉 Finished in {result['iterations']} iteration(s).")
    else:
        print(f"🧯 Failed after {result['iterations']} iterations. See {result['run_dir']} for details.")
