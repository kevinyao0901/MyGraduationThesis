import os
import re
import yaml
import subprocess
from datetime import datetime
from LLM import *
import time

# === Config ===






LLM_TYPE = "oss"
SHOT_PATH = "tasks/uncertain/task3/CAP_shots.yaml"
TASK_YAML   = "tasks/uncertain/task3/task.yaml"
CONDA_SH    = "/home/winter/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV   = "env_isaaclab"
TIMEOUT     = 300  # seconds
LOG_DIR     = "final_result"
MAX_ITERS   = 6







DST_PATH    = "/home/winter/robot/evaluaiton/monitor/test.py"
PROMPT_PATH = "tasks/interaction/task5/PRO.yaml"

# -------- Helpers --------
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
        code = "#!/usr/bin/env python\n# coding: utf-8\n" + "from ros_cmd_utils_quick import query_user, set_end_position, set_gripper_angles, get_obj_position, grasp, approach, get_operable_objs, send_nav_goal, move_forward, move_backward, turn_left, turn_right\nfrom assertion_utils import check_end_pose_reachable, check_gripper_angle_valid, check_object_exists, check_navigation_goal_feasible, check_grasp_safe, check_operable_objects_nonempty, check_object_position_valid\n" + code
        # code = "#!/usr/bin/env python\n# coding: utf-8\n" + code
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        os.chmod(path, 0o755)
    except Exception:
        pass

def run_script_in_conda(path: str, timeout: int = 300):
    try:
        proc = subprocess.run(
            [
                "bash", "-c",
                f"source {CONDA_SH} && conda activate {CONDA_ENV} && "
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
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        so = e.stdout or ""
        se = (e.stderr or "") + f"\n[runner] Timeout after {timeout}s."
        return -1, so, se
    except Exception as e:
        return -2, "", f"[runner] Failed to run script: {e}"

def save_attempt(run_dir: str, idx: int, code: str, stdout: str, stderr: str, rc: int, note: str = ""):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, f"attempt_{idx:02d}.py"), "w", encoding="utf-8") as f:
        f.write(code)
    with open(os.path.join(run_dir, f"attempt_{idx:02d}_stdout.txt"), "w", encoding="utf-8") as f:
        f.write(stdout)
    with open(os.path.join(run_dir, f"attempt_{idx:02d}_stderr.txt"), "w", encoding="utf-8") as f:
        f.write(stderr)
    with open(os.path.join(run_dir, f"attempt_{idx:02d}_meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"return_code={rc}\n")
        if note:
            f.write(f"note={note}\n")

def extract_assertion_error(stdout: str, stderr: str) -> str:
    """
    提取与 AssertionError/断言相关的错误片段。
    """
    text = (stderr or "") + "\n" + (stdout or "")
    # 直接包含 AssertionError 的段落（含后续若干行）
    m = re.search(r"(AssertionError.*(?:\n.+)*)", text)
    if m:
        return m.group(1).strip()

    # 行内包含 'assert ' 的失败信息
    lines = [ln for ln in text.splitlines() if "assert " in ln.lower()]
    if lines:
        return "\n".join(lines[-5:]).strip()

    # 回溯尾部（若包含 assert/AssertionError）
    tb = re.search(r"(Traceback \(most recent call last\):[\s\S]+?$)", text.strip())
    if tb:
        tail = "\n".join(tb.group(1).splitlines()[-15:])
        if "assert" in tail.lower() or "assertionerror" in tail:
            return tail.strip()

    return ""

def extract_stderr_tail(stderr: str, max_lines: int = 60) -> str:
    """
    从 STDERR 中提取末尾若干行作为摘要（覆盖非断言的错误/告警）。
    """
    if not stderr:
        return ""
    lines = stderr.strip().splitlines()
    # 取末尾 max_lines 行，避免反馈过长
    tail = "\n".join(lines[-max_lines:])
    return tail.strip()

def build_feedback_task(base_task: str, assert_msg: str, stderr_tail: str, max_len: int = 6000) -> str:
    """
    构建给 LLM 的改进提示：包含断言片段（如有）与 STDERR 摘要（如有）。
    """
    parts = [base_task.strip(), "\n\n### runtime diagnostics"]
    if assert_msg:
        snippet = assert_msg[-3500:] if len(assert_msg) > 3500 else assert_msg
        parts.append("Assertion-related error snippet:\n```\n" + snippet + "\n```")
    if stderr_tail:
        tail = stderr_tail[-3500:] if len(stderr_tail) > 3500 else stderr_tail
        parts.append("STDERR tail (non-assert errors/warnings may appear here):\n```\n" + tail + "\n```")
    parts.append("Please fix the program accordingly and output ONLY the final program in the custom robot control language.")
    prompt = "\n".join(parts)
    return prompt[:max_len]

# -------- Main PRO flow --------
if __name__ == "__main__":
    # 1) Init LLM
    llm = LLM(model_type=LLM_TYPE,shots_path= SHOT_PATH, prompt_path=PROMPT_PATH, use_shots=True)

    # 2) Build task once from YAML
    base_task = load_first_two_keys_as_task(TASK_YAML)

    # 3) Prepare log dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(LOG_DIR, f"PRO_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    # 4) 迭代条件：只要有 STDERR（包含或不包含断言）就继续；无 STDERR 且无断言才停止
    task = base_task
    for i in range(1, MAX_ITERS + 1):
        print(f"\n===== ITERATION {i}/{MAX_ITERS} =====")
        code = llm.generate_code(task)
        print("===== GENERATED CODE (preview) =====")
        print(code[:1000] + ("...\n" if len(code) > 1000 else ""))

        write_code_to_file(code, DST_PATH)
        rc, out, err = run_script_in_conda(DST_PATH, timeout=TIMEOUT)

        print("===== STDOUT =====")
        print(out)
        print("===== STDERR =====")
        print(err)
        print(f"Return code: {rc}")

        # 提取诊断信息

        assert_snip = extract_assertion_error(out, err)
        stderr_tail = extract_stderr_tail(err, max_lines=80)
        has_stderr = bool(err.strip())

        # 保存当前尝试
        note_flags = []
        if assert_snip:
            note_flags.append("assertion_error_detected")
        if has_stderr:
            note_flags.append("stderr_detected")
        note = ",".join(note_flags)
        save_attempt(run_dir, i, code, out, err, rc, note=note)

        # 停止条件：无断言片段 且 无任何 STDERR
        if not assert_snip and not has_stderr:
            if rc == 0:
                print(f"✅ Success with clean run (no STDERR, no assertion). Logs saved to: {run_dir}")
                final_status = "success"
            else:
                print(f"⚠️  No assertion and no STDERR, but return code={rc}. Logs saved to: {run_dir}")
                final_status = "nonzero_rc_no_stderr"
            with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as f:
                f.write(f"final_iteration={i}\n")
                f.write(f"final_return_code={rc}\n")
                f.write(f"final_status={final_status}\n")
            break

        # 只要有 STDERR（无论是否断言），就继续反馈并迭代
        print("🔁 Diagnostics found (STDERR and/or assertion). Refining code with feedback to LLM...")
        if assert_snip:
            with open(os.path.join(run_dir, f"attempt_{i:02d}_assert.txt"), "w", encoding="utf-8") as f:
                f.write(assert_snip)
        if has_stderr:
            with open(os.path.join(run_dir, f"attempt_{i:02d}_stderr_tail.txt"), "w", encoding="utf-8") as f:
                f.write(stderr_tail)

        task = build_feedback_task(base_task, assert_snip, stderr_tail)

        # 达到迭代上限仍有 STDERR → 写入摘要并终止
        if i == MAX_ITERS:
            with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as f:
                f.write(f"final_iteration={i}\n")
                f.write("final_status=max_iters_reached_with_stderr\n")
            print(f"🧯 Reached max iterations while STDERR still present. Logs saved to: {run_dir}")
