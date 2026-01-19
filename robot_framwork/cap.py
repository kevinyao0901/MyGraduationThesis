import os
import yaml
import subprocess
from datetime import datetime
from LLM import *

# === Config ===
DST_PATH = "/home/winter/robot/evaluaiton/monitor/test.py"
PROMPT_PATH = "tasks/object_manipulation/task3/CAP.yaml"
TASK_YAML = "tasks/object_manipulation/task3/task.yaml"
CONDA_SH = "/home/winter/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV = "env_isaaclab"
TIMEOUT = 300  # seconds
LOG_DIR = "./cap_log"

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
    if not code.startswith("#!/usr/bin/env python"):
        code = "#!/usr/bin/env python\n# coding: utf-8\n" + code
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
                f"source {CONDA_SH} && conda activate {CONDA_ENV} && python3 {path}"
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

def save_log(run_dir: str, code: str, stdout: str, stderr: str, rc: int):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "code.py"), "w", encoding="utf-8") as f:
        f.write(code)
    with open(os.path.join(run_dir, "stdout.txt"), "w", encoding="utf-8") as f:
        f.write(stdout)
    with open(os.path.join(run_dir, "stderr.txt"), "w", encoding="utf-8") as f:
        f.write(stderr)
    with open(os.path.join(run_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"return_code={rc}\n")

if __name__ == "__main__":
    # 1) Init LLM
    llm = LLM(model_type="api", prompt_path=PROMPT_PATH, use_shots=False)

    # 2) Build single-shot task from YAML
    task = load_first_two_keys_as_task(TASK_YAML)

    # 3) Generate code once
    code = llm.generate_code(task)
    print("===== GENERATED CODE =====")
    print(code)

    # 4) Write and run once
    write_code_to_file(code, DST_PATH)
    rc, out, err = run_script_in_conda(DST_PATH, timeout=TIMEOUT)

    print("===== STDOUT =====")
    print(out)
    print("===== STDERR =====")
    print(err)
    print(f"Return code: {rc}")

    # 5) Save logs (always)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(LOG_DIR, f"run_{ts}")
    save_log(run_dir, code, out, err, rc)

    if rc != 0:
        print(f"❌ Error logged to: {run_dir}")
    else:
        print(f"✅ Script finished successfully. Log saved to: {run_dir}")
