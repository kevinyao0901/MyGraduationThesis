import yaml

def load_task(yaml_path: str):
    """
    从 YAML 文件中读取任务。

    支持格式：
    - task: "单个任务"  -> 返回 ["单个任务"]
    - task: ["任务1", "任务2"]  -> 返回 ["任务1", "任务2"]

    Returns:
        list: 任务列表（统一返回列表格式）
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    v = data.get("task", "")

    # 如果是列表，直接返回
    if isinstance(v, list):
        return [str(t).strip() for t in v]

    # 如果是字符串或其他类型，包装成列表
    if isinstance(v, dict):
        return [yaml.safe_dump(v, allow_unicode=True).strip()]

    return [str(v).strip()] if v else []
