import argparse
import yaml
from controller import SystemController
from utils.task_loader import *

def parse_args():
    parser = argparse.ArgumentParser(description="Robot System Controller")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # 读取配置文件
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 初始化系统控制器
    controller = SystemController(config)

    # 从 YAML 加载任务（可能是单个或多个）
    tasks = load_task(config["task_yaml_path"])

    # 启动执行（自动处理单个或多个任务）
    results = controller.launch_system(tasks)
