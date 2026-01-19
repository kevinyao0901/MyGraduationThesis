import sys
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, "/mnt/d/Graduation_Thesis/Vebot-pro")

from LLM import LLM


def load_tasks(task_yaml_path):
    with open(task_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    tasks = data.get('task') if isinstance(data, dict) else data
    # Normalize to list
    if isinstance(tasks, str):
        return [tasks]
    if isinstance(tasks, list):
        return tasks
    raise ValueError(f"Unsupported task format in {task_yaml_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=False,
                        default='/mnt/d/Graduation_Thesis/my_model/models--Qwen--Qwen3-14B-AWQ/snapshots/31c69efc29464b6bb0aee1398b5a7b50a99340c3',
                        help='Local Qwen snapshot path')
    parser.add_argument('--task-yaml', type=str, required=False,
                        default='/mnt/d/Graduation_Thesis/robot_framwork/tasks/interaction/task1/CAP.yaml',
                        help='Path to task yaml (can contain task: string or task: [list])')
    parser.add_argument('--prompt-path', type=str, required=False,
                        default='/mnt/d/Graduation_Thesis/Vebot-pro/tasks/prompts/pipeline/generation_prompt.yaml')
    parser.add_argument('--shots-path', type=str, required=False,
                        default='/mnt/d/Graduation_Thesis/robot_framwork/tasks/interaction/task1/CAP_shots.yaml')
    parser.add_argument('--use-gpu', action='store_true', help='Force use GPU for model loading')

    args = parser.parse_args()

    model_path = args.model_path
    prompt_path = args.prompt_path
    shots_path = args.shots_path

    tasks = load_tasks(args.task_yaml)

    llm_obj = LLM(
        model_type="qwen_local",
        model_name=model_path,
        prompt_path=prompt_path,
        shots_path=shots_path,
        use_shots=False,
        api_key=None,
        api_url=None,
        llm_kwargs={"use_gpu": bool(args.use_gpu)}
    )

    for idx, task in enumerate(tasks, start=1):
        print(f"\n--- Task {idx}/{len(tasks)} ---")
        try:
            output = llm_obj.generate_code(task)
            print(output)
        except Exception as e:
            print(f"Error generating for task {idx}: {e}")


if __name__ == '__main__':
    main()