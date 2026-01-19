#!/usr/bin/env python3
import sys
import argparse
import yaml
import os
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LLM import LLM


def load_tasks(task_yaml_path):
    with open(task_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    tasks = data.get('task') if isinstance(data, dict) else data
    if isinstance(tasks, str):
        return [tasks]
    if isinstance(tasks, list):
        return tasks
    raise ValueError(f"Unsupported task format in {task_yaml_path}")


def main():
    parser = argparse.ArgumentParser(description='Run LLM to generate RCL code using RCL prompt (no shots)')
    parser.add_argument('--model-path', type=str, required=False,
                        default='/mnt/d/Graduation_Thesis/my_model/models--Qwen--Qwen3-14B-AWQ/snapshots/31c69efc29464b6bb0aee1398b5a7b50a99340c3',
                        help='Local Qwen snapshot path')
    parser.add_argument('--task-yaml', type=str, required=False,
                        default='/mnt/d/Graduation_Thesis/Vebot-pro/tasks/test/iteration_1.yaml',
                        help='Path to task yaml (can contain task: string or task: [list])')
    parser.add_argument('--prompt-path', type=str, required=False,
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tasks', 'prompts', 'rcl', 'rcl_prompt.yaml')),
                        help='Path to RCL prompt yaml (default: tasks/prompts/rcl/rcl_prompt_strict.yaml)')
    parser.add_argument('--use-gpu', action='store_true', help='Force use GPU for model loading')
    parser.add_argument('--max-tokens', type=int, default=2048, help='Maximum new tokens for generation (max_new_tokens)')
    parser.add_argument('--do-sample', action='store_true', help='Enable sampling (do_sample) during generation')
    args = parser.parse_args()

    tasks = load_tasks(args.task_yaml)

    # Create LLM instance configured to use the RCL prompt and no shots
    generation_kwargs = {"max_new_tokens": int(args.max_tokens)}
    if args.do_sample:
        generation_kwargs.update({"do_sample": True})

    llm_obj = LLM(
        model_type="qwen_local",
        model_name=args.model_path,
        prompt_path=args.prompt_path,
        shots_path="",
        use_shots=False,
        api_key=None,
        api_url=None,
        llm_kwargs={"use_gpu": bool(args.use_gpu), "generation_kwargs": generation_kwargs}
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
