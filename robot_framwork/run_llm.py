"""Run LLM.generate_code with multiple underlying model implementations and save outputs.

This script will:
 - load the system prompt and (optionally) shots via the LLM wrapper
 - for each model_type in the chosen list, create an implementation using LLMFactory
   and replace the wrapper's `llm` with it
 - call generate_code(task) and save the returned text (or error) into separate files

Usage (WSL, after activating env_isaaclab):
  python run_llm.py --task "My task" --prompt tasks/uncertain/task1/CAP.yaml --shots tasks/uncertain/task1/CAP_shots.yaml --use-shots

Be aware: some model types (e.g. "api") will issue network requests and use keys
embedded in the repository. Local models require transformers/torch installed.
"""

import argparse
import os
from datetime import datetime
from LLM import LLM, LLMFactory


DEFAULT_MODELS = ["api", "llama", "qwen", "oss", "8b", "32b"]


def run_all_models(task: str, prompt_path: str, shots_path: str, use_shots: bool, models, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for m in models:
        model_type = m
        safe_name = model_type.replace('/', '_')
        out_file = os.path.join(out_dir, f"generated_{safe_name}_{timestamp}.txt")
        print(f"\n== Running model_type={model_type} -> {out_file} ==")

        try:
            # Create wrapper that will load system prompt & shots
            wrapper = LLM(model_type="api", prompt_path=prompt_path, shots_path=shots_path, use_shots=use_shots)

            # Create implementation via factory using default model_name
            try:
                impl = LLMFactory.create_llm(model_type)
            except Exception as e:
                # try passing None to let factory use defaults or rethrow
                try:
                    impl = LLMFactory.create_llm(model_type, model_name=None)
                except Exception:
                    raise

            # Replace and run
            wrapper.llm = impl
            output = wrapper.generate_code(task)

            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(output or "")

            print(f"Saved output to {out_file}")
        except Exception as e:
            err_file = os.path.join(out_dir, f"error_{safe_name}_{timestamp}.txt")
            with open(err_file, 'w', encoding='utf-8') as f:
                f.write(repr(e))
            print(f"Model {model_type} failed; recorded error in {err_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="Move the robot arm to three positions forming a triangle.", help="Task description to send to models")
    p.add_argument("--prompt", type=str, default="tasks/uncertain/task1/CAP.yaml", help="System prompt path")
    p.add_argument("--shots", type=str, default="tasks/uncertain/task1/CAP_shots.yaml", help="Shots path")
    p.add_argument("--use-shots", action="store_true", help="Attach shots to prompts")
    p.add_argument("--models", type=str, default=','.join(DEFAULT_MODELS), help="Comma-separated model types to run, e.g. api,llama,qwen")
    p.add_argument("--out-dir", type=str, default="output", help="Directory to save generated outputs")
    args = p.parse_args()

    models = [x.strip() for x in args.models.split(',') if x.strip()]

    run_all_models(args.task, args.prompt, args.shots, args.use_shots, models, args.out_dir)


if __name__ == '__main__':
    main()
