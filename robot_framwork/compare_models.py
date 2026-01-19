"""Compare outputs from multiple LLM backends / quantized models using the project's LLM wrapper.

This script exercises the `LLM` wrapper in `LLM.py` but lets you swap the underlying
model implementation (via `LLMFactory.create_llm`) so you can compare outputs from:
- remote API models (model_type="api")
- local llama models (model_type="llama") if your environment provides a compatible
  implementation registered in `LLMFactory` (or if you install a local mapping).

Usage examples:
  # simple run with built-in defaults (edit defaults below)
  python compare_models.py --task "Move arm to triangle" --out results.json

Notes:
 - `LLM` requires both prompt_path and shots_path; set them with the flags or edit defaults.
 - This script will NOT modify `LLM.py`; it will instantiate `LLM` objects and replace
   their internal `.llm` implementation with objects created by `LLMFactory.create_llm`.
 - If a given model_type is not supported on your machine (e.g., local quantized model
   backend not installed), that model will be skipped with an error recorded.

"""
import argparse
import json
import time
from typing import List, Dict, Any

from LLM import LLM, LLMFactory


# Optional local HF-based loader for safetensors / local checkpoints.
class LocalHFLLM:
    """Light wrapper implementing create_chat_completion using HuggingFace transformers.

    This attempts to load models saved with safetensors (or regular HF format).
    It's optional and will raise informative ImportError if dependencies are missing.
    """
    def __init__(self, model_name: str, device: str = None, **kwargs):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
        except Exception as e:
            raise ImportError("transformers/torch not available: " + repr(e))

        self.model_name = model_name
        self.device = device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Try to load model with sensible defaults; allow HF to pick safetensors automatically
        model_kwargs = {}
        # prefer fp16 if CUDA available
        if torch.cuda.is_available():
            model_kwargs.update({"torch_dtype": torch.float16})

        # from_pretrained will accept a folder or model id; safetensors files are supported
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Build text-generation pipeline
        pipe_kwargs = {"model": self.model, "tokenizer": self.tokenizer}
        self.pipe = pipeline("text-generation", **pipe_kwargs)

    def create_chat_completion(self, messages):
        # Convert messages into a single prompt string. Keep simple formatting.
        prompt_parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt_parts.append(f"[{role}] {content}")
        prompt = "\n".join(prompt_parts)

        # Generate text
        out = self.pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)
        # pipeline returns list of dicts with 'generated_text'
        return out[0]["generated_text"]


def run_for_model(model_spec: Dict[str, Any], task: str, prompt_path: str, shots_path: str, use_shots: bool):
    """Run the provided task using an LLM wrapper whose underlying implementation
    is created from model_spec via LLMFactory.

    model_spec keys:
      - id: short name
      - type: model_type passed to LLMFactory (e.g., 'api', 'llama', 'oss', 'qwen')
      - model_name: optional model_name/path passed to create_llm
      - kwargs: optional dict of extra kwargs passed to create_llm
    """
    result = {
        "model_id": model_spec.get("id"),
        "model_type": model_spec.get("type"),
        "model_name": model_spec.get("model_name"),
        "ok": False,
        "error": None,
        "output": None,
        "time_sec": None,
    }

    try:
        # Create an LLM wrapper (it will load system prompt and shots)
        llm_wrapper = LLM(model_type="api", prompt_path=prompt_path, shots_path=shots_path, use_shots=use_shots)

        # Build underlying model implementation
        factory_kwargs = model_spec.get("kwargs", {}) or {}
        model_name = model_spec.get("model_name")
        model_type = model_spec.get("type")

        model_impl = None

        # If the spec explicitly requests a local/safetensors model, try LocalHFLLM first
        wants_local = (str(model_type).lower() in ("local", "safetensors")) or (isinstance(model_name, str) and model_name.endswith(".safetensors"))

        if wants_local:
            try:
                model_impl = LocalHFLLM(model_name, **factory_kwargs)
            except Exception as e:
                # fall through to factory attempt and record error later if both fail
                local_err = e

        if model_impl is None:
            try:
                model_impl = LLMFactory.create_llm(model_type, model_name=model_name, **factory_kwargs)
            except Exception as e:
                # If factory fails and we haven't tried LocalHFLLM yet, try it as a fallback
                if not wants_local and isinstance(model_name, str) and model_name.endswith(".safetensors"):
                    try:
                        model_impl = LocalHFLLM(model_name, **factory_kwargs)
                    except Exception as e_local:
                        raise RuntimeError(f"Factory error: {e}; Local HF fallback error: {e_local}")
                else:
                    raise

        # Replace the LLM wrapper's internal model with our implementation
        llm_wrapper.llm = model_impl

        start = time.time()
        output = llm_wrapper.generate_code(task)
        elapsed = time.time() - start

        result.update({"ok": True, "output": output, "time_sec": elapsed})
    except Exception as e:
        result.update({"ok": False, "error": repr(e)})

    return result


def compare_models(models: List[Dict[str, Any]], task: str, prompt_path: str, shots_path: str, use_shots: bool):
    results = []
    for m in models:
        print(f"Running model {m.get('id')} ({m.get('type')})...")
        r = run_for_model(m, task, prompt_path, shots_path, use_shots)
        if r["ok"]:
            print(f"  -> OK in {r['time_sec']:.2f}s, output length={len(r['output'] or '')}")
        else:
            print(f"  -> ERROR: {r['error']}")
        results.append(r)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Move the robot arm to three positions forming a triangle.", help="Task description to send to the models")
    parser.add_argument("--prompt", type=str, default="tasks/mobile_manipulation/task1/CAP.yaml", help="System prompt path (prompt_path)")
    parser.add_argument("--shots", type=str, default="tasks/mobile_manipulation/task1/CAP_shots.yaml", help="Few-shot examples path (shots_path)")
    parser.add_argument("--use-shots", action="store_true", help="Whether to attach shots to the prompt")
    parser.add_argument("--out", type=str, default="compare_results.json", help="Output JSON file to save results")
    args = parser.parse_args()

    # Define model specs to compare. Edit this list to include your quantized model paths/types.
    # Examples:
    #  - API model
    #  - local llama pointing to a quantized gguf/ggml file (ensure your LLMFactory supports 'llama' mapping)
    models = [
        {
            "id": "api_gpt4o",
            "type": "api",
            "model_name": "gpt-4o",
            "kwargs": {}
        },
        # Add your local quantized model here. Example:
        # {
        #     "id": "llama_q4",
        #     "type": "llama",
        #     "model_name": "/path/to/your/model-gguf-q4_0.gguf",
        #     "kwargs": {"n_ctx": 2048}
        # },
        # {
        #     "id": "local_example_safetensors",
        #     "type": "safetensors",
        #     "model_name": "/home/kevin/models/example_model_for_satetensors/",
        #     "kwargs": {}
        # },
    ]

    results = compare_models(models, args.task, args.prompt, args.shots, args.use_shots)

    # Save results
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"task": args.task, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {args.out}")


if __name__ == '__main__':
    main()
