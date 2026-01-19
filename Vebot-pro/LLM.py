import json
import re
from abc import ABC, abstractmethod
import requests
import yaml


class BaseLLM(ABC):
    def __init__(self, model_name: str):
        """
        基类的构造函数，确保所有子类都必须提供 model_name
        """
        self.model_name = model_name

    @abstractmethod
    def create_chat_completion(self, messages):
        pass

    @abstractmethod
    def get_model_info(self):
        pass


class APILLM(BaseLLM):
    """
    通用 OpenAI-compat LLM：
    - 只靠 model_name 区分模型（qwen / oss / llama / gpt 等）
    - 通过 OpenAI SDK + 自定义 base_url 调用
    """
    def __init__(self,
                 model_name: str = "gpt-4o",
                 api_url: str = "https://api.zhizengzeng.com/v1/",
                 api_key: str = ""):
        from openai import OpenAI
        super().__init__(model_name=model_name)
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self._client = OpenAI(api_key=self.api_key, base_url=f"{self.api_url}/")

    def create_chat_completion(self, messages):
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        return resp.choices[0].message.content

    def get_model_info(self):
        return f"APILLM model: {self.model_name} @ {self.api_url}"

class OllamaLLM(BaseLLM):
    """
    Ollama LLM implementation using Ollama's native API.
    API endpoint: http://localhost:11434/api/generate
    """
    def __init__(self,
                 model_name: str = "gemma3",
                 api_url: str = "http://localhost:11434",
                 api_key: str = ""):
        super().__init__(model_name=model_name)
        self.api_url = api_url.rstrip("/")
        self.generate_endpoint = f"{self.api_url}/api/generate"

    def create_chat_completion(self, messages):
        """
        Convert messages to Ollama's format and call the generate API.
        Ollama's /api/generate endpoint expects a single prompt string.
        """
        # Convert chat messages to a single prompt
        prompt = self._convert_messages_to_prompt(messages)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False  # Get complete response at once
        }

        try:
            response = requests.post(
                self.generate_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=600  # 5 minutes timeout
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API request failed: {str(e)}")

    def _convert_messages_to_prompt(self, messages):
        """
        Convert OpenAI-style messages to a single prompt string.
        """
        prompt_parts = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")

        # Add final prompt for assistant to respond
        prompt_parts.append("Assistant: ")

        return "\n".join(prompt_parts)

    def get_model_info(self):
        return f"OllamaLLM model: {self.model_name} @ {self.api_url}"


class QwenLocalLLM(BaseLLM):
    """
    Load a local Qwen model saved as safetensors (or similar) using Hugging Face
    transformers. The `model_name` or `model_path` should point to the local
    snapshot folder containing config.json, model.safetensors.index.json, etc.

    This loader tries to use `device_map='auto'` first and falls back to CPU if
    automatic placement isn't available in the environment.
    """
    def __init__(self, model_name: str = None, model_path: str = None, **kwargs):
        # model_name kept for compatibility; model_path is the local path to the snapshot
        super().__init__(model_name=model_name or model_path)
        self.model_path = model_path or model_name

        if not self.model_path:
            raise ValueError("QwenLocalLLM requires a local `model_path` or `model_name` pointing to the snapshot directory.")

        # Import heavy deps lazily to avoid hard dependency at module import time
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception as e:
            raise RuntimeError("Missing transformers/torch. Install transformers and torch to use local Qwen models: pip install transformers torch safetensors") from e

        self.torch = torch

        try:
            # tokenizer may require trust_remote_code for Qwen tokenizers
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

            # Allow caller to request GPU explicitly via kwargs: use_gpu=True or pass device_map directly.
            use_gpu = kwargs.pop("use_gpu", False)
            provided_device_map = kwargs.pop("device_map", None)
            # generation kwargs (passed to model.generate). Example: {"max_new_tokens":1024, "do_sample":False}
            self.generation_kwargs = kwargs.pop("generation_kwargs", {"max_new_tokens": 512})

            # Decide device_map to pass to transformers
            if provided_device_map is not None:
                device_map_to_use = provided_device_map
            elif use_gpu:
                # Require CUDA available when user explicitly requests GPU
                if not torch.cuda.is_available():
                    raise RuntimeError("use_gpu=True requested but no CUDA device is available (torch.cuda.is_available() is False).")
                # Map the whole model to the first CUDA device by default
                device_idx = torch.cuda.current_device()
                device_map_to_use = {"": f"cuda:{device_idx}"}
            else:
                # Default: try automatic placement first
                device_map_to_use = "auto"

            # Attempt to load model with the chosen device_map
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    device_map=device_map_to_use,
                )
            except Exception as load_exc:
                # If user explicitly requested GPU, surface a clear error
                if use_gpu or provided_device_map is not None:
                    raise RuntimeError(
                        f"Failed to load local Qwen model on requested device_map={device_map_to_use}: {load_exc}\n"
                        f"Ensure your environment has a compatible CUDA-enabled torch, and that any required libs for quantized models (autoawq/awq) are installed.")

                # If we attempted 'auto' and it failed, try a conservative CPU fallback only when the model isn't AWQ-locked
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                        device_map={"": "cpu"},
                    )
                except Exception as cpu_exc:
                    # If AWQ or similar requires GPU-only loading, give an actionable message
                    msg = str(cpu_exc)
                    if "AWQ" in msg or "requires auto-awq" in msg or "device_map that contains a CPU" in msg:
                        raise RuntimeError(
                            "The local model appears to be AWQ-quantized and cannot be loaded on CPU. "
                            "Please run with a CUDA-enabled environment and call LLMFactory.create_llm with use_gpu=True, "
                            "and ensure `autoawq`/`awq` and a compatible `transformers`/`torch` are installed.") from cpu_exc
                    else:
                        # Otherwise re-raise the original cpu exception
                        raise RuntimeError(f"Failed to load local Qwen model from {self.model_path}: {cpu_exc}") from cpu_exc

        except Exception as e:
            raise RuntimeError(f"Failed to load local Qwen model from {self.model_path}: {e}")

        # remember the initial GPU preference so we can reload later
        self._initial_use_gpu = use_gpu

    def release_cuda(self):
        """
        Attempt to release CUDA memory used by the loaded model.
        Strategy:
         - If model exists, try to move parameters to CPU (best-effort)
         - Delete the model reference and call torch.cuda.empty_cache()
        Returns True on success, False on failure.
        """
        try:
            if getattr(self, 'model', None) is None:
                return True

            # Try best-effort move to cpu to keep object usable
            try:
                self.model.to('cpu')
            except Exception:
                # not all sharded/accelerate models support .to('cpu')
                pass

            try:
                del self.model
            except Exception:
                self.model = None

            # free cached CUDA memory
            try:
                if hasattr(self.torch, 'cuda') and self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
            except Exception:
                pass

            self.model = None
            return True
        except Exception:
            return False

    def ensure_loaded(self, use_gpu: bool = None):
        """
        Ensure the model is loaded and placed on the requested device.
        If the model was previously released, this will attempt to re-load it
        with the same device_map selection logic used in __init__.
        """
        # default to original preference
        if use_gpu is None:
            use_gpu = getattr(self, '_initial_use_gpu', False)

        # If model already present and on desired device, do nothing
        if getattr(self, 'model', None) is not None:
            try:
                device = next(self.model.parameters()).device
                devstr = str(device)
                if use_gpu and devstr.startswith('cuda'):
                    return
                if not use_gpu and devstr == 'cpu':
                    return
            except Exception:
                pass

        # (Re-)load the model according to requested device
        try:
            from transformers import AutoModelForCausalLM
            if use_gpu:
                if not self.torch.cuda.is_available():
                    raise RuntimeError("Requested GPU but CUDA is not available to reload model.")
                device_idx = self.torch.cuda.current_device()
                device_map_to_use = {"": f"cuda:{device_idx}"}
            else:
                device_map_to_use = {"": "cpu"}

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map=device_map_to_use,
            )
            return
        except Exception as e:
            raise RuntimeError(f"Failed to (re)load local Qwen model from {self.model_path}: {e}") from e

    def create_chat_completion(self, messages):
        """
        Create a completion from chat-style messages. Prefer tokenizer's
        `apply_chat_template` if available (Qwen tokenizer provides it);
        otherwise, fall back to concatenating messages.
        Returns the generated assistant text as a string.
        """
        # Try to use tokenizer's chat helper if present
        try:
            if hasattr(self.tokenizer, 'apply_chat_template'):
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            else:
                # Fallback: simple concatenation similar to OllamaLLM
                prompt_parts = []
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "system":
                        prompt_parts.append(f"System: {content}\n")
                    elif role == "user":
                        prompt_parts.append(f"User: {content}\n")
                    elif role == "assistant":
                        prompt_parts.append(f"Assistant: {content}\n")
                prompt_parts.append("Assistant: ")
                prompt = "\n".join(prompt_parts)
                inputs = self.tokenizer(prompt, return_tensors="pt")

            # Move inputs to the model device
            try:
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception:
                # If moving tensors fails, continue (model may already be on CPU)
                pass

            # Merge provided generation kwargs with a safe default
            gen_kwargs = dict(self.generation_kwargs or {})
            # Ensure inputs are not overwritten
            outputs = self.model.generate(**inputs, **gen_kwargs)

            # decode only the newly generated tokens
            if hasattr(inputs, 'get') and isinstance(inputs, dict) and 'input_ids' in inputs:
                prompt_len = inputs['input_ids'].shape[-1]
                gen = outputs[0][prompt_len:]
                decoded = self.tokenizer.decode(gen, skip_special_tokens=True)
            else:
                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return decoded

        except Exception as e:
            raise RuntimeError(f"Local Qwen model generation failed: {e}") from e

    def get_model_info(self):
        return f"QwenLocal model: {self.model_name} @ {self.model_path}"

class LLMFactory:
    @staticmethod
    def create_llm(model_type, model_name=None, **kwargs):
        import os

        # 支持两种传入方式：
        # 1) model_type 形如 "qwen_local@/absolute/path/to/snapshot"（常见于 CLI）
        # 2) model_type="qwen_local" 且把路径作为 model_name 或通过 kwargs 传入 model_path
        model_path = None
        if isinstance(model_type, str) and "@" in model_type:
            model_type, after = model_type.split("@", 1)
            if after:
                model_path = after

        # 如果没有通过 model_type@path 传递，但 model_name 看起来像路径，也当作 model_path
        if model_path is None and model_name:
            # 在 Windows 上也支持带盘符的路径（含 ':'）
            if any(sep in model_name for sep in (os.sep, '/', '\\')) or (os.name == 'nt' and ':' in model_name):
                model_path = model_name
                model_name = None

        model_map = {
            "api": APILLM,
            "ollama": OllamaLLM,
            "qwen_local": QwenLocalLLM,
            # 以后添加新模型时，只需要在这里添加映射即可
        }

        # 获取对应的 LLM 类
        model_class = model_map.get(model_type)

        if not model_class:
            raise ValueError(f"Unsupported model type: {model_type}")

        # 如果有明确的本地 model_path，则把它传递给模型构造函数
        if model_path is not None:
            return model_class(model_name=model_name or model_path, model_path=model_path, **kwargs)

        # 否则按原有逻辑：如果没有传递 model_name，则使用默认构造；否则把 model_name 传入
        if model_name is None:
            return model_class(**kwargs)
        else:
            return model_class(model_name=model_name, **kwargs)


class LLM:
    def __init__(self,
                 model_type: str,
                 model_name: str,
                 prompt_path: str,
                 shots_path: str,
                 use_shots: bool,
                 api_key: str,
                 api_url: str,
                 llm_kwargs: dict = None):
        """
        Minimal, explicit initializer (no config dict).
        Keep the external behavior of generate_code / generate_code_notp.
        """
        self.model_type = model_type
        self.model_name = model_name
        self.prompt_path = prompt_path
        self.shots_path = shots_path
        self.use_shots = use_shots

        # 初始化底层后端（沿用你已有的工厂）
        # 只传递非 None 和非空字符串的参数
        kwargs = {}
        if api_key is not None and api_key != "":
            kwargs["api_key"] = api_key
        if api_url is not None and api_url != "":
            kwargs["api_url"] = api_url

        # 将额外的 llm_kwargs（比如 use_gpu / device_map）合并并传递给工厂
        if llm_kwargs:
            # only include simple keys (avoid passing unexpected nested configs)
            for k, v in llm_kwargs.items():
                kwargs[k] = v

        self.llm = LLMFactory.create_llm(
            model_type=self.model_type,
            model_name=self.model_name,
            **kwargs
        )

        # Prompt & shots
        self.messages = self._load_system_message()
        self.initial_messages = self.messages.copy()  # 保存初始系统消息用于重置
        self.shots = self._load_shots() if self.use_shots else ""

        # 代码块提取正则
        self.block_pattern = r"```(?:\w*)(?:\s*)([^`]*)```"

    def _load_system_message(self):
        # Load and format system messages from YAML
        with open(self.prompt_path, 'r', encoding='utf-8') as file:
            message = yaml.safe_load(file)
        
        # Directly join the values of all keys in the dictionary in order
        prompt = "".join(message[key] for key in message) + "\n"

        system_message = [{"role": "system", "content": prompt}]
        return system_message

    def _load_shots(self):
        """
        Load prompt examples from YAML to ensure that the generated shots 
        align with the desired format.
        """
        shots = ["Here are some reference examples. You can use these instances to generate code for a custom-designed robot control language:\n"]
        
        with open(self.shots_path, 'r', encoding='utf-8') as file:
            shot_items = yaml.safe_load(file)  # YAML 支持 dict / list
        
        # 遍历 YAML，支持 {key: {task:..., program:...}} 这种结构
        for name, content in shot_items.items():
            shots.append(f"### {name}")
            if isinstance(content, dict):
                for key, value in content.items():
                    shots.append(f"{key}: {value.strip() if isinstance(value, str) else value}")
            else:
                # 如果不是 dict，就直接输出
                shots.append(str(content))
        
        return '\n'.join(shots)


    def add_message(self, role, content):
        """
        添加一条消息到会话中。
        """
        self.messages.append({"role": role, "content": content})

    def reset_messages(self):
        """
        重置消息历史，只保留系统提示词。
        用于在多个 task 之间清除对话历史。
        """
        self.messages = self.initial_messages.copy()

    def generate_code(self, task, error=''):
        # Generate code based on task and model settings
        # 只需要持续调用generate_code就行了，会自动记录之前的对话的

        self.add_message("user", task + " : \n" + error + ("\n" + self.shots + "\n" if self.use_shots else ""))
        print(self.messages)
        output = self.llm.create_chat_completion(messages=self.messages)
        self.add_message("assistant", output)
        print("LLM生成:\n" + output + "\n")
        
        # Extract code block from output
        matches = re.findall(self.block_pattern, output)
        if matches:
            code = matches[0]
        else:
            # raise ValueError("LLM is unable to understand the code generation task and did not generate any code.")
            code = output
        return code
    
    def generate_code_notp(self, task, error=''):
        # Generate code based on task and model settings
        # 只需要持续调用generate_code就行了，会自动记录之前的对话的

        self.add_message("user", task + " : \n" + error + ("\n" + self.shots + "\n" if self.use_shots else ""))
        print(self.messages)
        output = self.llm.create_chat_completion(messages=self.messages)
        self.add_message("assistant", output)
        print("LLM生成:\n" + output + "\n")
        
        # Extract code block from output
        return output

    def save_output(self, local_path, stdout):
        # Save compilation output to file
        with open(local_path, 'w') as file:
            file.write(stdout)

    # ---------------- GPU management helpers ----------------
    def release_gpu(self):
        """
        Request the underlying implementation to free GPU memory if possible.
        Returns True if a release action was performed, False otherwise.
        """
        if hasattr(self.llm, 'release_cuda'):
            return self.llm.release_cuda()
        return False

    def ensure_gpu_loaded(self):
        """
        Ensure the underlying implementation is loaded on GPU (best-effort).
        Returns True on success, False otherwise.
        """
        if hasattr(self.llm, 'ensure_loaded'):
            return self.llm.ensure_loaded(use_gpu=True)
        return False


class LLMManager:
    """
    Create and manage multiple LLM instances from a unified config.

    期望的 config 结构（示例见下）：
    config["llms"] = {
        "default": "main",          # 可选，默认实例名
        "models": {
            "main": {
                "llm_model_type": "api",
                "llm_model_name": "gpt-4o",
                "prompt_path": "prompts/main.yml",
                "shots_path": "prompts/main_shots.yml",
                "use_shots": true,
                "api_key": "...",
                "api_url": "https://api.cxhao.com/v1/chat/completions",
            },
            "verifier": {
                "llm_model_type": "qwen",
                "llm_model_name": "qwen3-32b",
                "prompt_path": "prompts/verifier.yml",
                "use_shots": false,
                "api_key": "...",
            }
        }
    }
    """

    def __init__(self, config: dict):
        llms_cfg = (config or {}).get("llms", {})
        models_cfg = llms_cfg.get("models", {}) or {}
        self.default_name = llms_cfg.get("default") or (next(iter(models_cfg)) if models_cfg else None)

        self._instances = {}
        for name, p in models_cfg.items():
            # 安全取值
            model_type = p.get("llm_model_type")
            model_name = p.get("llm_model_name")
            prompt_path = p.get("prompt_path")
            shots_path = p.get("shots_path")
            use_shots  = p.get("use_shots")

            api_key = p.get("api_key")
            api_url = p.get("api_url")

            self._instances[name] = LLM(
                model_type=model_type,
                model_name=model_name,
                prompt_path=prompt_path,
                shots_path=shots_path,
                use_shots=use_shots,
                api_key=api_key,
                api_url=api_url,
                llm_kwargs=p.get("llm_kwargs", {}),
            )

    # -------- 获取实例 --------
    def get(self, name: str = None) -> LLM:
        if name is None:
            name = self.default_name
        if name is None:
            raise ValueError("No LLM is configured.")
        if name not in self._instances:
            raise KeyError(f"LLM '{name}' not found.")
        return self._instances[name]

    def names(self):
        return list(self._instances.keys())

    def has(self, name: str) -> bool:
        return name in self._instances

    # -------- 便捷路由（可选）--------
    def generate_code(self, task: str, error: str = "", name: str = None):
        return self.get(name).generate_code(task, error)

    def generate_code_notp(self, task: str, error: str = "", name: str = None):
        return self.get(name).generate_code_notp(task, error)
