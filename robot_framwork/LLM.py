import json
import os
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


# class LlamaLLM(BaseLLM):
#     def __init__(self, model_name = "./llama/gemma-2-9b-it-Q6_K.gguf", n_ctx=4096, chat_format="chatml", n_gpu_layers=-1):
#         from llama_cpp import Llama
#         self.model = Llama(model_path=model_name, n_ctx=n_ctx, chat_format=chat_format, n_gpu_layers=n_gpu_layers)

#     def create_chat_completion(self, messages):
#         return self.model.create_chat_completion(messages=messages)['choices'][0]['message']['content']

#     def get_model_info(self):
#         return f"Llama model at {self.model.model_name}"
    


class APILLM(BaseLLM):
    def __init__(self, model_name="gpt-4o", api_url="https://api.cxhao.com/v1/chat/completions", api_key="sk-FZwRGbVq8YbbFQVRA4Cf69246367496a9f9363D7746477A2"):
        """
        APILLM class to interact with an external API.
        :param model_name: The name of the model (default: "gpt-4o").
        :param api_url: The URL of the API endpoint.
        :param api_key: The API key for authentication.
        """
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key

    def create_chat_completion(self, messages):
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }
        
        payload = json.dumps({
            "model": self.model_name,
            "messages": messages
        })

        response = requests.post(self.api_url, headers=headers, data=payload)

        # Check if the response is successful
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            response.raise_for_status()

    def get_model_info(self):
        return f"APILLM model: {self.model_name}"
    
class LlamaLLM(BaseLLM):
    def __init__(self, model_name="llama-3.2-11b-vision-instruct", api_url="https://api.cxhao.com/v1/chat/completions", api_key="sk-FZwRGbVq8YbbFQVRA4Cf69246367496a9f9363D7746477A2"):
        """
        APILLM class to interact with an external API.
        :param model_name: The name of the model (default: "gpt-4o").
        :param api_url: The URL of the API endpoint.
        :param api_key: The API key for authentication.
        """
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key

    def create_chat_completion(self, messages):
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }
        
        payload = json.dumps({
            "model": self.model_name,
            "messages": messages
        })

        response = requests.post(self.api_url, headers=headers, data=payload)

        # Check if the response is successful
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            response.raise_for_status()

    def get_model_info(self):
        return f"APILLM model: {self.model_name}"
    
class QwenLLM(BaseLLM):
    # def __init__(self, model_name="qwen3-14b", api_url="https://api.cxhao.com/v1/chat/completions", api_key="sk-FZwRGbVq8YbbFQVRA4Cf69246367496a9f9363D7746477A2"):
    #     """
    #     APILLM class to interact with an external API.
    #     :param model_name: The name of the model (default: "gpt-4o").
    #     :param api_url: The URL of the API endpoint.
    #     :param api_key: The API key for authentication.
    #     """
    #     self.model_name = model_name
    #     self.api_url = api_url
    #     self.api_key = api_key

    # def create_chat_completion(self, messages):
    #     headers = {
    #         'Accept': 'application/json',
    #         'Authorization': f'Bearer {self.api_key}',
    #         'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    #         'Content-Type': 'application/json'
    #     }
        
    #     payload = json.dumps({
    #         "model": self.model_name,
    #         "messages": messages
    #     })

    #     response = requests.post(self.api_url, headers=headers, data=payload)

    #     # Check if the response is successful
    #     if response.status_code == 200:
    #         result = response.json()
    #         return result['choices'][0]['message']['content']
    #     else:
    #         response.raise_for_status()

    # def get_model_info(self):
    #     return f"APILLM model: {self.model_name}"
    def __init__(self,
                 model_name="qwen3-14b",
                 base_url=None,
                 api_key=None):
        """
        OSSLLM：接口与 APILLM 完全一致（create_chat_completion / get_model_info），
        但内部用 OpenAI SDK + 自定义 base_url 调用 gpt-oss-20b。
        """
        from openai import OpenAI
        self.model_name = model_name
        # prefer explicit api_key, otherwise read OPENAI_API_KEY from env
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Please set environment variable OPENAI_API_KEY or pass api_key to the constructor.")

        # base_url optional: if provided, use it (allow custom deployments); otherwise let SDK use default
        self.base_url = base_url.rstrip("/") if base_url else None
        if self.base_url:
            self._client = OpenAI(api_key=self.api_key, base_url=f"{self.base_url}/")
        else:
            self._client = OpenAI(api_key=self.api_key)

    def create_chat_completion(self, messages):
        """
        messages: 直接传入 OpenAI Chat 格式的消息数组，如：
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]
        """
        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
        except Exception as e:
            # SDK/network/auth error — raise with underlying message
            raise RuntimeError(f"OSS/Qwen API call failed: {e}") from e

        # 与 APILLM 保持一致：返回生成文本（第一个 choice）
        try:
            return resp.choices[0].message.content
        except Exception as e:
            # 尝试提取有用的调试信息，避免 NoneType 导致的难以理解的 TypeError
            try:
                resp_dump = getattr(resp, '__dict__', repr(resp))
            except Exception:
                resp_dump = repr(resp)
            raise RuntimeError(f"Unexpected response structure from OSS/Qwen API: {resp_dump}") from e

    def get_model_info(self):
        return f"OSSLLM model: {self.model_name} @ {self.base_url}"
    
class OSSLLM(BaseLLM):
    def __init__(self,
                 model_name="gpt-oss-20b",
                 base_url=None,
                 api_key=None):
        """
        OSSLLM：接口与 APILLM 完全一致（create_chat_completion / get_model_info），
        但内部用 OpenAI SDK + 自定义 base_url 调用 gpt-oss-20b。
        """
        from openai import OpenAI
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Please set environment variable OPENAI_API_KEY or pass api_key to the constructor.")
        self.base_url = base_url.rstrip("/") if base_url else None
        if self.base_url:
            self._client = OpenAI(api_key=self.api_key, base_url=f"{self.base_url}/")
        else:
            self._client = OpenAI(api_key=self.api_key)

    def create_chat_completion(self, messages):
        """
        messages: 直接传入 OpenAI Chat 格式的消息数组，如：
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]
        """
        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
        except Exception as e:
            raise RuntimeError(f"OSSLLM API call failed: {e}") from e

        try:
            return resp.choices[0].message.content
        except Exception as e:
            try:
                resp_dump = getattr(resp, '__dict__', repr(resp))
            except Exception:
                resp_dump = repr(resp)
            raise RuntimeError(f"Unexpected response structure from OSSLLM API: {resp_dump}") from e

    def get_model_info(self):
        return f"OSSLLM model: {self.model_name} @ {self.base_url}"

class qwen8bLLM(BaseLLM):
    def __init__(self,
                 model_name="qwen3-8b",
                 base_url=None,
                 api_key=None):
        """
        OSSLLM：接口与 APILLM 完全一致（create_chat_completion / get_model_info），
        但内部用 OpenAI SDK + 自定义 base_url 调用 qwen3-8b。
        """
        from openai import OpenAI
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Please set environment variable OPENAI_API_KEY or pass api_key to the constructor.")
        self.base_url = base_url.rstrip("/") if base_url else None
        if self.base_url:
            self._client = OpenAI(api_key=self.api_key, base_url=f"{self.base_url}/")
        else:
            self._client = OpenAI(api_key=self.api_key)

    def create_chat_completion(self, messages):
        """
        messages: 直接传入 OpenAI Chat 格式的消息数组，如：
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]
        """
        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
        except Exception as e:
            raise RuntimeError(f"qwen8bLLM API call failed: {e}") from e

        try:
            return resp.choices[0].message.content
        except Exception as e:
            try:
                resp_dump = getattr(resp, '__dict__', repr(resp))
            except Exception:
                resp_dump = repr(resp)
            raise RuntimeError(f"Unexpected response structure from qwen8bLLM API: {resp_dump}") from e

    def get_model_info(self):
        return f"OSSLLM model: {self.model_name} @ {self.base_url}"

class qwen32bLLM(BaseLLM):
    def __init__(self,
                 model_name="qwen3-32b",
                 base_url=None,
                 api_key=None):
        """
        OSSLLM：接口与 APILLM 完全一致（create_chat_completion / get_model_info），
        但内部用 OpenAI SDK + 自定义 base_url 调用 qwen3-32b。
        """
        from openai import OpenAI
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Please set environment variable OPENAI_API_KEY or pass api_key to the constructor.")
        self.base_url = base_url.rstrip("/") if base_url else None
        if self.base_url:
            self._client = OpenAI(api_key=self.api_key, base_url=f"{self.base_url}/")
        else:
            self._client = OpenAI(api_key=self.api_key)

    def create_chat_completion(self, messages):
        """
        messages: 直接传入 OpenAI Chat 格式的消息数组，如：
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]
        """
        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
        except Exception as e:
            raise RuntimeError(f"qwen32bLLM API call failed: {e}") from e

        try:
            return resp.choices[0].message.content
        except Exception as e:
            try:
                resp_dump = getattr(resp, '__dict__', repr(resp))
            except Exception:
                resp_dump = repr(resp)
            raise RuntimeError(f"Unexpected response structure from qwen32bLLM API: {resp_dump}") from e

    def get_model_info(self):
        return f"OSSLLM model: {self.model_name} @ {self.base_url}"

class LLMFactory:
    @staticmethod
    def create_llm(model_type, model_name=None, **kwargs):
        model_map = {
            "api":APILLM,
            "llama":LlamaLLM,
            "oss": OSSLLM,
            "qwen": QwenLLM,
            "8b": qwen8bLLM,
            "32b": qwen32bLLM
            # 以后添加新模型时，只需要在这里添加映射即可
        }

        # 获取对应的 LLM 类
        model_class = model_map.get(model_type)

        if not model_class:
            raise ValueError(f"Unsupported model type: {model_type}")

        # 如果没有传递 model_name，则使用默认值
        if model_name is None:
            # 创建模型时传递默认值
            return model_class(**kwargs)
        else:
            # 使用传递的 model_name
            return model_class(model_name=model_name, **kwargs)


class LLM:
    def __init__(self, model_type, prompt_path,shots_path, model_name=None,demo_path="./models/language/demo.txt", compiler_path="./models/language/RSLLang.jar", 
                 use_shots=False):
        self.model_type = model_type
        self.model_name = model_name
        self.demo_path = demo_path
        self.compiler_path = compiler_path
        self.prompt_path = prompt_path
        self.shots_path = shots_path
        self.use_shots = use_shots  # Add use_shots flag
        self.shots = self._load_shots() if self.use_shots else ""
        self.messages = self._load_system_message()
        self.pattern = r"```(?:\w*)(?:\s*)([^`]*)```"
        
        # Initialize Llama model
        self.llm = LLMFactory.create_llm(model_type=self.model_type, model_name=self.model_name)

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

    def generate_code(self, task, error=''):
        # Generate code based on task and model settings
        # 只需要持续调用generate_code就行了，会自动记录之前的对话的

        self.add_message("user", task + " : \n" + error + ("\n" + self.shots + "\n" if self.use_shots else ""))
        print(self.messages)
        output = self.llm.create_chat_completion(messages=self.messages)
        self.add_message("assistant", output)
        print("LLM生成:\n" + output + "\n")
        
        # Extract code block from output
        matches = re.findall(self.pattern, output)
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


# Example of how to use the LLM class
if __name__ == '__main__':
    llm = LLM(
        model_type="api",prompt_path="prompts/CAP.json",use_shots=False
    )
    task = "The robot arm sequentially moves its end-effector to three positions in space to form a triangle and sets the gripper to different angles at each point."
    local_path = "./output/result.txt"
    code = llm.generate_code(task)
    print(code)
    