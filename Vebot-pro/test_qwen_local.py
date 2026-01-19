import sys, traceback

# 把包含 LLM.py 的目录加入路径
sys.path.insert(0, "/mnt/d/Graduation_Thesis/Vebot-pro")

try:
    from LLM import LLMFactory
    model_path = "/mnt/d/Graduation_Thesis/my_model/models--Qwen--Qwen3-14B-AWQ/snapshots/31c69efc29464b6bb0aee1398b5a7b50a99340c3"

    print("Creating local Qwen LLM (this may take a while)...")
    # Force loading on GPU (avoid device_map='auto')
    llm = LLMFactory.create_llm(f"qwen_local@{model_path}", use_gpu=True)

    messages = [{"role": "user", "content": "Who are you?"}]
    print("Running generation...")
    resp = llm.create_chat_completion(messages)
    print("=== MODEL RESPONSE ===")
    print(resp)

except Exception as e:
    print("ERROR during test run:")
    traceback.print_exc()
