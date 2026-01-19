import socket
import os
import requests
from framework import *
HOST = '0.0.0.0'
PORT = 5001

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# Whisper API 配置
WHISPER_API_URL = "https://api.cxhao.com/v1/audio/transcriptions"
WHISPER_API_KEY = "sk-FZwRGbVq8YbbFQVRA4Cf69246367496a9f9363D7746477A2"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"[服务器] 正在监听端口 {PORT} ...")

    while True:
        conn, addr = server.accept()
        print(f"[连接] 来自 {addr}")

        with conn:
            filename = conn.recv(128).decode().strip()
            print(f"[接收] 文件名: {filename}")

            file_path = os.path.join(SAVE_DIR, filename)
            with open(file_path, 'wb') as f:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    f.write(data)
            print(f"[完成] 文件已保存至：{file_path}")

            # ---------- 调用 Whisper ----------
            try:
                headers = {
                    'Authorization': f'Bearer {WHISPER_API_KEY}'
                }
                data = {
                    'model': 'whisper-1',
                    'language': 'en',
                    'response_format': 'text'
                }

                with open(file_path, 'rb') as audio_file:
                    files = [
                        ('file', (filename, audio_file, 'application/octet-stream'))
                    ]
                    print(f"[识别] 正在调用 Whisper 接口...")
                    response = requests.post(WHISPER_API_URL, headers=headers, data=data, files=files)

                if response.status_code == 200:
                    result_text = response.text
                    result_path = os.path.join(SAVE_DIR, f"{filename}.txt")
                    with open(result_path, "w", encoding="utf-8") as f:
                        f.write(result_text)
                    print(f"[✅完成] 文本已保存到：{result_path}")

                    controller = SystemController(model='api') 
                    controller.initialize_system()
                    perceive = ""
                    task = perceive + result_text
                    code = controller.launch_system(task)

                    # ========== 向远程发送 "DONE" 和生成的代码 ==========
                    conn.sendall(b"DONE\n")
                    conn.sendall(code.encode('utf-8'))
                    print("[发送] 已将 DONE 和 code 返回给客户端")

                else:
                    print(f"[❌错误] Whisper API 失败：{response.status_code} {response.text}")

            except Exception as e:
                print(f"[识别失败] 错误信息：{e}")
