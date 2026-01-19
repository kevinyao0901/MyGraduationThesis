import os
import shutil
import subprocess
import uuid
import threading
import time

# 可选：只有选择 ROS 通知时才导入
import rospy
from std_msgs.msg import String


class Robot:
    def __init__(self):
        self.conda_env = "env_isaaclab"
        self.conda_python = "/home/winter/anaconda3/envs/env_isaaclab/bin/python"

    def _get_terminal_command(self, script_path):
        script_dir = os.path.dirname(os.path.abspath(script_path))
        bash_cmd = (
            f"cd {script_dir} &&"
            f"source /home/winter/anaconda3/etc/profile.d/conda.sh && "
            f"conda activate {self.conda_env} && "
            f"source /opt/ros/noetic/setup.bash && "
            f"source /home/winter/robot_ws/devel/setup.bash && "
            f"source /home/winter/catkin_ws/devel/setup.bash && "
            f"export PYTHONPATH=/home/winter/robot_ws/devel/lib/python3/dist-packages:"
            f"/home/winter/catkin_ws/devel/lib/python3/dist-packages:$PYTHONPATH && "
            f"sleep 1 && "
            f"{self.conda_python} {script_path}; "
            f"exec bash"
        )
        return ["gnome-terminal", "--", "bash", "-c", bash_cmd]
    
    def _ensure_ros_node(self):
        if not rospy.core.is_initialized():
            rospy.init_node("robot_send_code_waiter", anonymous=True, disable_signals=True)

    def wait_for_ros_done(
        self,
        topic="/task_done",
        expected_msg="done",
        timeout=600.0,
        require_fresh=True,
        fresh_window=0.2,
    ):
        """
        订阅 `topic`，等待一条等于 expected_msg 的 String 消息。
        - expected_msg: 期望的完成字符串，默认 "done"
        - require_fresh: 为 True 时，忽略订阅建立后 fresh_window 秒内收到的第一条消息
                         （防止 publisher 使用 latch=True 导致收到历史消息）
        - fresh_window: 判定“可能是历史消息”的时间窗（秒）
        返回 True=收到新完成信号；False=超时
        """
        self._ensure_ros_node()
        evt = threading.Event()
        start_time = rospy.get_time()
        seen_first = False  # 是否看到第一条可能是历史消息

        def _cb(msg: String):
            nonlocal seen_first
            # 只接受内容匹配的消息
            if msg.data != expected_msg:
                return

            now = rospy.get_time()
            # 如果需要“新消息”，忽略订阅刚建立就立刻到达的第一条（可能是 latched 老消息）
            if require_fresh and (now - start_time) < fresh_window and not seen_first:
                seen_first = True
                rospy.loginfo("[Robot] Ignored a possibly latched old message on /task_done.")
                return

            evt.set()

        sub = rospy.Subscriber(topic, String, _cb, queue_size=10)
        try:
            ok = evt.wait(timeout)
            return ok
        finally:
            sub.unregister()


    def send_code(
        self,
        stdout,
        wait=True,
        ros_topic="/task_done",
        expected_msg="done",
        timeout=600,
        require_fresh=True,
        fresh_window=0.2,
    ):
        src_cfg_path = "src/main/resources/output/cfg_output.json"
        dst_cfg_path = "/home/winter/robot/evaluaiton/monitor/cfg_output.json"
        dst_test_path = "/home/winter/robot/evaluaiton/monitor/test.py"
        server_path = "/home/winter/robot/evaluaiton/monitor/server.py"

        # 生成本次执行的唯一 run_id
        run_id = str(uuid.uuid4())

        # 覆盖 cfg_output.json
        if not os.path.exists(src_cfg_path):
            raise FileNotFoundError(f"[Robot] 源文件不存在: {src_cfg_path}")
        shutil.copyfile(src_cfg_path, dst_cfg_path)
        print(f"[Robot] write {dst_cfg_path}")
        

        # 覆盖 test.py
        with open(dst_test_path, "w", encoding="utf-8") as f:
            f.write(stdout)
        print(f"[Robot] write {dst_test_path}")

        # 打开终端运行 test.py
        subprocess.Popen(self._get_terminal_command(dst_test_path))
        print(f"[Robot] launch {dst_test_path}")

        # 打开终端运行 server.py
        subprocess.Popen(self._get_terminal_command(server_path))
        print(f"[Robot] launch {server_path}")

        if not wait:
            return True

        ok = self.wait_for_ros_done(
            topic=ros_topic,
            expected_msg=expected_msg,
            timeout=timeout,
            require_fresh=require_fresh,
            fresh_window=fresh_window,
        )
        print(f"[Robot] ROS done signal received: {ok}")
        return ok

    def shutdown(self):
        print("[Robot] closed")


if __name__ == "__main__":
    # 测试用例
    test_script = "/home/winter/robot/evaluaiton/monitor/test_terminal.py"
    with open(test_script, "w", encoding="utf-8") as f:
        f.write("print('Hello from conda env_isaaclab!')\n")
        f.write("input('Press Enter to exit...')\n")

    robot = Robot()
    cmd = robot._get_terminal_command(test_script)
    print("即将执行命令:", cmd)
    subprocess.Popen(cmd)
