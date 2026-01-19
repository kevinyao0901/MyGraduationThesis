import subprocess

def split_code(code):
    # 分割代码的第一行，这个后面应该交由编译器完成，而不是在这里完成，目前先暂时用简单的手段进行分割
    lines = code.splitlines()
    if not lines:
        return "", ""  # 如果代码为空，返回两个空字符串
    first_line = lines[0]  # 获取第一行
    remaining_code = "\n".join(lines[1:])  # 剩余代码
    return first_line, remaining_code

class Compiler:
    def __init__(self, compiler_path, code_path):
        self.compiler_path = compiler_path
        self.code_path = code_path

    def compile_code(self, program):
        # 使用分割函数分割代码
        # first_line, remaining_code = split_code(program)
        
        # 编译第一行
        stdout, stderr = self.compile_program(program)

        # 返回编译结果和剩余未编译的代码
        return stdout, stderr#, remaining_code, first_line

    def compile_program(self, program):
        with open(self.code_path, 'w', encoding='utf-8') as file: 
            print(program, file=file)
        command = ['java', '-jar', self.compiler_path, self.code_path]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout, result.stderr

# 示例使用
if __name__ == "__main__":
    demo_path = r"./models/language/demo.txt"
    compiler_path = r"./models/language/RSLLang.jar"
    
    compiler = Compiler(compiler_path, demo_path)  # 使用指定的编译器和代码路径
    code = """approach bottle;
    grasp bottle;
    """
    stdout, stderr, remaining_code, first_line = compiler.compile_code(code)
    
    print("编译输出:", stdout)
    print("错误输出:", stderr)
    print("未编译的代码:", remaining_code)
    print("已经编译的代码:" ,first_line)