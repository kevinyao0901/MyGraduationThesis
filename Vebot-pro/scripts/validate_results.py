#!/usr/bin/env python3
"""
Validate generated Python code files found in the `results/` folder.

This script looks for files matching a pattern (default: iteration_*.txt),
extracts the generated Python program starting from the required header
"# Python 2D robot control script. Start with:", and uses the project's
`Compiler` (Python mode) to check syntax (via compile()).

It prints a short summary to stdout and writes per-file reports under
`results/validation_reports/` when `--write-reports` is enabled.
"""
import argparse
import glob
import os
import re
import sys


def load_compiler():
    # Make sure the repository root (Vebot-pro) is importable
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from compiler import Compiler
    except Exception as e:
        raise ImportError(f"Failed to import Compiler from {repo_root}: {e}")

    return Compiler


HEADER_MARKER = "# Python 2D robot control script. Start with:"


RCL_KNOWN_TOKENS = {
    'forward', 'backward', 'turnright', 'turnleft', 'goto', 'approach', 'grasp',
    'release', 'slam', 'say', 'get_operable_objs', 'get_obj_position', 'query_user',
    'set_end', 'set_grip', 'if', 'else', 'for', 'while'
}


def extract_program_from_file(text: str, mode: str = 'python') -> str:
    """Extract program text from an LLM output blob.

    Modes:
      - 'python': look for the Python header marker and return the tail starting
        from its last occurrence (existing behavior).
      - 'rcl': try several heuristics to extract RCL source:
          1) code fences ``` or ```rcl
          2) the longest contiguous block of lines containing at least one
             RCL-known token or a semicolon (typical of the language)

    Returns an empty string when extraction fails.
    """
    if mode == 'python':
        idx = text.rfind(HEADER_MARKER)
        if idx == -1:
            return ""
        return text[idx:]

    # mode == 'rcl'
    # 1) try fenced code blocks first
    # prefer the last fenced code block if multiple exist
    matches = list(re.finditer(r"```(?:rcl)?\n(.*?)\n```", text, flags=re.S | re.I))
    if matches:
        return matches[-1].group(1).strip()

    # 2) scan for the longest contiguous block where lines contain known tokens or ';'
    lines = text.splitlines()

    blocks = []  # list of (start, end) index pairs (inclusive start, exclusive end)
    cur_start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            # blank line breaks a code block
            if cur_start is not None:
                blocks.append((cur_start, i))
                cur_start = None
            continue

        # consider line part of RCL-like block if it contains a semicolon or any known token
        has_semicolon = ';' in s
        has_token = any(tok in s.split() for tok in RCL_KNOWN_TOKENS)
        # also allow lines that look like commands starting with a token (e.g., "grasp \"cube\";")
        starts_with_token = bool(re.match(r"^\s*(?:[a-z_][a-z0-9_]*)\b", s)) and s.split()[0] in RCL_KNOWN_TOKENS

        if has_semicolon or has_token or starts_with_token:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                blocks.append((cur_start, i))
                cur_start = None

    if cur_start is not None:
        blocks.append((cur_start, len(lines)))

    if not blocks:
        return ""

    # pick the block closest to the end of file (prefer last few lines)
    best = max(blocks, key=lambda t: t[1])
    start, end = best
    snippet = "\n".join(lines[start:end]).strip()
    # require a minimal confidence: presence of at least one semicolon or known token
    if ';' not in snippet and not any(tok in snippet.split() for tok in RCL_KNOWN_TOKENS):
        return ""
    return snippet


def validate_file(compiler_cls, file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    program = extract_program_from_file(content)
    if not program:
        return False, "Header marker not found; could not extract program."

    # Instantiate Compiler in Python-mode
    compiler = compiler_cls({"use_rcl": False})
    _, err = compiler.compile_python_program(program)

    if err:
        return False, err
    return True, "OK"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default=None,
                   help='Directory containing generated result files (defaults to ../results/python_results or ../results/rcl_results when --rcl is used)')
    p.add_argument('--pattern', default='iteration_*.txt', help='Glob pattern to match result files')
    p.add_argument('--write-reports', action='store_true', help='Write per-file report files under results/validation_reports')
    p.add_argument('--rcl', action='store_true', help='Run RCL-mode compilation (requires Java and an RCL compiler jar)')
    p.add_argument('--compiler-jar', dest='compiler_jar', help='Path to RCL compiler jar (default: models/language/RCL_Compiler.jar)')
    p.add_argument('--code-path', dest='code_path', help='Temporary file path to write generated code for RCL compiler (default: models/language/generated_rcl.txt)')
    args = p.parse_args()

    # choose sensible defaults depending on mode
    if args.results_dir:
        results_dir = os.path.abspath(args.results_dir)
    else:
        if args.rcl:
            results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'rcl_results'))
        else:
            results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'python_results'))
    pattern = os.path.join(results_dir, args.pattern)
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found matching: {pattern}")
        return 2

    Compiler = load_compiler()

    report_dir = os.path.join(results_dir, 'validation_reports')
    if args.write_reports and not os.path.exists(report_dir):
        os.makedirs(report_dir, exist_ok=True)

    summary = []
    for fpath in files:
        if args.rcl:
            # RCL mode: need compiler_path and code_path
            compiler_path = args.compiler_jar or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'language', 'RCL_Compiler.jar')
            code_path = args.code_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'language', 'generated_rcl.txt')

            # Instantiate Compiler in RCL-mode
            compiler = Compiler({"use_rcl": True, "compiler_path": compiler_path, "code_path": code_path})

            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Try to extract RCL program via heuristics
            program = extract_program_from_file(content, mode='rcl')
            if not program:
                # As a fallback, detect if file contains the Python header and use Python check
                py_prog = extract_program_from_file(content, mode='python')
                if py_prog:
                    py_compiler = Compiler({"use_rcl": False})
                    _, err = py_compiler.compile_python_program(py_prog)
                    if err:
                        ok, info = False, f"(Python check) " + err
                    else:
                        ok, info = True, "OK (checked as Python, skipped RCL)"
                        program = py_prog
                else:
                    ok, info = False, "Could not extract RCL program from file (no fenced block or recognizable commands)."
            else:
                # Have an RCL program; run the RCL compiler
                stdout, stderr = compiler.compile_rcl_program(program)
                if stderr and stderr.strip():
                    ok, info = False, stderr
                else:
                    ok, info = True, stdout or "OK"
        else:
            ok, info = validate_file(Compiler, fpath)
        name = os.path.basename(fpath)
        summary.append((name, ok, info))
        status = 'PASS' if ok else 'FAIL'
        print(f"{name}: {status}")
        if not ok:
            # Print a short excerpt of the error
            print(info)

        if args.write_reports:
            rpt_path = os.path.join(report_dir, name + '.report.txt')
            with open(rpt_path, 'w', encoding='utf-8') as rpt:
                rpt.write(f"File: {fpath}\n")
                rpt.write(f"Result: {status}\n\n")
                if ok:
                    rpt.write("OK\n\n")
                else:
                    rpt.write("ERROR:\n")
                    rpt.write(info + "\n\n")

                # include the extracted program when available for easier debugging
                try:
                    with open(fpath, 'r', encoding='utf-8') as fr:
                        content = fr.read()
                    extracted = extract_program_from_file(content, mode='rcl') or extract_program_from_file(content, mode='python')
                    rpt.write("--- Extracted program (if any) ---\n")
                    if extracted:
                        rpt.write(extracted + "\n")
                    else:
                        rpt.write("<none>\n")
                except Exception:
                    rpt.write("<could not include extracted program>\n")

    # Summary
    total = len(summary)
    passed = sum(1 for _n, ok, _i in summary if ok)
    failed = total - passed
    print('\nValidation summary:')
    print(f'  Total: {total}, Passed: {passed}, Failed: {failed}')

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
