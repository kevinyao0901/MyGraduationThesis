# phases.py

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from utils.error_processing import CompileErrorProcessor
from utils.phase_logger import log_phase_execution
# optional: dynamic creation of LLM instances (for local Qwen use)
try:
    from LLM import LLM
except Exception:
    LLM = None


def _create_local_llm_from_cfg(cfg):
    """Helper: create an LLM(...) instance for local Qwen using the provided cfg dict.

    Raises RuntimeError if LLM class is not importable.
    """
    if LLM is None:
        raise RuntimeError("Local Qwen support requested but LLM class is not importable.")
    gen_kwargs = cfg.get("generation_kwargs") or {}
    return LLM(
        model_type="qwen_local",
        model_name=cfg.get("model_path"),
        prompt_path=cfg.get("prompt_path"),
        shots_path=cfg.get("shots_path", ""),
        use_shots=cfg.get("use_shots", False),
        api_key=None,
        api_url=None,
        llm_kwargs={"use_gpu": bool(cfg.get("use_gpu", False)), "generation_kwargs": gen_kwargs}
    )

_PHASE_REGISTRY = {}

def register_phase(name):
    """Decorator to register a Phase class."""
    def decorator(cls):
        _PHASE_REGISTRY[name] = cls
        return cls
    return decorator


class PhaseBase:
    def __init__(self, config, context):
        self.config = config
        self.context = context

    def run(self, ctx):
        raise NotImplementedError()

    def _maybe_release_llm(self):
        """
        If this phase has a local_llm and the config requested GPU usage,
        attempt to ask the LLM wrapper to release GPU memory. Best-effort only.
        """
        try:
            # prefer phase-local attribute
            llm_wrapper = getattr(self, 'local_llm', None)
            cfg = getattr(self, 'local_qwen_cfg', None)
            if llm_wrapper and cfg and cfg.get('use_gpu', False):
                # LLM wrapper exposes release_gpu()
                try:
                    llm_wrapper.release_gpu()
                except Exception:
                    # fallback: check underlying .llm for release_cuda
                    try:
                        if hasattr(llm_wrapper, 'llm') and hasattr(llm_wrapper.llm, 'release_cuda'):
                            llm_wrapper.llm.release_cuda()
                    except Exception:
                        pass
        except Exception:
            pass

###################pipeline####################
@register_phase("GENERATION")
class GenerationPhase(PhaseBase):
    def __init__(self, config, context):
        self.config = config
        self.llm_manager = context.llm_manager
        self.compiler = context.compiler
        self.max_iter = config.get("max_iteration", 6)

        self.llm_name = config.get("generation_llm_name", "main")

        # ---- NEW: instantiate postprocessor ----
        self.error_processor = CompileErrorProcessor(config)
        self.local_qwen_cfg = config.get("local_qwen")
        # defer creating the heavy local LLM until run() to avoid loading many
        # models at SystemController init time (which causes OOM)
        self.local_llm = None
        self.local_qwen_cfg = config.get("local_qwen")
        # defer creating the heavy local LLM until run() to avoid loading many
        # models at SystemController init time (which causes OOM)
        self.local_llm = None
        # Optional local Qwen config: if provided, GenerationPhase can instantiate
        # a dedicated local LLM instead of using llm_manager.get(). Expected format:
        # config["local_qwen"] = {
        #   "model_path": "/abs/path/to/qwen/snapshot",
        #   "prompt_path": "tasks/prompts/..",
        #   "shots_path": "...",            # optional
        #   "use_shots": False,              # optional
        #   "use_gpu": True,                 # optional
        #   "generation_kwargs": {...}       # optional
        # }
        self.local_qwen_cfg = config.get("local_qwen")
        # If requested, prepare a local LLM instance (created once)
        self.local_llm = None
        if self.local_qwen_cfg:
            self.local_llm = _create_local_llm_from_cfg(self.local_qwen_cfg)

    @log_phase_execution
    def run(self, ctx):
        task = ctx["task"]
        last_err = ""
        ctx["iteration_history"] = []  # 记录迭代历史

        # lazy-create the local LLM only when actually needed
        if self.local_qwen_cfg and self.local_llm is None:
            self.local_llm = _create_local_llm_from_cfg(self.local_qwen_cfg)

        if self.local_llm:
            llm = self.local_llm
        else:
            llm = self.llm_manager.get(self.llm_name)

        for i in range(1, self.max_iter + 1):
            # If a local Qwen config is provided, instantiate an LLM wrapper on demand
            if self.local_llm:
                if not last_err:
                    code = self.local_llm.generate_code(task, error="")
                else:
                    code = self.local_llm.generate_code(f"{task}\n\n### compile error\n{last_err}", error=last_err)
            else:
                if not last_err:
                    code = llm.generate_code(task, error="")
                else:
                    code = llm.generate_code(
                        f"{task}\n\n### compile error\n{last_err}",
                        error=last_err
                    )

            stdout, stderr_raw = self.compiler.compile_code(code)

            if not stderr_raw:
                ctx["iteration_history"].append({
                    "iteration": i,
                    "status": "success",
                    "error": None
                })
                ctx["compiled_code"] = stdout
                # phase finished using local model; free GPU if requested
                self._maybe_release_llm()
                return "TRANSFORM"

            # ---- NEW: postprocess compile stderr ----
            last_err = self.error_processor.process_compile_error(stderr_raw)

            # 记录失败的迭代
            ctx["iteration_history"].append({
                "iteration": i,
                "status": "failed",
                "error": last_err
            })

        raise RuntimeError("GenerationPhase failed")

@register_phase("GENERATION_ALT")
class GenerationAltPhase(PhaseBase):
    """
    Alternative generation phase that uses compile_code_alternative
    instead of compile_code for different compilation logic.
    """
    def __init__(self, config, context):
        self.config = config
        self.llm_manager = context.llm_manager
        self.compiler = context.compiler
        self.max_iter = config.get("max_iteration", 6)

        self.llm_name = config.get("generation_llm_name", "main")

        # ---- NEW: instantiate postprocessor ----
        self.error_processor = CompileErrorProcessor(config)
        self.local_qwen_cfg = config.get("local_qwen")
        self.local_llm = None

    @log_phase_execution
    def run(self, ctx):
        task = ctx["task"]
        last_err = ""
        ctx["iteration_history"] = []  # 记录迭代历史

        if self.local_llm:
            llm = self.local_llm
        else:
            llm = self.llm_manager.get(self.llm_name)
        if self.local_llm:
            llm = self.local_llm

        for i in range(1, self.max_iter + 1):
            if not last_err:
                code = llm.generate_code(task, error="")
            else:
                code = llm.generate_code(
                    f"{task}\n\n### compile error\n{last_err}",
                    error=last_err
                )

            # Use alternative compilation method
            stdout, stderr_raw = self.compiler.compile_code_alternative(code)

            if not stderr_raw:
                ctx["iteration_history"].append({
                    "iteration": i,
                    "status": "success",
                    "error": None
                })
                ctx["compiled_code"] = stdout
                # release local LLM GPU memory if configured
                self._maybe_release_llm()
                return "TRANSFORM"

            # ---- NEW: postprocess compile stderr ----
            last_err = self.error_processor.process_compile_error(stderr_raw)

            # 记录失败的迭代
            ctx["iteration_history"].append({
                "iteration": i,
                "status": "failed",
                "error": last_err
            })

        raise RuntimeError("GenerationAltPhase failed")
    

@register_phase("TRANSFORM")
class TransformPhase(PhaseBase):

    def __init__(self, config, context):
        self.config = config
        self.llm_manager = context.llm_manager

        # 两个模型名称（可配置、可使用不同模型）
        self.code1_llm_name = config.get("transform_llm_code1", "main")
        self.code2_llm_name = config.get("transform_llm_code2", "verifier")
        # support optional local qwen for transform phase; delay instantiation
        self.local_qwen_cfg = config.get("local_qwen")
        self.local_llm1 = None
        self.local_llm2 = None

    @log_phase_execution
    def run(self, ctx):
        task = ctx["task"]                  # 和 GenerationPhase 对齐，直接使用 task
        compiled = ctx["compiled_code"]     # 可用于 prompt，但你要求不使用专门函数

        # 取两种 LLM
        # lazy-create transform local LLM(s) if configured; reuse one instance
        if self.local_qwen_cfg and self.local_llm1 is None:
            self.local_llm1 = _create_local_llm_from_cfg(self.local_qwen_cfg)
            # reuse same instance for both roles to avoid double-loading
            self.local_llm2 = self.local_llm1

        if self.local_llm1 and self.local_llm2:
            llm1 = self.local_llm1
            llm2 = self.local_llm2
        else:
            llm1 = self.llm_manager.get(self.code1_llm_name)
            llm2 = self.llm_manager.get(self.code2_llm_name)

        mid_prompt = "The Python robot-control script is as follows:\n" + compiled + "\n"

        # 并行执行两个 LLM 调用
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交两个任务
            future1 = executor.submit(llm1.generate_code, task + mid_prompt, "")
            future2 = executor.submit(llm2.generate_code, task + mid_prompt, "")

            # 等待两个任务完成并获取结果
            ctx["code1"] = future1.result()
            ctx["code2"] = future2.result()

        # we've finished using the local transform LLM(s) — release GPU memory if requested
        self._maybe_release_llm()
        return "RUNTIME"


@register_phase("RUNTIME")
class RuntimePhase(PhaseBase):

    def __init__(self, config, context):
        self.robot = context.robot

    @log_phase_execution
    def run(self, ctx):
        r1 = self.robot.run_program(ctx["code1"])
        r2 = self.robot.run_program(ctx["code2"])

        if r1.error or r2.error:
            ctx["runtime_error"] = r1.error or r2.error
            return "REPAIR"

        return "DONE"

@register_phase("REPAIR")
class RepairPhase(PhaseBase):

    def __init__(self, config, context):
        self.config = config
        self.llm_manager = context.llm_manager

        # 修复代码使用的 llm 名称（可配置）
        self.repair_llm_name = config.get("repair_llm_name", "fixer")
        self.local_qwen_cfg = config.get("local_qwen")
        self.local_llm = None
        if self.local_qwen_cfg:
            self.local_llm = _create_local_llm_from_cfg(self.local_qwen_cfg)

    @log_phase_execution
    def run(self, ctx):
        err = ctx["runtime_error"]

        # 获取修复使用的 LLM
        if self.local_llm:
            llm = self.local_llm
        else:
            llm = self.llm_manager.get(self.repair_llm_name)

        # ---- 使用 generate_code 修复 ----
        # task = 原任务（ctx["task"]）
        # error = runtime_error
        # 让 LLM 基于任务 + 错误 修复并生成新代码
        repaired_code = llm.generate_code(ctx["task"], error=err)

        # ---- 如何决定跳转？ ----
        # 以下逻辑可根据你自己的系统需求调整：
        # 建议：
        # - 若错误涉及逻辑问题 → 回到 TRANSFORM 重新生成 code1 / code2
        # - 否则直接更新 code1 / code2 继续 runtime

        if "logic" in err.lower():
            # Re-run Phase2
            ctx["compiled_code"] = repaired_code
            # release local LLM from GPU if possible before switching phase
            self._maybe_release_llm()
            return "TRANSFORM"

        # 否则，直接让 repaired_code 更新两个 runtime 程序
        ctx["code1"] = repaired_code
        ctx["code2"] = repaired_code
        # release any GPU memory used by the local LLM
        self._maybe_release_llm()
        return "RUNTIME"
###################pipeline####################

###################one_time####################
@register_phase("ONE_TIME")
class OneTimePhase(PhaseBase):
    """
    One-time phase that generates all outputs (CONTROL, ASSERT, MONITOR)
    in a single LLM call and prints the result directly.
    No compilation or validation is performed.
    """
    def __init__(self, config, context):
        self.config = config
        self.llm_manager = context.llm_manager
        self.llm_name = config.get("generation_llm_name", "main")
        self.local_qwen_cfg = config.get("local_qwen")
        self.local_llm = None

    @log_phase_execution
    def run(self, ctx):
        task = ctx["task"]

        # lazy-create local LLM if configured
        if self.local_qwen_cfg and self.local_llm is None:
            self.local_llm = _create_local_llm_from_cfg(self.local_qwen_cfg)

        if self.local_llm:
            llm = self.local_llm
        else:
            llm = self.llm_manager.get(self.llm_name)

        # Generate all content in one call
        output = llm.generate_code(task, error="")

        # Print the complete output
        print("\n" + "="*80)
        print("ONE-TIME GENERATION OUTPUT")
        print("="*80)
        print(output)
        print("="*80 + "\n")

        # Store output in context
        ctx["one_time_output"] = output

        # End execution — try to release GPU memory used by local LLM
        self._maybe_release_llm()
        return "DONE"
###################one_time####################

###################baseline####################
@register_phase("BASELINE")
class BaselinePhase(PhaseBase):
    """
    Baseline phase with iterative code generation and compilation.
    Similar to GenerationPhase but for baseline experiments.
    """
    def __init__(self, config, context):
        self.config = config
        self.llm_manager = context.llm_manager
        self.compiler = context.compiler
        self.max_iter = config.get("max_iteration", 6)

        self.llm_name = config.get("generation_llm_name", "main")

        # ---- NEW: instantiate postprocessor ----
        self.error_processor = CompileErrorProcessor(config)

    @log_phase_execution
    def run(self, ctx):
        task = ctx["task"]
        last_err = ""
        ctx["iteration_history"] = []  # 记录迭代历史

        # lazy-create local LLM if configured
        if self.local_qwen_cfg and self.local_llm is None:
            self.local_llm = _create_local_llm_from_cfg(self.local_qwen_cfg)

        if self.local_llm:
            llm = self.local_llm
        else:
            llm = self.llm_manager.get(self.llm_name)

        for i in range(1, self.max_iter + 1):
            if not last_err:
                code = llm.generate_code(task, error="")
            else:
                code = llm.generate_code(
                    f"{task}\n\n### compile error\n{last_err}",
                    error=last_err
                )

            stdout, stderr_raw = self.compiler.compile_code(code)

            if not stderr_raw:
                ctx["iteration_history"].append({
                    "iteration": i,
                    "status": "success",
                    "error": None
                })
                ctx["compiled_code"] = stdout
                # release local LLM resources if configured to use GPU
                self._maybe_release_llm()
                return "DONE"

            # ---- NEW: postprocess compile stderr ----
            last_err = self.error_processor.process_compile_error(stderr_raw)

            # 记录失败的迭代
            ctx["iteration_history"].append({
                "iteration": i,
                "status": "failed",
                "error": last_err
            })

        raise RuntimeError("BaselinePhase failed")
###################baseline####################