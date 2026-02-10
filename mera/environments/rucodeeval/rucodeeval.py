from __future__ import annotations

import ast
import asyncio
import contextlib
import io
import os
import platform
import re
import signal
import string
import tempfile
from typing import Any, Dict, List, Optional

import multiprocess as mp
import verifiers as vf
from datasets import Dataset, load_dataset

DATASET_ID = "MERA-evaluation/MERA"
TASK_ID = "rucodeeval"

_FORMATTER = string.Formatter()


def _extract_first_def_name(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except Exception:
        match = re.search(r"(?m)^\s*def\s+([A-Za-z_]\w*)\s*\(", code)
        return match.group(1) if match else None

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None


def _normalize_tests_literal(tests_obj: Any) -> str:
    if isinstance(tests_obj, str):
        return tests_obj
    return repr(tests_obj)


def _parse_tests_literal(tests_literal: str) -> List[Dict[str, Any]]:
    try:
        value = ast.literal_eval(tests_literal)
    except Exception:
        return []
    if not isinstance(value, list):
        return []
    tests: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            tests.append(item)
    return tests


def _resolve_entry_point(meta_entry_point: str | None, template: str, canonical_solution: str) -> str:
    template_name = _extract_first_def_name(template) or ""
    canonical_name = _extract_first_def_name(canonical_solution) or ""
    meta_name = (meta_entry_point or "").strip()

    for candidate in (template_name, canonical_name, meta_name):
        if candidate:
            return candidate
    raise ValueError("Unable to resolve entry point")


def format_prompt(instruction: str, inputs: Any, context: str = "") -> str:
    fields = [field for _, field, _, _ in _FORMATTER.parse(instruction) if field is not None]
    named_fields = [field for field in fields if field and not field.isdigit()]
    positional_fields = [field for field in fields if field == "" or (field and field.isdigit())]

    mapping: Dict[str, Any] = {}
    if isinstance(inputs, dict):
        mapping.update(inputs)
    else:
        for name in named_fields:
            mapping[name] = inputs

    if "context" in fields and "context" not in mapping:
        mapping["context"] = context

    for name in named_fields:
        mapping.setdefault(name, "")

    if positional_fields:
        pos_val = "" if isinstance(inputs, dict) else inputs
        pos_args = [pos_val for _ in positional_fields]
        return instruction.format(*pos_args, **mapping)

    return instruction.format(**mapping)


def _extract_code_block(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip("\n")
    return text


def _maybe_indent_body(code: str, indent: str = "    ") -> str:
    lines = code.splitlines()
    first_nonempty = next((ln for ln in lines if ln.strip()), "")
    if not first_nonempty:
        return code
    if first_nonempty.startswith((" ", "\t")):
        return code
    indented = []
    for ln in lines:
        if not ln.strip():
            indented.append(ln)
        else:
            indented.append(indent + ln)
    return "\n".join(indented)


def build_candidate_program(template: str, completion: str, entry_point: str) -> str:
    code = _extract_code_block(completion)
    if re.search(rf"(?m)^\s*def\s+{re.escape(entry_point)}\b", code):
        return code
    body = _maybe_indent_body(code)
    return template + "\n" + body


def run_reference_solution(
    canonical_solution: str,
    tests: List[Dict[str, Any]],
    entry_point_candidates: List[str],
) -> List[str]:
    ns: Dict[str, Any] = {"__name__": "__main__"}
    exec(canonical_solution, ns, ns)

    fn = None
    for name in entry_point_candidates:
        if not name:
            continue
        candidate = ns.get(name)
        if callable(candidate):
            fn = candidate
            break
    if fn is None:
        names = ", ".join([n for n in entry_point_candidates if n])
        raise ValueError(f"Reference entry point not found (candidates: {names})")

    outputs: List[str] = []
    for test in tests:
        value = fn(**test)
        if value is None:
            raise ValueError("Reference returned None")
        outputs.append(str(value))
    return outputs


def check_solution(true_outputs: List[Any], pred_outputs: List[Any]) -> bool:
    if not pred_outputs or len(true_outputs) != len(pred_outputs):
        return False
    for expected, actual in zip(true_outputs, pred_outputs):
        if str(expected) != str(actual):
            return False
    return True


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


class WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs):
        raise OSError

    def readline(self, *args, **kwargs):
        raise OSError

    def readlines(self, *args, **kwargs):
        raise OSError

    def readable(self, *args, **kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):
    _stream = "stdin"


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def chdir(root: str):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


def reliability_guard(maximum_memory_bytes: int | None = None) -> None:
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    import builtins

    builtins.exit = None
    builtins.quit = None

    os.environ["OMP_NUM_THREADS"] = "1"
    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def unsafe_execute(
    program: str,
    tests: List[Dict[str, Any]],
    entry_point: str,
    result,
    timeout: float,
) -> None:
    with create_tempdir():
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir_ = os.chdir

        reliability_guard()
        try:
            exec_globals: Dict[str, Any] = {"__name__": "__main__"}
            with swallow_io():
                with time_limit(timeout):
                    exec(program, exec_globals, exec_globals)

            fn = exec_globals.get(entry_point)
            if not callable(fn):
                result.append([None] * len(tests))
                return

            outputs = []
            for test in tests:
                try:
                    value = fn(**test)
                    outputs.append(str(value) if value is not None else None)
                except Exception:
                    outputs.append(None)
            result.append(outputs)
        except Exception:
            result.append([None] * len(tests))
        finally:
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir_


def check_correctness(
    program: str,
    tests: List[Dict[str, Any]],
    entry_point: str,
    timeout: float,
) -> List[Any]:
    manager = mp.Manager()
    result = manager.list()
    process = mp.Process(target=unsafe_execute, args=(program, tests, entry_point, result, timeout))
    process.start()
    process.join(timeout=timeout + 1)
    if process.is_alive():
        process.kill()
    if not result:
        result.append([])
    return list(result[0])


def _build_dataset(split: str, cache_dir: Optional[str] = None) -> Dataset:
    raw = load_dataset(DATASET_ID, TASK_ID, split=split, cache_dir=cache_dir)

    def to_example(x: Dict[str, Any]) -> Dict[str, Any]:
        instruction = x.get("instruction", "")
        inputs = x.get("inputs") or {}
        meta = x.get("meta") or {}

        template = str(inputs.get("function", ""))
        tests_literal = _normalize_tests_literal(inputs.get("tests", "[]"))
        tests = _parse_tests_literal(tests_literal)
        canonical_solution = str(meta.get("canonical_solution", ""))

        meta_entry_point = meta.get("entry_point")
        try:
            entry_point = _resolve_entry_point(str(meta_entry_point) if meta_entry_point is not None else None, template, canonical_solution)
        except Exception:
            entry_point = ""

        valid = False
        expected_outputs: List[str] = []
        if template and canonical_solution and tests and entry_point:
            try:
                expected_outputs = run_reference_solution(
                    canonical_solution,
                    tests,
                    [
                        entry_point,
                        _extract_first_def_name(canonical_solution) or "",
                        _extract_first_def_name(template) or "",
                        str(meta_entry_point or ""),
                    ],
                )
                valid = bool(expected_outputs)
            except Exception:
                valid = False
                expected_outputs = []

        question = format_prompt(str(instruction), inputs, context="")

        return {
            "question": question,
            "answer": expected_outputs,
            "meta": meta,
            "inputs": {"function": template, "tests": tests_literal, "entry_point": entry_point},
            "valid": valid,
        }

    ds = raw.map(to_example, remove_columns=raw.column_names)
    ds = ds.filter(lambda row: bool(row.get("valid")))
    return ds.remove_columns(["valid"])


def load_environment(
    split: str = "test",
    system_prompt: str | None = None,
    cache_dir: str | None = None,
    max_eval_examples: int | None = None,
    timeout_s: float = 3.0,
    **_kwargs: Any,
) -> vf.Environment:
    ds = _build_dataset(split, cache_dir=cache_dir)
    if max_eval_examples is not None:
        ds = ds.select(range(min(int(max_eval_examples), len(ds))))

    parser = vf.MaybeThinkParser(_extract_code_block)

    async def reward_fn(
        parser_obj: vf.Parser,
        completion: vf.Messages,
        answer: Any,
        state: vf.State,
        **_kw: Any,
    ) -> float:
        if not answer:
            return 0.0
        code = parser_obj.parse_answer(completion) or ""
        if not code.strip():
            return 0.0

        input_obj = state.get("input", {})
        if not isinstance(input_obj, dict):
            return 0.0
        meta = input_obj.get("meta", {})
        inputs = input_obj.get("inputs", {})
        if not isinstance(meta, dict) or not isinstance(inputs, dict):
            return 0.0

        entry_point = str(inputs.get("entry_point", "")) or str(meta.get("entry_point", ""))
        template = str(inputs.get("function", ""))
        tests_literal = inputs.get("tests", "[]")
        if not isinstance(tests_literal, str):
            tests_literal = repr(tests_literal)
        tests = _parse_tests_literal(tests_literal)
        if not entry_point or not template or not tests:
            return 0.0

        program = build_candidate_program(template, code, entry_point)
        pred_outputs = await asyncio.to_thread(check_correctness, program, tests, entry_point, float(timeout_s))
        return 1.0 if check_solution(list(answer), pred_outputs) else 0.0

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(reward_fn)

    return vf.SingleTurnEnv(
        dataset=ds,
        eval_dataset=ds,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
