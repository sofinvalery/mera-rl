from __future__ import annotations

import ast
import contextlib
import io
import os
import platform
import signal
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import multiprocess as mp


def preprocess_generation(generation: str) -> str:
    if not generation:
        return generation
    first = generation[0]
    if first in {" ", "\n", "\t"}:
        return generation
    begin_pattern = first * 3 + "python"
    end_pattern = first * 3
    if generation.startswith(begin_pattern) and generation.endswith(end_pattern):
        return generation[len(begin_pattern) : -len(end_pattern)]
    return generation


def run_code_tests(
    completion: str, inputs: Dict[str, Any], meta: Dict[str, Any], outputs: Any
) -> bool:
    if not outputs:
        return False
    try:
        tests = ast.literal_eval(inputs["tests"])
    except Exception:
        return False
    entry_point = meta.get("entry_point")
    if not entry_point:
        return False
    full_source = inputs["function"] + "\n" + preprocess_generation(completion)
    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(check_correctness, full_source, tests, entry_point, 3.0)
        result = future.result()
    return check_solution(outputs, result)


def check_solution(true_outputs: List[Any], pred_outputs: List[Any]) -> bool:
    if not pred_outputs or len(true_outputs) != len(pred_outputs):
        return False
    for expected, actual in zip(true_outputs, pred_outputs):
        if str(expected) != str(actual):
            return False
    return True


def check_correctness(program: str, tests: List[Dict[str, Any]], entry_point: str, timeout: float):
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


def unsafe_execute(
    program: str, tests: List[Dict[str, Any]], entry_point: str, result, timeout: float
) -> None:
    with create_tempdir():
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        reliability_guard()
        try:
            with swallow_io():
                with time_limit(timeout):
                    exec(program)
            outputs = []
            for test in tests:
                try:
                    value = locals()[entry_point](**test)
                    outputs.append(str(value) if value is not None else None)
                except Exception:
                    outputs.append(None)
            result.append(outputs)
        except Exception:
            result.append([None] * len(tests))
        finally:
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir


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


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


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
