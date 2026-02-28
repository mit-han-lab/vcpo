"""
Tool sandbox module for safe code execution and tool management.

This module provides:
- PythonSandbox: Safe Python code execution environment
- ToolRegistry: Tool registration and execution management
- Memory management utilities
"""

import argparse
import ast
import asyncio
import gc
import json
import os
import re
import subprocess
import tempfile
import time
import threading
from contextlib import contextmanager
from typing import Any, Optional

import psutil
from fastapi import FastAPI, Request
import uvicorn

# Configuration for tool execution
TOOL_CONFIGS = {
    "max_turns": 16,
    "max_tool_calls": 16,
    "tool_concurrency": 32,
    # Python interpreter settings
    "python_timeout": 15,
    "python_memory_limit": "4GB",  # 4GB per Python process
    "python_cpu_limit": 1,
    # Memory management settings
    "max_memory_usage": 12288,  # 12GB total (75% of 16GB)
    "cleanup_threshold": 6144,  # 6GB
    "aggressive_cleanup_threshold": 3072,  # 3GB
    "force_cleanup_threshold": 9216,  # 9GB
}

# Global semaphore for controlling concurrent tool executions
SEMAPHORE = asyncio.Semaphore(TOOL_CONFIGS["tool_concurrency"])
THREAD_SEMAPHORE = threading.Semaphore(TOOL_CONFIGS["tool_concurrency"])
TOOL_METRIC_LOCK = threading.Lock()
TOOL_REQUESTS_TOTAL = 0
TOOL_TIMEOUTS_TOTAL = 0


def _record_tool_timeout(is_timeout: bool) -> float:
    global TOOL_REQUESTS_TOTAL, TOOL_TIMEOUTS_TOTAL
    with TOOL_METRIC_LOCK:
        TOOL_REQUESTS_TOTAL += 1
        if is_timeout:
            TOOL_TIMEOUTS_TOTAL += 1
        return (TOOL_TIMEOUTS_TOTAL / TOOL_REQUESTS_TOTAL) * 100.0


def _get_server_load_metrics(wait_time: float, active_requests: int) -> dict[str, float]:
    try:
        cpu_percent = psutil.cpu_percent(interval=None)
    except Exception:
        cpu_percent = 0.0
    try:
        load_1, load_5, load_15 = os.getloadavg()
    except Exception:
        load_1, load_5, load_15 = 0.0, 0.0, 0.0
    try:
        rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        rss_mb = 0.0
    return {
        "queue_wait_s": float(wait_time),
        "active_requests": float(active_requests),
        "tool_concurrency": float(TOOL_CONFIGS["tool_concurrency"]),
        "cpu_percent": float(cpu_percent),
        "load_1": float(load_1),
        "load_5": float(load_5),
        "load_15": float(load_15),
        "rss_mb": float(rss_mb),
    }

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def cleanup_memory():
    """Force garbage collection to free memory"""
    gc.collect()
    if hasattr(gc, "collect"):
        gc.collect()

def aggressive_cleanup_memory():
    """More aggressive memory cleanup"""
    # Force multiple garbage collection cycles
    for _ in range(3):
        gc.collect()

    # Clear Python's internal caches
    import sys

    # Note: sys.intern doesn't have a clear method, so we skip this
    # Clear module cache if possible
    if hasattr(sys, "modules"):
        # Don't clear all modules, but clear some common ones that might cache data
        modules_to_clear = ["numpy", "pandas", "matplotlib", "scipy"]
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                if hasattr(module, "clear_cache"):
                    module.clear_cache()


def check_and_cleanup_memory():
    """Check memory usage and perform appropriate cleanup"""
    current_memory = get_memory_usage()

    if current_memory > TOOL_CONFIGS["force_cleanup_threshold"]:
        # Force aggressive cleanup
        aggressive_cleanup_memory()
        return f"Warning: High memory usage ({current_memory:.1f}MB), performed aggressive cleanup"
    elif current_memory > TOOL_CONFIGS["cleanup_threshold"]:
        # Normal cleanup
        cleanup_memory()
        return f"Info: Memory usage ({current_memory:.1f}MB), performed cleanup"
    elif current_memory > TOOL_CONFIGS["aggressive_cleanup_threshold"]:
        # Light cleanup
        gc.collect()
        return f"Info: Memory usage ({current_memory:.1f}MB), performed light cleanup"

    return None


class PythonSandbox:
    """Python code sandbox, provides safe code execution environment"""

    def __init__(self, timeout: int = 10, memory_limit: str = "100MB"):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.allowed_modules = {
            "math",
            "time",
            "random",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "operator",
            "statistics",
            "decimal",
            "fractions",
            "numpy",
            "sympy",
            "scipy",
        }

    def _check_code_safety(self, code: str) -> tuple[bool, str]:
        """Check code safety by scanning for dangerous patterns"""
        # Check for dangerous operations
        dangerous_patterns = [
            r"import\s+os",
            r"import\s+sys",
            r"import\s+subprocess",
            r"import\s+shutil",
            r"import\s+glob",
            r"import\s+pathlib",
            r"__import__",
            r"eval\s*\(",
            r"exec\s*\(",
            r"open\s*\(",
            r"file\s*\(",
            r"input\s*\(",
            r"raw_input\s*\(",
            r"compile\s*\(",
            r"execfile\s*\(",
            r"getattr\s*\(",
            r"setattr\s*\(",
            r"delattr\s*\(",
            r"hasattr\s*\(",
            r"globals\s*\(",
            r"locals\s*\(",
            r"vars\s*\(",
            r"dir\s*\(",
            r"type\s*\(",
            r"isinstance\s*\(",
            r"issubclass\s*\(",
            r"super\s*\(",
            r"property\s*\(",
            r"staticmethod\s*\(",
            r"classmethod\s*\(",
            r"__\w+__",  # double underscore methods
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Code contains dangerous pattern: {pattern}"

        # Check imported modules using AST to avoid false positives.
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, f"Syntax error: {exc}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_name = alias.name.split(".")[0]
                    if root_name not in self.allowed_modules:
                        return False, f"Import of '{root_name}' is not allowed"
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                root_name = node.module.split(".")[0]
                if root_name not in self.allowed_modules:
                    return False, f"Import of '{root_name}' is not allowed"

        return True, "Code is safe"

    @contextmanager
    def _create_safe_environment(self):
        """Create safe execution environment with temporary directory"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="python_sandbox_")

        try:
            # Create safe Python script
            script_path = os.path.join(temp_dir, "code.py")

            # Set environment variables
            env = os.environ.copy()
            env["PYTHONPATH"] = temp_dir
            env["PYTHONUNBUFFERED"] = "1"

            yield script_path, env, temp_dir

        finally:
            # Clean up temporary directory
            try:
                import shutil

                shutil.rmtree(temp_dir)
            except Exception:
                pass

    def _wrap_code(self, code: str, memory_limit_mb: Optional[int], output_mode: str) -> str:
        limit_bytes = 4 * 1024 * 1024 * 1024
        if memory_limit_mb:
            limit_bytes = memory_limit_mb * 1024 * 1024

        code_literal = repr(code)

        return f"""import json
import sys
import traceback
from io import StringIO
import resource
import ast

try:
    resource.setrlimit(resource.RLIMIT_AS, ({limit_bytes}, -1))
except Exception:
    pass

old_stdout = sys.stdout
old_stderr = sys.stderr
stdout_capture = StringIO()
stderr_capture = StringIO()
sys.stdout = stdout_capture
sys.stderr = stderr_capture
error_msg = None

try:
    code_str = {code_literal}
    parsed = ast.parse(code_str, mode="exec")
    last_expr = None
    if parsed.body and isinstance(parsed.body[-1], ast.Expr):
        last_expr = parsed.body.pop().value
    exec_globals = {{"__builtins__": __builtins__}}
    exec(compile(parsed, "<sandbox>", "exec"), exec_globals)
    if last_expr is not None:
        if not stdout_capture.getvalue() and not stderr_capture.getvalue():
            value = eval(compile(ast.Expression(last_expr), "<sandbox>", "eval"), exec_globals)
            if value is not None:
                print(value)
        else:
            last_expr_module = ast.Module([ast.Expr(last_expr)], type_ignores=[])
            ast.fix_missing_locations(last_expr_module)
            exec(compile(last_expr_module, "<sandbox>", "exec"), exec_globals)
except Exception as e:
    error_msg = f"{{str(e)}}\\nTraceback:\\n{{traceback.format_exc()}}"
finally:
    stdout_output = stdout_capture.getvalue()
    stderr_output = stderr_capture.getvalue()
    sys.stdout = old_stdout
    sys.stderr = old_stderr

if "{output_mode}" == "json":
    payload = {{"stdout": stdout_output, "stderr": stderr_output, "error": error_msg}}
    print(json.dumps(payload))
else:
    if error_msg:
        print(f"Error: {{error_msg}}")
    else:
        result = ""
        if stdout_output:
            result += f"Output:\\n{{stdout_output}}"
        if stderr_output:
            result += f"\\nErrors:\\n{{stderr_output}}"
        print(result)"""

    def _run_code(
        self,
        code: str,
        stdin: Optional[str],
        timeout: Optional[int],
        memory_limit_mb: Optional[int],
        output_mode: str,
    ) -> dict[str, Any]:
        current_memory = get_memory_usage()
        if current_memory > TOOL_CONFIGS["max_memory_usage"]:
            aggressive_cleanup_memory()
            return {
                "status": "Error",
                "stdout": "",
                "stderr": "Error: Memory usage too high, please try again",
                "return_code": 1,
                "execution_time": 0.0,
            }

        is_safe, message = self._check_code_safety(code)
        if not is_safe:
            return {
                "status": "Error",
                "stdout": "",
                "stderr": f"Error: {message}",
                "return_code": 1,
                "execution_time": 0.0,
            }

        wrapped_code = self._wrap_code(code, memory_limit_mb, output_mode)
        run_timeout = timeout if timeout is not None else self.timeout
        start = time.monotonic()

        with self._create_safe_environment() as (script_path, env, temp_dir):
            with open(script_path, "w") as f:
                f.write(wrapped_code)

            try:
                process = subprocess.Popen(
                    ["python3", script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    env=env,
                    cwd=temp_dir,
                    text=True,
                )
                try:
                    stdout, stderr = process.communicate(input=stdin, timeout=run_timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    return {
                        "status": "TimeLimitExceeded",
                        "stdout": "",
                        "stderr": f"Error: Code execution timed out after {run_timeout} seconds",
                        "return_code": None,
                        "execution_time": time.monotonic() - start,
                    }
            except Exception as e:
                return {
                    "status": "Error",
                    "stdout": "",
                    "stderr": f"Error: Failed to execute code: {str(e)}",
                    "return_code": 1,
                    "execution_time": time.monotonic() - start,
                }

        cleanup_message = check_and_cleanup_memory()
        if cleanup_message:
            print(f"Memory cleanup: {cleanup_message}")

        stdout = stdout.strip()
        stderr = stderr.strip()
        if output_mode == "json":
            try:
                payload = json.loads(stdout) if stdout else {}
            except json.JSONDecodeError as e:
                return {
                    "status": "Error",
                    "stdout": "",
                    "stderr": f"Error: Invalid sandbox output: {e}",
                    "return_code": 1,
                    "execution_time": time.monotonic() - start,
                }
            error_msg = payload.get("error")
            if error_msg:
                return {
                    "status": "Error",
                    "stdout": payload.get("stdout", ""),
                    "stderr": error_msg,
                    "return_code": 1,
                    "execution_time": time.monotonic() - start,
                }
            return {
                "status": "Finished",
                "stdout": payload.get("stdout", ""),
                "stderr": payload.get("stderr", ""),
                "return_code": 0,
                "execution_time": time.monotonic() - start,
            }

        error_detected = stdout.lstrip().startswith("Error:")
        status = "Error" if error_detected or process.returncode else "Finished"
        return {
            "status": status,
            "stdout": "" if error_detected else stdout,
            "stderr": stdout if error_detected else stderr,
            "return_code": 1 if status != "Finished" else 0,
            "execution_time": time.monotonic() - start,
        }

    async def execute_code(self, code: str, stdin: Optional[str] = None) -> str:
        """Execute Python code in sandbox with safety checks"""
        result = self._run_code(code, stdin=stdin, timeout=None, memory_limit_mb=None, output_mode="text")
        if result["status"] == "Finished":
            return result["stdout"]
        return result["stderr"]

    def execute_code_api(
        self,
        code: str,
        stdin: Optional[str],
        timeout: Optional[int],
        memory_limit_mb: Optional[int],
    ) -> dict[str, Any]:
        return self._run_code(code, stdin=stdin, timeout=timeout, memory_limit_mb=memory_limit_mb, output_mode="json")


class ToolRegistry:
    """Tool registry, manages available tools and their execution"""

    def __init__(self):
        self.tools = {}
        self.python_sandbox = PythonSandbox(
            timeout=TOOL_CONFIGS["python_timeout"], memory_limit=TOOL_CONFIGS["python_memory_limit"]
        )
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools in the registry"""
        # Python code interpreter
        self.register_tool(
            "code_interpreter",
            {
                "type": "function",
                "function": {
                    "name": "code_interpreter",
                    "description": "A tool for executing Python code in a safe sandbox environment.",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string", "description": "The Python code to execute"}},
                        "required": ["code"],
                    },
                },
            },
        )

    def register_tool(self, name: str, tool_spec: dict[str, Any]):
        """Register a new tool in the registry"""
        self.tools[name] = tool_spec

    def get_tool_specs(self) -> list[dict[str, Any]]:
        """Get all tool specifications as a list"""
        return list(self.tools.values())

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call with the given arguments"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"

        async with SEMAPHORE:
            if tool_name == "code_interpreter":
                return await self._execute_python(arguments)
            else:
                return f"Error: Tool '{tool_name}' not implemented"

    async def _execute_python(self, arguments: dict[str, Any]) -> str:
        """Execute Python code using the sandbox"""
        code = arguments.get("code", "")
        if not code.strip():
            return "Error: No code provided"

        # Execute code in sandbox
        result = await self.python_sandbox.execute_code(code)
        return result


# Global tool registry instance
tool_registry = ToolRegistry()


def _handle_sandbox_payload(sandbox: PythonSandbox, payload: dict[str, Any]) -> dict[str, Any]:
    wait_start = time.monotonic()
    THREAD_SEMAPHORE.acquire()
    wait_time = time.monotonic() - wait_start
    try:
        active_requests = TOOL_CONFIGS["tool_concurrency"] - THREAD_SEMAPHORE._value
    except Exception:
        active_requests = -1
    load_metrics = _get_server_load_metrics(wait_time, active_requests)
    try:
        code = payload.get("code", "")
        stdin = payload.get("stdin")

        requested_timeout = float(payload.get("run_timeout", sandbox.timeout))
        run_timeout = max(0.0, requested_timeout - wait_time)
        memory_limit_mb = payload.get("memory_limit_MB")
        language = payload.get("language", "python")
        if language != "python":
            return {
                "status": "Failed",
                "compile_result": {"status": "Finished", "execution_time": 0.0, "stderr": "", "return_code": 0},
                "run_result": {
                    "status": "Error",
                    "stdout": "",
                    "stderr": f"Unsupported language: {language}",
                    "return_code": 1,
                    "execution_time": 0.0,
                },
            }
        if run_timeout <= 0.0:
            timeout_pct = _record_tool_timeout(True)
            print(
                f"[METRIC] tool_timeout_pct={timeout_pct:.2f}% "
                f"(timeouts={TOOL_TIMEOUTS_TOTAL} total={TOOL_REQUESTS_TOTAL})"
            )
            return {
                "status": "Failed",
                "compile_result": {"status": "Finished", "execution_time": 0.0, "stderr": "", "return_code": 0},
                "run_result": {
                    "status": "Error",
                    "stdout": "",
                    "stderr": (
                        "Error: Code execution timed out after "
                        f"{requested_timeout:.0f} seconds (queue wait {wait_time:.2f}s)"
                    ),
                    "return_code": None,
                    "execution_time": 0.0,
                    "timeout_pct": timeout_pct,
                    "is_timeout": True,
                    "load_metrics": load_metrics,
                },
            }

        result = sandbox.execute_code_api(
            code, stdin=stdin, timeout=run_timeout, memory_limit_mb=memory_limit_mb
        )
        is_timeout = result.get("status") == "TimeLimitExceeded"
        timeout_pct = _record_tool_timeout(is_timeout)

        print(
            "======================\n"
            ">>> New Request\n"
            f"{code}\n"
            "**********************\n"
            f">>> Code executed in {result['execution_time']:.4f}s\n"
            f">>> timeout_pct {timeout_pct:.2f}% (timeouts={TOOL_TIMEOUTS_TOTAL} total={TOOL_REQUESTS_TOTAL})\n"
            ">>> load_metrics\n"
            f"{json.dumps(load_metrics, ensure_ascii=False)}\n"
            ">>> stdout\n"
            f"{result['stdout']}\n"
            ">>> stderr\n"
            f"{result['stderr']}\n"
            "======================\n"
        )

        api_status = "Success" if result["status"] == "Finished" else "Failed"
        return {
            "status": api_status,
            "compile_result": {"status": "Finished", "execution_time": 0.0, "stderr": "", "return_code": 0},
            "run_result": {
                "status": result["status"],
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "return_code": result["return_code"],
                "execution_time": result["execution_time"],
                "timeout_pct": timeout_pct,
                "is_timeout": is_timeout,
                "load_metrics": load_metrics,
            },
        }
    finally:
        THREAD_SEMAPHORE.release()


def create_sandbox_app(sandbox: PythonSandbox) -> FastAPI:
    app = FastAPI()

    @app.post("/")
    async def run_code_root(request: Request) -> dict[str, Any]:
        try:
            payload = await request.json()
        except Exception as e:
            return {"status": "SandboxError", "message": f"Invalid request: {e}"}
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, _handle_sandbox_payload, sandbox, payload)
        except Exception as e:
            return {"status": "SandboxError", "message": f"Server error: {e}"}

    @app.post("/run_code")
    async def run_code_alias(request: Request) -> dict[str, Any]:
        return await run_code_root(request)

    return app


def run_sandbox_api_server(host: str, port: int, timeout: int, memory_limit_mb: int, workers: int) -> None:
    sandbox = PythonSandbox(timeout=timeout, memory_limit=f"{memory_limit_mb}MB")
    app = create_sandbox_app(sandbox)
    print(f"==== Starting Server at {host, port} ====", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning", workers=workers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local sandbox fusion-compatible API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=TOOL_CONFIGS["python_timeout"])
    parser.add_argument("--memory-limit-mb", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run_sandbox_api_server(args.host, args.port, args.timeout, args.memory_limit_mb, args.workers)


if __name__ == "__main__":
    main()
