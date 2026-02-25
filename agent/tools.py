import pathlib
import subprocess
import datetime
import os
from typing import Tuple, Optional
from pathlib import Path
from langchain_core.tools import tool

# ─────────────────────────────────────────────────────────────────────────────
# Dynamic project root per run
# ─────────────────────────────────────────────────────────────────────────────
PROJECTS_BASE = pathlib.Path.cwd() / "generated_projects"
PROJECT_ROOT: Optional[pathlib.Path] = None  # will be set by init_project_root()


def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

def _ensure_projects_base():
    PROJECTS_BASE.mkdir(parents=True, exist_ok=True)


def _unique_run_folder(prefix: str = "run") -> pathlib.Path:
    """
    Create a time-stamped run folder name like:
      run_YYYYmmdd_HHMMSS_mmm
    If collision occurs (very unlikely), append _1, _2, ...
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
    base_name = f"{prefix}_{ts}"
    candidate = PROJECTS_BASE / base_name
    if not candidate.exists():
        return candidate

    # Extremely rare: fallback to numeric suffixes
    i = 1
    while True:
        alt = PROJECTS_BASE / f"{base_name}_{i}"
        if not alt.exists():
            return alt
        i += 1

def set_project_root(path: Path) -> None:
    """Set the project root for all file tools."""
    global PROJECT_ROOT
    PROJECT_ROOT = Path(path)
    # Persist in environment so re-imports / subprocs can recover it
    os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

def _require_project_root() -> pathlib.Path:
    """
    Ensure PROJECT_ROOT is initialized before using any file tools.
    """
    global PROJECT_ROOT

    if PROJECT_ROOT is None:
        env = os.environ.get("PROJECT_ROOT")
        if env:
            PROJECT_ROOT = Path(env)
    if PROJECT_ROOT is None:
        raise RuntimeError(
            "PROJECT_ROOT is not initialized. Call init_project_root() first."
        )
    return PROJECT_ROOT




def safe_path_for_project(path: str) -> pathlib.Path:
    """
    Resolve a path under the current PROJECT_ROOT and prevent traversal outside.
    """
    root = _require_project_root()
    p = (root / path).resolve()
    root_resolved = root.resolve()
    # Allow: root itself, a direct file inside, or descendants
    if root_resolved not in p.parents and root_resolved != p.parent and root_resolved != p:
        raise ValueError("Attempt to write outside project root")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────
@tool
def init_project_root() -> str:
    """
    Initializes a new unique project directory under 'generated_projects/' and
    sets it as the active PROJECT_ROOT for this process.
    Returns the absolute path of the new project root.

    Example return:
      C:\\...\\generated_projects\\run_20260212_134512_123
    """
    prefix= "run"
    _ensure_projects_base()
    root = _unique_run_folder(prefix=prefix)
    root.mkdir(parents=True, exist_ok=True)
    set_project_root(root)
    return str(root)


@tool
def write_file(path: str, content: str) -> str:
    """Writes content to a file at the specified path within the project root."""
    p = safe_path_for_project(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return f"WROTE:{p}"


@tool
def read_file(path: str) -> str:
    """Reads content from a file at the specified path within the project root."""
    p = safe_path_for_project(path)
    if not p.exists():
        return ""
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


@tool
def get_current_directory() -> str:
    """Returns the current project root directory."""
    root = _require_project_root()
    return str(root)


@tool
def list_file(directory: str = ".") -> str:
    """Lists all files in the specified directory within the project root."""
    p = safe_path_for_project(directory)
    if not p.is_dir():
        return f"ERROR: {p} is not a directory"
    # compute root for relative paths
    root = _require_project_root()
    files = [str(f.relative_to(root)) for f in p.glob("**/*") if f.is_file()]
    return "\n".join(files) if files else "No files found."


@tool
def run_cmd(cmd: str, cwd: str = None, timeout: int = 30) -> Tuple[int, str, str]:
    """
    Runs a shell command in the specified directory (under the project root)
    and returns (returncode, stdout, stderr).
    """
    root = _require_project_root()
    cwd_dir = safe_path_for_project(cwd) if cwd else root
    res = subprocess.run(
        cmd,
        shell=True,
        cwd=str(cwd_dir),
        capture_output=True,
        text=True,
        timeout=timeout
    )
    return res.returncode, res.stdout, res.stderr