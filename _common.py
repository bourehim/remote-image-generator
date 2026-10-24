"""Shared helpers for the Kaggle bootstrap/launch scripts (stdlib only).

These scripts intentionally depend on nothing outside the Python standard
library: they must run before/independently of the project's virtualenv.

Two dataset layouts are supported:

* **repo layout** — the dataset contains the whole repository
  (``server/``, ``requirements/``, ``kaggle/`` ...).
* **flat layout** — the dataset contains only the launcher files
  (``bootstrap.py``, ``launch.py``, ``_common.py`` ...), as in
  ``/kaggle/input/datasets/<owner>/<slug>/``. In that case the server code must
  come from ``--source`` (git URL / archive / local directory), from a second
  full-repo dataset, or from a previous bootstrap run in
  ``/kaggle/working/remote-image``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

CLOUDFLARED_VERSION = os.environ.get("RIMAGE_CLOUDFLARED_VERSION", "2026.10.3")
CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/download/"
    f"{CLOUDFLARED_VERSION}/cloudflared-linux-amd64"
)
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "images", "node_modules"}
QUICK_TUNNEL_RE = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com")

PROJECT_MARKER = Path("server") / "rimage_server" / "main.py"
LAUNCHER_FILES = (
    "bootstrap.py",
    "launch.py",
    "_common.py",
    "install_cloudflared.sh",
    "NOTEBOOK.md",
)
ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz", ".tar")
DEFAULT_SEARCH_ROOTS = ("/kaggle/input/datasets", "/kaggle/input", "/kaggle/working")
WORKING_COPY = Path("/kaggle/working/remote-image")


def log(message: str = "") -> None:
    print(message, flush=True)


def warn(message: str) -> None:
    print(f"[warn] {message}", flush=True)


def die(message: str, code: int = 1) -> None:
    print(f"[error] {message}", file=sys.stderr, flush=True)
    raise SystemExit(code)


# ---------------------------------------------------------------------------
# layout discovery
# ---------------------------------------------------------------------------
class ProjectNotFound(RuntimeError):
    """No directory containing server/rimage_server/main.py was found."""

    def __init__(self, checked: Iterable[str], scripts_dir: Path | None = None) -> None:
        self.checked = list(checked)
        self.scripts_dir = scripts_dir
        super().__init__("project root (server/rimage_server/main.py) not found")


@dataclass
class Layout:
    """Where the launchers live and where the runnable project lives."""

    scripts_dir: Path
    project_root: Path | None
    mode: str  # "repo" | "working" | "flat"
    checked: list[str] = field(default_factory=list)

    @property
    def is_flat(self) -> bool:
        return self.mode == "flat"

    def require_project(self) -> Path:
        if self.project_root is None:
            raise ProjectNotFound(self.checked, self.scripts_dir)
        return self.project_root

    def summary(self) -> str:
        if self.project_root is None:
            return f"flat launcher dataset at {self.scripts_dir} (no project source)"
        return f"{self.mode} project at {self.project_root} (launchers: {self.scripts_dir})"


def is_project_dir(path: Path) -> bool:
    return (Path(path) / PROJECT_MARKER).is_file()


def _mode_for(path: Path) -> str:
    try:
        Path(path).resolve().relative_to(Path("/kaggle/working"))
        return "working"
    except (ValueError, OSError):
        return "repo"


def _find_project_in_tree(tree: Path) -> Path | None:
    tree = Path(tree)
    if is_project_dir(tree):
        return tree.resolve()
    matches = sorted(tree.rglob(str(PROJECT_MARKER)), key=lambda p: len(p.parts))
    for marker in matches:
        return marker.parent.parent.parent.resolve()
    return None


def detect_layout(
    explicit_project: str | Path | None = None,
    *,
    script_file: str | Path | None = None,
    search_roots: Iterable[str | Path] | None = None,
    working_copy: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> Layout:
    """Resolve launcher dir + project root, tolerating the flat dataset layout."""
    env = os.environ if env is None else env
    script_path = Path(script_file) if script_file else Path(__file__).resolve()
    scripts_dir = script_path.resolve().parent
    checked: list[str] = []

    candidates: list[Path] = []
    if explicit_project:
        candidates.append(Path(explicit_project))
    if env.get("RIMAGE_PROJECT_DIR"):
        candidates.append(Path(env["RIMAGE_PROJECT_DIR"]))
    candidates.extend([scripts_dir, scripts_dir.parent, scripts_dir.parent.parent])
    if working_copy:
        candidates.append(Path(working_copy))
    candidates.append(WORKING_COPY)

    for candidate in candidates:
        checked.append(str(candidate))
        if is_project_dir(candidate):
            resolved = candidate.resolve()
            return Layout(scripts_dir, resolved, _mode_for(resolved), checked)

    roots = [Path(root) for root in (search_roots if search_roots is not None else DEFAULT_SEARCH_ROOTS)]
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen or not root.is_dir():
            continue
        seen.add(key)
        try:
            markers = sorted(root.rglob(str(PROJECT_MARKER)), key=lambda p: len(p.parts))
        except OSError:
            continue
        for marker in markers[:20]:
            project = marker.parent.parent.parent
            checked.append(str(project))
            if is_project_dir(project):
                resolved = project.resolve()
                return Layout(scripts_dir, resolved, _mode_for(resolved), checked)

    return Layout(scripts_dir, None, "flat", checked)


def find_project(explicit: str | None = None, **kwargs: Any) -> Path:
    """Backwards-compatible helper: return the project root or raise ProjectNotFound."""
    layout = detect_layout(explicit, **kwargs)
    return layout.require_project()


def flat_dataset_hint(scripts_dir: Path) -> str:
    return (
        f"the dataset at {scripts_dir} contains only the launcher scripts "
        f"({', '.join(LAUNCHER_FILES[:3])} ...), not the server code.\n"
        "Fix it in one of these ways:\n"
        f"  1. bootstrap from a full-repo source:\n"
        f"       !python {scripts_dir / 'bootstrap.py'} --source <git-url|tar.gz|zip|dir>\n"
        "     (or set RIMAGE_SOURCE to the same value)\n"
        "  2. attach the full repository as a second Kaggle dataset, then:\n"
        f"       !python {scripts_dir / 'bootstrap.py'} --project-dir /kaggle/input/<that-dataset>\n"
        "  3. upload the whole repository (server/, requirements/, kaggle/) as the dataset,\n"
        "     not just its kaggle/ folder.\n"
        "After one successful bootstrap the runnable copy lives in /kaggle/working/remote-image,\n"
        "so launch.py works from this dataset path as well."
    )


# ---------------------------------------------------------------------------
# fetching project source (git / archive / local dir)
# ---------------------------------------------------------------------------
def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://", "git@", "ssh://"))


def _is_archive(value: str) -> bool:
    lowered = value.lower().split("?", 1)[0]
    return lowered.endswith(ARCHIVE_SUFFIXES)


def _safe_member_name(name: str) -> str:
    candidate = name.replace("\\", "/").strip()
    if not candidate or candidate.startswith("/") or re.match(r"^[A-Za-z]:", candidate):
        raise ValueError(f"unsafe archive member: {name!r}")
    parts = [part for part in candidate.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ValueError(f"unsafe archive member: {name!r}")
    return "/".join(parts)


def _extract_zip(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            name = _safe_member_name(info.filename)
            if info.is_dir():
                continue
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ValueError(f"refusing symlink in archive: {info.filename!r}")
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
    return dest


def _extract_tar(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            name = _safe_member_name(member.name)
            if member.isdir():
                (dest / name).mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"refusing non-regular archive member: {member.name!r}")
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                continue
            with source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
    return dest


def extract_archive(archive: Path, dest: Path) -> Path:
    name = str(archive).lower()
    if name.endswith(".zip"):
        return _extract_zip(Path(archive), Path(dest))
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        return _extract_tar(Path(archive), Path(dest))
    raise ValueError(f"unsupported archive type: {archive}")


def _download(url: str, target: Path, *, timeout: float = 180.0) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "rimage-bootstrap/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)
    return target


def fetch_source(source: str, destination: Path) -> Path:
    """Materialise the project from a git URL, archive (URL or path) or directory."""
    value = (source or "").strip()
    if not value:
        raise ValueError("empty source")
    destination = Path(destination).resolve()
    workdir = Path(tempfile.mkdtemp(prefix="rimage-src-"))
    try:
        if _is_url(value) and not _is_archive(value):
            log(f"cloning project: {value}")
            result = subprocess.run(
                ["git", "clone", "--depth", "1", "--quiet", value, str(workdir / "clone")],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "git clone failed (private repos need a token-bearing URL or a dataset): "
                    + (result.stderr or "").strip()[:300]
                )
            tree = workdir / "clone"
        elif _is_url(value):
            archive = workdir / (urllib.parse.urlparse(value).path.rsplit("/", 1)[-1] or "source.zip")
            log(f"downloading archive: {value}")
            _download(value, archive)
            tree = extract_archive(archive, workdir / "extract")
        elif _is_archive(value):
            tree = extract_archive(Path(value), workdir / "extract")
        elif Path(value).is_dir():
            tree = Path(value)
        else:
            raise FileNotFoundError(f"source not found or unsupported: {value}")

        project = _find_project_in_tree(tree)
        if project is None:
            raise RuntimeError(
                f"source {value!r} does not contain server/rimage_server/main.py"
            )
        copy_project(project, destination)
        return destination
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def ensure_launchers(project: Path, scripts_dir: Path) -> None:
    """Make sure <project>/kaggle/ holds this version's launcher scripts."""
    target = Path(project) / "kaggle"
    if (target / "launch.py").is_file() and (target / "bootstrap.py").is_file():
        return
    target.mkdir(parents=True, exist_ok=True)
    for name in LAUNCHER_FILES:
        source = Path(scripts_dir) / name
        if source.is_file():
            shutil.copy2(source, target / name)


# ---------------------------------------------------------------------------
# project operations
# ---------------------------------------------------------------------------
def copy_project(source: Path, destination: Path) -> Path:
    destination = Path(destination).resolve()
    if Path(source).resolve() == destination:
        return destination
    if destination.exists():
        shutil.rmtree(destination)
    log(f"copying project: {source} -> {destination}")
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(*SKIP_DIRS, "*.pyc"),
        dirs_exist_ok=True,
    )
    return destination


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    log("$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None, env=env)


def install_requirements(project: Path, files: Iterable[str]) -> None:
    requirement_files = [project / "requirements" / name for name in files]
    missing = [str(path) for path in requirement_files if not path.is_file()]
    if missing:
        die(
            "missing requirements files: "
            + ", ".join(missing)
            + "\nthis usually means the dataset only contains the launcher scripts; "
            "pass --source <git-url|archive|dir> to bootstrap.py",
            2,
        )
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q"]
    for path in requirement_files:
        cmd += ["-r", str(path)]
    if run(cmd) != 0:
        die("dependency installation failed; see the pip output above", 2)


def http_json(url: str, *, token: str | None = None, timeout: float = 10.0) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            try:
                return response.status, json.loads(body)
            except json.JSONDecodeError:
                return response.status, {"raw": body[:200]}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"raw": body[:200]}
    except Exception as exc:  # URLError, timeouts, TLS
        return 0, {"error": f"{type(exc).__name__}: {exc}"}


def wait_for_health(base_url: str, *, token: str | None = None, timeout: float = 120.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    last: dict[str, Any] = {}
    while time.time() < deadline:
        status, payload = http_json(f"{base_url.rstrip('/')}/health", token=token, timeout=8)
        if status == 200:
            return payload
        last = payload
        time.sleep(2)
    raise TimeoutError(f"server did not become healthy at {base_url}: {last}")


def require_gpu() -> dict[str, Any]:
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return {
                "available": False,
                "reason": "CUDA not available; set Settings -> Accelerator -> GPU and restart",
                "torch": torch.__version__,
            }
        props = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "name": props.name,
            "total_vram_mb": int(props.total_memory / 1024 / 1024),
            "capability": f"{props.major}.{props.minor}",
            "torch": torch.__version__,
        }
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def download_cloudflared(dest_dir: Path, *, version: str = CLOUDFLARED_VERSION) -> Path:
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        die(f"cloudflared binary is pinned for linux-amd64, this machine is {platform.machine()}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    binary = dest_dir / "cloudflared"
    if binary.is_file():
        result = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=20)
        if version in (result.stdout + result.stderr):
            return binary
        warn(f"existing cloudflared does not match {version}; reinstalling")

    url = (
        "https://github.com/cloudflare/cloudflared/releases/download/"
        f"{version}/cloudflared-linux-amd64"
    )
    log(f"downloading cloudflared {version} ...")
    tmp = binary.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    expected = os.environ.get("RIMAGE_CLOUDFLARED_SHA256", "").strip().lower()
    if expected:
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != expected:
            tmp.unlink(missing_ok=True)
            die("cloudflared checksum mismatch (RIMAGE_CLOUDFLARED_SHA256)")
    else:
        warn(
            "cloudflared checksum not pinned; set RIMAGE_CLOUDFLARED_SHA256 to verify the "
            "binary from Cloudflare's GitHub release"
        )
    tmp.chmod(0o755)
    tmp.replace(binary)
    result = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=20)
    if version not in (result.stdout + result.stderr):
        die(f"downloaded cloudflared is not version {version}: {result.stdout.strip()}")
    log(f"cloudflared ready: {binary} ({version})")
    return binary


def tail_file(path: Path, lines: int = 40) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(no log)"
    return "\n".join(content[-lines:])


def ensure_dir(path: Path, mode: int = 0o750) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, mode)
    except OSError:
        pass
    return path


def write_private(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
