#!/usr/bin/env python3
"""Kaggle notebook cell 2: run the headless server behind a Cloudflare tunnel.

    !python /kaggle/working/remote-image/kaggle/launch.py

Also works straight from a launcher-only dataset, e.g.
/kaggle/input/datasets/janemerser/remote-image-generator/launch.py, provided
bootstrap has already produced /kaggle/working/remote-image (or --source is
given).

What it does, in order:
  1. resolves the staged project + loads/generates the API token
     (env -> Kaggle Secret -> file -> new random)
  2. starts uvicorn on 127.0.0.1 only
  3. starts cloudflared (Quick Tunnel by default) and captures the real URL
  4. verifies the public /health endpoint and that auth is enforced
  5. prints the URL + client setup steps (never the token) and keeps running

Interrupt the cell (or stop the session) to shut everything down.
"""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit

from _common import (
    QUICK_TUNNEL_RE,
    Layout,
    detect_layout,
    die,
    download_cloudflared,
    ensure_dir,
    ensure_launchers,
    fetch_source,
    flat_dataset_hint,
    http_json,
    log,
    tail_file,
    wait_for_health,
    warn,
)

WORKING = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
RUNTIME = WORKING / ".rimage"
LOGS = WORKING / "rimage-logs"
OUTPUT = WORKING / "rimage-output"
TOKEN_FILE = RUNTIME / "api_token"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", help="project root (default: auto-detect)")
    parser.add_argument(
        "--source",
        default=os.environ.get("RIMAGE_SOURCE"),
        help="git URL, archive or directory with the project (used when the "
        "dataset only has the launcher scripts and no staged copy exists)",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tunnel", choices=["auto", "quick", "named", "none"], default="auto")
    parser.add_argument("--public-url", help="hostname of a named tunnel (or RIMAGE_PUBLIC_URL)")
    parser.add_argument("--mock", action="store_true", help="start the synthetic mock backend")
    parser.add_argument("--no-verify", action="store_true", help="skip public endpoint verification")
    parser.add_argument("--no-keep-alive", action="store_true", help="exit after verification")
    return parser.parse_args()


def resolve_token(project: Path) -> tuple[str, str]:
    """Use the server's own token resolution (env, Kaggle Secret, file, generate)."""
    sys.path.insert(0, str(project / "server"))
    try:
        from rimage_server.config import ConfigError, resolve_token as _resolve  # noqa: PLC0415

        try:
            return _resolve(os.environ, token_file=TOKEN_FILE)
        except ConfigError as exc:
            die(f"token configuration error: {exc}", 2)
    finally:
        sys.path.pop(0)
    raise AssertionError("unreachable")


class StreamCollector(threading.Thread):
    """Drains a subprocess stream into a log file, a ring buffer and a callback."""

    def __init__(self, stream, log_path: Path, lines: deque[str], on_line=None) -> None:
        super().__init__(daemon=True)
        self.stream = stream
        self.log_path = log_path
        self.lines = lines
        self.on_line = on_line

    def run(self) -> None:
        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                for raw in iter(self.stream.readline, ""):
                    line = raw.rstrip("\n")
                    handle.write(raw)
                    handle.flush()
                    self.lines.append(line)
                    if self.on_line:
                        self.on_line(line)
        except Exception:
            pass
        finally:
            try:
                self.stream.close()
            except Exception:
                pass


def start_quick_tunnel(binary: Path, port: int, log_path: Path) -> tuple[subprocess.Popen, str]:
    lines: deque[str] = deque(maxlen=200)
    found: dict[str, str] = {}
    ready = threading.Event()

    def scan(line: str) -> None:
        match = QUICK_TUNNEL_RE.search(line)
        if match and "url" not in found:
            found["url"] = match.group(0)
            ready.set()

    proc = subprocess.Popen(
        [
            str(binary),
            "tunnel",
            "--no-autoupdate",
            "--loglevel",
            "info",
            "--url",
            f"http://127.0.0.1:{port}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env={**os.environ, "NO_AUTOUPDATE": "true"},
    )
    collectors = [
        StreamCollector(proc.stdout, log_path, lines, scan),
        StreamCollector(proc.stderr, log_path, lines, scan),
    ]
    for collector in collectors:
        collector.start()

    deadline = time.time() + 90
    while time.time() < deadline:
        if ready.wait(0.5):
            return proc, found["url"]
        if proc.poll() is not None:
            break
    tail = "\n".join(list(lines)[-25:])
    try:
        proc.terminate()
    except Exception:
        pass
    die(f"cloudflared did not produce a Quick Tunnel URL.\n--- cloudflared output ---\n{tail}", 4)
    raise AssertionError("unreachable")


def start_named_tunnel(binary: Path, log_path: Path) -> tuple[subprocess.Popen, str | None]:
    tunnel_token = os.environ.get("CLOUDFLARE_TUNNEL_TOKEN", "").strip()
    if not tunnel_token:
        try:  # pragma: no cover - Kaggle-only import
            from kaggle_secrets import UserSecretsClient  # type: ignore

            tunnel_token = (UserSecretsClient().get_secret("CLOUDFLARE_TUNNEL_TOKEN") or "").strip()
        except Exception:
            tunnel_token = ""
    if not tunnel_token:
        die(
            "named tunnel requested but CLOUDFLARE_TUNNEL_TOKEN is not set "
            "(env or Kaggle Secret)",
            2,
        )
    env = {**os.environ, "TUNNEL_TOKEN": tunnel_token}
    # The token is passed via the environment, not argv, so it never appears in `ps`.
    proc = subprocess.Popen(
        [str(binary), "tunnel", "--no-autoupdate", "--loglevel", "info", "run"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    collector = StreamCollector(proc.stderr, log_path, deque(maxlen=200))
    collector.start()
    return proc, None


def verify_public(base_url: str, token: str) -> dict:
    log("verifying the public endpoint ...")
    health = wait_for_health(base_url, timeout=120)
    status, payload = http_json(f"{base_url}/v1/models", timeout=15)
    if status == 200:
        warn("the public API answered WITHOUT a token - this is a security problem, aborting")
        die("unauthenticated access detected", 5)
    if status not in {401, 403}:
        warn(f"unexpected unauthenticated status {status}: {payload}")

    status, payload = http_json(f"{base_url}/v1/models", token=token, timeout=20)
    if status != 200:
        die(f"authenticated request failed through the tunnel (HTTP {status}): {payload}", 5)
    log(f"public endpoint verified: /health 200, /v1/models 401 without token, 200 with token")
    return health


def print_summary(public_url: str, port: int, token_source: str, health: dict, log_path: Path) -> None:
    gpu = health.get("gpu", {})
    backend = health.get("backend", {})
    line = "=" * 74
    log("")
    log(line)
    log(" remote-image server is LIVE")
    log(line)
    log(f" Public URL   : {public_url}")
    log(f" Local URL    : http://127.0.0.1:{port}  (loopback only)")
    log(f" GPU          : {gpu.get('name') or 'none'} "
        f"({gpu.get('total_vram_mb') or 0} MB, bf16={gpu.get('bf16_supported')})")
    log(f" Backend      : {backend.get('type')}  models ready: {len(health.get('models', {}).get('available', []))}")
    log(f" Auth         : bearer token required ({token_source}); token is NOT printed")
    if TOKEN_FILE.is_file():
        log(f" Token file   : {TOKEN_FILE}  (open it in the Kaggle file browser)")
    else:
        log(" Token file   : - (token supplied via environment/secret)")
    log(f" Server log   : {log_path}")
    log(f" Output dir   : {OUTPUT}")
    log(line)
    log(" On your Linux machine (copy the token from the token file when prompted):")
    log(f"   remote-image configure --url {public_url}")
    log("   remote-image models")
    log(f"   remote-image generate --model sd15 --prompt \"a desert landscape at dusk\"")
    log("")
    log(" Quick Tunnel URLs are temporary: they change every time cloudflared restarts")
    log(" and stop working when this notebook session ends. Keep this cell running.")
    log(" Stop with the notebook's interrupt button; that terminates server + tunnel.")
    log(line)


def resolve_project(args: argparse.Namespace) -> tuple[Path, Layout]:
    """Find the staged project (or fetch it) and refresh the launchers."""
    layout = detect_layout(args.project_dir, script_file=__file__)
    log(f"launchers : {layout.scripts_dir}")

    project = layout.project_root
    if project is None:
        source = (args.source or "").strip()
        if not source:
            print(flat_dataset_hint(layout.scripts_dir), file=sys.stderr, flush=True)
            die("no runnable project found; run bootstrap.py first", 3)
        project = fetch_source(source, WORKING / "remote-image")

    log(f"project   : {project}  [{layout.mode if layout.project_root else 'fetched'}]")
    try:
        ensure_launchers(project, layout.scripts_dir)
    except OSError as exc:
        warn(f"could not refresh launcher scripts in {project}: {exc}")
    return project, layout


def main() -> int:
    args = parse_args()
    project, _layout = resolve_project(args)
    ensure_dir(RUNTIME, 0o700)
    ensure_dir(LOGS, 0o750)
    ensure_dir(OUTPUT, 0o750)

    token, token_source = resolve_token(project)
    if token_source == "generated":
        log(f"generated a new API token -> {TOKEN_FILE} (mode 0600)")
    else:
        log(f"API token loaded from {token_source} (value not printed)")

    server_log = LOGS / "server.log"
    server_env = {
        **os.environ,
        "RIMAGE_API_TOKEN": token,
        "RIMAGE_HOST": "127.0.0.1",
        "RIMAGE_PORT": str(args.port),
        "RIMAGE_OUTPUT_DIR": str(OUTPUT),
        "RIMAGE_DATA_DIR": str(RUNTIME),
        "RIMAGE_LOG_FILE": str(server_log),
        "RIMAGE_LOG_LEVEL": "INFO",
        "PYTHONPATH": str(project / "server"),
        "PYTHONUNBUFFERED": "1",
    }
    if args.mock:
        server_env["RIMAGE_ENABLE_MOCK"] = "1"

    log(f"starting server on 127.0.0.1:{args.port} ...")
    server_out = server_log.open("a", encoding="utf-8")
    server = subprocess.Popen(
        [sys.executable, "-m", "rimage_server", "--host", "127.0.0.1", "--port", str(args.port)],
        cwd=str(project / "server"),
        env=server_env,
        stdout=server_out,
        stderr=subprocess.STDOUT,
    )
    _PROCS["server"] = server

    try:
        health = wait_for_health(f"http://127.0.0.1:{args.port}", token=token, timeout=240)
    except Exception as exc:
        die(f"server failed to start: {exc}\n--- server log ---\n{tail_file(server_log, 50)}", 3)

    log(f"server healthy: backend={health.get('backend', {}).get('type')} "
        f"models={health.get('models', {}).get('count')}")

    tunnel = None
    public_url = ""
    tunnel_log = LOGS / "cloudflared.log"
    mode = args.tunnel
    if mode == "auto":
        mode = "named" if os.environ.get("CLOUDFLARE_TUNNEL_TOKEN") else "quick"

    if mode == "quick":
        binary = download_cloudflared(WORKING / "bin")
        tunnel, public_url = start_quick_tunnel(binary, args.port, tunnel_log)
        _PROCS["tunnel"] = tunnel
        log(f"cloudflared Quick Tunnel: {public_url}")
    elif mode == "named":
        binary = download_cloudflared(WORKING / "bin")
        public_url = (args.public_url or os.environ.get("RIMAGE_PUBLIC_URL") or "").rstrip("/")
        if not public_url:
            die("named tunnel needs --public-url or RIMAGE_PUBLIC_URL (the hostname in your CF config)", 2)
        tunnel, _ = start_named_tunnel(binary, tunnel_log)
        _PROCS["tunnel"] = tunnel
        log(f"cloudflared named tunnel running; expected public URL: {public_url}")
        time.sleep(5)
    else:
        public_url = f"http://127.0.0.1:{args.port}"
        warn("tunnel disabled (--tunnel none); the API is only reachable on this machine")

    if not args.no_verify and mode != "none":
        try:
            health = verify_public(public_url, token)
        except SystemExit:
            raise
        except Exception as exc:
            die(f"public verification failed: {exc}", 4)

    if public_url.startswith("https://"):
        (WORKING / "rimage-public-url.txt").write_text(public_url + "\n", encoding="utf-8")

    print_summary(public_url, args.port, token_source, health, server_log)

    if args.no_keep_alive:
        log("--no-keep-alive: stopping server and tunnel now")
        return 0

    stop = threading.Event()

    def handle(signum, _frame):
        log(f"\nreceived signal {signum}; shutting down ...")
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handle)
        except (ValueError, OSError):
            pass

    last_beat = time.time()
    while not stop.is_set():
        stop.wait(15)
        if stop.is_set():
            break
        if server.poll() is not None:
            warn("server process exited unexpectedly")
            log(tail_file(server_log, 60))
            return 1
        if tunnel is not None and tunnel.poll() is not None:
            warn("cloudflared exited; the public URL is dead")
            log(tail_file(tunnel_log, 30))
            return 1
        if time.time() - last_beat > 300:
            last_beat = time.time()
            status, payload = http_json(f"http://127.0.0.1:{args.port}/health", token=token, timeout=10)
            queue = payload.get("queue", {}) if status == 200 else {}
            log(f"[alive] jobs queued={queue.get('queued', '?')} "
                f"active={queue.get('active_job') or '-'} "
                f"loaded_model={(payload.get('backend') or {}).get('loaded_model') or '-'}")
    return 0


_PROCS: dict[str, subprocess.Popen | None] = {"server": None, "tunnel": None}


def cleanup() -> None:
    """Terminate the tunnel first, then the server (idempotent, atexit-safe)."""
    for name in ("tunnel", "server"):
        proc = _PROCS.get(name)
        if proc is None or proc.poll() is not None:
            continue
        log(f"stopping {name} ...")
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        except Exception:
            pass
        _PROCS[name] = None


atexit.register(cleanup)


if __name__ == "__main__":
    raise SystemExit(main())
