#!/usr/bin/env python3
"""Kaggle notebook cell 1: stage the project, install pinned deps, inspect the GPU.

Works with either dataset layout:

* full repository dataset:
    !python /kaggle/input/remote-image-generator/kaggle/bootstrap.py
* flat launcher dataset (only bootstrap.py/launch.py/_common.py, e.g.
  /kaggle/input/datasets/janemerser/remote-image-generator/bootstrap.py):
    !python /kaggle/input/datasets/janemerser/remote-image-generator/bootstrap.py \
        --source https://github.com/<you>/remote-image-generator.git

The runnable copy is always left at /kaggle/working/remote-image.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from _common import (
    Layout,
    copy_project,
    detect_layout,
    die,
    ensure_launchers,
    fetch_source,
    flat_dataset_hint,
    install_requirements,
    log,
    require_gpu,
    warn,
)

DEFAULT_DEST = Path("/kaggle/working/remote-image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project-dir", help="repository root to copy/use")
    parser.add_argument(
        "--source",
        default=os.environ.get("RIMAGE_SOURCE"),
        help="git URL, archive (URL or path) or directory containing the project "
        "(required when the dataset only has the launcher scripts)",
    )
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help="writable copy of the project")
    parser.add_argument("--skip-install", action="store_true", help="do not run pip install")
    parser.add_argument("--skip-copy", action="store_true", help="run in place")
    parser.add_argument("--json", action="store_true", help="machine-readable summary")
    return parser.parse_args()


def dependency_versions(project: Path) -> dict[str, str | None]:
    sys.path.insert(0, str(project / "server"))
    try:
        from rimage_server.gpu import environment_report  # noqa: PLC0415

        return environment_report(None)["packages"]
    except Exception as exc:
        warn(f"could not inspect dependency versions: {type(exc).__name__}: {exc}")
        return {}
    finally:
        sys.path.pop(0)


def diagnose(project: Path) -> dict:
    sys.path.insert(0, str(project / "server"))
    try:
        from rimage_server.gpu import environment_report  # noqa: PLC0415
        from rimage_server.registry import ModelRegistry  # noqa: PLC0415

        return environment_report(None, ModelRegistry())
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        sys.path.pop(0)


def stage_project(args: argparse.Namespace, layout: Layout) -> Path:
    """Return a runnable project directory, fetching or copying when necessary."""
    dest = Path(args.dest)
    source = (args.source or "").strip()

    if layout.project_root is None:
        if not source:
            die(flat_dataset_hint(layout.scripts_dir), 3)
        project = fetch_source(source, dest)
    elif source:
        log(f"--source given; staging from {source} instead of {layout.project_root}")
        project = fetch_source(source, dest)
    elif args.skip_copy:
        project = layout.project_root
    else:
        project = copy_project(layout.project_root, dest)

    try:
        ensure_launchers(project, layout.scripts_dir)
    except OSError as exc:
        warn(f"could not refresh launcher scripts in {project}: {exc}")

    if not (project / "requirements").is_dir():
        die(
            f"{project} has no requirements/ directory - it is not a complete project copy.\n"
            "Pass --source <git-url|archive|dir> pointing at the full repository.",
            2,
        )
    return project


def main() -> int:
    args = parse_args()
    log("=" * 72)
    log(" remote-image-generator | Kaggle bootstrap")
    log("=" * 72)

    layout = detect_layout(args.project_dir, script_file=__file__)
    log(f"launchers : {layout.scripts_dir}")
    if layout.project_root is not None:
        log(f"project   : {layout.project_root}  [{layout.mode}]")
    else:
        warn("launcher-only dataset detected (no server/ or requirements/ here)")

    gpu = require_gpu()
    if gpu.get("available"):
        log(f"GPU       : {gpu['name']}  {gpu['total_vram_mb']} MB  cc {gpu['capability']}")
        log(f"torch     : {gpu.get('torch')}")
    else:
        warn(f"no usable GPU: {gpu.get('reason')}")
        warn("the server will start but every diffusion model will be reported unavailable")

    project = stage_project(args, layout)
    log(f"working   : {project}")

    if not args.skip_install:
        log("\ninstalling pinned dependencies (torch is left untouched) ...")
        install_requirements(project, ["api.txt", "ml.txt"])

    log("\nverifying imports ...")
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch, diffusers, transformers, accelerate, fastapi, uvicorn;"
            "print('torch', torch.__version__, 'cuda', torch.cuda.is_available());"
            "print('diffusers', diffusers.__version__, 'transformers', transformers.__version__)",
        ],
        capture_output=True,
        text=True,
    )
    if check.stdout.strip():
        log(check.stdout.strip())
    if check.returncode != 0:
        die("import check failed:\n" + (check.stderr or "")[-1500:], 2)

    report = diagnose(project)
    versions = report.get("packages", {})
    log("\ndependency versions:")
    for name in ("torch", "diffusers", "transformers", "accelerate", "huggingface-hub", "fastapi"):
        log(f"  {name:<18} {versions.get(name)}")

    models = report.get("models", {})
    if models:
        log("\nmodel compatibility on this GPU:")
        for model_id, info in models.items():
            flag = "OK  " if info["available"] else "NO  "
            reason = f"  <- {info['reason']}" if info["reason"] else ""
            log(f"  [{flag}] {model_id:<15} min {info['min_vram_mb']:>6} MB  {info['precision']}{reason}")

    launch_script = project / "kaggle" / "launch.py"
    if args.json:
        print(
            json.dumps(
                {
                    "gpu": gpu,
                    "project": str(project),
                    "scripts_dir": str(layout.scripts_dir),
                    "layout": layout.mode,
                    "report": report,
                },
                default=str,
            )
        )

    log("\nnext step:")
    log(f"  !python {launch_script}")
    log(f"  (or from this dataset: !python {layout.scripts_dir / 'launch.py'})")
    log("\nReminder: enable Internet in the notebook settings; Quick Tunnel needs outbound access.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
