from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
DIST = ROOT / "dist"
DEFAULT_ARCHITECTURES = "75;80;86;89;90"


def run(command: list[str], *, capture: bool = False) -> str:
    print(f"[release] {' '.join(command)}")
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout.strip() if capture else ""


def require(command: str) -> None:
    if shutil.which(command) is None:
        raise SystemExit(f"required command not found: {command}")


def require_github_cli() -> None:
    require("gh")
    version = subprocess.run(
        ["gh", "--version"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if not version.startswith("gh version "):
        raise SystemExit("gh on PATH is not the official GitHub CLI")

    authenticated = subprocess.run(
        ["gh", "auth", "status"],
        text=True,
        capture_output=True,
    )
    if authenticated.returncode != 0:
        raise SystemExit("GitHub CLI is not authenticated; run: gh auth login")


def verify_source(tag: str) -> None:
    status = run(["git", "status", "--porcelain"], capture=True)
    if status:
        raise SystemExit("release builds require a clean working tree")

    head = run(["git", "rev-parse", "HEAD"], capture=True)
    tagged = run(["git", "rev-list", "-n", "1", tag], capture=True)
    if head != tagged:
        raise SystemExit(f"{tag} does not point at HEAD")


def verify_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()

    required = {
        "carl/__init__.py",
        "carl/gymnasium/__init__.py",
        "carl/gymnasium/torch.py",
    }
    missing = required.difference(names)
    if missing:
        raise SystemExit(f"wheel is missing: {', '.join(sorted(missing))}")

    if any("__pycache__" in name or name.endswith(".pyc") for name in names):
        raise SystemExit("wheel contains generated Python bytecode")

    native_suffix = ".pyd" if platform.system() == "Windows" else ".so"
    if not any(
        name.startswith("carl/_carl") and name.endswith(native_suffix) for name in names
    ):
        raise SystemExit(f"wheel does not contain a native {native_suffix} extension")


def build_wheel(python: str, architectures: str) -> Path:
    require("uv")
    require("nvcc")

    if BUILD.exists():
        shutil.rmtree(BUILD)

    command = [
        "uv",
        "build",
        "--wheel",
        "--clear",
        "--python",
        python,
        "--out-dir",
        str(DIST),
    ]

    env = os.environ.copy()
    env["CARL_CUDA_ARCHITECTURES"] = architectures
    print(f"[release] CUDA architectures: {architectures}")
    print(f"[release] {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, check=True, env=env)

    wheels = list(DIST.glob("carl-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one wheel in {DIST}, found {len(wheels)}")

    verify_wheel(wheels[0])
    return wheels[0]


def publish(tag: str) -> None:
    release = json.loads(
        run(
            [
                "gh",
                "release",
                "view",
                tag,
                "--json",
                "assets,isPrerelease",
            ],
            capture=True,
        )
    )
    assets = [asset["name"] for asset in release["assets"]]

    has_linux = any("linux" in name and name.endswith(".whl") for name in assets)
    has_windows = any("win_amd64" in name and name.endswith(".whl") for name in assets)
    if not has_linux or not has_windows:
        raise SystemExit(
            "release requires both Linux and Windows wheels before promotion"
        )

    run(["gh", "release", "edit", tag, "--prerelease=false", "--draft=false"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build, validate, and upload a native CARL wheel."
    )
    parser.add_argument("--tag", default="v0.1.0")
    parser.add_argument("--python", default="3.11")
    parser.add_argument("--architectures", default=DEFAULT_ARCHITECTURES)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    require("git")
    if args.upload or args.publish:
        require_github_cli()
        verify_source(args.tag)

    if not args.publish or args.upload:
        wheel = build_wheel(args.python, args.architectures)
        print(f"[release] Validated {wheel}")

        if args.upload:
            run(["gh", "release", "upload", args.tag, str(wheel), "--clobber"])
    if args.publish:
        publish(args.tag)


if __name__ == "__main__":
    main()
