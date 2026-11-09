import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


def run(cmd, **kwargs):
    print(f"[carl] {' '.join(cmd)}")
    subprocess.check_call(cmd, **kwargs)


def find_cuda_toolkit():
    """Locate a system toolkit; explicit CUDA_HOME must not silently fall back."""
    home = os.environ.get("CUDA_HOME")
    if home:
        root = Path(home).expanduser().resolve()
        if not (root / "bin" / "nvcc").is_file():
            raise RuntimeError(f"CARL: CUDA_HOME={root} does not contain bin/nvcc")
        return root

    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).resolve().parents[1]
    for root in (
        Path("/usr/local/cuda-12.8"),
        Path("/usr/local/cuda"),
        Path("/opt/cuda"),
    ):
        if (root / "bin" / "nvcc").is_file():
            return root.resolve()
    return None


def cuda_cmake_args():
    if sys.platform != "linux":
        raise RuntimeError("CARL: source builds currently support Linux only")
    root = find_cuda_toolkit()
    if root is None:
        raise RuntimeError(
            "CARL: source builds require the CUDA 12.8 development toolkit and a "
            "compatible C++ compiler (for example GCC 12). Install the toolkit, "
            "set CUDA_HOME to its root or put nvcc on PATH, then rerun uv sync. "
            "PyTorch's runtime libraries do not provide nvcc. Alternatively, "
            "install a compatible prebuilt CARL wheel."
        )
    nvcc = root / "bin" / "nvcc"
    try:
        version = subprocess.check_output([str(nvcc), "--version"], text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"CARL: could not execute {nvcc}") from exc
    if not re.search(r"\brelease 12\.8\b", version):
        raise RuntimeError(f"CARL: CUDA 12.8 is required; {nvcc} reported:\n{version}")

    targets = [
        root,
        root / "targets" / "x86_64-linux",
        root / "targets" / "aarch64-linux",
    ]
    if not any((target / "include" / "cuda_runtime.h").is_file() for target in targets):
        raise RuntimeError(
            f"CARL: CUDA headers missing under {root}; install the development toolkit"
        )
    if not any(
        (target / lib / "libcudart_static.a").is_file()
        for target in targets
        for lib in ("lib64", "lib")
    ):
        raise RuntimeError(
            f"CARL: libcudart_static.a missing under {root}; install the development toolkit"
        )

    print(f"[carl] CUDA compiler: {nvcc}")
    return [
        f"-DCMAKE_CUDA_COMPILER={nvcc}",
        f"-DCUDAToolkit_ROOT={root}",
    ]


def cuda_architectures():
    override = os.environ.get("CARL_CUDA_ARCHITECTURES")
    if override:
        print(f"[carl] CUDA architectures: {override} (override)")
        return override

    try:
        capabilities = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        ).splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    architectures = sorted(
        {capability.strip().replace(".", "") for capability in capabilities}
    )
    if not architectures or not all(
        architecture.isdigit() for architecture in architectures
    ):
        return None

    detected = ";".join(architectures)
    print(f"[carl] CUDA architectures: {detected} (detected)")
    return detected


def hip_cmake_args():
    if sys.platform != "linux":
        raise RuntimeError("CARL: source builds currently support Linux only")
    root = Path(os.environ.get("ROCM_PATH", "/opt/rocm")).expanduser().resolve()
    candidates = (
        root / "llvm/bin/clang++",
        root / "lib/llvm/bin/clang++",
    )
    compiler = next((path for path in candidates if path.is_file()), None)
    if compiler is None or not (root / "include/hip/hip_runtime.h").is_file():
        raise RuntimeError(
            f"CARL: ROCm development toolkit missing under {root}; "
            "install ROCm with hipCUB/rocPRIM and set ROCM_PATH"
        )

    args = [
        f"-DCMAKE_HIP_COMPILER={compiler}",
        f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={root}",
    ]
    architectures = os.environ.get("CARL_HIP_ARCHITECTURES")
    if architectures:
        args.append(f"-DCMAKE_HIP_ARCHITECTURES={architectures}")

    return args


def gpu_cmake_args():
    backend = os.environ.get("CARL_GPU_BACKEND", "CUDA").upper()
    if backend == "CUDA":
        args = ["-DCARL_GPU_BACKEND=CUDA", *cuda_cmake_args()]
        architectures = cuda_architectures()
        if architectures:
            args.append(f"-DCMAKE_CUDA_ARCHITECTURES={architectures}")
        return args

    if backend == "HIP":
        return ["-DCARL_GPU_BACKEND=HIP", *hip_cmake_args()]

    raise RuntimeError("CARL_GPU_BACKEND must be CUDA or HIP")


class CMakeBuild(build_ext):

    def run(self):
        toolkit_args = gpu_cmake_args()
        build_dir    = Path(self.build_temp)

        print("[carl] Locating pybind11...")
        pybind_dir = subprocess.check_output(
            [sys.executable, "-c", "import pybind11; print(pybind11.get_cmake_dir())"],
            text=True,
        ).strip()
        print(f"[carl] pybind11: {pybind_dir}")

        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

        print("[carl] Configuring CMake...")
        cmake_args = [
            "cmake",
            "-S",
            ".",
            "-B",
            str(build_dir),
            f"-Dpybind11_DIR={pybind_dir}",
            f"-DPython_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        ]

        cmake_args.extend(toolkit_args)

        run(cmake_args)

        print("[carl] Building carl_module...")
        run(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--config",
                "Release",
                "--target",
                "carl_module",
                "--parallel",
            ]
        )

        suffix          = sysconfig.get_config_var("EXT_SUFFIX")
        extension_files = list(build_dir.rglob(f"_carl*{suffix}"))
        if len(extension_files) != 1:
            found = ", ".join(str(path) for path in extension_files) or "none"
            raise RuntimeError(f"expected one _carl extension, found: {found}")

        dest = Path(self.build_lib) / "carl"
        dest.mkdir(parents=True, exist_ok=True)

        extension = extension_files[0]
        print(f"[carl] Installing {extension.name}")
        shutil.copy(extension, dest / extension.name)
        asset_dest = dest / "assets"
        asset_dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path("assets/arena.obj"), asset_dest / "arena.obj")

        if getattr(self, "editable_mode", False):
            source_dest = Path("python") / "carl"
            print(f"[carl] Installing editable extension into {source_dest}")
            shutil.copy(extension, source_dest / extension.name)

        print("[carl] Done")


setup(
    packages    = find_packages(where="python"),
    package_dir = {"": "python"},
    ext_modules = [Extension("carl._carl", sources=[])],
    cmdclass    = {"build_ext": CMakeBuild},
)
