import os
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


class CMakeBuild(build_ext):
    def run(self):
        build_dir = Path(self.build_temp)

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

        architectures = os.environ.get("CARL_CUDA_ARCHITECTURES")
        if architectures:
            cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={architectures}")

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

        suffix = sysconfig.get_config_var("EXT_SUFFIX")
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
        print("[carl] Done")


setup(
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[Extension("carl._carl", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
)
