import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    def run(self):
        build_dir = Path(self.build_temp)
        build_dir.mkdir(parents=True, exist_ok=True)

        pybind_dir = subprocess.check_output(
            [sys.executable, "-c",
             "import pybind11; print(pybind11.get_cmake_dir())"],
            text=True).strip()

        subprocess.check_call([
            "cmake", "-S", ".", "-B", str(build_dir),
            f"-Dpybind11_DIR={pybind_dir}",
        ])

        subprocess.check_call([
            "cmake", "--build", str(build_dir),
            "--target", "carl_module", "--parallel",
        ])

        so_files = list(build_dir.glob("carl*.so"))
        if not so_files:
            raise RuntimeError("carl module .so not found")

        dest = Path(self.build_lib) / "carl"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(so_files[0], dest / so_files[0].name)


setup(
    name="carl",
    version="0.1.0",
    description="CUDA Rocket League simulation",
    ext_modules=[Extension("carl", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
)
