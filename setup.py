import shutil
import subprocess
import sys
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
            [sys.executable, "-c",
             "import pybind11; print(pybind11.get_cmake_dir())"],
            text=True).strip()
        print(f"[carl] pybind11: {pybind_dir}")

        build_dir.mkdir(parents=True, exist_ok=True)

        print("[carl] Configuring CMake...")
        run([
            "cmake", "-S", ".", "-B", str(build_dir),
            f"-Dpybind11_DIR={pybind_dir}",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        ])

        print("[carl] Building carl_module...")
        run([
            "cmake", "--build", str(build_dir),
            "--target", "carl_module", "--parallel",
        ])

        so_files = list(build_dir.glob("_carl*.so"))
        if not so_files:
            raise RuntimeError("_carl module .so not found")

        dest = Path(self.build_lib) / "carl"
        dest.mkdir(parents=True, exist_ok=True)

        print(f"[carl] Installing {so_files[0].name}")
        shutil.copy(so_files[0], dest / so_files[0].name)
        print("[carl] Done")


setup(
    name="carl",
    version="0.1.0",
    description="CUDA Rocket League simulation",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[Extension("carl._carl", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
)
