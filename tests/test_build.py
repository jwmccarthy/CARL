import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path
from unittest import mock

SETUP_PY = Path(__file__).resolve().parents[1] / "setup.py"


class BuildConfigurationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with (
            mock.patch("setuptools.setup") as setup,
            mock.patch(
                "subprocess.check_call", side_effect=AssertionError("unexpected build")
            ),
            mock.patch(
                "subprocess.check_output",
                side_effect=AssertionError("unexpected command"),
            ),
        ):
            namespace = runpy.run_path(str(SETUP_PY))

        setup.assert_called_once()
        cls.find_toolkit = staticmethod(namespace["find_cuda_toolkit"])
        cls.cmake_args = staticmethod(namespace["cuda_cmake_args"])
        cls.gpu_args = staticmethod(namespace["gpu_cmake_args"])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()

        self.start_patch(mock.patch.dict(os.environ, {}, clear=True))
        self.which = self.start_patch(mock.patch("shutil.which", return_value=None))
        self.start_patch(mock.patch.object(sys, "platform", "linux"))
        self.output = self.start_patch(
            mock.patch(
                "subprocess.check_output",
                return_value="Cuda compilation tools, release 12.8, V12.8.93\n",
            )
        )
        self.start_patch(
            mock.patch(
                "subprocess.check_call", side_effect=AssertionError("unexpected build")
            )
        )

    def start_patch(self, patcher):
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def touch(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def toolkit(self, target="", library="lib64"):
        self.touch("bin/nvcc")
        self.touch(Path(target) / "include/cuda_runtime.h")
        self.touch(Path(target) / library / "libcudart_static.a")

        os.environ["CUDA_HOME"] = str(self.root)

    def test_explicit_cuda_home(self):
        self.touch("bin/nvcc")
        os.environ["CUDA_HOME"] = str(self.root)

        self.assertEqual(self.find_toolkit(), self.root)
        self.which.assert_not_called()
        self.output.assert_not_called()

    def test_invalid_explicit_cuda_home_does_not_fall_back(self):
        os.environ["CUDA_HOME"] = str(self.root)
        self.which.return_value = "/other/cuda/bin/nvcc"

        with self.assertRaisesRegex(
            RuntimeError, "CUDA_HOME=.*does not contain bin/nvcc"
        ):
            self.find_toolkit()

        self.which.assert_not_called()

    def test_path_nvcc_resolves_symlink_to_toolkit(self):
        compiler = self.touch("toolkit/bin/nvcc")
        link = self.root / "nvcc"
        link.symlink_to(compiler)
        self.which.return_value = str(link)

        self.assertEqual(self.find_toolkit(), compiler.parents[1])
        self.which.assert_called_once_with("nvcc")

    def test_standard_toolkit_roots_in_priority_order(self):
        roots = [
            Path("/usr/local/cuda-12.8"),
            Path("/usr/local/cuda"),
            Path("/opt/cuda"),
        ]
        with (
            mock.patch.object(Path, "is_file", autospec=True) as is_file,
            mock.patch.object(Path, "resolve", autospec=True, side_effect=lambda path: path),
        ):
            for index, expected in enumerate(roots):
                available = {root / "bin/nvcc" for root in roots[index:]}
                is_file.side_effect = lambda path: path in available

                with self.subTest(root=expected):
                    self.assertEqual(self.find_toolkit(), expected)

    def test_no_toolkit_found(self):
        with mock.patch.object(Path, "is_file", return_value=False):
            self.assertIsNone(self.find_toolkit())

    def test_cuda_128_returns_compiler_and_toolkit_arguments(self):
        self.toolkit()

        self.assertEqual(
            self.cmake_args(),
            [
                f"-DCMAKE_CUDA_COMPILER={self.root / 'bin/nvcc'}",
                f"-DCUDAToolkit_ROOT={self.root}",
            ],
        )
        self.output.assert_called_once_with(
            [str(self.root / "bin/nvcc"), "--version"], text=True
        )

    def test_target_specific_headers_and_static_libraries(self):
        self.touch("bin/nvcc")
        os.environ["CUDA_HOME"] = str(self.root)
        targets = ("targets/x86_64-linux", "targets/aarch64-linux")
        for target, library in product(targets, ("lib", "lib64")):
            with self.subTest(target=target, library=library):
                self.check_target_toolkit(target, library)

    def check_target_toolkit(self, target, library):
        header = self.touch(Path(target) / "include/cuda_runtime.h")
        archive = self.touch(Path(target) / library / "libcudart_static.a")

        try:
            self.assertIn(f"-DCUDAToolkit_ROOT={self.root}", self.cmake_args())
        finally:
            header.unlink()
            archive.unlink()

    def test_rejects_cuda_129(self):
        self.toolkit()
        self.output.return_value = "Cuda compilation tools, release 12.9, V12.9.41"

        with self.assertRaisesRegex(
            RuntimeError, r"CUDA 12\.8 is required;.*reported:"
        ):
            self.cmake_args()

    def test_missing_headers(self):
        self.toolkit()
        (self.root / "include/cuda_runtime.h").unlink()

        with self.assertRaisesRegex(
            RuntimeError, "CUDA headers missing.*development toolkit"
        ):
            self.cmake_args()

    def test_missing_static_runtime(self):
        self.toolkit()
        (self.root / "lib64/libcudart_static.a").unlink()
        self.touch("lib64/libcudart.so")

        with self.assertRaisesRegex(
            RuntimeError, r"libcudart_static\.a missing.*development toolkit"
        ):
            self.cmake_args()

    def test_missing_toolkit_has_actionable_error(self):
        with (
            mock.patch.object(Path, "is_file", return_value=False),
            self.assertRaises(RuntimeError) as raised,
        ):
            self.cmake_args()

        message = str(raised.exception)
        for hint in (
            "CUDA 12.8 development toolkit",
            "C++ compiler",
            "CUDA_HOME",
            "PATH",
            "prebuilt CARL wheel",
        ):
            self.assertIn(hint, message)

        self.output.assert_not_called()

    def test_requires_linux(self):
        with (
            mock.patch.object(sys, "platform", "darwin"),
            self.assertRaisesRegex(RuntimeError, "Linux only"),
        ):
            self.cmake_args()

        self.which.assert_not_called()
        self.output.assert_not_called()

    def test_gpu_backend_defaults_to_cuda(self):
        self.toolkit()
        os.environ["CARL_CUDA_ARCHITECTURES"] = "90"
        expected = [
            "-DCARL_GPU_BACKEND=CUDA",
            *self.cmake_args(),
            "-DCMAKE_CUDA_ARCHITECTURES=90",
        ]
        self.assertEqual(self.gpu_args(), expected)
        os.environ["CARL_GPU_BACKEND"] = "cuda"
        self.assertEqual(self.gpu_args(), expected)

    def test_hip_configuration(self):
        os.environ.update(
            CARL_GPU_BACKEND="hip",
            CARL_CUDA_ARCHITECTURES="90",
            CUDA_HOME=str(self.root / "missing-cuda"),
        )
        for index, compiler in enumerate(("llvm/bin/clang++", "lib/llvm/bin/clang++")):
            root = self.root / str(index)
            self.touch(root / compiler)
            self.touch(root / "include/hip/hip_runtime.h")
            os.environ["ROCM_PATH"] = str(root)
            for arch in ("", "gfx1100"):
                with self.subTest(compiler=compiler, arch=arch):
                    os.environ["CARL_HIP_ARCHITECTURES"] = arch
                    expected = [
                        "-DCARL_GPU_BACKEND=HIP",
                        f"-DCMAKE_HIP_COMPILER={root / compiler}",
                        f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={root}",
                    ]
                    if arch:
                        expected.append(f"-DCMAKE_HIP_ARCHITECTURES={arch}")
                    self.assertEqual(self.gpu_args(), expected)
        self.which.assert_not_called()
        self.output.assert_not_called()

    def test_gpu_backend_errors(self):
        os.environ["CARL_GPU_BACKEND"] = "METAL"
        with self.assertRaisesRegex(RuntimeError, "must be CUDA or HIP"):
            self.gpu_args()
        os.environ["CARL_GPU_BACKEND"] = "HIP"
        cases = ((), ("llvm/bin/clang++",), ("include/hip/hip_runtime.h",))
        for index, files in enumerate(cases):
            root = self.root / str(index)
            os.environ["ROCM_PATH"] = str(root)
            for path in files:
                self.touch(root / path)
            with self.subTest(files=files), self.assertRaisesRegex(
                RuntimeError, "ROCm development toolkit"
            ):
                self.gpu_args()
        with mock.patch.object(sys, "platform", "darwin"), self.assertRaisesRegex(
            RuntimeError, "Linux only"
        ):
            self.gpu_args()
        self.which.assert_not_called()
        self.output.assert_not_called()

    def test_unusable_nvcc_has_clear_error(self):
        self.toolkit()
        for error in (
            OSError("not executable"),
            subprocess.CalledProcessError(1, "nvcc"),
        ):
            with self.subTest(error=type(error).__name__):
                self.check_unusable_nvcc(error)

    def check_unusable_nvcc(self, error):
        self.output.side_effect = error

        with self.assertRaisesRegex(RuntimeError, "could not execute.*bin/nvcc") as raised:
            self.cmake_args()

        self.assertIs(raised.exception.__cause__, error)


if __name__ == "__main__":
    unittest.main()
