"""Build the portable C ABI alongside Python, including editable installs."""

import json
import shutil
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


class RustBuild(build_ext):
    def build_extension(self, ext):
        root = Path(__file__).parent.resolve()
        completed = subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--locked",
                "-p",
                "argusnet-native",
                "--message-format=json",
            ],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        artifacts = [
            json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")
        ]
        library = next(
            Path(name)
            for item in artifacts
            if item.get("reason") == "compiler-artifact"
            and item.get("target", {}).get("name") == "argusnet_native"
            for name in item["filenames"]
            if name.endswith((".so", ".dylib", ".dll"))
        )
        destination = Path(self.get_ext_fullpath(ext.name))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(library, destination)


class BuildPython(build_py):
    def run(self):
        super().run()
        if self.editable_mode:
            return  # Editable installs read the schema from the source checkout.
        destination = Path(self.build_lib) / "argusnet" / "evaluation" / "replay-schema.json"
        shutil.copy2(Path(__file__).parent / "docs" / "replay-schema.json", destination)


setup(
    ext_modules=[Extension("argusnet._native", sources=[])],
    cmdclass={"build_ext": RustBuild, "build_py": BuildPython},
)
