# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from setuptools import find_packages, setup

import versioneer


def get_requires():
    def read_requires(requirements_file):
        with open(requirements_file, "r", encoding="utf-8") as f:
            file_content = f.read()
            lines = [
                line.strip() for line in file_content.strip().split("\n") if not line.startswith("#") and line != ""
            ]
            return lines

    return read_requires("requirements.txt")


extra_require = {
    "test": [],
    "gpu": [],
    "math": [],
    "vllm": [],
    "sglang": [],
    "mcore": [],
    "transferqueue": [],
}


def main():
    setup(
        name="relax",
        version=versioneer.get_version(),
        author="Relax Contributors",
        author_email="Xiaohongshu-AI Platform",
        description="Relax: Reinforcement Engine Leveraging Agentic X-modality",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords=["RL", "Agentic", "Multi-modality", "pytorch", "deep learning", "megatron", "vllm", "sglang"],
        license="Apache-2.0",
        url="https://github.com/redai-studio/Relax",
        package_dir={"": "."},
        package_data={"": ["*.yaml"]},
        packages=find_packages(),
        python_requires=">=3.10.0",
        install_requires=get_requires(),
        extras_require=extra_require,
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        cmdclass=dict(
            versioneer.get_cmdclass(),
        ),
    )


if __name__ == "__main__":
    main()
