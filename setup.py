"""
Setup script for mono-bev package
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_long_description():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mono-bev",
    version="1.0.0",
    author="Nick Pai",
    author_email="weichenpai57@example.com",
    description="Monocular 2D-to-BEV Detection Pipeline for nuScenes",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/nick8592/mono-bev",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "monobev-train=scripts.train:main",
            "monobev-infer=scripts.pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "README.md", "LICENSE"],
    },
)
