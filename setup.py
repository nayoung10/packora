#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="packora",
    version="0.0.1",
    description="Flow-matching model for molecular crystal structure prediction",
    author="Packora authors",
    license="PolyForm-Noncommercial-1.0.0",
    author_email="",
    url="https://github.com/nayoung10/packora",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(
        include=["src", "src.*", "eval", "eval.*", "configs", "configs.*"]
    ),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "packora-predict = src.prediction.api.cli:main",
        ]
    },
)
