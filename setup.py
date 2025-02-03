#!/usr/bin/env python3

from pathlib import Path
from setuptools import setup

directory = Path(__file__).resolve().parent
with open(directory / 'README.md', encoding='utf-8') as f:
  long_description = f.read()

  setup(name='robot',
    version='0.0.0',
    description='Open source bimanual robot',
    author='NYU GRAIL',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['robot'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    install_requires=["phoenix6", "numpy", "pyzmq", "rerun-sdk"],
    python_requires='>=3.10',
    extras_require={},
    include_package_data=True,
  )