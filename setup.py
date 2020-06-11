#!/usr/bin/env python
from setuptools import setup
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()


setup(
    name='IETK-Ret',
    version='0.0.5',
    description='Image Enhancement Toolkit for Retinal Fundus Images IETK-Ret',
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/adgaudio/ietk-ret",
    author='Alex Gaudio',
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
    ],
    include_package_data=True,
    packages=['ietk'],
    install_requires=[
       "argparse", "opencv-python", "opencv-contrib-python",
        "matplotlib", "numpy", "pillow", "scikit-image",
        "scipy",
    ]
)
