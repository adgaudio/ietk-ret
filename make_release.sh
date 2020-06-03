#!/usr/bin/env bash

set -e
set -u
# Increment the MAJOR version when you make incompatible API changes.
# Increment the MINOR version when you add functionality in a backwards-compatible manner.
# Increment the PATCH version when you make backwards-compatible bug fixes. (Source)

part="$1"  # major | minor | patch
bumpversion "$part"

python3 setup.py sdist bdist_wheel
twine upload dist/*

rm -rf dist/

git push
git push --tags
# py
# pip install --index-url https://test.pypi.org/simple/ simplepytorch
