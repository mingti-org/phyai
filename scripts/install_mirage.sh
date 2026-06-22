#!/usr/bin/env bash
git submodule update --init --recursive
cd third_party/mirage
uv pip install -e . -v
export MIRAGE_HOME=$(pwd)
