#!/bin/bash
set -e
uv run python -m pytest tests/web_app/test_api.py
