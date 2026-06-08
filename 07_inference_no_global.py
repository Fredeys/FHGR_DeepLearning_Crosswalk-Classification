#!/usr/bin/env python3
"""External inference on DeepL_Datenset/no_global negative images."""

from __future__ import annotations

import importlib


inference = importlib.import_module("07_inference")


if __name__ == "__main__":
    inference.no_global_main()
