"""phyai_kernel imports and probes aiter only when an adapter runs."""

from __future__ import annotations

import subprocess
import sys

MODULES = ("activation", "attention", "availability", "gemm", "norm", "rope")


def _run(code: str) -> None:
    subprocess.run([sys.executable, "-c", code], check=True)


def test_modules_import_without_aiter():
    _run(
        "import os, sys\n"
        "os.environ.pop('AITER_CONFIG_GEMM_BF16', None)\n"
        "sys.modules['aiter'] = None\n"
        "import phyai_kernel\n"
        f"for name in {MODULES!r}:\n"
        "    __import__(f'phyai_kernel.aiter.{name}')\n"
        "assert 'AITER_CONFIG_GEMM_BF16' not in os.environ\n"
    )


def test_probe_fails_without_aiter():
    _run(
        "import sys\n"
        "sys.modules['aiter'] = None\n"
        "import torch\n"
        "from phyai_kernel.aiter.availability import available\n"
        "if torch.version.hip is None:\n"
        "    assert available() is False\n"
        "else:\n"
        "    try:\n"
        "        available()\n"
        "    except ImportError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError('available() succeeded without aiter')\n"
    )


def test_probe_skips_aiter_on_non_rocm_torch():
    _run(
        "import sys, torch\n"
        "torch.version.hip = None\n"
        "from phyai_kernel.aiter.availability import available\n"
        "assert available() is False\n"
        "assert 'aiter' not in sys.modules\n"
    )
