"""Built-in engine plugin loading tests."""

from __future__ import annotations

import os
import sys
import subprocess


def test_builtin_plugins_load_on_demand() -> None:
    script = """
import sys

from phyai.engine import Engine, EngineCore

model_modules = {
    "phyai.models.pi0.main_pi0",
    "phyai.models.pi05.main_pi05",
    "phyai.models.cosmos3.main_cosmos3",
    "phyai.models.cosmos3.main_cosmos3_policy",
    "phyai.models.cosmos3.main_cosmos3_staged",
    "phyai.models.gr00t_n17.main_gr00t_n17",
    "phyai.models.qwen_image_21.main_qwen_image_21",
    "phyai.models.minicpm_gr00t.main_minicpm_gr00t",
    "phyai.models.dreamzero.main_dreamzero",
}
assert model_modules.isdisjoint(sys.modules)
assert "pi05" in Engine.registered()
assert "cosmos3_staged" in Engine.registered()
assert "qwen_image_21" in Engine.registered()
assert "dreamzero" in Engine.registered()

assert EngineCore.plugin_class("pi05").name == "pi05"
assert "phyai.models.pi05.main_pi05" in sys.modules
assert "phyai.models.qwen_image_21.main_qwen_image_21" not in sys.modules
assert "phyai_utils_tools" not in sys.modules

assert EngineCore.plugin_class("cosmos3_staged").name == "cosmos3_staged"
assert "phyai.models.cosmos3.main_cosmos3_staged" in sys.modules
assert "phyai.models.qwen_image_21.main_qwen_image_21" not in sys.modules

assert EngineCore.plugin_class("qwen_image_21").name == "qwen_image_21"
assert "phyai.models.qwen_image_21.main_qwen_image_21" in sys.modules

assert EngineCore.plugin_class("dreamzero").name == "dreamzero"
assert "phyai.models.dreamzero.main_dreamzero" in sys.modules
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    subprocess.run([sys.executable, "-c", script], check=True, env=env)
