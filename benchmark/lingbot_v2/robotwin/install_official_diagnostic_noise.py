"""Install explicit-noise support in the Official LingBot V2 deploy wrapper.

This patches only the deployment boundary used for parity diagnostics. The
Official model already accepts an optional ``noise`` tensor; its WebSocket
wrapper normally does not expose it.
"""

from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path

DIAGNOSTIC_NOISE_KEY = "_lingbot_diagnostic_noise"
PATCH_MARKER = "LingBot explicit-noise diagnostic compatibility"


def patch_source(source: str) -> str:
    if PATCH_MARKER in source:
        return source

    prepare_start = source.index("    def _prepare_model_input(self, observation):")
    prepare_end = source.index("\n    @staticmethod", prepare_start)
    prepare = source[prepare_start:prepare_end]
    prepare_anchor = """        observation = dict(observation)\n        self.resize_image(observation)\n"""
    if prepare_anchor not in prepare:
        raise RuntimeError("Official _prepare_model_input anchor was not found.")
    prepare = prepare.replace(
        prepare_anchor,
        """        observation = dict(observation)\n        # LingBot explicit-noise diagnostic compatibility.\n        diagnostic_noise = observation.pop(\n            \"_lingbot_diagnostic_noise\", None\n        )\n        self.resize_image(observation)\n""",
        1,
    )
    return_anchor = """        if self.use_bf16:\n            observation['state'] = observation['state'].to(torch.bfloat16)\n        return observation\n"""
    if return_anchor not in prepare:
        raise RuntimeError("Official _prepare_model_input return anchor was not found.")
    prepare = prepare.replace(
        return_anchor,
        """        if self.use_bf16:\n            observation['state'] = observation['state'].to(torch.bfloat16)\n        if diagnostic_noise is not None:\n            observation[\"_lingbot_diagnostic_noise\"] = torch.as_tensor(\n                diagnostic_noise\n            )\n        return observation\n""",
        1,
    )
    source = source[:prepare_start] + prepare + source[prepare_end:]

    batch_start = source.index("    def sample_actions_batch(")
    batch_end = source.index("\nclass LingBotVlaV2InferencePolicy", batch_start)
    batch = source[batch_start:batch_end]
    state_anchor = """        state = observation[\"state\"]\n        image_grid_thw = observation.get(\"image_grid_thw\", None)\n"""
    if state_anchor not in batch:
        raise RuntimeError("Official sample_actions_batch state anchor was not found.")
    batch = batch.replace(
        state_anchor,
        """        state = observation[\"state\"]\n        diagnostic_noise = observation.get(\n            \"_lingbot_diagnostic_noise\", None\n        )\n        if diagnostic_noise is not None:\n            diagnostic_noise = diagnostic_noise.to(dtype=dtype, device=device)\n        image_grid_thw = observation.get(\"image_grid_thw\", None)\n""",
        1,
    )
    call_anchor = """                            image_grid_thw=self._to_device_image_grid_thw(image_grid_thw, device),\n"""
    call_count = batch.count(call_anchor)
    if call_count != 3:
        raise RuntimeError(
            "Expected three Official sample_actions_batch calls, found "
            f"{call_count}."
        )
    batch = batch.replace(
        call_anchor,
        """                            noise=diagnostic_noise,\n                            image_grid_thw=self._to_device_image_grid_thw(image_grid_thw, device),\n""",
    )
    source = source[:batch_start] + batch + source[batch_end:]
    ast.parse(source)
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    path = args.file.resolve()
    source = path.read_text(encoding="utf-8")
    patched = patch_source(source)
    if args.check:
        if patched == source and PATCH_MARKER in source:
            print(f"Official diagnostic-noise patch installed: {path}")
        else:
            print(f"Official diagnostic-noise patch can be installed: {path}")
        return
    if patched == source:
        print(f"Official diagnostic-noise patch already installed: {path}")
        return

    backup = path.with_suffix(path.suffix + ".before-diagnostic-noise")
    if not backup.exists():
        shutil.copy2(path, backup)
    temporary = path.with_suffix(path.suffix + ".diagnostic-noise.tmp")
    temporary.write_text(patched, encoding="utf-8")
    temporary.replace(path)
    print(f"Official diagnostic-noise patch installed: {path}")
    print(f"Backup: {backup}")


if __name__ == "__main__":
    main()
