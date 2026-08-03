from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


MODEL_RELATIVE_PATH = Path(
    "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
)

ATTENTION_REPLACEMENTS = {
    'vlm_config._attn_implementation = "flash_attention_2"': (
        "vlm_config._attn_implementation = " "self.config.attention_implementation"
    ),
    'vlm_config.text_config._attn_implementation = "flash_attention_2"': (
        "vlm_config.text_config._attn_implementation = "
        "self.config.attention_implementation"
    ),
    ("self.config.qwen_expert_config._attn_implementation = " '"flash_attention_2"'): (
        "self.config.qwen_expert_config._attn_implementation = "
        "self.config.attention_implementation"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the official LingBot V2 source and make its hard-coded "
            "FlashAttention2 selection honor the existing config field."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_source(source: Path, output: Path) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    if not (source / MODEL_RELATIVE_PATH).is_file():
        raise FileNotFoundError(
            f"official model source is missing: {source / MODEL_RELATIVE_PATH}"
        )
    if output.exists():
        raise FileExistsError(
            f"output already exists; choose a new directory: {output}"
        )

    shutil.copytree(
        source,
        output,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
        ),
    )
    model_path = output / MODEL_RELATIVE_PATH
    before_hash = sha256_file(model_path)
    source_text = model_path.read_text(encoding="utf-8")
    patched_text = source_text
    for old, new in ATTENTION_REPLACEMENTS.items():
        count = patched_text.count(old)
        if count != 1:
            raise RuntimeError(
                f"expected exactly one official attention line, found {count}: "
                f"{old}"
            )
        patched_text = patched_text.replace(old, new)
    model_path.write_text(patched_text, encoding="utf-8", newline="\n")
    after_hash = sha256_file(model_path)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source": str(source),
        "output": str(output),
        "model_relative_path": str(MODEL_RELATIVE_PATH),
        "model_sha256_before": before_hash,
        "model_sha256_after": after_hash,
        "changes": [
            {
                "before": old,
                "after": new,
                "purpose": (
                    "select the official eager attention implementation on "
                    "Thor through the existing config field"
                ),
            }
            for old, new in ATTENTION_REPLACEMENTS.items()
        ],
        "weight_or_model_equation_changes": False,
        "operator_backend_change": True,
        "operator_backend_change_detail": (
            "FlashAttention2 selection is replaced by the official eager "
            "attention selection because the Thor stack has no usable "
            "flash_attn package."
        ),
    }
    manifest_path = output / "PHYAI_THOR_COMPAT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = prepare_source(args.source, args.output)
    print("Prepared official Thor source")
    print(f"source model SHA256 : {manifest['model_sha256_before']}")
    print(f"patched model SHA256: {manifest['model_sha256_after']}")
    print(f"output              : {args.output.resolve()}")


if __name__ == "__main__":
    main()
