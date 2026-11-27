"""Entry-point configuration parsing and processor interface settings."""

import pytest

from examples.deployment import agilex, tianji
from examples.deployment.pi05 import Pi05ProcessorConfig
from examples.deployment.client import validate_client
from examples.deployment.serve_pi05 import DEFAULT_CONFIG, Pi05ServerConfig
from examples.deployment.configuration import parse_config


@pytest.mark.parametrize(
    "module, schema, action_hz, horizon",
    [
        (tianji, tianji.TianjiClientConfig, 25.0, 20),
        (agilex, agilex.AgilexClientConfig, 30.0, 25),
    ],
)
def test_bundled_client_configs_are_valid(module, schema, action_hz, horizon):
    config = parse_config(
        schema, default_config=module.DEFAULT_CONFIG, argv=[], validate=validate_client
    )
    assert config.schedule.action_hz == action_hz
    assert config.schedule.execution_horizon == horizon
    assert config.control.to_options().control_hz == 200.0


def test_overrides_reach_nested_robot_settings():
    config = parse_config(
        tianji.TianjiClientConfig,
        default_config=tianji.DEFAULT_CONFIG,
        argv=["robot.mode=1", "robot.wrench_topics.left=/w", "schedule.action_hz=50"],
        validate=validate_client,
    )
    assert config.robot.mode == 1
    assert config.robot.wrench_topics == {"left": "/w"}
    assert config.robot.domain_id == 24
    assert config.schedule.create().action_hz == 50


@pytest.mark.parametrize(
    "override", ["schedule.strategy=unknown", "robot.mode=2", "control.control_hz=0"]
)
def test_invalid_client_settings_exit_before_any_io(override):
    with pytest.raises(SystemExit):
        parse_config(
            tianji.TianjiClientConfig,
            default_config=tianji.DEFAULT_CONFIG,
            argv=[override],
            validate=validate_client,
        )


def test_server_config_requires_an_existing_checkpoint(tmp_path):
    with pytest.raises(SystemExit):
        parse_config(
            Pi05ServerConfig,
            default_config=DEFAULT_CONFIG,
            argv=["task=pick", f"policy.checkpoint={tmp_path / 'missing'}"],
            validate=Pi05ServerConfig.validate,
        )
    config = parse_config(
        Pi05ServerConfig,
        default_config=DEFAULT_CONFIG,
        argv=["task=pick", f"policy.checkpoint={tmp_path}"],
        validate=Pi05ServerConfig.validate,
    )
    assert config.processor.image_resize_backend == "pil"


def test_processor_config_validation():
    Pi05ProcessorConfig(camera_names=["a"], action_dim=3).validate()
    for bad in (
        Pi05ProcessorConfig(camera_names=[]),
        Pi05ProcessorConfig(camera_names=["a", "a"]),
        Pi05ProcessorConfig(action_dim=0),
        Pi05ProcessorConfig(image_resize_backend="cv2"),
        Pi05ProcessorConfig(action_dtype="int64"),
    ):
        with pytest.raises(ValueError):
            bad.validate()


@pytest.mark.parametrize("key", ["camera_resize_hw", "camera_size"])
@pytest.mark.parametrize("source", ["cli", "yaml"])
def test_camera_size_overrides_bundled_default(tmp_path, key, source):
    argv = [f"robot.{key}=[300,400]"]
    if source == "yaml":
        path = tmp_path / "legacy.yaml"
        path.write_text(f"robot:\n  {key}: [300, 400]\n")
        argv = ["--config", str(path)]
    config = parse_config(
        tianji.TianjiClientConfig,
        default_config=tianji.DEFAULT_CONFIG,
        argv=argv,
        validate=validate_client,
    )
    assert config.robot.camera_size == [300, 400]
    assert config.robot.camera_resize_hw is None


@pytest.mark.parametrize("canonical", [[540, 960], None])
def test_conflicting_camera_aliases_raise_value_error(canonical):
    from omegaconf import OmegaConf

    from examples.deployment.configuration import normalize_aliases

    layer = OmegaConf.create(
        {"robot": {"camera_size": canonical, "camera_resize_hw": [300, 400]}}
    )
    with pytest.raises(ValueError, match="Conflicting.*camera_resize_hw.*camera_size"):
        normalize_aliases(layer, tianji.TianjiClientConfig.config_aliases)


def test_conflicting_camera_cli_reports_usage_error(capsys):
    with pytest.raises(SystemExit) as error:
        parse_config(
            tianji.TianjiClientConfig,
            default_config=tianji.DEFAULT_CONFIG,
            argv=["robot.camera_size=[540,960]", "robot.camera_resize_hw=[300,400]"],
        )
    assert error.value.code == 2
    assert "Conflicting" in capsys.readouterr().err


def test_identical_camera_aliases_are_accepted():
    config = parse_config(
        tianji.TianjiClientConfig,
        default_config=tianji.DEFAULT_CONFIG,
        argv=["robot.camera_size=[300,400]", "robot.camera_resize_hw=[300,400]"],
        validate=validate_client,
    )
    assert config.robot.camera_size == [300, 400]


def test_cli_camera_size_overrides_legacy_yaml(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text("robot:\n  camera_resize_hw: [300, 400]\n")
    config = parse_config(
        tianji.TianjiClientConfig,
        default_config=tianji.DEFAULT_CONFIG,
        argv=["--config", str(path), "robot.camera_size=null"],
        validate=validate_client,
    )
    assert config.robot.camera_size is None


@pytest.mark.parametrize("canonical", [None, [300, 400]])
def test_direct_tianji_config_accepts_camera_alias(canonical):
    config = tianji.TianjiConfig(camera_size=canonical, camera_resize_hw=[300, 400])
    config.validate()
    assert config.camera_size == [300, 400]


def test_direct_tianji_config_rejects_conflicting_camera_alias():
    config = tianji.TianjiConfig(camera_size=[540, 960], camera_resize_hw=[300, 400])
    with pytest.raises(ValueError, match="Conflicting"):
        config.validate()
