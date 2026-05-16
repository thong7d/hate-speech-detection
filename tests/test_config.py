from src.utils.config import load_yaml_config, resolve_path


def test_required_configs_exist() -> None:
    for path in ["configs/train.yaml", "configs/model.yaml", "configs/api.yaml"]:
        config = load_yaml_config(path)
        assert isinstance(config, dict)
        assert config


def test_resolve_path_returns_absolute_path() -> None:
    assert resolve_path("configs/train.yaml").is_absolute()
