"""Round-trip checks for the Pydantic config schema."""

from pathlib import Path

import pytest
import yaml

from nanotse.utils.config import Config


@pytest.fixture
def smoke_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "smoke.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "seed": 0,
                "device": "cpu",
                "data": {"root": str(tmp_path / "data")},
                "train": {"steps": 10},
                "model": {"name": "tdse"},
            }
        )
    )
    return p


def test_config_round_trip(smoke_yaml: Path) -> None:
    cfg = Config.from_yaml(smoke_yaml)
    assert cfg.train.steps == 10
    assert cfg.data.sample_rate == 16000  # default kicked in
    assert cfg.data.batch_size == 4  # default kicked in
    assert cfg.model.name == "tdse"
    assert cfg.device == "cpu"


def test_config_loads_repo_smoke_yaml() -> None:
    """The checked-in configs/smoke.yaml must validate against the schema."""
    repo_root = Path(__file__).resolve().parents[1]
    cfg = Config.from_yaml(repo_root / "configs" / "smoke.yaml")
    assert cfg.data.num_clips == 200
    assert cfg.train.steps == 500


def test_config_loads_repo_a100_yaml() -> None:
    """The checked-in configs/a100.yaml must validate against the schema."""
    repo_root = Path(__file__).resolve().parents[1]
    cfg = Config.from_yaml(repo_root / "configs" / "a100.yaml")
    assert cfg.device == "cuda"
    assert cfg.model.name == "nanotse"


def test_config_rejects_bad_device(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "device": "tpu",  # not in the Literal
                "data": {"root": str(tmp_path)},
                "train": {"steps": 1},
                "model": {"name": "tdse"},
            }
        )
    )
    with pytest.raises(ValueError):  # noqa: PT011 - pydantic raises ValidationError ⊂ ValueError
        Config.from_yaml(p)
