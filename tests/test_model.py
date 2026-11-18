"""Q-network architecture, initialization and checkpoint round-trip tests."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from config import Config
from model import QNetwork, build_q_network, predict_q_values
from utils import load_checkpoint, save_checkpoint, set_seed


def test_default_mlp_architecture() -> None:
    network = QNetwork()
    linear = [module for module in network.network if isinstance(module, nn.Linear)]
    assert len(linear) == 3
    assert linear[0].in_features == 192 and linear[0].out_features == 128
    assert linear[1].in_features == 128 and linear[1].out_features == 128
    assert linear[2].in_features == 128 and linear[2].out_features == 4
    assert not any(isinstance(module, (nn.Dropout, nn.BatchNorm1d)) for module in network.modules())


def test_all_biases_start_at_zero_and_init_is_once() -> None:
    network = QNetwork()
    for module in network.modules():
        if isinstance(module, nn.Linear):
            assert torch.count_nonzero(module.bias).item() == 0


def test_forward_batch_and_single_state_shapes() -> None:
    network = QNetwork()
    batch = torch.zeros(7, 192)
    output = network(batch)
    assert output.shape == (7, 4)
    single = network(torch.zeros(192))
    assert single.shape == (4,)
    assert torch.equal(output[0], single)
    assert torch.equal(network.get_q_values(batch), output)
    with torch.no_grad():
        assert predict_q_values(network, batch[0].numpy(), "cpu").shape == (1, 4)


def test_forward_rejects_wrong_state_dimension() -> None:
    network = QNetwork()
    try:
        network(torch.zeros(3, 191))
    except ValueError as exc:
        assert "192" in str(exc)
    else:
        raise AssertionError("Expected a dimension error")


def test_initialization_is_reproducible_with_seed() -> None:
    set_seed(123)
    first = QNetwork()
    set_seed(123)
    second = QNetwork()
    for (name_a, tensor_a), (name_b, tensor_b) in zip(
        first.state_dict().items(), second.state_dict().items()
    ):
        assert name_a == name_b
        assert torch.equal(tensor_a, tensor_b)


def test_configurable_hidden_capacity() -> None:
    config = Config().copy_with(hidden_dims=(64, 64))
    network = build_q_network(config)
    linear = [module for module in network.network if isinstance(module, nn.Linear)]
    assert [layer.out_features for layer in linear] == [64, 64, 4]


def test_checkpoint_save_and_load_produces_identical_outputs(tmp_path: Path) -> None:
    config = Config()
    set_seed(7)
    network = build_q_network(config)
    target = build_q_network(config)
    target.load_state_dict(network.state_dict())
    checkpoint_path = tmp_path / "best_model.pt"
    save_checkpoint(
        checkpoint_path,
        network,
        target_model=target,
        episode=17,
        config=config,
        extra={"note": "unit test"},
    )
    payload = load_checkpoint(checkpoint_path)
    reloaded = build_q_network(config)
    reloaded.load_state_dict(payload["model_state_dict"])
    reloaded.eval()
    network.eval()
    probe = torch.rand(11, config.state_dim)
    with torch.no_grad():
        expected = network(probe)
        actual = reloaded(probe)
    assert torch.allclose(expected, actual, atol=0, rtol=0)
    assert payload["episode"] == 17
    assert payload["note"] == "unit test"
    assert payload["config"]["hidden_dims"] == [128, 128]
    assert "target_model_state_dict" in payload
    assert next(reloaded.parameters()).device.type == "cpu"