"""Q-learning target, CUDA placement, replay, sampler and update tests."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pytest
import torch

from check_cuda import assert_batch_on_cuda, assert_model_on_cuda, require_cuda
from config import Config
from generator import load_dataset
from model import build_q_network
from replay_buffer import ReplayBuffer
from search import QGuidedWAStar, SearchResult, TransitionExperience
from trainer import (
    CATEGORY_SATISFICED,
    CATEGORY_SOLVED,
    CATEGORY_UNSOLVED,
    InstancePoolSampler,
    QTrainer,
    validate_network,
)
from utils import set_seed


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The project requires an NVIDIA CUDA device for training tests",
)

DEVICE = "cuda:0"


def _constant_target(trainer: QTrainer, values: tuple[float, float, float, float]) -> None:
    output_layer = trainer.target_model.network[-1]
    with torch.no_grad():
        output_layer.weight.zero_()
        output_layer.bias.copy_(torch.tensor(values, dtype=torch.float32, device=DEVICE))


def _batch(batch_size: int = 4) -> Dict[str, torch.Tensor]:
    states = torch.zeros((batch_size, 192), dtype=torch.float32, device=DEVICE)
    actions = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)
    next_states = torch.zeros((batch_size, 192), dtype=torch.float32, device=DEVICE)
    return_bounds = torch.full(
        (batch_size,), -math.inf, dtype=torch.float32, device=DEVICE
    )
    return_bounds[3] = -1.5
    terminal = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)
    terminal[1] = True
    dead_end = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)
    dead_end[2] = True
    next_mask = torch.zeros((batch_size, 4), dtype=torch.bool, device=DEVICE)
    next_mask[0, 1] = True
    next_mask[1, 2] = True
    next_mask[3, 1] = True
    return {
        "states": states,
        "actions": actions,
        "next_states": next_states,
        "return_bounds": return_bounds,
        "terminal": terminal,
        "dead_end": dead_end,
        "next_valid_mask": next_mask,
    }


def _dummy_experience(action: int) -> TransitionExperience:
    state = np.zeros(192, dtype=np.float32)
    state[64 + action] = 1.0
    next_state = np.zeros(192, dtype=np.float32)
    next_state[64 + action] = 1.0
    return TransitionExperience(
        state=state,
        action=action,
        next_state=next_state,
        return_bound=-math.inf,
        terminal=False,
        dead_end=False,
        next_valid_mask=np.ones(4, dtype=np.bool_),
    )


def test_cuda_requirement_and_model_placement() -> None:
    device = require_cuda(DEVICE, verbose=False)
    assert device.type == "cuda"
    config = Config()
    set_seed(5)
    model = build_q_network(config).to(device)
    trainer = QTrainer(config, device=device, model=model)
    assert_model_on_cuda(trainer.model, device)
    assert_model_on_cuda(trainer.target_model, device)
    assert next(trainer.model.parameters()).device.type == "cuda"
    with pytest.raises(RuntimeError):
        QTrainer(config, device="cpu")


def test_bellman_mask_solution_bound_terminal_and_dead_end_targets() -> None:
    config = Config()
    trainer = QTrainer(config, device=DEVICE)
    _constant_target(trainer, (100.0, -2.0, 50.0, 70.0))
    batch = _batch()
    assert_batch_on_cuda(batch, DEVICE)
    target = trainer.compute_targets(batch)
    assert target.shape == (4,)
    # Row 0: illegal actions carry high Q but only legal action 1 is used.
    assert target[0].item() == pytest.approx(-3.0)
    # Row 1: terminal goal, no bootstrap.
    assert target[1].item() == pytest.approx(-1.0)
    # Row 2: true dead end.
    assert target[2].item() == pytest.approx(-64.0)
    # Row 3: finite search bound -1.5 beats bootstrap -1 + (-2) = -3.
    assert target[3].item() == pytest.approx(-1.5)


def test_no_solution_bounds_ablation_uses_only_bellman_target() -> None:
    config = Config().copy_with(use_solution_bounds=False)
    trainer = QTrainer(config, device=DEVICE)
    _constant_target(trainer, (100.0, -2.0, 50.0, 70.0))
    batch = _batch()
    target = trainer.compute_targets(batch)
    assert target[3].item() == pytest.approx(-3.0)


def test_bad_device_batch_is_rejected() -> None:
    config = Config()
    trainer = QTrainer(config, device=DEVICE)
    cpu_batch = {
        key: tensor.cpu() if isinstance(tensor, torch.Tensor) else tensor
        for key, tensor in _batch().items()
    }
    with pytest.raises(RuntimeError):
        trainer.train_step(cpu_batch)


def test_backprop_updates_online_network_not_target() -> None:
    config = Config().copy_with(learning_rate=1e-3, target_update_interval=1000)
    trainer = QTrainer(config, device=DEVICE)
    _constant_target(trainer, (100.0, -2.0, 50.0, 70.0))
    before = [parameter.detach().clone() for parameter in trainer.model.parameters()]
    target_before = [parameter.detach().clone() for parameter in trainer.target_model.parameters()]
    stats = trainer.train_step(_batch())
    assert stats.gradient_updates == 1
    assert stats.target_network_updated is False
    assert math.isfinite(stats.loss)
    assert math.isfinite(stats.q_mean)
    assert stats.q_min <= stats.q_max
    changed = any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, trainer.model.parameters())
    )
    assert changed, "Backpropagation must modify online-network parameters"
    assert all(
        torch.equal(previous, current.detach())
        for previous, current in zip(target_before, trainer.target_model.parameters())
    )
    assert all(not parameter.requires_grad for parameter in trainer.target_model.parameters())


def test_target_network_sync_frequency() -> None:
    config = Config().copy_with(target_update_interval=2)
    trainer = QTrainer(config, device=DEVICE)
    _constant_target(trainer, (10.0, 20.0, 30.0, 40.0))
    target_initial = {
        name: value.detach().clone() for name, value in trainer.target_model.state_dict().items()
    }
    batch = _batch()
    first = trainer.train_step(batch)
    assert first.target_network_updated is False
    assert all(
        torch.equal(target_initial[name], value)
        for name, value in trainer.target_model.state_dict().items()
    ), "Target must not sync before the configured interval"
    second = trainer.train_step(batch)
    assert second.target_network_updated is True
    assert trainer.target_updates == 1
    assert all(
        torch.equal(online, target)
        for online, target in zip(
            trainer.model.state_dict().values(), trainer.target_model.state_dict().values()
        )
    )


def test_replay_buffer_fifo_sampling_and_tensor_collation() -> None:
    buffer = ReplayBuffer(capacity=5, seed=3)
    for index in range(7):
        experience = _dummy_experience(index % 4)
        experience.return_bound = float(index)
        buffer.add(experience)
    assert len(buffer) == 5
    assert [experience.action for experience in buffer.get_all()] == [2, 3, 0, 1, 2]
    assert [experience.return_bound for experience in buffer.get_all()] == [2.0, 3.0, 4.0, 5.0, 6.0]
    batch = buffer.sample(3, device=DEVICE)
    assert set(batch) == {
        "states",
        "actions",
        "next_states",
        "return_bounds",
        "terminal",
        "dead_end",
        "next_valid_mask",
    }
    assert batch["states"].shape == (3, 192)
    assert batch["actions"].shape == (3,)
    assert batch["next_valid_mask"].shape == (3, 4)
    assert batch["states"].device.type == "cuda"
    assert_batch_on_cuda(batch, DEVICE)
    with pytest.raises(ValueError):
        buffer.sample(6, device=DEVICE)
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0)


def test_instance_pool_sampler_is_mutually_exclusive() -> None:
    config = Config()
    sampler = InstancePoolSampler(config, [0, 1, 2, 3], seed=42)
    assert sampler.counts() == {
        CATEGORY_UNSOLVED: 4,
        CATEGORY_SOLVED: 0,
        CATEGORY_SATISFICED: 0,
    }
    success_short = SearchResult(True, expansions=3, actions=[0, 1, 2])
    success_long = SearchResult(True, expansions=8, actions=[0, 1, 2])
    failure = SearchResult(False, expansions=8)
    assert InstancePoolSampler.classify(success_short) == CATEGORY_SOLVED
    assert InstancePoolSampler.classify(success_long) == CATEGORY_SATISFICED
    assert InstancePoolSampler.classify(failure) == CATEGORY_UNSOLVED
    sampler.update(0, CATEGORY_SOLVED)
    sampler.update(1, CATEGORY_SATISFICED)
    assert sampler.counts() == {
        CATEGORY_UNSOLVED: 2,
        CATEGORY_SOLVED: 1,
        CATEGORY_SATISFICED: 1,
    }
    assert sampler.category_of(0) == CATEGORY_SOLVED
    sampler.update(0, CATEGORY_UNSOLVED)
    assert sampler.category_of(0) == CATEGORY_UNSOLVED
    assert sampler.category_of(0) not in (CATEGORY_SOLVED, CATEGORY_SATISFICED)
    chosen = {sampler.choose() for _ in range(60)}
    assert chosen.issubset({0, 1, 2, 3})

    uniform = InstancePoolSampler(
        config.copy_with(use_instance_sampling=False), [0, 1, 2, 3], seed=42
    )
    assert all(uniform.choose() in {0, 1, 2, 3} for _ in range(30))


def test_search_replay_train_step_integration() -> None:
    config = Config().copy_with(batch_size=8, target_update_interval=50)
    device = require_cuda(DEVICE, verbose=False)
    set_seed(config.training_seed)
    model = build_q_network(config).to(device)
    trainer = QTrainer(config, device=device, model=model)
    instances, _ = load_dataset(config.train_path)
    search = QGuidedWAStar(model, config, device)
    result = search.search_instance(instances[0])
    assert result.experiences
    buffer = ReplayBuffer(config.batch_size, seed=1)
    buffer.add_many(result.experiences[: config.batch_size])
    while len(buffer) < config.batch_size:
        buffer.add(result.experiences[0])
    batch = buffer.sample(config.batch_size, device=device)
    assert_batch_on_cuda(batch, DEVICE)
    stats = trainer.train_step(batch)
    assert stats.gradient_updates == 1
    assert math.isfinite(stats.loss)


def test_validation_does_not_require_optimizer_updates() -> None:
    config = Config().copy_with(search_budget=16)
    device = require_cuda(DEVICE, verbose=False)
    instances, _ = load_dataset(config.validation_path)
    model = build_q_network(config).to(device)
    metrics = validate_network(model, instances[:5], config, device)
    assert metrics.num_instances == 5
    assert 0.0 <= metrics.success_rate <= 1.0
    assert 0 <= metrics.num_solved <= 5
    assert metrics.mean_expansions_all >= 0
    # Model remains in inference mode after validation.
    assert model.training is False