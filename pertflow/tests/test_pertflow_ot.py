import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from pertflow import (
    ConditionKey,
    DirectPathFlow,
    FlowHead,
    SelfFlow,
    build_condition_index,
    compute_transport_plan,
    compute_pair_metrics,
    sample_hard_pairs,
)


class RecordingVectorField(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_x = None
        self.last_times = None
        self.last_cond = None

    def forward(self, x, times, cond):
        self.last_x = x.detach().clone()
        self.last_times = times.detach().clone()
        self.last_cond = cond.detach().clone()
        return torch.zeros_like(x)


class ConstantVelocityField(nn.Module):
    def __init__(self, velocity):
        super().__init__()
        self.velocity = velocity

    def forward(self, x, times, cond):
        return torch.full_like(x, self.velocity)


class FakeOTWithStabilizedFallback:
    def __init__(self, fallback_plan):
        self.fallback_plan = fallback_plan
        self.bregman = self
        self.sinkhorn_calls = 0
        self.stabilized_calls = 0

    def sinkhorn(self, a, b, cost, reg, **kwargs):
        self.sinkhorn_calls += 1
        return torch.zeros_like(cost)

    def sinkhorn_stabilized(self, a, b, cost, reg, **kwargs):
        self.stabilized_calls += 1
        return self.fallback_plan.clone()


class FakeOTThatAlwaysFails:
    def __init__(self):
        self.bregman = self
        self.sinkhorn_calls = 0
        self.stabilized_calls = 0

    def sinkhorn(self, a, b, cost, reg, **kwargs):
        self.sinkhorn_calls += 1
        return torch.zeros_like(cost)

    def sinkhorn_stabilized(self, a, b, cost, reg, **kwargs):
        self.stabilized_calls += 1
        return torch.zeros_like(cost)


class TestPertflowOT(unittest.TestCase):
    def test_build_condition_index_groups_controls_by_celltype_and_filters_unmatched_targets(self):
        obs = pd.DataFrame(
            {
                "celltype": ["alpha", "alpha", "beta", "beta", "gamma"],
                "condition": ["ctrl", "ctrl+G1", "ctrl", "ctrl+G2", "ctrl+G3"],
            }
        )
        pert_indices = np.array([0, 1, 0, 2, 3], dtype=np.int64)

        control_by_celltype, target_by_condition = build_condition_index(obs, pert_indices)

        self.assertEqual(set(control_by_celltype), {"alpha", "beta"})
        self.assertEqual(control_by_celltype["alpha"].tolist(), [0])
        self.assertEqual(control_by_celltype["beta"].tolist(), [2])
        self.assertEqual(set(target_by_condition), {
            ConditionKey(celltype="alpha", pert_idx=1),
            ConditionKey(celltype="beta", pert_idx=2),
        })
        np.testing.assert_array_equal(
            target_by_condition[ConditionKey(celltype="alpha", pert_idx=1)],
            np.array([1], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            target_by_condition[ConditionKey(celltype="beta", pert_idx=2)],
            np.array([3], dtype=np.int64),
        )

    def test_sample_hard_pairs_only_uses_nonzero_transport_mass(self):
        plan = torch.tensor(
            [
                [0.0, 0.7, 0.0],
                [0.3, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        source_idx, target_idx = sample_hard_pairs(
            plan,
            num_pairs=32,
            generator=torch.Generator().manual_seed(0),
        )

        sampled_pairs = set(zip(source_idx.tolist(), target_idx.tolist()))
        self.assertTrue(sampled_pairs.issubset({(0, 1), (1, 0)}))
        self.assertEqual(len(source_idx), 32)
        self.assertEqual(len(target_idx), 32)

    def test_compute_transport_plan_uses_stabilized_sinkhorn_when_base_plan_is_empty(self):
        fake_ot = FakeOTWithStabilizedFallback(
            torch.tensor([[0.4, 0.1], [0.1, 0.4]], dtype=torch.float32)
        )
        source = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
        target = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

        with patch("pertflow.require_ot", return_value=fake_ot):
            plan = compute_transport_plan(source, target, reg=0.05)

        self.assertEqual(fake_ot.sinkhorn_calls, 1)
        self.assertEqual(fake_ot.stabilized_calls, 1)
        self.assertGreater(float(plan.sum()), 0.0)
        self.assertTrue(torch.all(plan >= 0))
        self.assertTrue(torch.allclose(plan, fake_ot.fallback_plan))

    def test_compute_transport_plan_falls_back_to_deterministic_pairs_when_ot_fails(self):
        fake_ot = FakeOTThatAlwaysFails()
        source = torch.tensor([[0.0], [10.0]], dtype=torch.float32)
        target = torch.tensor([[0.1], [9.9]], dtype=torch.float32)

        with patch("pertflow.require_ot", return_value=fake_ot):
            plan = compute_transport_plan(source, target, reg=0.05)

        self.assertEqual(fake_ot.sinkhorn_calls, 1)
        self.assertEqual(fake_ot.stabilized_calls, 1)
        self.assertGreater(float(plan.sum()), 0.0)
        self.assertTrue(torch.all(plan >= 0))
        self.assertEqual(int((plan > 0).sum().item()), 2)
        self.assertGreater(float(plan[0, 0]), 0.0)
        self.assertGreater(float(plan[1, 1]), 0.0)

    def test_direct_path_flow_uses_source_target_interpolation_and_velocity_target(self):
        vector_field = RecordingVectorField()
        flow = DirectPathFlow(vector_field, loss_fn=F.mse_loss)

        source = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        target = torch.tensor([[2.0, 5.0]], dtype=torch.float32)
        cond = torch.ones((1, 2, 3), dtype=torch.float32)
        times = torch.tensor([0.25], dtype=torch.float32)

        loss = flow(source=source, target=target, cond=cond, times=times)

        expected_xt = source.lerp(target, times.unsqueeze(-1))
        expected_velocity = target - source
        expected_loss = F.mse_loss(torch.zeros_like(expected_velocity), expected_velocity)

        self.assertTrue(torch.allclose(vector_field.last_x, expected_xt))
        self.assertTrue(torch.allclose(vector_field.last_times, times))
        self.assertTrue(torch.allclose(vector_field.last_cond, cond))
        self.assertTrue(torch.isclose(loss, expected_loss))

    def test_direct_path_flow_sampling_starts_from_source_state(self):
        flow = DirectPathFlow(ConstantVelocityField(velocity=0.5))

        source = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        cond = torch.zeros((1, 2, 4), dtype=torch.float32)

        sampled = flow.sample(source=source, cond=cond, steps=4)

        self.assertTrue(torch.allclose(sampled, source + 0.5))

    def test_self_flow_sample_can_infer_shape_from_source_batch(self):
        flow = SelfFlow(FlowHead(8))
        source = torch.randn(3, 5)
        cond = torch.randn(3, 5, 8)

        sampled = flow.sample(source=source, cond=cond, steps=2)

        self.assertEqual(tuple(sampled.shape), (3, 5))

    def test_self_flow_rejects_patch_sizes_that_do_not_divide_sequence_length(self):
        flow = SelfFlow(FlowHead(8), patch_size=4)
        source = torch.randn(2, 10)
        target = torch.randn(2, 10)
        cond = torch.randn(2, 10, 8)

        with self.assertRaisesRegex(ValueError, "patch_size"):
            flow(source=source, target=target, cond=cond)

    def test_compute_pair_metrics_uses_explicit_pairwise_metric_names(self):
        preds = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        targets = np.array([[2.0, 1.0], [0.0, 6.0]], dtype=np.float32)

        metrics = compute_pair_metrics(preds, targets)

        self.assertIn("pair_cell_mean_mae", metrics)
        self.assertNotIn("dist_cell_mean_wasserstein", metrics)
        self.assertAlmostEqual(metrics["pair_mse"], 3.25)
        self.assertAlmostEqual(metrics["pair_mae"], 1.75)
        self.assertAlmostEqual(metrics["pair_cell_mean_mae"], 0.25)


if __name__ == "__main__":
    unittest.main()
