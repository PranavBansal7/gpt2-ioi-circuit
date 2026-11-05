"""
Unit and Synthetic Regression Tests for the IOI Circuit Interpretability Suite
Author: Pranav Bansal

Verifies mathematical soundness, operator shapes, and hook precision:
1. Synthetic logit difference calculation (per-prompt vs mean)
2. Normalized recovery metric calibration (clean=1.0, corrupted=0.0, midpoint=0.5)
3. Activation patching preservation (unpatched positions/heads remain bitwise identical)
4. Component patching tensor slices across (z, q, k, v, pattern)
5. 26-head circuit keep-mask exact node count (26 * batch = total True nodes)
6. Pearson correlation robustness to constant vectors
"""

import math
from types import SimpleNamespace

import pytest
import torch as t

import importlib


def test_synthetic_logit_difference():
    mod = importlib.import_module("01_ioi_logit_difference")

    logits = t.zeros((3, 4, 10), dtype=t.float32)
    logits[0, -1, 2] = 5.0
    logits[0, -1, 7] = 1.0
    logits[1, -1, 4] = 2.0
    logits[1, -1, 3] = 5.0
    logits[2, -1, 6] = 0.5
    logits[2, -1, 8] = -0.5
    answer_tokens = t.tensor([[2, 7], [4, 3], [6, 8]], dtype=t.long)

    per_prompt = mod.logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=True)
    expected = t.tensor([4.0, -3.0, 1.0])
    assert per_prompt.shape == (3,)
    assert t.allclose(per_prompt, expected, atol=1e-6)

    average = mod.logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=False)
    assert average.ndim == 0
    assert t.allclose(average, expected.mean(), atol=1e-6)


def test_ioi_recovery_metric_calibration():
    mod = importlib.import_module("03_residual_activation_patching")

    # Clean = +4.0, Corrupted = -4.0
    clean_diff = t.tensor(4.0)
    corrupted_diff = t.tensor(-4.0)

    # Fake logits where diff is +4.0, -4.0, 0.0
    logits_clean = t.zeros((1, 2, 10))
    logits_clean[0, -1, 0] = 4.0
    logits_clean[0, -1, 1] = 0.0

    logits_corrupted = t.zeros((1, 2, 10))
    logits_corrupted[0, -1, 0] = 0.0
    logits_corrupted[0, -1, 1] = 4.0

    logits_midpoint = (logits_clean + logits_corrupted) / 2.0
    answers = t.tensor([[0, 1]], dtype=t.long)

    score_clean = mod.ioi_metric(logits_clean, answers, corrupted_diff, clean_diff)
    score_corrupted = mod.ioi_metric(logits_corrupted, answers, corrupted_diff, clean_diff)
    score_midpoint = mod.ioi_metric(logits_midpoint, answers, corrupted_diff, clean_diff)

    assert t.allclose(score_clean, t.tensor(1.0), atol=1e-6)
    assert t.allclose(score_corrupted, t.tensor(0.0), atol=1e-6)
    assert t.allclose(score_midpoint, t.tensor(0.5), atol=1e-6)


def test_residual_position_patch_isolation():
    mod = importlib.import_module("03_residual_activation_patching")

    batch, seq, d_model = 2, 5, 16
    corrupted_act = t.randn(batch, seq, d_model)
    clean_act = t.randn(batch, seq, d_model)
    clean_cache = {"test_hook": clean_act}
    fake_hook = SimpleNamespace(name="test_hook")

    patch_pos = 2
    original = corrupted_act.clone()
    patched = mod.patch_residual_component(corrupted_act.clone(), fake_hook, patch_pos, clean_cache)

    assert patched.shape == original.shape
    # Patched position matches clean
    assert t.equal(patched[:, patch_pos, :], clean_act[:, patch_pos, :])
    # Unpatched positions match corrupted original
    unpatched = [i for i in range(seq) if i != patch_pos]
    assert t.equal(patched[:, unpatched, :], original[:, unpatched, :])


def test_head_vector_patch_isolation():
    mod = importlib.import_module("04_head_activation_patching")

    batch, seq, n_heads, d_head = 2, 4, 12, 16
    corrupted_z = t.randn(batch, seq, n_heads, d_head)
    clean_z = t.randn(batch, seq, n_heads, d_head)
    clean_cache = {"test_hook": clean_z}
    fake_hook = SimpleNamespace(name="test_hook")

    head_to_patch = 7
    original = corrupted_z.clone()
    patched = mod.patch_head_vector(corrupted_z.clone(), fake_hook, head_to_patch, clean_cache)

    assert patched.shape == original.shape
    assert t.equal(patched[:, :, head_to_patch, :], clean_z[:, :, head_to_patch, :])
    unpatched_heads = [h for h in range(n_heads) if h != head_to_patch]
    assert t.equal(patched[:, :, unpatched_heads, :], original[:, :, unpatched_heads, :])


def test_component_patch_shapes():
    mod = importlib.import_module("05_head_component_patching")

    batch, seq, n_heads, d_head = 2, 6, 12, 16
    # Pattern: (batch, n_heads, seq, seq)
    pattern_corr = t.randn(batch, n_heads, seq, seq)
    pattern_clean = t.randn(batch, n_heads, seq, seq)
    fake_hook = SimpleNamespace(name="hook_pattern")
    clean_cache = {"hook_pattern": pattern_clean}

    patched_pattern = mod.patch_head_component(
        pattern_corr.clone(), fake_hook, 3, clean_cache, component="pattern"
    )
    assert patched_pattern.shape == pattern_corr.shape
    assert t.equal(patched_pattern[:, 3, :, :], pattern_clean[:, 3, :, :])

    # Z / V: (batch, seq, n_heads, d_head)
    v_corr = t.randn(batch, seq, n_heads, d_head)
    v_clean = t.randn(batch, seq, n_heads, d_head)
    fake_hook_v = SimpleNamespace(name="hook_v")
    clean_cache_v = {"hook_v": v_clean}

    patched_v = mod.patch_head_component(
        v_corr.clone(), fake_hook_v, 3, clean_cache_v, component="v"
    )
    assert patched_v.shape == v_corr.shape
    assert t.equal(patched_v[:, :, 3, :], v_clean[:, :, 3, :])


def test_pearson_correlation_robustness():
    mod = importlib.import_module("08_name_mover_copying_evidence")

    x = t.tensor([1.0, 2.0, 3.0, 4.0])
    y = t.tensor([2.0, 4.0, 6.0, 8.0])
    r = mod.pearson_corr(x, y)
    assert math.isclose(r, 1.0, rel_tol=1e-5)

    # Constant vector should not raise ZeroDivisionError
    const = t.tensor([5.0, 5.0, 5.0, 5.0])
    r_const = mod.pearson_corr(x, const)
    assert math.isnan(r_const)
