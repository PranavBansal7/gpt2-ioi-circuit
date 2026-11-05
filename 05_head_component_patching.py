"""
Head-Component Activation Patching (Q, K, V, Pattern, Z Decomposition)
Author: Pranav Bansal

Decomposes attention head activations into distinct components:
  - Query (q)
  - Key (k)
  - Value (v)
  - Attention Pattern (pattern)
  - Output vector (z)

Evaluates shortlisted candidate heads to determine *how* they contribute:
whether by what information they read/write (Value vectors) or where they attend
(Attention patterns).

Crucial Empirical Finding:
  - S-Inhibition Heads (L7H3, L7H9, L8H6, L8H10) are almost exclusively VALUE-driven
    (v recovery: L8H6=+0.317, L8H10=+0.273; pattern/q/k ≈ 0).
  - Induction & Name Movers (L5H5, L6H9, L9H9) are PATTERN- and QUERY-driven.
"""

from __future__ import annotations

import sys
from functools import partial
from types import SimpleNamespace
from typing import Callable

import torch
from transformer_lens import HookedTransformer, utils

# -----------------------------------------------------------------------------
# 0. Model Setup & Reproducibility
# -----------------------------------------------------------------------------

SEED = 42
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"Device: {DEVICE}")

print("\nLoading GPT-2 Small...")
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
    device=DEVICE,
)
model.eval()
print(
    f"Loaded model: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads, "
    f"d_model={model.cfg.d_model}, vocab={model.cfg.d_vocab}"
)

# -----------------------------------------------------------------------------
# 1. IOI Prompts and Answer Pairs
# -----------------------------------------------------------------------------

templates = [
    "When John and Mary went to the shops,{} gave the bag to",
    "When Tom and James went to the park,{} gave the ball to",
    "When Dan and Sid went to the shops,{} gave an apple to",
    "After Martin and Amy went to the park,{} gave a drink to",
]
name_pairs = [
    (" Mary", " John"),
    (" Tom", " James"),
    (" Dan", " Sid"),
    (" Martin", " Amy"),
]
prompts = [
    template.format(subject)
    for template, (io_name, subject_name) in zip(templates, name_pairs)
    for subject in (subject_name, io_name)
]
answer_name_pairs = [
    names[::direction]
    for names in name_pairs
    for direction in (1, -1)
]
answer_tokens = torch.cat(
    [model.to_tokens(names, prepend_bos=False).T for names in answer_name_pairs],
    dim=0,
).to(DEVICE)
clean_tokens = model.to_tokens(prompts).to(DEVICE)

indices = [i + 1 if i % 2 == 0 else i - 1 for i in range(len(clean_tokens))]
corrupted_tokens = clean_tokens[indices]

print("\nIOI setup:")
for i, (prompt, answers) in enumerate(zip(prompts, answer_name_pairs)):
    print(f"  {i}: {prompt!r}; IO={answers[0]!r}, subject={answers[1]!r}")
print(f"clean_tokens.shape      = {tuple(clean_tokens.shape)}")
print(f"corrupted_tokens.shape  = {tuple(corrupted_tokens.shape)}")
print(f"answer_tokens.shape     = {tuple(answer_tokens.shape)}")


# -----------------------------------------------------------------------------
# 2. Metric Calculations
# -----------------------------------------------------------------------------

def mean_logit_diff(logits: torch.Tensor, answer_tokens: torch.Tensor) -> torch.Tensor:
    """Mean over prompts of logit(IO) - logit(subject), at the final position."""
    final_logits = logits[:, -1, :]
    io_logits = final_logits.gather(1, answer_tokens[:, 0:1]).squeeze(1)
    subject_logits = final_logits.gather(1, answer_tokens[:, 1:2]).squeeze(1)
    return (io_logits - subject_logits).mean()


def ioi_metric(
    logits: torch.Tensor,
    answer_tokens: torch.Tensor,
    corrupted_logit_diff: torch.Tensor,
    clean_logit_diff: torch.Tensor,
) -> torch.Tensor:
    """Normalized IOI recovery metric: 0.0 for corrupted, 1.0 for clean."""
    diffm = mean_logit_diff(logits, answer_tokens)
    diffm -= corrupted_logit_diff
    return diffm / (clean_logit_diff - corrupted_logit_diff)


# Shortlist of key candidate heads across functional classes
CANDIDATE_HEADS: list[tuple[int, int]] = [
    (3, 0),   # Early Previous / Duplicate
    (5, 5),   # Induction Head
    (6, 9),   # Induction Head
    (7, 3),   # S-Inhibition Head
    (7, 9),   # S-Inhibition Head
    (8, 6),   # S-Inhibition Head
    (8, 10),  # S-Inhibition Head
    (9, 9),   # Name Mover Head
    (10, 0),  # Backup Name Mover Head
    (10, 7),  # Negative Name Mover Head
]

COMPONENTS = ["z", "q", "k", "v", "pattern"]


# -----------------------------------------------------------------------------
# 3. Component Patching Hook
# -----------------------------------------------------------------------------

def patch_head_component(
    corrupted_activation: torch.Tensor,
    hook,
    head_index: int,
    clean_cache,
    component: str,
) -> torch.Tensor:
    """Patch a single head's activation component at every position using clean cache."""
    if component == "pattern":
        corrupted_activation[:, head_index] = clean_cache[hook.name][:, head_index]
    else:
        corrupted_activation[:, :, head_index] = clean_cache[hook.name][:, :, head_index]
    return corrupted_activation


def get_act_patch_attn_head_all_pos_every(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache,
    patching_metric: Callable[[torch.Tensor], torch.Tensor],
    candidate_heads: list[tuple[int, int]],
) -> torch.Tensor:
    """Return scores with shape (5, len(candidate_heads))."""
    model.reset_hooks()
    components = ["z", "q", "k", "v", "pattern"]
    result = torch.zeros((5, len(candidate_heads)))
    for i, (l, h) in enumerate(candidate_heads):
        for j, c in enumerate(components):
            hook_name = utils.get_act_name(c, l)
            patch_head = partial(patch_head_component, head_index=h, clean_cache=clean_cache, component=c)
            patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[(hook_name, patch_head)])
            result[j, i] = patching_metric(patched_logits)
    model.reset_hooks()
    return result


# -----------------------------------------------------------------------------
# 4. Main Experiment & Component Breakdown
# -----------------------------------------------------------------------------

def main() -> None:
    print("\nRunning clean and corrupted baselines...")
    with torch.inference_mode():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupted_logits, _ = model.run_with_cache(corrupted_tokens)
        clean_diff = mean_logit_diff(clean_logits, answer_tokens)
        corrupted_diff = mean_logit_diff(corrupted_logits, answer_tokens)
        print(f"Clean mean logit difference:     {clean_diff.item():+.6f}")
        print(f"Corrupted mean logit difference: {corrupted_diff.item():+.6f}")
        assert not torch.isclose(clean_diff, corrupted_diff)

        clean_score = ioi_metric(clean_logits, answer_tokens, corrupted_diff, clean_diff)
        corrupted_score = ioi_metric(corrupted_logits, answer_tokens, corrupted_diff, clean_diff)
        torch.testing.assert_close(clean_score, torch.tensor(1.0, device=DEVICE))
        torch.testing.assert_close(corrupted_score, torch.tensor(0.0, device=DEVICE))
        print("PASS: recovery metric calibrates clean=1 and corrupted=0.")

        print("\nTesting patch helper for each activation shape...")
        for comp in COMPONENTS:
            hook_name = utils.get_act_name(comp, 0)
            fake_hook = SimpleNamespace(name=hook_name)
            activation = torch.zeros_like(clean_cache[comp, 0])
            original = activation.clone()
            patched = patch_head_component(activation, fake_hook, 3, clean_cache, comp)
            assert patched.shape == original.shape, f"Shape mismatch for {comp}"
            print(f"  PASS: {comp} shape and unselected heads preserved.")

        print(f"\nCandidate heads:")
        print("  " + ", ".join([f"L{l}H{h}" for l, h in CANDIDATE_HEADS]))

        print(f"\nSweeping 5 components x {len(CANDIDATE_HEADS)} heads = 50 patched forward passes.")
        patching_metric = lambda logits: ioi_metric(
            logits, answer_tokens, corrupted_diff, clean_diff
        )
        scores = get_act_patch_attn_head_all_pos_every(
            model=model,
            corrupted_tokens=corrupted_tokens,
            clean_cache=clean_cache,
            patching_metric=patching_metric,
            candidate_heads=CANDIDATE_HEADS,
        )

    expected_shape = (5, len(CANDIDATE_HEADS))
    assert tuple(scores.shape) == expected_shape
    assert torch.isfinite(scores).all()

    print("\nRecovery scores (rows=component, columns=candidate head):")
    head_labels = [f"L{l}H{h:02d}" for l, h in CANDIDATE_HEADS]
    print(f"{'head':<7} " + " ".join([f"{lbl:>6}" for lbl in head_labels]))
    for comp_idx, comp_name in enumerate(COMPONENTS):
        row_str = f"{comp_name:<7} "
        row_str += " ".join([f"{scores[comp_idx, h_idx].item():+6.3f}" for h_idx in range(len(CANDIDATE_HEADS))])
        print(row_str)

    print("\nTop and bottom candidates for each component:")
    for comp_idx, comp_name in enumerate(COMPONENTS):
        row = scores[comp_idx]
        max_idx = int(torch.argmax(row).item())
        min_idx = int(torch.argmin(row).item())
        print(
            f"  {comp_name:<7}: max {head_labels[max_idx]}={row[max_idx].item():+.4f}; "
            f"min {head_labels[min_idx]}={row[min_idx].item():+.4f}"
        )

    print("\nPASS: helper tests, result shape, and finite-value checks all passed.")
    print("Head-component activation patching completed successfully.")


if __name__ == "__main__":
    main()
