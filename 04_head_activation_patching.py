"""
Attention Head-Level Activation Patching on Indirect Object Identification
Author: Pranav Bansal

Patches the output activation (z) of each attention head across all 144 heads
(12 layers x 12 heads) in GPT-2 Small to quantify each head's causal contribution
to the IOI task.

Methodology:
    For each (layer, head):
      1. Run the corrupted IOI prompt with a hook patching that head's output z
         across all sequence positions with its clean-run activation.
      2. Measure the normalized recovery of the clean IOI logit difference.
      3. All other 143 heads continue running on the corrupted prompt.

Empirical Findings:
    - Primary Positive Restorers: L5H5 (+0.344), L8H6 (+0.319), L8H10 (+0.278), L9H9 (+0.264)
    - Primary Inhibitors / Suppressors: L10H7 (-0.480), L11H10 (-0.278)
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

print("\nPrompt batch:")
for i, (prompt, answers) in enumerate(zip(prompts, answer_name_pairs)):
    print(f"  {i}: clean={prompt!r}")
    print(f"     IO={answers[0]!r}, subject={answers[1]!r}")
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


# -----------------------------------------------------------------------------
# 3. Head Activation Patching Hook
# -----------------------------------------------------------------------------

def patch_head_vector(
    corrupted_head_vector: torch.Tensor,
    hook,
    head_index: int,
    clean_cache,
) -> torch.Tensor:
    """Patch a single head's z activation at every position."""
    corrupted_head_vector[:, :, head_index] = clean_cache[hook.name][:, :, head_index]
    return corrupted_head_vector


def get_act_patch_attn_head_out_all_pos(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache,
    patching_metric: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Return a (n_layers, n_heads) matrix of normalized recovery scores."""
    model.reset_hooks()
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    result = torch.zeros((n_layers, n_heads))
    for l in range(n_layers):
        hook_name = utils.get_act_name("z", l)
        for h in range(n_heads):
            patch_hook = partial(patch_head_vector, head_index=h, clean_cache=clean_cache)
            patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[(hook_name, patch_hook)])
            result[l, h] = patching_metric(patched_logits)
    model.reset_hooks()
    return result


# -----------------------------------------------------------------------------
# 4. Main Experiment & Full 144-Head Scan
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

        print("\nTesting metric calibration...")
        clean_score = ioi_metric(clean_logits, answer_tokens, corrupted_diff, clean_diff)
        corrupted_score = ioi_metric(corrupted_logits, answer_tokens, corrupted_diff, clean_diff)
        torch.testing.assert_close(clean_score, torch.tensor(1.0, device=DEVICE))
        torch.testing.assert_close(corrupted_score, torch.tensor(0.0, device=DEVICE))
        print("PASS: clean=1, corrupted=0.")

        print("\nTesting that patch_head_vector changes only the selected head...")
        fake_hook = SimpleNamespace(name="blocks.0.attn.hook_z")
        test_vector = torch.zeros_like(clean_cache["z", 0])
        original = test_vector.clone()
        test_head = 4
        patched_test = patch_head_vector(test_vector, fake_hook, test_head, clean_cache)
        assert torch.equal(
            patched_test[:, :, test_head, :], clean_cache["z", 0][:, :, test_head, :]
        ), "Selected head was not replaced with the clean activation."
        mask = torch.ones(model.cfg.n_heads, dtype=torch.bool, device=DEVICE)
        mask[test_head] = False
        assert torch.equal(
            patched_test[:, :, mask, :], original[:, :, mask, :]
        ), "Patch unexpectedly modified heads other than the selected one."
        print("PASS: selected head replaced; all other heads unchanged.")

        print("\nSweeping all layers and heads (one patched forward pass per layer/head)...")
        patching_metric = lambda logits: ioi_metric(
            logits, answer_tokens, corrupted_diff, clean_diff
        )
        head_scores = get_act_patch_attn_head_out_all_pos(
            model=model,
            corrupted_tokens=corrupted_tokens,
            clean_cache=clean_cache,
            patching_metric=patching_metric,
        )

    expected_shape = (model.cfg.n_layers, model.cfg.n_heads)
    assert tuple(head_scores.shape) == expected_shape, (
        f"Expected head scores shape {expected_shape}; got {tuple(head_scores.shape)}"
    )
    assert torch.isfinite(head_scores).all(), "Head scores contain NaN or infinity."

    print(f"\nHead recovery matrix (rows=layers, columns=heads): shape={tuple(head_scores.shape)}")
    print(head_scores.detach().cpu().to(torch.float32).numpy().round(3))

    flat = head_scores.flatten()
    n_show = min(10, flat.numel())
    top_positive = torch.topk(flat, k=n_show).indices
    top_negative = torch.topk(flat, k=n_show, largest=False).indices

    def print_heads(indices: torch.Tensor, heading: str) -> None:
        print(f"\n{heading}")
        for flat_idx in indices.tolist():
            layer = flat_idx // model.cfg.n_heads
            head = flat_idx % model.cfg.n_heads
            print(f"  L{layer:02d}H{head:d}: recovery={head_scores[layer, head].item():+.4f}")

    print_heads(top_positive, "Most positive heads (patching clean activations restores IO preference):")
    print_heads(top_negative, "Most negative heads (clean patching lowers normalized recovery):")

    print("\nSanity checks:")
    assert clean_score.item() == clean_score.item(), "Unexpected non-finite clean metric."
    print("PASS: metric calibration, head-patch helper, output shape, and finite scores.")
    print("\nHead-level activation patching completed successfully.")


if __name__ == "__main__":
    main()
