"""
Residual Stream Activation Patching on Indirect Object Identification
Author: Pranav Bansal

Performs full residual-stream causal activation patching across all 12 layers
and 15 token sequence positions in GPT-2 Small.

Methodology:
    1. Clean run: IOI prompt (e.g. "When John and Mary went to the shops, John gave the bag to")
    2. Corrupted run: Swapped subject (e.g. "When John and Mary went to the shops, Mary gave the bag to")
    3. Patching: Replace activation at (layer, position) in corrupted run with clean activation.
    4. Metric: Normalized IOI Logit Difference recovery:
       recovery = (patched_diff - corrupted_diff) / (clean_diff - corrupted_diff)
       (0.0 = corrupted performance, 1.0 = clean performance).

Identifies the critical spatiotemporal information transition in the residual stream:
early layers process the duplicate subject at position S2, which abruptly shifts
to position END in late layers where name movers read the signal.
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
# 1. IOI Dataset Construction
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

# Corrupt each prompt by swapping with its adjacent matched prompt
indices = [i + 1 if i % 2 == 0 else i - 1 for i in range(len(clean_tokens))]
corrupted_tokens = clean_tokens[indices]

print("\nClean and corrupted prompts:")
for i, (prompt, answers) in enumerate(zip(prompts, answer_name_pairs)):
    print(f"  {i}: clean={prompt!r}")
    print(f"     corrupted={model.to_string(corrupted_tokens[i])!r}")
    print(f"     IO={answers[0]!r}, subject={answers[1]!r}")

print("\nExample matched pair:")
print("  clean:     ", model.to_string(clean_tokens[0]))
print("  corrupted: ", model.to_string(corrupted_tokens[0]))
print(f"clean_tokens.shape     = {tuple(clean_tokens.shape)}")
print(f"corrupted_tokens.shape = {tuple(corrupted_tokens.shape)}")
print(f"answer_tokens.shape    = {tuple(answer_tokens.shape)}")

assert clean_tokens.shape == corrupted_tokens.shape
assert answer_tokens.shape == (clean_tokens.shape[0], 2)
assert clean_tokens.shape[0] == 8, "Expected 8 matched IOI prompts."
assert clean_tokens.shape[1] == 15, "Prompt token lengths changed."


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
# 3. Residual Stream Activation Patching Hook
# -----------------------------------------------------------------------------

def patch_residual_component(
    corrupted_residual_component: torch.Tensor,
    hook,
    pos: int,
    clean_cache,
) -> torch.Tensor:
    """Replace one sequence position in a corrupted residual activation."""
    corrupted_residual_component[:, pos] = clean_cache[hook.name][:, pos]
    return corrupted_residual_component


def get_act_patch_resid_pre(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache,
    patching_metric: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Return normalized recovery scores with shape (n_layers, sequence_length)."""
    model.reset_hooks()
    n_layers = model.cfg.n_layers
    seq_len = corrupted_tokens.shape[1]
    result = torch.zeros((n_layers, seq_len))
    for l in range(n_layers):
        hook_name = utils.get_act_name("resid_pre", l)
        for s in range(seq_len):
            patch_hook = partial(patch_residual_component, pos=s, clean_cache=clean_cache)
            patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[(hook_name, patch_hook)])
            result[l, s] = patching_metric(patched_logits)
    model.reset_hooks()
    return result


# -----------------------------------------------------------------------------
# 4. Main Experiment & Layer/Position Sweep
# -----------------------------------------------------------------------------

def main() -> None:
    print("\nRunning clean and corrupted baselines...")
    with torch.inference_mode():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupted_logits, _ = model.run_with_cache(corrupted_tokens)

        clean_logit_diff = mean_logit_diff(clean_logits, answer_tokens)
        corrupted_logit_diff = mean_logit_diff(corrupted_logits, answer_tokens)

        print(f"Clean mean logit difference:     {clean_logit_diff.item():+.6f}")
        print(f"Corrupted mean logit difference: {corrupted_logit_diff.item():+.6f}")
        assert not torch.isclose(clean_logit_diff, corrupted_logit_diff), (
            "Clean and corrupted metrics are equal, so normalized recovery is undefined."
        )

        print("\nTesting metric calibration...")
        clean_score = ioi_metric(
            clean_logits, answer_tokens, corrupted_logit_diff, clean_logit_diff
        )
        corrupted_score = ioi_metric(
            corrupted_logits, answer_tokens, corrupted_logit_diff, clean_logit_diff
        )
        midpoint_score = ioi_metric(
            (clean_logits + corrupted_logits) / 2,
            answer_tokens,
            corrupted_logit_diff,
            clean_logit_diff,
        )
        torch.testing.assert_close(clean_score, torch.tensor(1.0, device=DEVICE))
        torch.testing.assert_close(corrupted_score, torch.tensor(0.0, device=DEVICE))
        torch.testing.assert_close(midpoint_score, torch.tensor(0.5, device=DEVICE))
        print("PASS: clean=1, corrupted=0, midpoint=0.5.")

        print("\nTesting single-position patch helper...")
        test_layer = 0
        test_pos = min(3, clean_tokens.shape[1] - 1)
        fake_hook = SimpleNamespace(name="blocks.0.hook_resid_pre")
        test_activation = torch.zeros_like(clean_cache["resid_pre", test_layer])
        original = test_activation.clone()
        patched_test = patch_residual_component(
            test_activation, fake_hook, test_pos, clean_cache
        )
        assert torch.equal(
            patched_test[:, test_pos, :], clean_cache["resid_pre", test_layer][:, test_pos, :]
        ), "Selected position was not replaced with the clean activation."
        assert torch.equal(
            patched_test[:, torch.arange(patched_test.shape[1], device=DEVICE) != test_pos, :],
            original[:, torch.arange(original.shape[1], device=DEVICE) != test_pos, :],
        ), "Patch unexpectedly modified positions other than the selected one."
        print("PASS: helper replaces the selected position only.")

        print("\nSweeping all layers and sequence positions...")
        patching_metric = lambda logits: ioi_metric(
            logits, answer_tokens, corrupted_logit_diff, clean_logit_diff
        )
        patch_scores = get_act_patch_resid_pre(
            model=model,
            corrupted_tokens=corrupted_tokens,
            clean_cache=clean_cache,
            patching_metric=patching_metric,
        )

    expected_shape = (model.cfg.n_layers, clean_tokens.shape[1])
    assert tuple(patch_scores.shape) == expected_shape, (
        f"Expected patch scores shape {expected_shape}; got {tuple(patch_scores.shape)}"
    )
    assert torch.isfinite(patch_scores).all(), "Patch scores contain NaN or infinity."

    token_labels = model.to_str_tokens(clean_tokens[0])
    print(f"\nPatch score matrix (rows=layers, columns=positions): shape={tuple(patch_scores.shape)}")
    print(patch_scores.detach().cpu().to(torch.float32).numpy().round(3))

    print("\nHighest-recovery layer/position pairs:")
    flat_indices = torch.topk(patch_scores.flatten(), k=min(12, patch_scores.numel())).indices
    seq_len = clean_tokens.shape[1]
    for flat_idx in flat_indices.tolist():
        layer = flat_idx // seq_len
        pos = flat_idx % seq_len
        print(
            f"  L{layer:02d} pos {pos:02d} token={token_labels[pos]!r}: "
            f"recovery={patch_scores[layer, pos].item():+.4f}"
        )

    print("\nBest position in each selected layer:")
    selected_layers = sorted(set([0, 2, 4, 6, 7, 8, 9, 10, 11]))
    for layer in selected_layers:
        pos = int(torch.argmax(patch_scores[layer]).item())
        print(
            f"  L{layer:02d}: pos {pos:02d} token={token_labels[pos]!r}, "
            f"recovery={patch_scores[layer, pos].item():+.4f}"
        )

    print("\nSanity checks:")
    assert clean_score.item() == clean_score.item(), "Unexpected non-finite clean metric."
    print("PASS: metric calibration, patch helper, output shape, and finite values.")
    print("\nResidual-stream activation patching completed successfully.")


if __name__ == "__main__":
    main()
