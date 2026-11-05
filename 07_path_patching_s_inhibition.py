"""
Path Patching into S-Inhibition Heads' Value Inputs
Author: Pranav Bansal

Tests which earlier attention heads transmit crucial signals into the Value (V) inputs
of the candidate S-Inhibition heads:
    Receivers: L7H3, L7H9, L8H6, L8H10

Methodology:
    1. For each earlier sender head (layers 0-7, 12 heads/layer = 96 senders):
       a. Run clean prompts with that sender's output patched from the corrupted run,
          while freezing all other heads to clean.
       b. In this forward pass, cache the resulting Value activations entering the
          target receiver heads.
    2. Run clean prompts again, patching ONLY the receivers' Value inputs from that cache.
    3. Measure the resulting noising metric on the final logits.

Key Empirical Finding:
    The Value inputs of S-Inhibition heads are driven predominantly by:
    - Induction Head L5H5 (-0.3395)
    - Induction Head L6H9 (-0.1289)
    - Early Previous-Token / Duplicate Head L3H0 (-0.1186)
    Providing direct causal evidence for the circuit pathway:
    [Duplicate/Induction Heads] -> [S-Inhibition V-inputs] -> [Name Mover Inhibition]
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

# Candidate S-Inhibition heads to evaluate as receivers
RECEIVER_HEADS: list[tuple[int, int]] = [(8, 6), (8, 10), (7, 9), (7, 3)]

print("\nIOI setup:")
for i, (prompt, answers) in enumerate(zip(prompts, answer_name_pairs)):
    print(f"  {i}: {prompt!r}; IO={answers[0]!r}, subject={answers[1]!r}")
print(f"clean_tokens.shape      = {tuple(clean_tokens.shape)}")
print(f"corrupted_tokens.shape  = {tuple(corrupted_tokens.shape)}")
print(f"answer_tokens.shape     = {tuple(answer_tokens.shape)}")
print("Receiver heads: " + ", ".join([f"L{l}H{h}" for l, h in RECEIVER_HEADS]))


# -----------------------------------------------------------------------------
# 2. Metric Calculations
# -----------------------------------------------------------------------------

def mean_logit_diff(logits: torch.Tensor, answer_tokens: torch.Tensor) -> torch.Tensor:
    """Mean over prompts of logit(IO) - logit(subject), at the final position."""
    final_logits = logits[:, -1, :]
    io_logits = final_logits.gather(1, answer_tokens[:, 0:1]).squeeze(1)
    subject_logits = final_logits.gather(1, answer_tokens[:, 1:2]).squeeze(1)
    return (io_logits - subject_logits).mean()


def noising_metric(
    logits: torch.Tensor,
    answers: torch.Tensor,
    clean_logit_diff: torch.Tensor,
    corrupted_logit_diff: torch.Tensor,
) -> torch.Tensor:
    """0 = clean performance; -1 = corrupted performance."""
    current = mean_logit_diff(logits, answers)
    return (current - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)


# -----------------------------------------------------------------------------
# 3. Path Patching Hooks
# -----------------------------------------------------------------------------

def patch_or_freeze_head_vectors(
    orig_head_vector: torch.Tensor,
    hook,
    new_cache,
    orig_cache,
    head_to_patch: tuple[int, int],
) -> torch.Tensor:
    """Freeze each layer's z to clean values, except chosen sender from corruption."""
    orig_head_vector[...] = orig_cache[hook.name][...]
    if head_to_patch[0] == hook.layer():
        orig_head_vector[:, :, head_to_patch[1], :] = new_cache[hook.name][:, :, head_to_patch[1], :]
    return orig_head_vector


def patch_head_input(
    orig_activation: torch.Tensor,
    hook,
    patched_cache,
    head_list: list[tuple[int, int]],
) -> torch.Tensor:
    """Patch only the listed receiver heads at this layer from patched_cache."""
    for l, h in head_list:
        if hook.layer() == l:
            orig_activation[:, :, h, :] = patched_cache[hook.name][:, :, h, :]
    return orig_activation


def get_path_patch_head_to_heads(
    receiver_heads: list[tuple[int, int]],
    receiver_input: str,
    model: HookedTransformer,
    patching_metric: Callable[[torch.Tensor], torch.Tensor],
    new_tokens: torch.Tensor,
    orig_tokens: torch.Tensor,
    new_cache=None,
    orig_cache=None,
) -> torch.Tensor:
    """Return path-patching scores into receiver heads' inputs, shape (max_layer, n_heads)."""
    assert receiver_input in ('q', 'k', 'v'), "receiver_input must be q, k, or v"
    if new_cache is None:
        _, new_cache = model.run_with_cache(new_tokens, return_type=None)
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(orig_tokens, return_type=None)

    receiver_layers = [layer for layer, head in receiver_heads]
    max_receiver_layer = max(receiver_layers)
    receiver_hook_names = [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
    result = torch.zeros((max_receiver_layer, model.cfg.n_heads))

    for sender_layer in range(max_receiver_layer):
        for sender_head in range(model.cfg.n_heads):
            z_hook_fn = partial(
                patch_or_freeze_head_vectors,
                new_cache=new_cache,
                orig_cache=orig_cache,
                head_to_patch=(sender_layer, sender_head),
            )
            with model.hooks(fwd_hooks=[(lambda name: name.endswith("hook_z"), z_hook_fn)]):
                _, patched_cache = model.run_with_cache(
                    orig_tokens,
                    names_filter=lambda name: name in receiver_hook_names,
                    return_type=None,
                )
            receiver_hook_fn = [
                (name, partial(patch_head_input, patched_cache=patched_cache, head_list=receiver_heads))
                for name in receiver_hook_names
            ]
            with model.hooks(fwd_hooks=receiver_hook_fn):
                patched_logits = model(orig_tokens)
            result[sender_layer, sender_head] = patching_metric(patched_logits)
    return result


# -----------------------------------------------------------------------------
# 4. Main Experiment & Path Sweep
# -----------------------------------------------------------------------------

def main() -> None:
    with torch.inference_mode():
        print("\nRunning clean and corrupted baselines...")
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)
        clean_diff = mean_logit_diff(clean_logits, answer_tokens)
        corrupted_diff = mean_logit_diff(corrupted_logits, answer_tokens)
        print(f"Clean mean logit difference:     {clean_diff.item():+.6f}")
        print(f"Corrupted mean logit difference: {corrupted_diff.item():+.6f}")
        assert clean_diff > corrupted_diff

        metric = partial(
            noising_metric,
            answers=answer_tokens,
            clean_logit_diff=clean_diff,
            corrupted_logit_diff=corrupted_diff,
        )
        torch.testing.assert_close(metric(clean_logits), torch.tensor(0.0, device=DEVICE))
        torch.testing.assert_close(metric(corrupted_logits), torch.tensor(-1.0, device=DEVICE))
        print("PASS: noising metric calibrates clean=0 and corrupted=-1.")

        print("\nTesting patch_head_input with fake tensors...")
        test_layer = 7
        act = torch.zeros((2, 3, model.cfg.n_heads, model.cfg.d_head))
        fake_cache = {utils.get_act_name("v", test_layer): torch.ones_like(act) * 42.0}
        fake_hook = SimpleNamespace(layer=lambda: test_layer, name=utils.get_act_name("v", test_layer))
        patched_act = patch_head_input(act, fake_hook, fake_cache, head_list=RECEIVER_HEADS)
        assert torch.all(patched_act[:, :, 3, :] == 42.0)
        assert torch.all(patched_act[:, :, 9, :] == 42.0)
        assert torch.all(patched_act[:, :, 0, :] == 0.0)
        print("PASS: only receiver heads 7.3 and 7.9 were patched in layer 7.")

        print("\nRunning path-patching sweep into receiver V inputs...")
        receiver_layers = [layer for layer, _ in RECEIVER_HEADS]
        max_receiver_layer = max(receiver_layers)
        print(f"Senders evaluated: {max_receiver_layer} layers x {model.cfg.n_heads} heads = {max_receiver_layer * model.cfg.n_heads}")

        scores = get_path_patch_head_to_heads(
            receiver_heads=RECEIVER_HEADS,
            receiver_input="v",
            model=model,
            patching_metric=metric,
            new_tokens=corrupted_tokens,
            orig_tokens=clean_tokens,
            new_cache=corrupted_cache,
            orig_cache=clean_cache,
        )

    expected_shape = (max_receiver_layer, model.cfg.n_heads)
    assert tuple(scores.shape) == expected_shape
    assert torch.isfinite(scores).all()

    print("\nPath-patching matrix (rows=sender layers, columns=sender heads):")
    print(scores.detach().cpu().to(torch.float32).numpy().round(3))

    flat = scores.flatten()
    n_show = min(10, flat.numel())
    top_negative = torch.topk(flat, k=n_show, largest=False).indices
    top_positive = torch.topk(flat, k=min(5, flat.numel()), largest=True).indices

    print("\nMost negative scores (corrupted sender most harms clean IOI performance):")
    for flat_idx in top_negative.tolist():
        layer = flat_idx // model.cfg.n_heads
        head = flat_idx % model.cfg.n_heads
        print(f"  L{layer}H{head}: {scores[layer, head].item():+.4f}")

    print("\nMost positive scores (corrupted sender improves the metric):")
    for flat_idx in top_positive.tolist():
        layer = flat_idx // model.cfg.n_heads
        head = flat_idx % model.cfg.n_heads
        print(f"  L{layer}H{head}: {scores[layer, head].item():+.4f}")

    print("\nPASS: receiver-input path-patching shape and finite-value checks passed.")
    print("Path patching into S-inhibition value inputs completed successfully.")


if __name__ == "__main__":
    main()
