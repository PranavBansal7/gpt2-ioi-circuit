"""
Path Patching from Attention Heads to the Final Residual Stream
Author: Pranav Bansal

Distinguishes between direct logit effects (heads writing directly into the
unembedding via the final residual stream) and indirect effects (heads writing
into downstream heads).

Methodology:
    1. Clean tokens are run through the model.
    2. Every attention head's output (z) is FROZEN to its clean-run cache, EXCEPT
       for one target sender head, which is patched with its corrupted-run activation.
    3. The model output is read strictly from the final residual stream at the END position.
    4. Metric:
       noising_metric = (current_diff - clean_diff) / (clean_diff - corrupted_diff)
       (0.0 = clean performance; -1.0 = corrupted performance).

Findings:
    - Primary Positive Direct Writers: L9H9 (-0.922), L9H6 (-0.401), L10H0 (-0.168), L10H10 (-0.138).
      (Noising their output collapses clean performance by up to 92%).
    - Inhibitory Direct Writers: L10H7 (+0.541), L11H10 (+0.278).
      (Noising their output actually improves IO preference because their negative suppression is disabled).
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
# 3. Path Patching Hook (Freezing other heads, patching sender)
# -----------------------------------------------------------------------------

def patch_or_freeze_head_vectors(
    orig_head_vector: torch.Tensor,
    hook,
    new_cache,
    orig_cache,
    head_to_patch: tuple[int, int],
) -> torch.Tensor:
    """Freeze all z activations to the clean/original cache, except one sender."""
    orig_head_vector[...] = orig_cache[hook.name]
    sender_layer, sender_head = head_to_patch
    if hook.layer() == sender_layer:
        orig_head_vector[..., sender_head, :] = new_cache[hook.name][..., sender_head, :]
    return orig_head_vector


def get_path_patch_head_to_final_resid_post(
    model: HookedTransformer,
    patching_metric: Callable[[torch.Tensor], torch.Tensor],
    new_tokens: torch.Tensor,
    orig_tokens: torch.Tensor,
    new_cache=None,
    orig_cache=None,
) -> torch.Tensor:
    """Return one path-patching score per sender head, shape (n_layers, n_heads)."""
    result = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(orig_tokens, return_type=None)
    if new_cache is None:
        _, new_cache = model.run_with_cache(new_tokens, return_type=None)

    z_name_filter = lambda name: name.endswith("z")
    resid_post_hook_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1)

    for l in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            hook_fn = partial(
                patch_or_freeze_head_vectors,
                orig_cache=orig_cache,
                new_cache=new_cache,
                head_to_patch=(l, h),
            )
            with model.hooks(fwd_hooks=[(z_name_filter, hook_fn)]):
                _, patched_cache = model.run_with_cache(
                    orig_tokens,
                    names_filter=lambda name: name == resid_post_hook_name,
                    return_type=None,
                )
                patched_act = patched_cache[resid_post_hook_name]
                patched_ltok = patched_act[:, -1:, :]
                logits = model.unembed(model.ln_final(patched_ltok))
                final = patching_metric(logits)
                result[l, h] = final
    return result


# -----------------------------------------------------------------------------
# 4. Main Experiment & Path Patching Sweep
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

        print("\nTesting patch/freeze helper with fake cache values...")
        fake_clean = {"blocks.0.attn.hook_z": torch.zeros((2, 3, model.cfg.n_heads, model.cfg.d_head))}
        fake_corrupted = {"blocks.0.attn.hook_z": torch.ones((2, 3, model.cfg.n_heads, model.cfg.d_head))}
        test_z = torch.full_like(fake_clean["blocks.0.attn.hook_z"], 999.0)
        fake_hook = SimpleNamespace(layer=lambda: 0, name="blocks.0.attn.hook_z")
        patched_z = patch_or_freeze_head_vectors(
            test_z, fake_hook, fake_corrupted, fake_clean, head_to_patch=(0, 2)
        )
        assert torch.equal(patched_z[:, :, 2, :], fake_corrupted["blocks.0.attn.hook_z"][:, :, 2, :])
        non_sender_mask = torch.ones(model.cfg.n_heads, dtype=torch.bool)
        non_sender_mask[2] = False
        assert torch.equal(patched_z[:, :, non_sender_mask, :], fake_clean["blocks.0.attn.hook_z"][:, :, non_sender_mask, :])
        print("PASS: helper freezes all heads to clean, patches only the selected sender, and preserves caches.")

        print("\nRunning path-patching sweep: 12 layers x 12 heads = 144 senders.")
        scores = get_path_patch_head_to_final_resid_post(
            model=model,
            patching_metric=metric,
            new_tokens=corrupted_tokens,
            orig_tokens=clean_tokens,
            new_cache=corrupted_cache,
            orig_cache=clean_cache,
        )

    expected_shape = (model.cfg.n_layers, model.cfg.n_heads)
    assert tuple(scores.shape) == expected_shape
    assert torch.isfinite(scores).all()

    print("\nPath-patching noising-effect matrix (rows=layers, columns=heads):")
    print(scores.detach().cpu().to(torch.float32).numpy().round(3))

    flat = scores.flatten()
    n_show = min(10, flat.numel())
    top_degrading = torch.topk(flat, k=n_show, largest=False).indices
    top_improving = torch.topk(flat, k=min(5, flat.numel()), largest=True).indices

    print("\nHeads whose corrupted sender activation most degrades clean performance:")
    for flat_idx in top_degrading.tolist():
        layer = flat_idx // model.cfg.n_heads
        head = flat_idx % model.cfg.n_heads
        print(f"  L{layer:02d}H{head:02d}: metric={scores[layer, head].item():+.4f}")

    print("\nHeads whose corrupted sender activation most improves the metric:")
    for flat_idx in top_improving.tolist():
        layer = flat_idx // model.cfg.n_heads
        head = flat_idx % model.cfg.n_heads
        print(f"  L{layer:02d}H{head:02d}: metric={scores[layer, head].item():+.4f}")

    print("\nPASS: path-patching result shape and finite-value checks passed.")
    print("Path patching from each head to final resid_post completed successfully.")


if __name__ == "__main__":
    main()
