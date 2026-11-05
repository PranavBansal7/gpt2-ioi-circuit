"""
Direct Logit Attribution (DLA) on Indirect Object Identification
Author: Pranav Bansal

Decomposes GPT-2 Small's final residual stream into individual attention head
contributions to the IO-vs-subject logit difference:
    DLA(l, h) = (LN-scale * head_output[l, h])^T (W_U[:, IO] - W_U[:, S])

Identifies candidate Name Mover Heads (positive attribution, promoting the indirect object)
and Negative Name Mover Heads (negative attribution, promoting the repeated subject).
"""

from __future__ import annotations

import sys
import traceback
from typing import Tuple

import torch
from transformer_lens import HookedTransformer


# -----------------------------------------------------------------------------
# 0. Reproducibility and model setup
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
tokens = model.to_tokens(prompts).to(DEVICE)

print("\nPrompt batch:")
for i, (prompt, names) in enumerate(zip(prompts, answer_name_pairs)):
    print(f"  {i}: {prompt!r}")
    print(f"     IO={names[0]!r}, subject={names[1]!r}")
print(f"tokens.shape        = {tuple(tokens.shape)}")
print(f"answer_tokens.shape = {tuple(answer_tokens.shape)}")

assert tokens.ndim == 2 and tokens.shape[0] == 8, "Expected eight prompts."
assert answer_tokens.shape == (8, 2), "Expected (batch=8, [IO, subject]) answer IDs."
assert tokens.shape[1] == 15, "Prompt token lengths changed."


# -----------------------------------------------------------------------------
# 2. Residual-Space Unembedding Directions
# -----------------------------------------------------------------------------

def get_logit_diff_directions(
    model: HookedTransformer,
    answer_tokens: torch.Tensor,
) -> torch.Tensor:
    """Return one IO-minus-subject unembedding direction per prompt, shape (batch, d_model).

    Args:
        model: HookedTransformer instance.
        answer_tokens: Shape (batch, 2), with column 0 = IO, column 1 = S.

    Returns:
        Direction tensor of shape (batch, d_model).
    """
    inter = model.tokens_to_residual_directions(answer_tokens)
    io, subj = inter.unbind(dim=1)
    return io - subj


# -----------------------------------------------------------------------------
# 3. Residual-Stack-to-Logit-Difference Attribution
# -----------------------------------------------------------------------------

def residual_stack_to_logit_diff(
    residual_stack: torch.Tensor,
    cache,
    logit_diff_directions: torch.Tensor,
) -> torch.Tensor:
    """Attribute IO-vs-subject logit difference to residual components.

    Args:
        residual_stack: Shape (components, batch, d_model).
        cache: TransformerLens ActivationCache from model.run_with_cache.
        logit_diff_directions: Shape (batch, d_model).

    Returns:
        Batch-mean contribution tensor shaped like residual_stack.shape[:-2].
    """
    scaled = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    direct = torch.einsum("c b d, b d -> c", scaled, logit_diff_directions)
    batch_size, _ = logit_diff_directions.shape
    return direct / batch_size


# -----------------------------------------------------------------------------
# 4. Evaluation and DLA Analysis
# -----------------------------------------------------------------------------

def run_tests_and_analysis() -> None:
    print("\nBuilding prompt-specific logit-difference directions...")
    directions = get_logit_diff_directions(model, answer_tokens)
    expected_direction_shape = (tokens.shape[0], model.cfg.d_model)
    assert tuple(directions.shape) == expected_direction_shape, (
        f"Expected directions shape {expected_direction_shape}; got {tuple(directions.shape)}"
    )
    assert directions.device == tokens.device, "Directions and tokens must share a device."
    print(f"directions.shape = {tuple(directions.shape)}")

    print("\nRunning GPT-2 and caching activations...")
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    batch_indices = torch.arange(tokens.shape[0], device=DEVICE)
    final_logits = logits[batch_indices, -1, :]
    io_logits = final_logits.gather(1, answer_tokens[:, 0:1]).squeeze(1)
    subject_logits = final_logits.gather(1, answer_tokens[:, 1:2]).squeeze(1)
    per_prompt_logit_diffs = io_logits - subject_logits
    mean_logit_diff = per_prompt_logit_diffs.mean()

    print("\nActual IO-vs-subject logit differences:")
    for i, (io_name, subject_name) in enumerate(answer_name_pairs):
        print(
            f"  {i}: IO {io_name.strip():>6s} vs subject {subject_name.strip():>6s}: "
            f"{per_prompt_logit_diffs[i].item():+8.4f}"
        )
    print(f"Mean logit difference: {mean_logit_diff.item():+.6f}")

    # Reconstruct logit difference from final residual post-LayerNorm
    final_resid = cache["resid_post", -1][:, -1, :]
    direct_from_final_resid = residual_stack_to_logit_diff(
        final_resid.unsqueeze(0), cache, directions
    )[0]
    print(
        "Mean logit difference reconstructed from final residual: "
        f"{direct_from_final_resid.item():+.6f}"
    )
    assert torch.allclose(
        direct_from_final_resid,
        mean_logit_diff,
        atol=3e-2,
        rtol=3e-3,
    ), "Final residual attribution did not reconstruct the model's logit difference."
    print("PASS: final residual reconstructs the actual mean logit difference.")

    print("\nStacking attention-head results for DLA...")
    head_results = cache.stack_head_results(layer=-1, pos_slice=-1)
    head_attributions_flat = residual_stack_to_logit_diff(head_results, cache, directions)
    head_attributions = head_attributions_flat.reshape(model.cfg.n_layers, model.cfg.n_heads)

    print("\nPer-head mean direct logit attribution (rows=layers, columns=heads):")
    print(head_attributions.detach().cpu().to(torch.float32).numpy().round(3))

    flat = head_attributions.flatten()
    n_show = min(10, flat.numel())
    top_positive = torch.topk(flat, k=n_show).indices
    top_negative = torch.topk(flat, k=n_show, largest=False).indices

    def print_heads(indices: torch.Tensor, heading: str) -> None:
        print(f"\n{heading}")
        for flat_index in indices.tolist():
            layer = flat_index // model.cfg.n_heads
            head = flat_index % model.cfg.n_heads
            value = head_attributions[layer, head].item()
            print(f"  L{layer}H{head}: {value:+.5f}")

    print_heads(top_positive, "Most positive heads (promote IO over subject):")
    print_heads(top_negative, "Most negative heads (push toward subject over IO):")

    print("\nSanity checks:")
    assert torch.isfinite(directions).all(), "Found non-finite values in directions."
    assert torch.isfinite(head_attributions).all(), "Found non-finite head attributions."
    print("PASS: all direction and head-attribution values are finite.")
    print("\nDirect Logit Attribution analysis completed successfully.")


if __name__ == "__main__":
    try:
        run_tests_and_analysis()
    except Exception as e:
        print("\nAnalysis failed. Full traceback:")
        traceback.print_exc()
        sys.exit(1)
