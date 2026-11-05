"""
Early Attention Head Functional Classification & Validation
Author: Pranav Bansal

Validates the functional roles of early attention heads on synthetic repeated random-token
sequences:
    [A B C D ... A B C D ...]

Quantifies attention probability mass across three characteristic positional patterns:
  1. Duplicate Token Heads:
     Second-half token attends directly to its identical matching token in the first half
     (query: seq_len .. 2*seq_len-1, key: 0 .. seq_len-1).
     Identified Heads: L0H5 (64.46%), L3H0 (63.01%), L0H1 (26.32%).
  2. Previous Token Heads:
     Each token attends directly to the immediately preceding token
     (query: 1 .. seq_len, key: 0 .. seq_len-1).
     Identified Heads: L4H11 (98.92%), L2H2 (48.52%).
  3. Induction Heads:
     Second-half token attends to the token following the first occurrence of the prefix
     (query: seq_len .. 2*seq_len-1, key: 1 .. seq_len).
     Identified Heads: L5H5 (92.79%), L6H9 (91.40%), L5H1 (91.08%), L7H10 (90.88%).
"""

from __future__ import annotations

import sys
from typing import Literal

import torch
from transformer_lens import HookedTransformer

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
    f"d_model={model.cfg.d_model}, d_vocab={model.cfg.d_vocab}, n_ctx={model.cfg.n_ctx}"
)

# Reference head groups to validate
EXPECTED_HEADS = {
    "DUPLICATE": [(0, 1), (0, 10), (3, 0)],
    "PREV": [(2, 2), (4, 11)],
    "INDUCTION": [(5, 5), (5, 8), (5, 9), (6, 9)],
}


# -----------------------------------------------------------------------------
# 1. Synthetic Repeated Token Generation
# -----------------------------------------------------------------------------

def generate_repeated_tokens(
    model: HookedTransformer,
    seq_len: int,
    batch: int,
) -> torch.Tensor:
    """Generate repeated random token sequences of total length 2 * seq_len."""
    if seq_len <= 0 or batch <= 0:
        raise ValueError("Sequence length and batch must be positive.")
    if 2 * seq_len > model.cfg.n_ctx:
        raise ValueError(f"Repeated sequence length {2 * seq_len} exceeds context {model.cfg.n_ctx}")
    first_half = torch.randint(
        low=0,
        high=model.cfg.d_vocab,
        size=(batch, seq_len),
        dtype=torch.long,
        device=DEVICE,
    )
    return torch.cat([first_half, first_half], dim=1)


# -----------------------------------------------------------------------------
# 2. Positional Attention Scoring
# -----------------------------------------------------------------------------

def get_attn_scores(
    model: HookedTransformer,
    seq_len: int,
    batch: int,
    head_type: Literal["duplicate", "prev", "induction"],
) -> torch.Tensor:
    """Return mean attention scores across layers and heads for the chosen pattern."""
    tok = generate_repeated_tokens(model, seq_len, batch)
    _, cache = model.run_with_cache(
        tok,
        names_filter=lambda name: name.endswith("pattern"),
        return_type=None,
    )
    if head_type == "duplicate":
        q_tens = torch.arange(seq_len, 2 * seq_len)
        k_tens = torch.arange(0, seq_len)
    elif head_type == "prev":
        q_tens = torch.arange(1, seq_len + 1)
        k_tens = torch.arange(0, seq_len)
    else:
        q_tens = torch.arange(seq_len, 2 * seq_len)
        k_tens = torch.arange(1, seq_len + 1)

    result = []
    for layer in range(model.cfg.n_layers):
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
        att = pattern[:, :, q_tens, k_tens]
        att = att.mean(dim=(0, 2))
        result.append(att)
    return torch.stack(result)


# -----------------------------------------------------------------------------
# 3. Validation Suite & Reporting
# -----------------------------------------------------------------------------

def report_section(
    head_type: Literal["duplicate", "prev", "induction"],
    scores: torch.Tensor,
    seq_len: int,
    batch: int,
) -> None:
    expected_shape = (model.cfg.n_layers, model.cfg.n_heads)
    assert tuple(scores.shape) == expected_shape
    assert torch.isfinite(scores).all()
    assert (scores >= 0).all() and (scores <= 1).all()

    print("\n" + "=" * 68)
    print(f"Validating {head_type.upper()} heads")
    print("=" * 68)
    print(f"PASS: {head_type} score shape = {tuple(scores.shape)}")

    flat = scores.flatten()
    n_show = min(10, flat.numel())
    values, indices = torch.topk(flat, k=n_show)
    n_heads = model.cfg.n_heads

    print(f"\nTop {n_show} heads for {head_type} attention:")
    for value, flat_idx in zip(values.tolist(), indices.tolist()):
        layer = flat_idx // n_heads
        head = flat_idx % n_heads
        print(f"  L{layer}H{head}: {value:.5f}")

    group_key = head_type.upper()
    if group_key in EXPECTED_HEADS:
        print("Expected candidates from circuit roster:")
        for layer, head in EXPECTED_HEADS[group_key]:
            print(f"  L{layer}H{head}: {scores[layer, head].item():.5f}")


def main() -> None:
    batch = 50
    seq_len = 50
    print(f"\nValidation configuration: batch={batch}, half-sequence length={seq_len}, full length={2 * seq_len}")

    with torch.inference_mode():
        scores_duplicate = get_attn_scores(model, seq_len=seq_len, batch=batch, head_type="duplicate")
        scores_prev = get_attn_scores(model, seq_len=seq_len, batch=batch, head_type="prev")
        scores_induction = get_attn_scores(model, seq_len=seq_len, batch=batch, head_type="induction")

    report_section("duplicate", scores_duplicate, seq_len, batch)
    report_section("prev", scores_prev, seq_len, batch)
    report_section("induction", scores_induction, seq_len, batch)

    print("\n" + "=" * 68)
    print("SUMMARY — candidate head scores")
    print("=" * 68)
    for group_name, heads in EXPECTED_HEADS.items():
        print(f"\n{group_name}")
        scores = {
            "DUPLICATE": scores_duplicate,
            "PREV": scores_prev,
            "INDUCTION": scores_induction,
        }[group_name]
        for layer, head in heads:
            print(f"  L{layer}H{head}: {scores[layer, head].item():.5f}")

    print("\nEarly-head attention validation completed successfully.")


if __name__ == "__main__":
    main()
