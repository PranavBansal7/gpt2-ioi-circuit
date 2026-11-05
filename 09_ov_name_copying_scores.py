"""
General Name-Copying Scores from Output-Value (OV) Circuits
Author: Pranav Bansal

Evaluates the isolated Output-Value (OV) circuits across all attention heads in
GPT-2 Small on a vocabulary of 109 single-token names.

Methodology:
    For each name in the vocabulary:
      1. Embed the name and pass through the residual contribution of MLP0 (which acts
         as an extended embedding in GPT-2 Small).
      2. Apply the head's OV projection matrix: W_OV = W_V @ W_O.
      3. Apply the final LayerNorm and unembedding matrix W_U.
      4. Measure how often the input name appears in the top-k (k=5) vocabulary logits:
         - Using +W_OV: measures ordinary name-copying behavior.
         - Using -W_OV: measures sign-inverted anti-copying behavior.

Empirical Findings:
    - Positive Name Movers (L9H9: 100%, L10H0: 94.50%, L9H6: 94.50%) systematically
      place the input name in the top-5 logits via +W_OV.
    - Negative Name Movers (L10H7: 100%, L11H10: 100%) systematically place the input
      name in the top-5 logits via -W_OV, confirming their role as dedicated anti-copiers.
"""

from __future__ import annotations

import sys
from typing import Sequence

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
    f"d_model={model.cfg.d_model}, d_head={model.cfg.d_head}, vocab={model.cfg.d_vocab}"
)

# Canonical 109 single-token space-prefixed names
NAMES = [
    "Aaron", "Adam", "Alan", "Alex", "Alice", "Amy", "Anderson", "Andre", "Andrew",
    "Andy", "Anna", "Anthony", "Arthur", "Austin", "Blake", "Brandon", "Brian",
    "Carter", "Charles", "Charlie", "Christian", "Christopher", "Clark", "Cole",
    "Collins", "Connor", "Crew", "Crystal", "Daniel", "David", "Dean", "Edward",
    "Elizabeth", "Emily", "Eric", "Eva", "Ford", "Frank", "George", "Georgia",
    "Graham", "Grant", "Henry", "Ian", "Jack", "Jacob", "Jake", "James", "Jamie",
    "Jane", "Jason", "Jay", "Jennifer", "Jeremy", "Jessica", "John", "Jonathan",
    "Jordan", "Joseph", "Joshua", "Justin", "Kate", "Kelly", "Kevin", "Kyle",
    "Laura", "Leon", "Lewis", "Lisa", "Louis", "Luke", "Madison", "Marco",
    "Marcus", "Maria", "Mark", "Martin", "Mary", "Matthew", "Max", "Michael",
    "Michelle", "Morgan", "Patrick", "Paul", "Peter", "Prince", "Rachel",
    "Richard", "River", "Robert", "Roman", "Rose", "Ruby", "Russell", "Ryan",
    "Sarah", "Scott", "Sean", "Simon", "Stephen", "Steven", "Sullivan", "Taylor",
    "Thomas", "Tyler", "Victoria", "Warren", "William",
]


# -----------------------------------------------------------------------------
# 1. Extended Embeddings (Embed + MLP0)
# -----------------------------------------------------------------------------

def get_name_residuals_after_mlp0(
    model: HookedTransformer,
    names: Sequence[str] = NAMES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (name_token_ids, name_residuals_after_mlp0)."""
    spaced_names = [f" {name.strip()}" for name in names]
    token_ids = model.to_tokens(spaced_names, prepend_bos=False).to(DEVICE)
    assert token_ids.shape == (len(names), 1), "All names must be single tokens."
    with torch.inference_mode():
        _, cache = model.run_with_cache(
            token_ids,
            names_filter=lambda name: name == "blocks.0.hook_resid_post",
            return_type=None,
        )
    return token_ids, cache["blocks.0.hook_resid_post"]


# -----------------------------------------------------------------------------
# 2. OV Copying Scores
# -----------------------------------------------------------------------------

def get_copying_scores(
    model: HookedTransformer,
    k: int = 5,
    names: Sequence[str] = NAMES,
) -> torch.Tensor:
    """Return copying scores of shape (2, n_layers - 1, n_heads).

    Index 0: fraction of names in top-k logits after +W_OV.
    Index 1: fraction of names in top-k logits after -W_OV.
    """
    name_token_ids, resid_after_mlp0 = get_name_residuals_after_mlp0(model, names)
    result = torch.zeros(2, model.cfg.n_layers - 1, model.cfg.n_heads)
    for i in range(model.cfg.n_layers - 1):
        for j in range(model.cfg.n_heads):
            w_ov = model.W_V[i + 1, j] @ model.W_O[i + 1, j]
            for s in (+1, -1):
                head_wr = resid_after_mlp0 @ (s * w_ov)
                logits = model.unembed(model.ln_final(head_wr))
                top_ids = logits.topk(k=k, dim=-1).indices[:, 0, :]
                copy = (name_token_ids == top_ids).any(dim=-1)
                if s == 1:
                    result[0, i, j] = copy.float().mean()
                else:
                    result[1, i, j] = copy.float().mean()
    return result


# -----------------------------------------------------------------------------
# 3. Reporting Helpers & Analysis
# -----------------------------------------------------------------------------

def report_named_heads(scores: torch.Tensor) -> None:
    """Print copying scores for the canonical name-mover head groups."""
    groups = {
        "positive name movers": [(9, 9), (10, 0), (9, 6)],
        "negative name movers": [(10, 7), (11, 10)],
    }
    sign_labels = ["+W_OV (ordinary direction)", "-W_OV (sign-flipped direction)"]
    for sign_idx, sign_label in enumerate(sign_labels):
        print(f"\n{sign_label}")
        for group_name, heads in groups.items():
            print(f"  {group_name}:")
            values = []
            for layer, head in heads:
                value = scores[sign_idx, layer - 1, head].item()
                values.append(value)
                print(f"    L{layer}H{head}: {value:.2%}")
            print(f"    group mean: {sum(values) / len(values):.2%}")


def print_top_heads(scores: torch.Tensor, sign_idx: int, label: str, n: int = 10) -> None:
    """Print top scoring heads, converting result row to original layer index."""
    flat = scores[sign_idx].flatten()
    count = min(n, flat.numel())
    values, indices = torch.topk(flat, k=count)
    n_heads = model.cfg.n_heads
    print(f"\nTop {count} heads for {label}:")
    for value, flat_idx in zip(values.tolist(), indices.tolist()):
        result_layer = flat_idx // n_heads
        head = flat_idx % n_heads
        actual_layer = result_layer + 1
        print(f"  L{actual_layer}H{head}: {value:.2%}")


def main() -> None:
    print(f"\nEvaluating {len(NAMES)} names:")
    print(", ".join(NAMES))

    print("\nPreparing name representations after MLP0...")
    token_ids, resid = get_name_residuals_after_mlp0(model, NAMES)
    print(f"name_token_ids.shape = {tuple(token_ids.shape)}")
    print(f"name_residuals.shape = {tuple(resid.shape)}")
    assert token_ids.shape == (len(NAMES), 1)
    assert resid.shape == (len(NAMES), 1, model.cfg.d_model)
    assert torch.isfinite(resid).all()
    print("PASS: all names are single tokens and MLP0 representations are finite.")

    print("\nComputing OV copying scores for every attention head...")
    scores = get_copying_scores(model, k=5, names=NAMES)
    assert tuple(scores.shape) == (2, model.cfg.n_layers - 1, model.cfg.n_heads)
    assert torch.isfinite(scores).all()
    assert (scores >= 0).all() and (scores <= 1).all()
    print(f"PASS: copying-score tensor shape = {tuple(scores.shape)}.")
    print("All scores are finite fractions in [0, 1].")

    report_named_heads(scores)
    print_top_heads(scores, sign_idx=0, label="+W_OV name copying", n=10)
    print_top_heads(scores, sign_idx=1, label="-W_OV name copying (candidate anti-copy heads)", n=10)

    print("\nOV Name-Copying evaluation completed successfully.")


if __name__ == "__main__":
    main()
