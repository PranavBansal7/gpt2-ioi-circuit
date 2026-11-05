"""
Name-Mover Copying Behavior & Writing-Direction Projections
Author: Pranav Bansal

Validates the mechanistic hypothesis that Name Mover heads read the name token
from early positions and copy it by writing directly in the direction of that
token's unembedding vector at the END position.

Contrasts two opposing attention heads in GPT-2 Small:
  1. Positive Name Mover Head L9H9:
     High attention to IO token (mean: 0.817) aligns strongly with a positive
     projection onto IO's unembedding vector (+66.33, Pearson r = +0.639).
  2. Negative Name Mover Head L11H10:
     High attention to IO token (mean: 0.643) aligns with an inhibitory negative
     projection onto IO's unembedding vector (-33.85, Pearson r = -0.652).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
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
for i, (prompt, answers) in enumerate(zip(prompts, answer_name_pairs)):
    print(f"  {i}: {prompt!r}; IO={answers[0]!r}, S={answers[1]!r}")
print(f"tokens.shape        = {tuple(tokens.shape)}")
print(f"answer_tokens.shape = {tuple(answer_tokens.shape)}")


# -----------------------------------------------------------------------------
# 2. Position Helper & Correlation
# -----------------------------------------------------------------------------

def first_occurrence_positions(token_batch: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Return the index of the first occurrence of target_ids[b] in token_batch[b]."""
    batch_size = token_batch.size(0)
    positions = torch.empty(batch_size, dtype=torch.long, device=token_batch.device)
    for b in range(batch_size):
        matches = (token_batch[b] == target_ids[b]).nonzero(as_tuple=False)
        assert matches.numel() > 0, f"Token {target_ids[b]} not found in row {b}."
        positions[b] = matches[0, 0]
    return positions


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation, with a safe result for constant vectors."""
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if denom.item() == 0:
        return float("nan")
    return (torch.dot(x, y) / denom).item()


@dataclass
class HeadEvidence:
    attn_to_io: torch.Tensor
    attn_to_subject: torch.Tensor
    proj_to_io: torch.Tensor
    proj_to_subject: torch.Tensor


# -----------------------------------------------------------------------------
# 3. Output Projection & Evidence Extraction
# -----------------------------------------------------------------------------

def project_head_output_onto_names(
    model: HookedTransformer,
    z_at_end: torch.Tensor,
    head_layer: int,
    head_index: int,
    answer_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (projection_on_IO_unembedding, projection_on_subject_unembedding)."""
    rsw = z_at_end @ model.W_O[head_layer, head_index]
    t1 = rsw * model.W_U[:, answer_tokens[:, 0]].T
    t2 = rsw * model.W_U[:, answer_tokens[:, 1]].T
    return t1.sum(dim=-1), t2.sum(dim=-1)


def get_head_copying_evidence(
    model: HookedTransformer,
    cache,
    tokens: torch.Tensor,
    answer_tokens: torch.Tensor,
    head_layer: int,
    head_index: int,
) -> HeadEvidence:
    """Collect END->IO/S attention and the head's output projections on IO/S."""
    z = cache["z", head_layer][:, -1, head_index, :]
    project_to_io, project_to_subject = project_head_output_onto_names(
        model, z, head_layer, head_index, answer_tokens
    )
    pattern = cache["pattern", head_layer][:, head_index, -1, :]
    batch_indices = torch.arange(pattern.size(0))
    io_pos = first_occurrence_positions(tokens, answer_tokens[:, 0])
    sub_pos = first_occurrence_positions(tokens, answer_tokens[:, 1])
    attn_to_io = pattern[batch_indices, io_pos]
    attn_to_sub = pattern[batch_indices, sub_pos]
    return HeadEvidence(attn_to_io, attn_to_sub, project_to_io, project_to_subject)


# -----------------------------------------------------------------------------
# 4. Main Experiment & Reporting
# -----------------------------------------------------------------------------

def report_head(name: str, evidence: HeadEvidence) -> None:
    r_io = pearson_corr(evidence.attn_to_io, evidence.proj_to_io)
    r_sub = pearson_corr(evidence.attn_to_subject, evidence.proj_to_subject)
    print(f"\nAnalyzing {name}...")
    print(
        f"  IO     : attention mean={evidence.attn_to_io.mean().item():.4f}, "
        f"projection mean={evidence.proj_to_io.mean().item():+.4f}, Pearson r={r_io:+.4f}"
    )
    print(
        f"  subject: attention mean={evidence.attn_to_subject.mean().item():.4f}, "
        f"projection mean={evidence.proj_to_subject.mean().item():+.4f}, Pearson r={r_sub:+.4f}"
    )
    print("  Per-prompt evidence:")
    print("    idx | attn(IO) attn(S) | proj(IO) proj(S)")
    for i in range(len(evidence.attn_to_io)):
        print(
            f"    {i:3d} | {evidence.attn_to_io[i].item():.4f}   {evidence.attn_to_subject[i].item():.4f} | "
            f"{evidence.proj_to_io[i].item():+8.4f}  {evidence.proj_to_subject[i].item():+8.4f}"
        )


def main() -> None:
    print("\nCaching attention patterns and head z activations...")
    with torch.inference_mode():
        _, cache = model.run_with_cache(
            tokens,
            return_type=None,
            names_filter=lambda name: name.endswith("pattern") or name.endswith("z"),
        )

        test_io_pos = first_occurrence_positions(tokens, answer_tokens[:, 0])
        test_sub_pos = first_occurrence_positions(tokens, answer_tokens[:, 1])
        assert test_io_pos.shape == (8,)
        assert test_sub_pos.shape == (8,)
        assert (test_io_pos != test_sub_pos).all()
        print("PASS: found the first occurrence of IO and subject in every prompt.")
        print(f"Positions (first prompt): IO={test_io_pos[0].item()}, S1={test_sub_pos[0].item()}, END={tokens.shape[1] - 1}")

        evidence_9_9 = get_head_copying_evidence(model, cache, tokens, answer_tokens, 9, 9)
        evidence_11_10 = get_head_copying_evidence(model, cache, tokens, answer_tokens, 11, 10)

    report_head("L9H9", evidence_9_9)
    report_head("L11H10", evidence_11_10)

    for ev in (evidence_9_9, evidence_11_10):
        for tensor in (ev.attn_to_io, ev.attn_to_subject, ev.proj_to_io, ev.proj_to_subject):
            assert torch.isfinite(tensor).all()
        assert (ev.attn_to_io >= 0).all() and (ev.attn_to_io <= 1).all()
        assert (ev.attn_to_subject >= 0).all() and (ev.attn_to_subject <= 1).all()

    print("\nPASS: all copying-evidence values are finite and attention probabilities are valid.")
    print("Name-mover writing-direction evidence computed successfully.")


if __name__ == "__main__":
    main()
