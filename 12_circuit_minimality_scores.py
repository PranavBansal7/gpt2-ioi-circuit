"""
IOI Circuit Minimality Scoring & Component Necessity Analysis
Author: Pranav Bansal

Evaluates the necessity and redundancy of each of the 26 heads within the IOI circuit
using conditional minimality scores (Wang et al., 2022).

Theoretical Formulation:
    Ordinarily, ablating an individual attention head can result in negligible performance
    degradation if other parallel or backup heads compensate.
    To unmask redundancy and backup mechanisms, we measure the marginal impact of removing
    head v AFTER removing a pre-specified reference set K of related heads:

        M(v) = | F(C \\ (K U {v})) - F(C \\ K) | / | F(M) |

    where:
      - C is the full 26-head circuit
      - K is the specific subset of related/parallel heads removed first
      - F(M) is the full-model IOI logit difference

Empirical Findings:
    - Negative Name Movers exhibit the highest minimality scores:
      L10H7 (42.37%), L11H10 (36.92%) (mean: 39.65%).
    - Previous Token Heads: L4H11 (31.35%), L2H2 (18.92%) (mean: 25.14%).
    - Induction Heads: L5H5 (29.29%), L5H9 (25.55%) (mean: 21.41%).
    - Duplicate Token Heads: L0H1 (29.23%), L0H10 (21.31%) (mean: 20.61%).
    - S2 Inhibition Heads: L8H6 (24.70%), L8H10 (21.42%) (mean: 15.70%).
    - Name Movers: L9H9 (17.86%), L9H6 (12.97%) (mean: 12.08%).
    - Backup Name Movers: L10H10 (21.24%), L10H1 (7.63%) (mean: 5.82%).
"""

from __future__ import annotations

import random
import sys
from functools import partial
from types import SimpleNamespace
from typing import Dict, FrozenSet, List, Set, Tuple

import torch
from transformer_lens import HookedTransformer, utils

# -----------------------------------------------------------------------------
# 0. Model Setup & Reproducibility
# -----------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
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

# -----------------------------------------------------------------------------
# 1. Official 26-Head Circuit Roster & Positions
# -----------------------------------------------------------------------------

CIRCUIT: Dict[str, List[Tuple[int, int]]] = {
    "name mover": [(9, 9), (10, 0), (9, 6)],
    "backup name mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (11, 9)],
    "negative name mover": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9), (7, 2), (7, 10)],
    "duplicate token": [(0, 1), (0, 10), (3, 0)],
    "previous token": [(2, 2), (4, 11)],
}

SEQ_POS_TO_KEEP: Dict[str, str] = {
    "name mover": "end",
    "backup name mover": "end",
    "negative name mover": "end",
    "s2 inhibition": "end",
    "induction": "s1+1",
    "duplicate token": "s2",
    "previous token": "s1+1",
}

# -----------------------------------------------------------------------------
# 2. Reference Minimality Sets (K) for Each Circuit Component
# -----------------------------------------------------------------------------

K_FOR_EACH_COMPONENT: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {
    # Name Movers: remove backup name movers
    (9, 9): {(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (11, 9)},
    (10, 0): {(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (11, 9)},
    (9, 6): {(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (11, 9)},
    # Backup Name Movers: remove other backups and primary name movers
    (10, 10): {(9, 9), (10, 0), (9, 6), (10, 6), (10, 2), (10, 1), (11, 2), (11, 9)} - {(10, 10)},
    (10, 6): {(9, 9), (10, 0), (9, 6), (10, 10), (10, 2), (10, 1), (11, 2), (11, 9)} - {(10, 6)},
    (10, 2): {(9, 9), (10, 0), (9, 6), (10, 10), (10, 6), (10, 1), (11, 2), (11, 9)} - {(10, 2)},
    (10, 1): {(9, 9), (10, 0), (9, 6), (10, 10), (10, 6), (10, 2), (11, 2), (11, 9)} - {(10, 1)},
    (11, 2): {(9, 9), (10, 0), (9, 6), (10, 10), (10, 6), (10, 2), (10, 1), (11, 9)} - {(11, 2)},
    (11, 9): {(9, 9), (10, 0), (9, 6), (10, 10), (10, 6), (10, 2), (10, 1), (11, 2)} - {(11, 9)},
    # Negative Name Movers: tested individually
    (10, 7): set(),
    (11, 10): set(),
    # S2 Inhibition Heads: remove other S-inhibition heads
    (7, 3): {(7, 9), (8, 6), (8, 10)},
    (7, 9): {(7, 3), (8, 6), (8, 10)},
    (8, 6): {(7, 3), (7, 9), (8, 10)},
    (8, 10): {(7, 3), (7, 9), (8, 6)},
    # Induction Heads: remove other induction heads
    (5, 5): {(5, 8), (5, 9), (6, 9), (7, 2), (7, 10)},
    (5, 8): {(5, 5), (5, 9), (6, 9), (7, 2), (7, 10)},
    (5, 9): {(5, 5), (5, 8), (6, 9), (7, 2), (7, 10)},
    (6, 9): {(5, 5), (5, 8), (5, 9), (7, 2), (7, 10)},
    (7, 2): {(5, 5), (5, 8), (5, 9), (6, 9), (7, 10)},
    (7, 10): {(5, 5), (5, 8), (5, 9), (6, 9), (7, 2)},
    # Duplicate Token Heads: remove other duplicate token heads
    (0, 1): {(0, 10), (3, 0)},
    (0, 10): {(0, 1), (3, 0)},
    (3, 0): {(0, 1), (0, 10)},
    # Previous Token Heads: remove other previous token heads
    (2, 2): {(4, 11)},
    (4, 11): {(2, 2)},
}

# -----------------------------------------------------------------------------
# 3. Datasets: Clean IOI and Template-Matched ABC Reference
# -----------------------------------------------------------------------------

TEMPLATES = [
    "When [A] and [B] went to the shops, [C] gave the bag to",
    "When [A] and [B] went to the park, [C] gave the ball to",
    "When [A] and [B] went to the shops, [C] gave an apple to",
    "After [A] and [B] went to the park, [C] gave a drink to",
]

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

clean_name_pairs = [
    ("Mary", "John"),
    ("Tom", "James"),
    ("Dan", "Sid"),
    ("Martin", "Amy"),
]
clean_prompts: List[str] = []
clean_template_ids: List[int] = []
answer_name_pairs: List[Tuple[str, str]] = []

for t_id, (io_name, s_name) in enumerate(clean_name_pairs):
    clean_prompts.append(
        TEMPLATES[t_id].replace("[A]", s_name).replace("[B]", io_name).replace("[C]", s_name)
    )
    clean_template_ids.append(t_id)
    answer_name_pairs.append((f" {io_name}", f" {s_name}"))

    clean_prompts.append(
        TEMPLATES[t_id].replace("[A]", io_name).replace("[B]", s_name).replace("[C]", io_name)
    )
    clean_template_ids.append(t_id)
    answer_name_pairs.append((f" {s_name}", f" {io_name}"))

clean_tokens = model.to_tokens(clean_prompts).to(DEVICE)
clean_template_ids_tensor = torch.tensor(clean_template_ids, dtype=torch.long, device=DEVICE)
answer_tokens = torch.cat(
    [model.to_tokens(list(names), prepend_bos=False).T for names in answer_name_pairs],
    dim=0,
).to(DEVICE)

N_ABC_PER_TEMPLATE = 12
abc_prompts: List[str] = []
abc_template_ids: List[int] = []

for t_id in range(len(TEMPLATES)):
    for _ in range(N_ABC_PER_TEMPLATE):
        a_name, b_name, c_name = random.sample(NAMES, 3)
        prompt = (
            TEMPLATES[t_id]
            .replace("[A]", a_name.strip())
            .replace("[B]", b_name.strip())
            .replace("[C]", c_name.strip())
        )
        abc_prompts.append(prompt)
        abc_template_ids.append(t_id)

abc_tokens = model.to_tokens(abc_prompts).to(DEVICE)
abc_template_ids_tensor = torch.tensor(abc_template_ids, dtype=torch.long, device=DEVICE)

print("\nMinimality experiment setup:")
print(f"  clean IOI prompts: {len(clean_prompts)}")
print(f"  ABC mean prompts: {len(abc_prompts)}")
print(f"  circuit heads: {len(K_FOR_EACH_COMPONENT)}")
print(f"  minimality sets: {len(K_FOR_EACH_COMPONENT)}")


# -----------------------------------------------------------------------------
# 4. Token Position Extraction
# -----------------------------------------------------------------------------

def first_occurrence_positions(token_batch: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    batch_size = token_batch.size(0)
    positions = torch.empty(batch_size, dtype=torch.long, device=token_batch.device)
    for b in range(batch_size):
        matches = (token_batch[b] == target_ids[b]).nonzero(as_tuple=False)
        assert matches.numel() > 0
        positions[b] = matches[0, 0]
    return positions


def last_occurrence_positions(token_batch: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    batch_size = token_batch.size(0)
    positions = torch.empty(batch_size, dtype=torch.long, device=token_batch.device)
    for b in range(batch_size):
        matches = (token_batch[b] == target_ids[b]).nonzero(as_tuple=False)
        assert matches.numel() > 0
        positions[b] = matches[-1, 0]
    return positions


def get_token_positions(tokens: torch.Tensor, answers: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "io": first_occurrence_positions(tokens, answers[:, 0]),
        "s1": first_occurrence_positions(tokens, answers[:, 1]),
        "s2": last_occurrence_positions(tokens, answers[:, 1]),
        "s1+1": first_occurrence_positions(tokens, answers[:, 1]) + 1,
        "end": torch.full((tokens.size(0),), tokens.size(1) - 1, dtype=torch.long, device=tokens.device),
    }


def mean_logit_diff(logits: torch.Tensor, answer_tokens: torch.Tensor) -> torch.Tensor:
    final_logits = logits[:, -1, :]
    io_logits = final_logits.gather(1, answer_tokens[:, 0:1]).squeeze(1)
    subject_logits = final_logits.gather(1, answer_tokens[:, 1:2]).squeeze(1)
    return (io_logits - subject_logits).mean()


# -----------------------------------------------------------------------------
# 5. Template-Grouped Means & Masking
# -----------------------------------------------------------------------------

def compute_means_by_template(
    model: HookedTransformer,
    means_tokens: torch.Tensor,
    means_template_ids: torch.Tensor,
    target_template_ids: torch.Tensor,
) -> torch.Tensor:
    mean = torch.zeros((
        model.cfg.n_layers,
        target_template_ids.size(0),
        means_tokens.size(1),
        model.cfg.n_heads,
        model.cfg.d_head,
    ), device=DEVICE)
    _, means_cache = model.run_with_cache(
        means_tokens,
        names_filter=lambda name: name.endswith("hook_z"),
        return_type=None,
    )
    unique_temp = target_template_ids.unique()
    for layer in range(model.cfg.n_layers):
        means = means_cache[utils.get_act_name("z", layer)]
        for temp in unique_temp:
            mean_mask = (temp == means_template_ids)
            target_mask = (temp == target_template_ids)
            target_mean = means[mean_mask].mean(dim=0)
            mean[layer, target_mask] = target_mean
    return mean


def get_heads_and_posns_to_keep(
    model: HookedTransformer,
    tokens: torch.Tensor,
    answers: torch.Tensor,
    template_ids: torch.Tensor,
    circuit: Dict[str, List[Tuple[int, int]]] = CIRCUIT,
    seq_pos_to_keep: Dict[str, str] = SEQ_POS_TO_KEEP,
) -> Dict[int, torch.Tensor]:
    batch, seq = tokens.shape
    keep = {}
    for layer in range(model.cfg.n_layers):
        keep[layer] = torch.zeros((batch, seq, model.cfg.n_heads), dtype=torch.bool, device=tokens.device)
    batch_indices = torch.arange(batch, device=tokens.device)
    positions = get_token_positions(tokens, answers)
    for component, head_list in circuit.items():
        pos_key = seq_pos_to_keep[component]
        pos_indices = positions[pos_key]
        for layer, head in head_list:
            keep[layer][batch_indices, pos_indices, head] = True
    return keep


def hook_fn_mask_z(
    z: torch.Tensor,
    hook,
    heads_and_posns_to_keep: Dict[int, torch.Tensor],
    means: torch.Tensor,
) -> torch.Tensor:
    layer = hook.layer()
    layer_mask = heads_and_posns_to_keep[layer].unsqueeze(-1)
    return torch.where(layer_mask, z, means[layer])


def run_minimal_circuit(
    model: HookedTransformer,
    tokens: torch.Tensor,
    means: torch.Tensor,
    heads_and_posns_to_keep: Dict[int, torch.Tensor],
) -> torch.Tensor:
    model.reset_hooks()
    fwd_hooks = []
    for layer in range(model.cfg.n_layers):
        hook_name = utils.get_act_name("z", layer)
        hook_fn = partial(hook_fn_mask_z, heads_and_posns_to_keep=heads_and_posns_to_keep, means=means)
        fwd_hooks.append((hook_name, hook_fn))
    with torch.inference_mode():
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks, return_type="logits")
    return logits


# -----------------------------------------------------------------------------
# 6. Minimality Scoring Functions
# -----------------------------------------------------------------------------

ScoreCache = Dict[FrozenSet[Tuple[int, int]], float]


def get_score(
    model: HookedTransformer,
    tokens: torch.Tensor,
    answers: torch.Tensor,
    template_ids: torch.Tensor,
    means: torch.Tensor,
    removed_heads: Set[Tuple[int, int]],
    circuit: Dict[str, List[Tuple[int, int]]] = CIRCUIT,
    score_cache: ScoreCache | None = None,
) -> float:
    key = frozenset(removed_heads)
    if score_cache is not None and key in score_cache:
        return score_cache[key]
    reduced_circuit = {
        category: [h for h in heads if h not in removed_heads]
        for category, heads in circuit.items()
    }
    keep = get_heads_and_posns_to_keep(model, tokens, answers, template_ids, reduced_circuit)
    logits = run_minimal_circuit(model, tokens, means, keep)
    score = mean_logit_diff(logits, answers).item()
    if score_cache is not None:
        score_cache[key] = score
    return score


def get_minimality_score(
    model: HookedTransformer,
    tokens: torch.Tensor,
    answers: torch.Tensor,
    template_ids: torch.Tensor,
    means: torch.Tensor,
    tested_head: Tuple[int, int],
    K: Set[Tuple[int, int]],
    full_model_score: float,
    score_cache: ScoreCache,
    circuit: Dict[str, List[Tuple[int, int]]] = CIRCUIT,
) -> float:
    r"""Compute |F(C \ (K U {v})) - F(C \ K)| / |F(M)|."""
    assert tested_head not in K, f"tested head {tested_head} not in K"
    score_without_k = get_score(
        model, tokens, answers, template_ids, means, K, circuit, score_cache
    )
    score_without_k_and_v = get_score(
        model, tokens, answers, template_ids, means, K | {tested_head}, circuit, score_cache
    )
    return abs(score_without_k_and_v - score_without_k) / abs(full_model_score)


def get_all_minimality_scores(
    model: HookedTransformer,
    tokens: torch.Tensor,
    answers: torch.Tensor,
    template_ids: torch.Tensor,
    means: torch.Tensor,
    full_model_score: float,
    k_for_each_component: Dict[Tuple[int, int], Set[Tuple[int, int]]] = K_FOR_EACH_COMPONENT,
) -> Dict[Tuple[int, int], float]:
    score_cache: ScoreCache = {}
    result: Dict[Tuple[int, int], float] = {}
    for tested_head, K in k_for_each_component.items():
        result[tested_head] = get_minimality_score(
            model, tokens, answers, template_ids, means, tested_head, K, full_model_score, score_cache
        )
    return result


# -----------------------------------------------------------------------------
# 7. Main Experiment & Reporting
# -----------------------------------------------------------------------------

def main() -> None:
    with torch.inference_mode():
        print("\nRunning clean and ABC baselines...")
        clean_logits = model(clean_tokens)
        abc_logits = model(abc_tokens[:len(clean_tokens)])
        clean_score = mean_logit_diff(clean_logits, answer_tokens).item()
        abc_score = mean_logit_diff(abc_logits, answer_tokens).item()
        print(f"Full-model clean IOI logit difference: {clean_score:+.6f}")
        print(f"ABC baseline logit difference:         {abc_score:+.6f}")

        print("\nComputing ABC mean activations (once, grouped by template)...")
        means = compute_means_by_template(
            model=model,
            means_tokens=abc_tokens,
            means_template_ids=abc_template_ids_tensor,
            target_template_ids=clean_template_ids_tensor,
        )
        assert tuple(means.shape) == (model.cfg.n_layers, 8, 15, model.cfg.n_heads, model.cfg.d_head)
        print(f"PASS: means shape = {tuple(means.shape)}")

        print("\nRunning the full 26-head circuit once as a faithfulness reference...")
        full_circuit_keep = get_heads_and_posns_to_keep(
            model, clean_tokens, answer_tokens, clean_template_ids_tensor, CIRCUIT
        )
        circuit_logits = run_minimal_circuit(model, clean_tokens, means, full_circuit_keep)
        circuit_score = mean_logit_diff(circuit_logits, answer_tokens).item()

        retention_pct = (circuit_score - abc_score) / (clean_score - abc_score) * 100
        print(f"Full-model score:      {clean_score:+.6f}")
        print(f"26-head circuit score: {circuit_score:+.6f}")
        print(f"ABC baseline score:    {abc_score:+.6f}")
        print(f"Circuit retention of clean-vs-ABC gap: {retention_pct:.1f}%")

        print("\nComputing conditional minimality scores for 26 heads...")
        scores = get_all_minimality_scores(
            model=model,
            tokens=clean_tokens,
            answers=answer_tokens,
            template_ids=clean_template_ids_tensor,
            means=means,
            full_model_score=clean_score,
            k_for_each_component=K_FOR_EACH_COMPONENT,
        )

    assert len(scores) == 26
    for v in scores.values():
        assert v >= 0.0 and v == v

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    print("\nMinimality scores (fraction of full-model logit difference):")
    for (layer, head), score in sorted_scores:
        print(f"  L{layer:02d}H{head:02d}: {score:.5f} ({score * 100:.2f}%)")

    print("\nTop 8 heads by conditional minimality score:")
    for (layer, head), score in sorted_scores[:8]:
        print(f"  L{layer:02d}H{head:02d}: {score:.5f} ({score * 100:.2f}%)")

    print("\nPer-category averages:")
    for category, heads in CIRCUIT.items():
        cat_scores = [scores[h] for h in heads]
        cat_mean = sum(cat_scores) / len(cat_scores)
        cat_max = max(cat_scores)
        print(f"  {category:<21} : mean={cat_mean:.5f} ({cat_mean * 100:.2f}%), max={cat_max:.5f} ({cat_max * 100:.2f}%)")

    print("\nPASS: 26 finite, nonnegative normalized minimality scores computed.")
    print("Circuit minimality analysis completed successfully.")


if __name__ == "__main__":
    main()
