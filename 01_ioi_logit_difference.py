"""
Indirect Object Identification (IOI) — Task Setup & Logit-Difference Metric
Author: Pranav Bansal

This module establishes the foundational behavioral metric for the Indirect Object
Identification (IOI) task in GPT-2 Small, replicating the experimental setup of
Wang et al. (2022) ("Interpretability in the Wild: a Circuit for Indirect Object
Identification in GPT-2 small").

Task Formulation:
    Given an IOI context such as:
    "When John and Mary went to the shops, John gave the bag to"
    the model must predict the indirect object (" Mary", IO) rather than
    the repeated subject (" John", S).

Metric:
    logit_diff = logit(IO) - logit(S) at the final sequence position (" to").
    A positive logit difference indicates that the model favors the correct indirect
    object over the duplicate subject.
"""

from __future__ import annotations

from typing import Union

import torch as t
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-small"
DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")

PROMPT_FORMAT = [
    "When John and Mary went to the shops,{} gave the bag to",
    "When Tom and James went to the park,{} gave the ball to",
    "When Dan and Sid went to the shops,{} gave an apple to",
    "After Martin and Amy went to the park,{} gave a drink to",
]

# Each pair is (indirect-object name, repeated-subject name), including the
# leading space because GPT-2's answer tokens are space-prefixed tokens.
NAME_PAIRS = [
    (" Mary", " John"),
    (" Tom", " James"),
    (" Dan", " Sid"),
    (" Martin", " Amy"),
]

# Each template is used symmetrically, swapping which name acts as the subject.
PROMPTS = [
    prompt.format(subject)
    for prompt, names in zip(PROMPT_FORMAT, NAME_PAIRS)
    for subject in names[::-1]
]
ANSWERS = [names[::direction] for names in NAME_PAIRS for direction in (1, -1)]


def logits_to_ave_logit_diff(
    logits: t.Tensor,
    answer_tokens: t.Tensor,
    per_prompt: bool = False,
) -> t.Tensor:
    """Return logit(IO) - logit(S), either for each prompt or averaged.

    Args:
        logits: Tensor of shape (batch, sequence, vocab).
        answer_tokens: Integer tensor of shape (batch, 2). Column 0 is the
            correct IO answer token ID; column 1 is the incorrect subject token ID.
        per_prompt: If True, return shape (batch). Otherwise return a scalar mean.

    Returns:
        Tensor of per-prompt logit differences, or scalar average logit difference.
    """
    final_logits = logits[:, -1]
    answer_logits = t.gather(final_logits, dim=1, index=answer_tokens)
    answer_logit_diff = answer_logits[:, 0] - answer_logits[:, 1]
    if per_prompt:
        return answer_logit_diff
    return answer_logit_diff.mean()


def run_synthetic_tests() -> None:
    """Test the metric independently of GPT-2's actual behavior."""
    logits = t.zeros((3, 4, 10), dtype=t.float32)
    logits[:, 0, :] = 100.0
    logits[:, 1, :] = -100.0
    logits[:, 2, :] = 50.0

    # Final position: known IO-vs-subject differences are 4, -3, and 1.
    logits[0, -1, 2] = 5.0
    logits[0, -1, 7] = 1.0
    logits[1, -1, 4] = 2.0
    logits[1, -1, 3] = 5.0
    logits[2, -1, 6] = 0.5
    logits[2, -1, 8] = -0.5
    answer_tokens = t.tensor([[2, 7], [4, 3], [6, 8]], dtype=t.long)

    per_prompt = logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=True)
    expected = t.tensor([4.0, -3.0, 1.0])
    assert per_prompt.shape == (3,), f"Expected (3,), got {tuple(per_prompt.shape)}"
    assert t.allclose(per_prompt, expected, atol=1e-6), (
        f"Per-prompt metric incorrect: got {per_prompt.tolist()}"
    )

    average = logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=False)
    assert average.ndim == 0, f"Expected scalar mean, got shape {tuple(average.shape)}"
    assert t.allclose(average, expected.mean(), atol=1e-6), (
        f"Average metric incorrect: got {average.item()}"
    )
    print("SYNTHETIC METRIC TESTS PASSED")


def build_answer_tokens(model: HookedTransformer) -> t.Tensor:
    """Convert each (correct, incorrect) name pair into a (batch, 2) ID tensor."""
    answer_rows = []
    for answer_pair in ANSWERS:
        pair_tokens = model.to_tokens(list(answer_pair), prepend_bos=False)
        assert pair_tokens.shape == (2, 1), (
            f"Expected both answer names to be single tokens, got {tuple(pair_tokens.shape)} "
            f"for answer pair {answer_pair!r}."
        )
        answer_rows.append(pair_tokens[:, 0])
    return t.stack(answer_rows).to(dtype=t.long, device=DEVICE)


def main() -> None:
    print("=" * 78)
    print(" Indirect Object Identification: Baseline Logit Difference")
    print("=" * 78)
    print()
    print("Task: Prefer the indirect-object name (IO) over the repeated subject (S).")
    print("Metric: logit(IO) - logit(S) at the final prompt position.")
    print("Positive score => IO preferred; negative score => subject preferred.")
    print()

    run_synthetic_tests()
    print()

    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=str(DEVICE),
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )
    model.eval()
    print("Model loaded.")
    print()

    tokens = model.to_tokens(PROMPTS, prepend_bos=True).to(DEVICE)
    answer_tokens = build_answer_tokens(model)

    assert tokens.shape[0] == len(PROMPTS)
    assert answer_tokens.shape == (len(PROMPTS), 2)

    print("PROMPTS AND ANSWERS")
    print("-" * 78)
    for i, (prompt, answers) in enumerate(zip(PROMPTS, ANSWERS)):
        print(f"{i + 1}. {prompt!r}")
        print(f"   IO (correct): {answers[0]!r}; S (incorrect): {answers[1]!r}")
        print(f"   answer token IDs: {answer_tokens[i].tolist()}")
    print()

    print(f"tokens.shape        = {tuple(tokens.shape)}")
    print(f"answer_tokens.shape = {tuple(answer_tokens.shape)}")
    print()

    with t.inference_mode():
        logits, cache = model.run_with_cache(tokens)
        per_prompt_diff = logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=True)
        average_diff = logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=False)

    print("PER-PROMPT LOGIT DIFFERENCES")
    print("-" * 78)
    for i, (answers, diff) in enumerate(zip(ANSWERS, per_prompt_diff.tolist())):
        interpretation = "IO preferred" if diff > 0 else "subject preferred" if diff < 0 else "tie"
        print(f"{i + 1}. IO={answers[0]!r:>9} vs S={answers[1]!r:>9} | diff={diff:+8.4f} | {interpretation}")
    print()
    print(f"Average IO-vs-subject logit difference: {average_diff.item():+.4f}")
    print(f"Cache type: {type(cache).__name__}")
    print(f"Example resid_pre layer 0 shape: {tuple(cache['resid_pre', 0].shape)}")
    print()

    assert per_prompt_diff.shape == (len(PROMPTS),)
    assert t.allclose(average_diff, per_prompt_diff.mean(), atol=1e-6)
    assert t.isfinite(per_prompt_diff).all() and t.isfinite(average_diff)

    print("REAL-DATA METRIC TESTS PASSED")
    print("Baseline IOI Logit Difference verification complete.")


if __name__ == "__main__":
    main()
