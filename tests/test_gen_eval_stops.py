"""Stop-string truncation + generation-task protocol tests (no NPU needed).

2026-09-17 protocol change: base models in few-shot completion never emit the
chat EOS, so generation runs to the full max_gen_tokens and the model keeps
writing hallucinated next rounds after the target answer. The fix cuts the
generated text at the few-shot delimiter re-appearing (lm-eval `until`
semantics). Gold-verified on the eval packs: the stop strings never occur
inside gsm8k/math gold answers or questions (N=1319/500).

These tests pin:
  - apply_stop_strings edge cases (earliest marker, boundaries, no-op)
  - the gsm8k first-#### extractor is protected by the cut
  - the math rfind-boxed extractor is protected by the cut (its real fix:
    a hallucinated next round's boxed answer would out-rfind the real one)
  - bare newlines inside math solutions (paragraphs, [asy] blocks) survive
  - the STEM_TASKS registry values (caps + stop strings == delimiters)
  - the NLL prompt budget stays frozen at the historical cap

Run: python tests/test_gen_eval_stops.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.core_eval import (  # noqa: E402
    apply_stop_strings, extract_answer, compare_answers, _NLL_FROZEN_GEN_CAP,
)
from scripts.base_eval import STEM_TASKS  # noqa: E402


def main():
    checks = []

    def check(name, ok):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        checks.append(ok)

    # --- apply_stop_strings edge cases ---
    text = "abc def ghi"
    check("no stops -> unchanged", apply_stop_strings(text, []) == text)
    check("absent stop -> unchanged", apply_stop_strings(text, ["zzz"]) == text)
    check("cut before the stop", apply_stop_strings("abcXdefYghi", ["Y"]) == "abcXdef")
    check("earliest of multiple stops wins", apply_stop_strings("abcXdefYghi", ["Y", "X"]) == "abc")
    check("stop at position 0 -> empty", apply_stop_strings("Xabc", ["X"]) == "")

    # --- gsm8k: first-#### extractor + hallucinated next round ---
    gsm_stops = ["\nAnswer: "]
    good = ("Janet sells 16 - 3 - 4 = 9 eggs a day.\n#### 18\n\n"
            "Natalia sold clips to 48 friends.\nAnswer: Natalia sold 48/2 = 24 clips.\n#### 24")
    cut = apply_stop_strings(good, gsm_stops)
    check("gsm8k hallucinated ANSWER cut (question text remains, bounded)",
          "#### 18" in cut and "#### 24" not in cut and "48/2" not in cut)
    check("gsm8k extraction correct after cut",
          compare_answers(extract_answer(cut, 'gsm8k'), "18", 'gsm8k'))

    # --- gsm8k: degenerate immediate stop -> bounded failure (None) ---
    check("gsm8k degenerate stop-at-0 judged wrong",
          extract_answer(apply_stop_strings("\nAnswer: 7", gsm_stops), 'gsm8k') is None)

    # --- gsm8k: fallback path no longer sees hallucinated ANSWER numbers
    # (the hallucinated question text itself remains in the window — bounded,
    # and only reachable for samples that never produced '####') ---
    no_mark = "She computes 16 - 3 = 13 eggs.\n\nTom likes apples.\nAnswer: Tom has 12 apples."
    cut2 = apply_stop_strings(no_mark, gsm_stops)
    check("gsm8k fallback: hallucinated answer removed", "12" not in cut2)
    check("gsm8k fallback extracts from real CoT", extract_answer(cut2, 'gsm8k') == "13")

    # --- math: rfind-boxed protected by the cut ---
    math_stops = ["\n\nSolution: "]
    boxed = ("We have $r = 3$.\nThe answer is $\\boxed{3}$.\n\n"
             "Find the derivative of $x^2$.\n\nSolution: We get $\\boxed{2x}$.")
    mcut = apply_stop_strings(boxed, math_stops)
    check("math hallucinated solution cut", "\\boxed{3}" in mcut and "2x" not in mcut)
    check("math rfind lands on target boxed", extract_answer(mcut, 'math') == "3")

    # --- math: internal paragraphs / [asy] blocks survive ---
    para = ("We have that $r = 3$.\n\n[asy]\nunitsize(0.8 cm);\n[/asy]\n\n"
            "Therefore the answer is $\\boxed{(3,\\pi/2)}$.")
    check("math internal blank lines NOT cut", apply_stop_strings(para, math_stops) == para)

    # --- math: multi-boxed single solution keeps last-boxed semantics ---
    multi = "First pass gives $\\boxed{5}$, but simplifying yields $\\boxed{6}$."
    check("math multi-boxed rfind semantics preserved", extract_answer(multi, 'math') == "6")
    check("math multi-boxed untouched by stops", apply_stop_strings(multi, math_stops) == multi)

    # --- protocol constants / registry pins ---
    check("NLL prompt budget frozen at historical cap", _NLL_FROZEN_GEN_CAP == 256)
    tasks = {x['label']: x for x in STEM_TASKS}
    g = tasks['gsm8k_cot']
    m = tasks['math_cot_500']
    check("gsm8k cap stays 256", g['max_gen_tokens'] == 256)
    check("gsm8k stop string == its delimiter", g['stop_strings'] == [g['continuation_delimiter']])
    check("math cap raised to 1024", m['max_gen_tokens'] == 1024)
    check("math stop string == its delimiter", m['stop_strings'] == [m['continuation_delimiter']])

    print(f"\n{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
