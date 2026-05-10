"""
GeekMAN — Geek-oriented username Matching Across online Networks.

This module is a clean, importable rewrite of the algorithm originally
implemented in `geekman_matching.ipynb` (https://github.com/mrayhanulmasud/geekman).

The publicly facing entry points are:

    * `load_word_list(path)`            — read the curated dictionary
    * `compute_similarity(u1, u2, ...)` — return a single similarity score
    * `compute_similarity_detailed(...)`— return the score together with
                                          per-feature scores and the
                                          chunkifications, so the UI can
                                          show *why* a pair scored the way
                                          it did
    * `compute_batch(df, ...)`          — vectorised batch version used by
                                          the "File Upload" tab
"""

from __future__ import annotations

import bisect
import re
import string
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd
from py_stringmatching.similarity_measure.jaccard import Jaccard
from py_stringmatching.similarity_measure.levenshtein import Levenshtein
from py_stringmatching.similarity_measure.monge_elkan import MongeElkan

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

ALPHABET_LOWER = list(string.ascii_lowercase)
ALPHABET_UPPER = list(string.ascii_uppercase)
DIGITS = list(string.digits)

#: minimum number of characters that can count as a dictionary "token"
MIN_LENGTH = 3

#: leet-speak / slang character substitution map
SLANGIFICATION_LINES = [
    "0od", "1iltj", "2z", "3es", "4ar", "5s",
    "6g", "7tljr", "8b", "9g", "@a", "!i", "$s",
]
SLANGIFICATION_MAP: dict[str, str] = {ln[0]: ln[1:] for ln in SLANGIFICATION_LINES}
SLANG_ALPHABET: list[str] = list(SLANGIFICATION_MAP.keys())

#: characters that count as "valid" (used inside dictionary chunkification)
_VALID_DICT_CHARS = set(ALPHABET_LOWER + SLANG_ALPHABET)


# --------------------------------------------------------------------------- #
# Dictionary                                                                  #
# --------------------------------------------------------------------------- #

def load_word_list(path: str) -> set[str]:
    """Read the curated dictionary from disk and return it as a set."""
    words: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            w = line.strip()
            if w:
                words.add(w)
    return words


# --------------------------------------------------------------------------- #
# Weighted Interval Scheduling                                                #
# (used to pick the optimal non-overlapping set of dictionary chunks)         #
# Original implementation: https://github.com/aladdinpersson/Algorithms-      #
# Collection-Python                                                           #
# --------------------------------------------------------------------------- #

class WeightedIntervalScheduling:
    """Greedy DP for picking the highest-coverage non-overlapping intervals."""

    def __init__(self, intervals):
        self.I = sorted(intervals, key=lambda t: t[1])
        self.OPT: list[int] = []
        self.solution: list = []

    def previous_intervals(self):
        finish = [t[1] for t in self.I]
        return [bisect.bisect(finish, self.I[i][0]) - 1 for i in range(len(self.I))]

    def find_solution(self, j):
        if j == -1:
            return
        if (self.I[j][2] + self.compute_opt(self.p[j])) > self.compute_opt(j - 1):
            self.solution.append(self.I[j])
            self.find_solution(self.p[j])
        else:
            self.find_solution(j - 1)

    def compute_opt(self, j):
        if j == -1:
            return 0
        if 0 <= j < len(self.OPT):
            return self.OPT[j]
        return max(
            self.I[j][2] + self.compute_opt(self.p[j]),
            self.compute_opt(j - 1),
        )

    def weighted_interval(self):
        if not self.I:
            return 0, self.solution
        self.p = self.previous_intervals()
        for j in range(len(self.I)):
            self.OPT.append(self.compute_opt(j))
        self.find_solution(len(self.I) - 1)
        return self.OPT[-1], self.solution[::-1]


# --------------------------------------------------------------------------- #
# Chunkification helpers                                                      #
# --------------------------------------------------------------------------- #

def _get_dict_chunks(username: str, word_list: set[str]):
    """Greedy dictionary chunking using weighted-interval scheduling.

    Returns ``(num_dict_chunks, all_intervals, total_dict_coverage)``.
    """
    n = len(username)
    next_list = [(-1, -1, -1) for _ in range(n)]

    for i in range(n - 1, -1, -1):
        next_list[i] = (i, i + 1, 1)
        if i + MIN_LENGTH <= n and username[i] in ALPHABET_LOWER:
            local_optimal = 0
            curr_length = MIN_LENGTH
            for _ in range(i + MIN_LENGTH - 1, n):
                if username[i: i + curr_length] in word_list:
                    next_length = 0
                    if i + curr_length < n:
                        next_end = next_list[i + curr_length][1]
                        next_length = next_end - (i + curr_length)
                    if curr_length + next_length >= local_optimal:
                        local_optimal = curr_length + next_length
                        next_list[i] = (i, i + curr_length, curr_length)
                curr_length += 1

    sched = WeightedIntervalScheduling(next_list)
    _, chunks = sched.weighted_interval()
    dict_chunk_lengths = [c[2] for c in chunks if c[2] > 1]
    return len(dict_chunk_lengths), chunks, sum(dict_chunk_lengths)


def _slangify(username: str) -> set[str]:
    """Return all leet-speak/slang transformations of ``username``."""
    found = [c for c in SLANGIFICATION_MAP if c in username]
    if not found:
        return set()
    transformed = [username]
    for slang_char in found:
        new_t: list[str] = []
        for replace_char in SLANGIFICATION_MAP[slang_char]:
            for tu in transformed:
                new_t.append(tu.replace(slang_char, replace_char))
        transformed = new_t
    return set(transformed)


def get_dict_token_based_chunkification(
    username: str, word_list: set[str]
) -> list[str]:
    """Dictionary-token based chunkification (with slang awareness)."""
    if username.isnumeric():
        return []

    username = "".join(c for c in username.lower() if c in _VALID_DICT_CHARS)
    no_slang = _get_dict_chunks(username, word_list)
    token_based = [username[c[0]: c[1]] for c in no_slang[1]]

    slangified = _slangify(username)
    opt_chunkification = no_slang
    opt_username: str | None = None
    if slangified:
        for su in slangified:
            cs = _get_dict_chunks(su, word_list)
            better = (cs[2] > opt_chunkification[2]) or (
                cs[2] == opt_chunkification[2] and cs[0] < opt_chunkification[0]
            )
            if better:
                opt_chunkification = cs
                opt_username = su
        if opt_username is not None:
            token_based = [
                opt_username[c[0]: c[1]] if c[2] > 1 else username[c[0]: c[1]]
                for c in opt_chunkification[1]
            ]

    return [t for t in token_based if len(t) >= MIN_LENGTH]


def get_digit_based_chunkification(username: str) -> tuple[list[str], list[str]]:
    """Split ``username`` into letter-only and digit-only groups."""
    username = username.lower()
    letters = [c.strip() for c in re.findall(r"([^0-9]+)", username)]
    digits = [c.strip() for c in re.findall(r"([0-9]+)", username)]
    return letters, digits


def get_symbol_based_chunkification(username: str) -> list[str]:
    """Split ``username`` on any non-alphanumeric character."""
    username = username.lower()
    return [c for c in re.split(r"[^a-zA-Z0-9]+", username) if len(c) > 0]


def get_capital_letter_based_chunkification(username: str) -> list[str]:
    """Split ``username`` at every uppercase letter (CamelCase aware)."""
    if username.islower():
        return []
    start = 0
    chunks: list[str] = []
    for i in range(1, len(username)):
        if username[i] in ALPHABET_UPPER:
            chunks.append(username[start: i].lower().strip())
            start = i
    chunks.append(username[start:].lower())
    return chunks


def get_username_without_symbol_digit(username: str) -> str:
    """Strip all non-letter characters from ``username``."""
    return "".join(re.split(r"[^a-z]+", username.lower()))


def _chunklen(chunks: Iterable[str]) -> int:
    return sum(len(c) for c in chunks)


# --------------------------------------------------------------------------- #
# Similarity                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class SimilarityBreakdown:
    """Full breakdown of how a similarity score was computed."""

    score: float
    component_scores: dict[str, float] = field(default_factory=dict)
    chunkifications_1: dict[str, list[str] | str] = field(default_factory=dict)
    chunkifications_2: dict[str, list[str] | str] = field(default_factory=dict)


def _safe_lev(lev: Levenshtein, a: str, b: str) -> float:
    return lev.get_sim_score(a, b) if (a and b) else 0.0


def compute_similarity_detailed(
    u1: str, u2: str, word_list: set[str]
) -> SimilarityBreakdown:
    """Return the GeekMAN similarity along with all intermediate signals."""
    u1, u2 = str(u1), str(u2)

    lower_1, lower_2 = u1.lower(), u2.lower()
    wo_1 = get_username_without_symbol_digit(u1)
    wo_2 = get_username_without_symbol_digit(u2)
    cap_1 = get_capital_letter_based_chunkification(u1)
    cap_2 = get_capital_letter_based_chunkification(u2)
    sym_1 = get_symbol_based_chunkification(u1)
    sym_2 = get_symbol_based_chunkification(u2)
    dig_1 = get_digit_based_chunkification(u1)[0]
    dig_2 = get_digit_based_chunkification(u2)[0]
    dict_1 = get_dict_token_based_chunkification(u1, word_list)
    dict_2 = get_dict_token_based_chunkification(u2, word_list)

    lev = Levenshtein()
    me = MongeElkan(sim_func=Levenshtein().get_sim_score)
    jac = Jaccard()

    # 1. lowercase Levenshtein
    s_lower = _safe_lev(lev, lower_1, lower_2)

    # 2. without symbols / digits
    s_wo = _safe_lev(lev, wo_1, wo_2)

    # 3. capitalisation Jaccard (longer list first)
    if _chunklen(cap_1) and _chunklen(cap_2):
        a, b = (cap_1, cap_2) if len(cap_1) >= len(cap_2) else (cap_2, cap_1)
        s_cap = jac.get_raw_score(a, b)
    else:
        s_cap = 0.0

    # 4. symbol chunks Monge-Elkan
    if _chunklen(sym_1) and _chunklen(sym_2):
        a, b = (sym_1, sym_2) if len(sym_1) >= len(sym_2) else (sym_2, sym_1)
        s_sym = me.get_raw_score(a, b)
    else:
        s_sym = 0.0

    # 5. digit chunks Monge-Elkan
    if _chunklen(dig_1) and _chunklen(dig_2):
        a, b = (dig_1, dig_2) if len(dig_1) >= len(dig_2) else (dig_2, dig_1)
        s_dig = me.get_raw_score(a, b)
    else:
        s_dig = 0.0

    # 6. dictionary token chunks (weighted Monge-Elkan)
    if _chunklen(dict_1) and _chunklen(dict_2):
        a, b = (dict_1, dict_2) if len(dict_1) >= len(dict_2) else (dict_2, dict_1)
        raw = me.get_raw_score(a, b)
        denom = (len(lower_1) + len(lower_2)) or 1
        weight = (_chunklen(dict_1) + _chunklen(dict_2)) / denom
        s_dict = raw * weight
    else:
        s_dict = 0.0

    components = {
        "Levenshtein (lowercase)": round(s_lower, 3),
        "Levenshtein (no symbols / digits)": round(s_wo, 3),
        "Jaccard (capitalisation chunks)": round(s_cap, 3),
        "Monge–Elkan (symbol chunks)": round(s_sym, 3),
        "Monge–Elkan (digit chunks)": round(s_dig, 3),
        "Monge–Elkan (dictionary chunks, weighted)": round(s_dict, 3),
    }
    final = round(max(components.values()), 3)

    return SimilarityBreakdown(
        score=final,
        component_scores=components,
        chunkifications_1={
            "lowercase": lower_1,
            "without symbols/digits": wo_1,
            "capitalisation chunks": cap_1,
            "symbol chunks": sym_1,
            "digit chunks": dig_1,
            "dictionary chunks": dict_1,
        },
        chunkifications_2={
            "lowercase": lower_2,
            "without symbols/digits": wo_2,
            "capitalisation chunks": cap_2,
            "symbol chunks": sym_2,
            "digit chunks": dig_2,
            "dictionary chunks": dict_2,
        },
    )


def compute_similarity(u1: str, u2: str, word_list: set[str]) -> float:
    """Return only the final similarity score (kept for backwards compat.)."""
    return compute_similarity_detailed(u1, u2, word_list).score


# --------------------------------------------------------------------------- #
# Batch helper                                                                #
# --------------------------------------------------------------------------- #

def compute_batch(
    df: pd.DataFrame,
    word_list: set[str],
    col_a: str = "username_1",
    col_b: str = "username_2",
    progress_callback=None,
) -> pd.DataFrame:
    """Compute similarities row-wise and return ``df`` with a ``sim_score``
    column appended.

    ``progress_callback`` (optional): callable receiving a float in [0, 1]
    for UI progress bars.
    """
    out = df.copy()
    out[col_a] = out[col_a].astype(str)
    out[col_b] = out[col_b].astype(str)

    n = len(out)
    scores: list[float] = []
    for i, (a, b) in enumerate(zip(out[col_a], out[col_b]), start=1):
        scores.append(compute_similarity(a, b, word_list))
        if progress_callback is not None and (i % max(1, n // 100) == 0 or i == n):
            progress_callback(i / n)
    out["sim_score"] = scores
    return out
