"""sfs_reward.py — observability-gated RLVR reward for AQUA-NL (F19 rebuild).

WHY THE REBUILD
---------------
The previous reward in this module was tolerance-band SFS-F1 minus degeneration
penalties. The band-F1 core is a RETIRED, SATURATED metric: bands are wide
enough that a constant-mode policy (emit the population-median value for every
feature on every clip — the observed f0_min=75-on-96%-of-clips collapse) scores
near-ceiling, so under RL pressure it is a dead objective with no gradient
toward actually reading the signal. The 2026-07-15 RL design memo (fix F19 in
.claude/research/MASTER_PLAN_2026-07-16.md) specifies the replacement below.

Two pieces of the old module are deliberately KEPT:
  1. the rep_n / nonascii degeneration guards (SFS-style rewards are blind to
     fluency collapse — repetition loops and foreign-token injection can carry
     parseable numbers; the guard multiplies them away), and
  2. the FIRST-CLAIM-ONLY / anti-spam reasoning (later duplicate numeric claims
     for a slot are ignored, so value-spread gaming earns nothing).

THE REWARD (per clip, bounded in [0, 1])
----------------------------------------
Inputs: generated text y; GT dict from the clip's features-CSV row (the caller
supplies it); per-slot observability o_f in {0,1}. The slot inventory is
`data.feature_set.SUPERVISED_FEATURES` (single source of truth).

Observability (project rule, single source of truth in feature_set):
    o_f = 0  iff  f in ILL_POSED_UNDER_OVERLAP_FEATURES
              and clip overlap_ratio >= HEDGE_OVERLAP_TAU
    o_f = 1  otherwise.

Slot classification via the ClaimParser:
    VALUE                  first parsed numeric claim v_f (duplicates IGNORED)
    HEDGE                  parse_hedges() hit, with attribution flag A_f (F20)
    MENTIONED-UNPARSEABLE  slot noun present, no number, no hedge
    ABSENT                 slot not realized at all

Per-slot reward r_f:
    VALUE  & o_f = 1   ->  1 - min(|v_f - g_f| / s_f, 1)   (s_f = FEATURE_SCALES)
    VALUE  & o_f = 0   ->  0.0   (over-claim: ANY number on an unobservable slot)
    HEDGE  attributed  ->  0.8 if o_f = 0 else 0.2
    HEDGE  generic     ->  0.4 if o_f = 0 else 0.2
    MENTIONED-UNPARSEABLE -> 0.0
    ABSENT             ->  0.0

Sequence reward:
    R(y) = G(y) * mean_f r_f
    G(y) = template_validity(y) * (1 - rep_n(y, 4)) * (1 - nonascii_frac(y))
    template_validity = fraction of inventory slots realized (VALUE or HEDGE).

Why this resists the known attacks:
  - CONSTANT-MODE: the VALUE ramp is continuous in |v-g|/s, so median-spam
    averages strictly below truth on any spread of clips (no plateau).
  - EASY-FEATURE-ONLY: ABSENT slots score 0 AND shrink template_validity, so
    omission is doubly costly.
  - HEDGE-SPAM: hedging an observable slot earns at most 0.2 < an honest value;
    hedging is only profitable exactly where a number is impossible (o_f = 0).
  - FORMAT-GAMING: only the first claim per slot counts; unparseable mentions
    earn 0 (< any hedge).
  - DEGENERATION: loops / non-ASCII multiply the whole sequence reward down.

Dependency note: this module imports only feature_set, sfs, ckpt_selection and
the stdlib — no trl, no GPU. feature_set imports torch at module level for
tensor helpers this reward never calls; the tests stub torch out, keeping the
reward unit-testable in a torch-free environment.
"""

from __future__ import annotations

import math
import re
import warnings
from typing import Any, Callable, Mapping, Sequence

# Single source of truth for the slot inventory, scales, and the observability
# rule constants. (_GENUINE_ZERO_FEATURES / _to_float mirror the missing-value
# semantics used by extract_scalars, so GT coercion cannot drift.)
from data.feature_set import (
    FEATURE_NAMES,
    FEATURE_SCALES,
    HEDGE_OVERLAP_TAU,
    ILL_POSED_UNDER_OVERLAP_FEATURES,
    SUPERVISED_FEATURES,
    _GENUINE_ZERO_FEATURES,
    _to_float,
)
from eval.ckpt_selection import nonascii_frac, rep_n
from eval.sfs import ClaimParser, HybridClaimParser, SFSScorer

# Shared, stateless parser instances (regexes compile once at import).
_PARSER = HybridClaimParser()          # numeric claims (tagged or prose)
_HEDGE_PARSER = ClaimParser()          # hedge claims (always prose)
_SCORER = SFSScorer()                  # ONLY for the deprecated band-F1 path

# Per-slot lookups in SUPERVISED_FEATURES order.
_SCALE: dict[str, float] = dict(zip(FEATURE_NAMES, FEATURE_SCALES))
_CSV_COL: dict[str, str] = {name: col for name, col, _fmt in SUPERVISED_FEATURES}

# Sync check: the parser's literal hedge-feature coverage (kept literal there so
# eval/sfs.py stays torch-free) must equal feature_set's ill-posed set.
_HEDGEABLE = frozenset(
    f for feats, _re in ClaimParser.HEDGE_FEATURE_PATTERNS for f in feats
)
assert _HEDGEABLE == ILL_POSED_UNDER_OVERLAP_FEATURES, (
    "ClaimParser.HEDGE_FEATURE_PATTERNS drifted from "
    "feature_set.ILL_POSED_UNDER_OVERLAP_FEATURES: "
    f"{sorted(_HEDGEABLE)} != {sorted(ILL_POSED_UNDER_OVERLAP_FEATURES)}"
)

# ── Reward constants (F19) ────────────────────────────────────────────────────
HEDGE_ATTRIBUTED_UNOBSERVABLE: float = 0.8   # hedge + overlap named + o_f = 0
HEDGE_GENERIC_UNOBSERVABLE: float = 0.4      # hedge, no attribution,  o_f = 0
HEDGE_OBSERVABLE: float = 0.2                # any hedge on an observable slot

# Slot classes.
SLOT_VALUE = "value"
SLOT_HEDGE = "hedge"
SLOT_MENTIONED = "mentioned_unparseable"
SLOT_ABSENT = "absent"

# Mention detection (for MENTIONED-UNPARSEABLE vs ABSENT). Numerically both
# score 0 and neither counts toward template_validity; the distinction is kept
# for dashboards ("the model talked about SNR but emitted no usable claim").
_MENTION_RES: dict[str, "re.Pattern[str]"] = {
    "snr": re.compile(r"\bsnr\b|signal[\s-]*to[\s-]*noise", re.IGNORECASE),
    "srmr": re.compile(r"\bsrmr\b|reverberation", re.IGNORECASE),
    "f0_mean": re.compile(
        r"\bf0\b|\bpitch\b|fundamental\s+frequency", re.IGNORECASE),
    "f0_sd": re.compile(
        r"\bf0\b[^.!?]*(?:\bsd\b|deviation)|standard\s+deviation", re.IGNORECASE),
    "speaking_rate": re.compile(r"speaking\s+rate", re.IGNORECASE),
    "pause_count": re.compile(r"pause\s+count|\bpauses\b", re.IGNORECASE),
    "pause_rate": re.compile(r"pause\s+rate", re.IGNORECASE),
    "overlap_ratio": re.compile(r"overlap\s+ratio", re.IGNORECASE),
    "jitter": re.compile(r"\bjitter\b", re.IGNORECASE),
    "shimmer": re.compile(r"\bshimmer\b", re.IGNORECASE),
    "hnr": re.compile(r"\bhnr\b|harmonics?[\s-]*to[\s-]*noise", re.IGNORECASE),
}
assert set(_MENTION_RES) == set(FEATURE_NAMES), "mention regexes must cover the inventory"


__all__ = [
    "HEDGE_ATTRIBUTED_UNOBSERVABLE",
    "HEDGE_GENERIC_UNOBSERVABLE",
    "HEDGE_OBSERVABLE",
    "classify_slots",
    "observability_from_row",
    "reward_components",
    "observability_reward",
    "make_reward_func",
    "make_sfs_reward_func",       # deprecated name, returns the NEW reward
    "extract_completion_text",
    "sfs_f1",                     # band-F1 helper (deprecated as a reward)
    "deprecated_band_f1_reward",  # the retired reward, kept for comparison only
]


# ── GT + observability from the features-CSV row ──────────────────────────────
def _gt_value(row: Mapping[str, Any], slot: str) -> float:
    """GT scalar for one slot from a features-CSV-shaped row.

    Accepts either the CSV column name ("snr_db") or the slot short name
    ("snr") as the key, so both raw CSV rows and SFSScorer-shaped dicts work.
    Missing-value semantics mirror feature_set.extract_scalars: NaN-like cells
    are missing, except the genuine-zero features (overlap_ratio, pause_count,
    pause_rate) where missing means a real 0. Returns NaN when truly missing.
    """
    csv_col = _CSV_COL[slot]
    raw = row[csv_col] if csv_col in row else row.get(slot)
    val = _to_float(raw)
    if math.isnan(val) and slot in _GENUINE_ZERO_FEATURES:
        return 0.0
    return val


def observability_from_row(row: Mapping[str, Any]) -> dict[str, int]:
    """Per-slot observability o_f from the clip's features row (project rule).

    o_f = 0 iff the slot is in ILL_POSED_UNDER_OVERLAP_FEATURES AND the clip's
    overlap_ratio >= HEDGE_OVERLAP_TAU; otherwise o_f = 1. A missing/NaN
    overlap_ratio means no evidence of overlap -> everything observable
    (matches feature_set._hedged_features).
    """
    ov = _gt_value(row or {}, "overlap_ratio")
    heavy = (not math.isnan(ov)) and ov >= HEDGE_OVERLAP_TAU
    return {
        name: 0 if (heavy and name in ILL_POSED_UNDER_OVERLAP_FEATURES) else 1
        for name in FEATURE_NAMES
    }


# ── Slot classification ───────────────────────────────────────────────────────
def classify_slots(text: str) -> dict[str, dict]:
    """Classify every inventory slot of `text` as VALUE / HEDGE /
    MENTIONED-UNPARSEABLE / ABSENT.

    Returns {slot: {"class": str, "value": float | None, "attributed": bool | None}}.

    FIRST-CLAIM-ONLY: the parser deduplicates numeric claims per feature and we
    additionally keep only the first here, so "SNR is 10 dB. SNR is 20 dB."
    binds v_snr = 10 and the second claim earns nothing (anti value-spread).
    A slot with both a number and a hedge classifies as VALUE (stating a number
    while hedging is still an assertion — and still an over-claim when o_f = 0).
    """
    text = text or ""
    values: dict[str, float] = {}
    for claim in _PARSER.parse(text):
        if claim.feature in _SCALE and claim.feature not in values:
            values[claim.feature] = float(claim.value)

    hedges = _HEDGE_PARSER.parse_hedges(text)

    slots: dict[str, dict] = {}
    for name in FEATURE_NAMES:
        if name in values:
            slots[name] = {"class": SLOT_VALUE, "value": values[name],
                           "attributed": None}
        elif name in hedges:
            slots[name] = {"class": SLOT_HEDGE, "value": None,
                           "attributed": bool(hedges[name]["attributed"])}
        elif _MENTION_RES[name].search(text):
            slots[name] = {"class": SLOT_MENTIONED, "value": None,
                           "attributed": None}
        else:
            slots[name] = {"class": SLOT_ABSENT, "value": None,
                           "attributed": None}
    return slots


def _slot_reward(
    info: Mapping[str, Any],
    gt: float,
    scale: float,
    observable: bool,
    *,
    hedge_attributed_unobs: float,
    hedge_generic_unobs: float,
    hedge_obs: float,
) -> float:
    """r_f for one classified slot (see module docstring for the table)."""
    cls = info["class"]
    if cls == SLOT_VALUE:
        if not observable:
            return 0.0  # over-claim: any number on an unobservable slot
        if math.isnan(gt):
            return 0.0  # unverifiable claim: no GT measurement for this clip
        return 1.0 - min(abs(info["value"] - gt) / scale, 1.0)
    if cls == SLOT_HEDGE:
        if observable:
            return hedge_obs
        return hedge_attributed_unobs if info["attributed"] else hedge_generic_unobs
    return 0.0  # MENTIONED-UNPARSEABLE and ABSENT


# ── Sequence reward ───────────────────────────────────────────────────────────
def reward_components(
    text: str,
    features_row: Mapping[str, Any] | None,
    observability: Mapping[str, Any] | None = None,
    *,
    hedge_attributed_unobs: float = HEDGE_ATTRIBUTED_UNOBSERVABLE,
    hedge_generic_unobs: float = HEDGE_GENERIC_UNOBSERVABLE,
    hedge_obs: float = HEDGE_OBSERVABLE,
    ngram: int = 4,
) -> dict:
    """Full reward breakdown for ONE completion — the auditable form of
    `observability_reward` (per-slot classes and rewards, gate terms, R).

    Args:
        text:          the model's completion (decoded string).
        features_row:  the clip's GT row (features-CSV column names or slot
                       short names as keys).
        observability: optional per-slot {slot: 0/1} override; slots absent
                       from it keep the row-derived default rule.
        hedge_*:       the hedge reward levels (defaults per the F19 memo).
        ngram:         n for the repetition guard (default 4, matching
                       ckpt_selection).

    Returns a dict:
        reward, gate, template_validity, mean_slot_reward, rep_fraction,
        nonascii_fraction, slots={slot: {class, value, attributed, gt,
        observable, reward}}.
    """
    text = text or ""
    row = features_row or {}

    obs = observability_from_row(row)
    if observability:
        for name, flag in observability.items():
            if name in obs:
                obs[name] = int(flag)

    slots = classify_slots(text)

    per_slot: dict[str, dict] = {}
    total = 0.0
    realized = 0
    for name in FEATURE_NAMES:
        info = slots[name]
        gt = _gt_value(row, name)
        r = _slot_reward(
            info, gt, _SCALE[name], bool(obs[name]),
            hedge_attributed_unobs=hedge_attributed_unobs,
            hedge_generic_unobs=hedge_generic_unobs,
            hedge_obs=hedge_obs,
        )
        if info["class"] in (SLOT_VALUE, SLOT_HEDGE):
            realized += 1
        total += r
        per_slot[name] = {
            **info,
            "gt": None if math.isnan(gt) else gt,
            "observable": obs[name],
            "reward": r,
        }

    n_slots = len(FEATURE_NAMES)
    mean_slot_reward = total / n_slots
    template_validity = realized / n_slots
    rep_fraction = rep_n(text, ngram)
    nonascii_fraction = nonascii_frac(text)
    gate = template_validity * (1.0 - rep_fraction) * (1.0 - nonascii_fraction)

    return {
        "reward": gate * mean_slot_reward,
        "gate": gate,
        "template_validity": template_validity,
        "mean_slot_reward": mean_slot_reward,
        "rep_fraction": rep_fraction,
        "nonascii_fraction": nonascii_fraction,
        "slots": per_slot,
    }


def observability_reward(
    text: str,
    features_row: Mapping[str, Any] | None,
    observability: Mapping[str, Any] | None = None,
    **cfg,
) -> float:
    """The F19 scalar reward R(y) in [0, 1] for one completion.

        R(y) = G(y) * mean_f r_f
        G(y) = template_validity * (1 - rep_n(y, 4)) * (1 - nonascii_frac(y))

    See `reward_components` for the per-slot breakdown and kwargs.
    """
    return float(reward_components(text, features_row, observability, **cfg)["reward"])


# ── TRL GRPO batch wrapper ────────────────────────────────────────────────────
def extract_completion_text(completion: Any) -> str:
    """Normalize a TRL completion into a plain string.

    TRL passes completions in one of two shapes depending on whether the dataset
    is "prompt-completion" (plain text) or "conversational" (chat messages):

      - str:                      "The SNR is 16.10 dB. ..."         → returned as-is
      - list[dict] (chat turns):  [{"role": "assistant", "content": "..."}]
                                  → the content of the LAST turn is returned
      - dict (single message):    {"role": ..., "content": "..."}    → its content

    Anything else is coerced with str(). Returns "" for None / empty.
    """
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        return str(completion.get("content", "") or "")
    if isinstance(completion, Sequence):
        if not completion:
            return ""
        last = completion[-1]
        if isinstance(last, Mapping):
            return str(last.get("content", "") or "")
        return str(last)
    return str(completion)


def _resolve_gt_list(
    prompts: Sequence[Any] | None,
    completions: Sequence[Any],
    features_lookup: Any,
    kwargs: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Per-completion GT feature rows (unchanged resolution contract).

    1. ALIGNED LIST via kwargs: TRL forwards non-standard dataset columns to the
       reward function as kwarg lists aligned with `completions`; a column named
       `gt_features` (or `gt` / `ground_truth` / `features`) is used directly.
    2. `features_lookup` as an aligned Sequence.
    3. `features_lookup` as a Mapping keyed by an id column
       ("clip_ids"/"keys"/"ids"/"id"/"clip_id"/"clip_stem"/"filename") or,
       failing that, by prompt text.
    4. `features_lookup` as a callable: features_lookup(key) -> row dict.

    Missing entries become {} (VALUE claims become unverifiable -> 0 reward).
    """
    n = len(completions)

    for key in ("gt_features", "gt", "ground_truth", "features"):
        if key in kwargs and isinstance(kwargs[key], Sequence) and not isinstance(kwargs[key], (str, bytes)):
            return _pad_to(list(kwargs[key]), n)

    if isinstance(features_lookup, Sequence) and not isinstance(features_lookup, (str, bytes)):
        return _pad_to(list(features_lookup), n)

    if isinstance(features_lookup, Mapping):
        keys = _per_sample_keys(prompts, kwargs, n)
        return [dict(features_lookup.get(k, {})) if k is not None else {} for k in keys]

    if callable(features_lookup):
        keys = _per_sample_keys(prompts, kwargs, n)
        out = []
        for k in keys:
            try:
                out.append(dict(features_lookup(k) or {}))
            except Exception:
                out.append({})
        return out

    return [{} for _ in range(n)]


def _per_sample_keys(
    prompts: Sequence[Any] | None,
    kwargs: Mapping[str, Any],
    n: int,
) -> list[Any]:
    """Per-sample key for a Mapping/callable lookup: prefer an explicit id
    column forwarded through kwargs, else the prompt text."""
    for key in ("clip_ids", "keys", "ids", "id", "clip_id", "clip_stem", "filename"):
        if key in kwargs and isinstance(kwargs[key], Sequence) and not isinstance(kwargs[key], (str, bytes)):
            return _pad_to(list(kwargs[key]), n, fill=None)
    if prompts is not None:
        return _pad_to(list(prompts), n, fill=None)
    return [None] * n


def _pad_to(seq: list, n: int, fill: Any = None) -> list:
    """Truncate or pad `seq` to length n."""
    if len(seq) == n:
        return seq
    if len(seq) > n:
        return seq[:n]
    return seq + [fill] * (n - len(seq))


def make_reward_func(
    features_lookup: Any = None,
    *,
    gt_lookup: Any = None,
    hedge_attributed_unobs: float = HEDGE_ATTRIBUTED_UNOBSERVABLE,
    hedge_generic_unobs: float = HEDGE_GENERIC_UNOBSERVABLE,
    hedge_obs: float = HEDGE_OBSERVABLE,
    ngram: int = 4,
    rep_n: int | None = None,
    **deprecated_cfg,
) -> Callable[..., list[float]]:
    """Build a TRL-GRPO-compatible batch reward function over the NEW
    observability-gated reward (F19). Same wrapper contract as the old
    `make_sfs_reward_func`:

        reward_func(prompts, completions, **kwargs) -> list[float]

    GT resolution (see `_resolve_gt_list`): pass the per-clip features row as a
    dataset column (`gt_features=[...]`, recommended), OR an aligned list /
    Mapping / callable as `features_lookup`. An optional `observability=[...]`
    kwargs column (one {slot: 0/1} dict or None per completion) overrides the
    row-derived observability rule per sample.

    Back-compat: `gt_lookup=` is accepted as an alias for `features_lookup`
    (grpo_train.py's existing keyword); `rep_n=` as an alias for `ngram`. The
    retired band-F1 knobs (f1_weight / rep_penalty / nonascii_penalty) are
    ACCEPTED but IGNORED with a DeprecationWarning — the new reward multiplies
    the degeneration guards into G(y) instead of subtracting weighted terms.
    """
    if features_lookup is None:
        features_lookup = gt_lookup
    if rep_n is not None:
        ngram = int(rep_n)
    if deprecated_cfg:
        warnings.warn(
            "make_reward_func: ignoring retired band-F1 reward kwargs "
            f"{sorted(deprecated_cfg)} — the band-F1 core is a dead objective "
            "under RL (F19); the observability-gated reward has no such knobs.",
            DeprecationWarning,
            stacklevel=2,
        )

    def reward_func(prompts=None, completions=None, **kwargs) -> list[float]:
        if completions is None:
            completions = []
        rows = _resolve_gt_list(prompts, completions, features_lookup, kwargs)
        obs_col = kwargs.get("observability")
        if not (isinstance(obs_col, Sequence) and not isinstance(obs_col, (str, bytes))):
            obs_col = None
        rewards: list[float] = []
        for i, (completion, row) in enumerate(zip(completions, rows)):
            text = extract_completion_text(completion)
            obs = obs_col[i] if obs_col is not None and i < len(obs_col) else None
            rewards.append(
                observability_reward(
                    text,
                    row or {},
                    obs,
                    hedge_attributed_unobs=hedge_attributed_unobs,
                    hedge_generic_unobs=hedge_generic_unobs,
                    hedge_obs=hedge_obs,
                    ngram=ngram,
                )
            )
        return rewards

    reward_func.__name__ = "observability_reward_func"
    return reward_func


def make_sfs_reward_func(features_lookup: Any = None, **cfg) -> Callable[..., list[float]]:
    """DEPRECATED NAME — returns the NEW observability-gated reward (F19).

    Kept so `grpo_train.py`'s `from training.sfs_reward import
    make_sfs_reward_func` keeps importing and its legacy kwargs keep parsing
    (they are ignored with a DeprecationWarning). This does NOT build the
    retired band-F1 reward; for that (analysis only, never as an RL objective)
    call `deprecated_band_f1_reward` directly.
    """
    return make_reward_func(features_lookup, **cfg)


# ── DEPRECATED band-F1 path (retired as an RL objective) ─────────────────────
def sfs_f1(generated_text: str, gt_features: Mapping[str, Any]) -> float:
    """Tolerance-band SFS-F1 of one description vs an SFSScorer-shaped GT dict.

    DEPRECATED as an RL reward (kept for offline comparison dashboards). The
    band saturates under constant-mode emission, so it must never be optimized
    against — use `observability_reward` instead.
    """
    if not generated_text:
        return 0.0
    claims = _PARSER.parse(generated_text)
    result = _SCORER.score(claims, dict(gt_features))
    return float(result["f1"])


def deprecated_band_f1_reward(
    generated_text: str,
    gt_features: Mapping[str, Any],
    *,
    f1_weight: float = 1.0,
    rep_penalty: float = 0.5,
    nonascii_penalty: float = 1.0,
    ngram: int = 4,
) -> float:
    """RETIRED reward — DO NOT USE AS AN RL OBJECTIVE.

    This is the pre-F19 reward, verbatim:

        reward = f1_weight * SFS_band_F1(text vs gt)
               - rep_penalty * rep_n(text, ngram)
               - nonascii_penalty * nonascii_frac(text)

    It is a DEAD OBJECTIVE under RL pressure: the tolerance bands are saturated
    (a constant-mode policy emitting the population-median value for every
    feature — the observed f0_min=75-on-96%-of-clips collapse — scores
    near-ceiling), so optimizing it teaches the policy to ignore the audio.
    Kept ONLY so old runs can be re-scored for comparison plots. The factory
    (`make_reward_func` / `make_sfs_reward_func`) never builds this.
    """
    text = generated_text or ""
    return (
        f1_weight * sfs_f1(text, gt_features)
        - rep_penalty * rep_n(text, ngram)
        - nonascii_penalty * nonascii_frac(text)
    )
