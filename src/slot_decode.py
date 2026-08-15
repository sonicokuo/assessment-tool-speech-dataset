"""Constrained slot-decode harness (fix F18, master plan 2026-07-16).

Decode-time harness that makes repetition loops structurally impossible:
the per-feature sentence FRAME is teacher-forced (never sampled), the LM
only fills numeric slots, and logits at slot positions are constrained to
digit/sign/point tokens (Willard & Louf, arXiv:2307.09702). In "verified"
mode the aux regression head's grounded value is substituted into the
emitted text while the LM's own value is recorded for the slot report
(SymGen-style verified substitution, arXiv:2311.09188). The first-value-
token digit distribution is captured per slot as a confidence/abstention
signal (Decoding-based Regression, arXiv:2501.19383).

Works on EXISTING checkpoints; no retrain required.

Public API:
    slot_generate(adapter, llm, tokenizer, audio_features, overlap_info,
                  prompt_ids, device, *, mode="free_slots",
                  overlap_ratio=None, abstain_mask=None, max_digits=10)
        -> (text, slot_report)
"""

import math
import os
import sys

import torch

try:  # canonical feature list / scales / hedging constants
    from data.feature_set import (
        SUPERVISED_FEATURES,
        ILL_POSED_UNDER_OVERLAP_FEATURES,
        HEDGE_OVERLAP_TAU,
    )
except ImportError:  # pragma: no cover - allow src/ on path directly
    from src.data.feature_set import (  # type: ignore
        SUPERVISED_FEATURES,
        ILL_POSED_UNDER_OVERLAP_FEATURES,
        HEDGE_OVERLAP_TAU,
    )

# Frame text comes from the builder — the single source of truth for the
# SFS-parseable canonical phrasing ("The <NAME> is <value> <unit>." forms;
# one GROUPED hedge sentence for all abstained pitch/voice features).
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from build_canonical_descriptions import (  # noqa: E402
    _canonical_clause,
    _hedge_sentence,
    _fmt,
)

# Characters the LM may emit inside a numeric slot. Qwen tokenizes numbers
# per-digit, so single-character token ids cover the whole value alphabet.
SLOT_CHARS = tuple("0123456789.-")
DIGIT_CHARS = tuple("0123456789")

# Private-use probe char: guaranteed absent from the clause templates, so
# clause.split(probe) recovers the value-independent (prefix, suffix) frame.
_PROBE = ""


# ---------------------------------------------------------------------------
# Sentence frames (canonical order), sourced from the builder
# ---------------------------------------------------------------------------

def slot_frames():
    """[(short_name, fmt, frame_prefix, frame_suffix)] in canonical slot order.

    The prefix/suffix are the text before/after the numeric value in the
    builder's canonical clause. All canonical clauses are value-INDEPENDENT
    around the slot (verified: build_canonical_descriptions._canonical_clause
    lines 223-256), so a probe-split is exact.
    """
    frames = []
    for short, _col, fmt in SUPERVISED_FEATURES:
        clause = _canonical_clause(short, _PROBE)
        prefix, suffix = clause.split(_PROBE)
        frames.append((short, fmt, prefix, suffix))
    return frames


def format_value(feature, fmt, value):
    """Format a numeric value the way the canonical builder does (int for
    pause_count, two-decimal otherwise per F15)."""
    return _fmt(feature, fmt, float(value))


# ---------------------------------------------------------------------------
# Tokenizer-side helpers
# ---------------------------------------------------------------------------

def build_slot_token_ids(tokenizer):
    """Map single-char slot tokens to ids.

    Returns (allowed, digits): {token_id: char} for the value alphabet, and
    the ordered {token_id: char} for '0'..'9' (first-digit softmax reporting).
    Unknown ids are dropped, so a toy tokenizer without '-' still works.
    """
    unk = getattr(tokenizer, "unk_token_id", None)

    def ids_for(chars):
        out = {}
        for ch in chars:
            tid = tokenizer.convert_tokens_to_ids(ch)
            if tid is not None and tid != unk:
                out[tid] = ch
        return out

    return ids_for(SLOT_CHARS), ids_for(DIGIT_CHARS)


def _feed(llm, embed_layer, token_ids, past, device):
    """Teacher-force a token span through the LM in ONE forward.

    Returns (new_past, last_logits); empty spans return the inputs unchanged.
    Frame tokens go through this path and are therefore never sampled.
    """
    if not token_ids:
        return past, None
    ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    out = llm(inputs_embeds=embed_layer(ids), past_key_values=past, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :]


def _gen_value(llm, embed_layer, past, last_logits, allowed, digits,
               max_digits, device):
    """Greedy digit-constrained value generation from the current cache state.

    Stops when the UNCONSTRAINED argmax leaves the value alphabet (the model
    wants to end the number — the natural terminator) or at max_digits.
    Returns (value_str, new_past, new_logits, first_digit_probs, entropy).
    """
    chars = []
    first_probs = None
    entropy = None
    logits = last_logits
    for step in range(max_digits):
        raw = logits[0].float()
        if step == 0 and digits:
            d_ids = list(digits.keys())
            probs = torch.softmax(raw[d_ids], dim=-1)
            first_probs = {digits[i]: float(p) for i, p in zip(d_ids, probs)}
            entropy = float(-sum(p * math.log(max(p, 1e-12))
                                 for p in first_probs.values()))
        if int(raw.argmax()) not in allowed and chars:
            break
        masked = torch.full_like(raw, float("-inf"))
        a_ids = list(allowed.keys())
        masked[a_ids] = raw[a_ids]
        nxt = int(masked.argmax())
        chars.append(allowed[nxt])
        past, logits = _feed(llm, embed_layer, [nxt], past, device)
    value = "".join(chars).rstrip(".")  # avoid "1.9039.." when the frame suffix adds the period
    return value, past, logits, first_probs, entropy


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def slot_generate(
    adapter,
    llm,
    tokenizer,
    audio_features,
    overlap_info,
    prompt_ids,
    device,
    *,
    mode="free_slots",
    overlap_ratio=None,
    abstain_mask=None,
    sigma_tau=None,
    gate="none",
    max_digits=10,
):
    """Constrained slot decoding over the canonical feature frames.

    mode="free_slots": the LM's own digits fill each slot.
    mode="verified":   the aux head's value is emitted (LM value recorded).

    ABSTENTION GATE (rewired 2026-08-14). `gate` selects WHO decides to withhold:

      "sigma"  — the model's own predicted uncertainty: abstain when
                 sigma_f > sigma_tau[f], with per-feature thresholds calibrated on dev.
                 This is the literature-standard reject rule for regression, which
                 thresholds the conditional variance (Zaoui, Denis & Hebiri, NeurIPS 2020,
                 arXiv:2006.16597) using a heteroscedastic head (Kendall & Gal, NeurIPS 2017,
                 arXiv:1703.04977). Sweeping sigma_tau traces the full risk-coverage curve.
      "oracle" — the ORIGINAL rule: feat in ILL_POSED_UNDER_OVERLAP and
                 overlap_ratio >= HEDGE_OVERLAP_TAU. This reads a GROUND-TRUTH overlap
                 label, so it is an ORACLE BASELINE and must be labelled as such in any
                 table. It is no longer reachable by accident.
      "none"   (DEFAULT) — emit everything (the full-coverage row).

    The default is "none" deliberately: it is the only setting that cannot FABRICATE an
    abstention statistic. Both abstaining gates must be requested BY NAME, so no table can
    ever report abstention without the caller having chosen its mechanism. The previous
    behaviour was the opposite — the oracle rule fired whenever nobody passed a mask, which
    is exactly how a ground-truth if-statement came to underlie the paper's headline claim.

    ⚠️ WHY THIS CHANGED. `abstain_mask` was a parameter with NO CALLER anywhere in the
    repo, so every run silently fell through to the oracle rule, and the sigma head fed
    ANALYSES ONLY. The paper's abstention claim was therefore a hard-coded if-statement on
    a ground-truth label rather than a model decision. The head was trained and evaluated
    all along; it was simply never connected to the decision.

    `abstain_mask` still wins where set, so an externally-computed mask (e.g. a residual
    error-predictor head) can be dropped in without touching this function.

    Returns (text, slot_report): slot_report[feat] = {lm_value, emitted_value,
    aux_value, digit_probs, digit_entropy, abstained, sigma, gate}, plus "_summary".
    """
    if gate not in ("sigma", "oracle", "none"):
        raise ValueError(f"unknown abstention gate: {gate!r}")
    if mode not in ("free_slots", "verified"):
        raise ValueError(f"unknown slot_decode mode: {mode!r}")
    audio_features = audio_features.unsqueeze(0).to(device)
    overlap_info = overlap_info.unsqueeze(0).to(device)
    if hasattr(llm, "dtype") and llm.dtype == torch.bfloat16:  # real LLM path
        audio_features = audio_features.to(torch.bfloat16)
        overlap_info = overlap_info.to(torch.bfloat16)
    out = adapter(audio_features, overlap_info)
    aux_logvar = None
    if isinstance(out, tuple):
        prefix_embeds, aux_pred = out[0], (out[1] if len(out) > 1 else None)
        # Reliability-head adapters nest (mean, log_var) in slot 1. The harness substitutes
        # the MEAN; log_var is now CAPTURED rather than discarded, because it is what the
        # sigma gate below thresholds. Previously it was dropped here and the abstention
        # decision fell through to the oracle overlap rule.
        if isinstance(aux_pred, tuple):
            aux_pred, aux_logvar = aux_pred[0], aux_pred[1]
    else:
        prefix_embeds, aux_pred = out, None
    if aux_logvar is not None and aux_logvar.dim() == 3:      # (B,N,F) pooling variants
        aux_logvar = aux_logvar.mean(dim=1)
    if gate == "sigma" and aux_logvar is None:
        raise ValueError(
            "gate='sigma' needs a reliability_head checkpoint (no log_var was returned). "
            "Pass gate='oracle' for the labelled oracle baseline or gate='none' for full "
            "coverage — but never report either as model-driven abstention.")
    if mode == "verified" and aux_pred is None:
        raise ValueError("mode='verified' requires an adapter with an aux head")

    embed_layer = llm.get_input_embeddings()
    first = torch.cat([prefix_embeds, embed_layer(prompt_ids)], dim=1)
    o = llm(inputs_embeds=first, use_cache=True)
    past, logits = o.past_key_values, o.logits[:, -1, :]

    allowed, digits = build_slot_token_ids(tokenizer)
    frames = slot_frames()
    report = {}
    clauses = []
    abstained_any = False

    from data.feature_set import SUPERVISED_FEATURES as _SF
    _short_names = [f[0] if isinstance(f, (tuple, list)) else str(f) for f in _SF]

    def _sigma_of(feat):
        """Predicted sd for one feature, or None. log_var -> sigma = exp(0.5*log_var)."""
        if aux_logvar is None or feat not in _short_names:
            return None
        j = _short_names.index(feat)
        if j >= aux_logvar.shape[-1]:
            return None
        return float(torch.exp(0.5 * aux_logvar[0, j]).detach().float().cpu())

    for idx, (short, fmt, f_prefix, f_suffix) in enumerate(frames):
        sig = _sigma_of(short)
        # An explicit mask always wins, so a residual error-predictor head can be swapped
        # in without editing this function. Otherwise the SELECTED gate decides -- and the
        # oracle rule is now reachable only when asked for by name.
        if (abstain_mask or {}).get(short, False):
            abstain = True
        elif gate == "sigma":
            tau = (sigma_tau or {}).get(short)
            abstain = bool(sig is not None and tau is not None and sig > tau)
        elif gate == "oracle":
            abstain = bool(
                short in ILL_POSED_UNDER_OVERLAP_FEATURES
                and overlap_ratio is not None
                and overlap_ratio >= HEDGE_OVERLAP_TAU
            )
        else:                                        # gate == "none"
            abstain = False
        if abstain:
            # `sigma` and `gate` are recorded on EVERY slot, abstained or not, so a
            # risk-coverage curve can be swept post-hoc from one decode pass without
            # re-running the LM: coverage is a threshold on the stored sigma.
            report[short] = {"abstained": True, "lm_value": None,
                             "emitted_value": None, "aux_value": None,
                             "digit_probs": None, "digit_entropy": None,
                             "sigma": sig, "gate": gate}
            abstained_any = True
            continue
        lead = ("" if not clauses else " ") + f_prefix
        past, logits = _feed(
            llm, embed_layer,
            tokenizer(lead, add_special_tokens=False)["input_ids"],
            past, device,
        )
        lm_value, past, logits, first_probs, entropy = _gen_value(
            llm, embed_layer, past, logits, allowed, digits, max_digits, device)
        aux_value = None
        if aux_pred is not None and idx < aux_pred.shape[-1]:
            aux_value = format_value(short, fmt, aux_pred[0, idx].float())
        emitted = aux_value if mode == "verified" else lm_value
        # Keep the LM's cache consistent with the EMITTED text: in verified
        # mode the substituted value is fed so later slots are conditioned on
        # what the reader will actually see.
        tail = (emitted if mode == "verified" else "") + f_suffix
        past, logits = _feed(
            llm, embed_layer,
            tokenizer(tail, add_special_tokens=False)["input_ids"],
            past, device,
        )
        clauses.append(f_prefix + emitted + f_suffix)
        report[short] = {"abstained": False, "lm_value": lm_value,
                         "emitted_value": emitted, "aux_value": aux_value,
                         "digit_probs": first_probs, "digit_entropy": entropy,
                         "sigma": sig, "gate": gate}

    text = " ".join(clauses)
    if abstained_any:
        # E5: `_hedge_sentence` embeds the overlap NUMBER, and the SFS parser then scores
        # that number as an `overlap_ratio` PREDICTION -- i.e. the metric partly scores an
        # oracle input against itself. Cite the number ONLY under the oracle gate, where it
        # is genuinely the stated justification; a sigma-gated abstention did not consult
        # overlap at all and must not quote it.
        # ⚠️ WORDING STILL OWED: the sentence says "Because the speakers overlap heavily",
        # which a sigma gate can fire without (e.g. high uncertainty on a clean clip).
        # Rewording touches the target distribution and the AbstentionDetector, so it is a
        # deliberate change, not a silent one -- left for the retrain that adopts this gate.
        text = (text + " " if text else "") + _hedge_sentence(
            overlap_ratio if gate == "oracle" else None)
    for _, _, f_prefix, _ in frames:  # structural guarantee: no frame repeats
        assert text.count(f_prefix) <= 1, f"frame repeated: {f_prefix!r}"
    report["_summary"] = {
        "mode": mode,
        "n_slots": len(frames),
        "n_abstained": sum(1 for k, v in report.items()
                           if k != "_summary" and v["abstained"]),
    }
    return text, report


def rep_n(text, n=4):
    """Fraction of repeated n-grams in `text` (0 = no repetition)."""
    words = text.split()
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)
