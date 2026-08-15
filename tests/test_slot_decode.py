"""F18 slot-decode tests — fake-LM protocol; imports ONLY slot_decode +
eval.sfs + eval.ckpt_selection (NOT inference.py, whose transformers import is
unavailable locally). Run: python3 tests/test_slot_decode.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from slot_decode import (  # noqa: E402
    slot_generate, slot_frames, build_slot_token_ids, format_value,
)
from eval.ckpt_selection import rep_n as real_rep_n  # noqa: E402
from data.feature_set import SUPERVISED_FEATURES  # noqa: E402


# ── Fake LM / tokenizer over a toy per-char vocab ───────────────────────────
class FakeTokenizer:
    """Single-char vocab: '0'-'9' -> 0-9, '.' -> 10, '-' -> 11, others from 12."""
    unk_token_id = 999

    def __init__(self):
        self.vocab = {str(d): d for d in range(10)}
        self.vocab["."] = 10
        self.vocab["-"] = 11
        self._next = 12

    def convert_tokens_to_ids(self, tok):
        return self.vocab.get(tok, self.unk_token_id)

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for ch in text:
            if ch not in self.vocab:
                self.vocab[ch] = self._next
                self._next += 1
            ids.append(self.vocab[ch])
        return {"input_ids": ids}


class FakeLLM:
    """Callable protocol: returns logits favoring a scripted digit sequence,
    then a non-value char (natural terminator). Records every fed id so tests
    can assert frame tokens were teacher-forced (fed), never sampled."""
    VOCAB = 700

    def __init__(self, script="42.5"):
        self.script = script
        self.pos = 0
        self.fed_ids = []
        self.n_forwards = 0
        self.dtype = torch.float32
        self._emb = torch.nn.Embedding(self.VOCAB, 8)

    def get_input_embeddings(self):
        return self._emb

    def __call__(self, inputs_embeds=None, past_key_values=None, use_cache=True):
        self.n_forwards += 1
        n = inputs_embeds.shape[1]
        # Recover fed ids by nearest-embedding lookup (exact for our Embedding).
        with torch.no_grad():
            flat = inputs_embeds[0]
            d = torch.cdist(flat, self._emb.weight)
            self.fed_ids.extend(int(i) for i in d.argmin(dim=-1))
        logits = torch.full((1, n, self.VOCAB), -10.0)
        # Position within the current value = length of the trailing run of
        # value-char ids (0-11) in everything fed so far. Frame feeds end with
        # non-value chars, resetting the run; each sampled digit is fed back by
        # the harness, advancing it. This stays in sync with the harness by
        # construction, regardless of how many teacher-forced feeds occur.
        run = 0
        for i in reversed(self.fed_ids):
            if i <= 11:
                run += 1
            else:
                break
        tok = FakeTokenizer()  # char->id mapping mirror for the script
        if run < len(self.script):
            want = tok.convert_tokens_to_ids(self.script[run])
        else:
            want = 30  # arbitrary non-value id => natural terminator
        logits[0, -1, want] = 10.0

        class Out:
            pass

        o = Out()
        o.logits = logits
        o.past_key_values = (past_key_values or ()) or None
        return o


class FakeAdapter:
    """Returns (prefix_embeds, aux_pred) like AdapterWithAuxHead."""
    def __init__(self, dim=8, n_feat=len(SUPERVISED_FEATURES)):
        self.prefix = torch.zeros(1, 4, dim)
        self.aux = torch.arange(1.0, n_feat + 1.0).unsqueeze(0) * 1.5

    def __call__(self, af, oi):
        return self.prefix, self.aux


def _run(mode="free_slots", **kw):
    llm = FakeLLM()
    tok = FakeTokenizer()
    text, report = slot_generate(
        FakeAdapter(), llm, tok,
        torch.zeros(10, 4), torch.zeros(10, 4),
        torch.tensor([[20, 21]]), torch.device("cpu"),
        mode=mode, **kw)
    return text, report, llm, tok


# ── Tests ───────────────────────────────────────────────────────────────────
def test_frames_cover_all_slots():
    frames = slot_frames()
    assert len(frames) == len(SUPERVISED_FEATURES)
    assert all(p and "is" in p for _, _, p, _ in frames)


def test_frame_tokens_forced_never_sampled():
    text, report, llm, tok = _run()
    # every frame-prefix char id must appear in fed_ids (teacher-forced)
    frame_ids = tok("The SNR is ")["input_ids"]
    assert all(i in llm.fed_ids for i in frame_ids)
    assert "The SNR is 42.5 dB." in text


def test_digit_constraint_and_terminator():
    text, report, llm, tok = _run()
    for feat, r in report.items():
        if feat == "_summary" or r["abstained"]:
            continue
        assert r["lm_value"] == "42.5"           # scripted digits only
        assert set(r["lm_value"]) <= set("0123456789.-")
        assert r["digit_entropy"] is not None
        assert abs(sum(r["digit_probs"].values()) - 1.0) < 1e-5


def test_verified_mode_substitutes_aux_and_records_lm():
    text, report, llm, tok = _run(mode="verified")
    r = report["snr"]
    assert r["lm_value"] == "42.5"
    assert r["emitted_value"] == r["aux_value"] != r["lm_value"]
    assert f"The SNR is {r['aux_value']} dB." in text


def test_abstain_emits_grouped_hedge():
    # gate="oracle" is now REQUIRED to exercise the overlap>=tau rule. It used to fire
    # whenever no abstain_mask was passed, which is how a ground-truth if-statement ended
    # up underlying the paper's abstention claim without any caller asking for it.
    text, report, llm, tok = _run(overlap_ratio=0.9, gate="oracle")
    assert report["f0_mean"]["abstained"] and report["jitter"]["abstained"]
    assert "cannot be reliably estimated" in text
    assert "The F0 mean is" not in text        # no number for abstained slot
    assert report["_summary"]["n_abstained"] >= 4


def test_output_has_zero_repetition():
    for kw in ({}, {"overlap_ratio": 0.9}, {"mode": "verified"}):
        text, _, _, _ = _run(**kw)
        assert real_rep_n(text, 4) < 0.05, text


def test_real_claimparser_roundtrip():
    from eval.sfs import HybridClaimParser
    text, report, _, _ = _run()
    claims = {c.feature: c.value for c in HybridClaimParser().parse(text)}
    assert abs(claims["snr"] - 42.5) < 1e-6
    assert abs(claims["srmr"] - 42.5) < 1e-6


def test_format_value_matches_builder():
    assert format_value("snr", "{:.2f}", 3.14159) == "3.14"
    assert format_value("pause_count", "{:.2f}", 4.7) == "5"  # int feature


# ── Sigma-gated abstention (wired 2026-08-14) ───────────────────────────────
class _RelAdapter(FakeAdapter):
    """Adapter whose aux slot nests (mean, log_var), like a reliability_head checkpoint."""

    def __init__(self, log_var):
        super().__init__()
        self.log_var = log_var

    def __call__(self, af, oi):
        return self.prefix, (self.aux, self.log_var)


def _run_rel(log_var, **kw):
    llm, tok = FakeLLM(), FakeTokenizer()
    text, report = slot_generate(
        _RelAdapter(log_var), llm, tok,
        torch.zeros(10, 4), torch.zeros(10, 4),
        torch.tensor([[20, 21]]), torch.device("cpu"),
        mode="free_slots", **kw)
    return text, report


def test_sigma_gate_abstains_above_threshold_and_not_below():
    n = len(SUPERVISED_FEATURES)
    names = [f[0] if isinstance(f, (tuple, list)) else str(f) for f in SUPERVISED_FEATURES]
    # sigma = exp(0.5*log_var); log_var=+4 -> sigma~7.39, log_var=-4 -> sigma~0.135
    lv = torch.full((1, n), -4.0)
    lv[0, names.index("f0_mean")] = 4.0
    _, rep = _run_rel(lv, gate="sigma", sigma_tau={k: 1.0 for k in names})
    assert rep["f0_mean"]["abstained"] is True, "high sigma must abstain"
    assert rep["snr"]["abstained"] is False, "low sigma must NOT abstain"
    # sigma is recorded on BOTH branches so a risk-coverage curve is sweepable post-hoc
    assert rep["f0_mean"]["sigma"] > rep["snr"]["sigma"]
    assert rep["snr"]["gate"] == "sigma"


def test_sigma_gate_ignores_overlap_and_omits_the_number():
    """A sigma gate must not consult overlap, nor quote it in the hedge (defect E5)."""
    n = len(SUPERVISED_FEATURES)
    names = [f[0] if isinstance(f, (tuple, list)) else str(f) for f in SUPERVISED_FEATURES]
    lv = torch.full((1, n), 4.0)                      # everything uncertain
    text, rep = _run_rel(lv, gate="sigma", overlap_ratio=0.9,
                         sigma_tau={k: 1.0 for k in names})
    assert "overlap ratio)" not in text, "sigma gate must not cite the oracle overlap number"
    # and with a huge threshold nothing abstains even though overlap_ratio is high
    _, rep2 = _run_rel(lv, gate="sigma", overlap_ratio=0.9,
                       sigma_tau={k: 1e6 for k in names})
    assert not any(v["abstained"] for k, v in rep2.items() if not k.startswith("_"))


def test_default_gate_never_abstains():
    """The default must be incapable of fabricating an abstention statistic."""
    _, rep, _, _ = _run(overlap_ratio=0.99)           # high overlap, no gate requested
    assert not any(v["abstained"] for k, v in rep.items() if not k.startswith("_"))


def test_sigma_gate_requires_reliability_head():
    import pytest
    with pytest.raises(ValueError, match="reliability_head"):
        _run(gate="sigma", sigma_tau={"snr": 1.0})


# NOTE: the script-runner must stay LAST — it collects globals() at execution time, so any
# test defined below it would be silently skipped in `python3 tests/test_slot_decode.py`
# mode (pytest collects regardless of position, which is how the gap would hide).
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
