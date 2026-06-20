"""Round-trip tests for slim checkpoints (FIX 1) + stratified val subset (FIX 2).

FIX 1 — slim checkpoints:
  (a) slim payload = LoRA + unfrozen rows + heads, NOT the frozen base
      (key set + total bytes << full).
  (b) save(best)->load round-trip produces IDENTICAL forward outputs.
  (c) save(last)->resume: optimizer/scheduler present in last.pt and load cleanly;
      best.pt has NO optimizer and the load path tolerates that.
  (d) backward-compat: a fat-format checkpoint (full base keys) still loads.

FIX 2 — val subset:
  (e) val_subset_size selects N clips deterministically (same set across calls),
      stratified across overlap bins when overlap_ratio is present.

The model tests build a TINY llama via AutoConfig + LoRA so they run on CPU in
seconds. They are skipped (not failed) if torch/peft/transformers aren't present.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ckpt_io import (  # noqa: E402
    CKPT_FORMAT_SLIM,
    SlimLoadError,
    slim_llm_state_dict,
    is_slim_state_dict,
    load_llm_state_dict,
    overlap_bin,
    overlap_strata_from_csv_map,
)
from ckpt_selection import seeded_val_indices  # noqa: E402

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from transformers import AutoModelForCausalLM, AutoConfig  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402


# ── tiny model factory ────────────────────────────────────────────────────────
def _tiny_config(tie=False):
    return AutoConfig.for_model(
        "llama", hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        vocab_size=64, tie_word_embeddings=tie,
    )


def _build_peft(seed=0, tie=False):
    """A deterministic tiny LoRA-wrapped llama. Same seed => identical base."""
    torch.manual_seed(seed)
    base = AutoModelForCausalLM.from_config(_tiny_config(tie=tie))
    lc = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.0)
    return get_peft_model(base, lc)


def _train_a_bit(m, steps=4, lr=0.5, unfreeze_embed=False):
    """Push the LoRA adapters (and optionally embed rows) off their init."""
    if unfreeze_embed:
        m.get_input_embeddings().weight.requires_grad_(True)
        out = m.get_output_embeddings()
        if out is not None:
            out.weight.requires_grad_(True)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    x = torch.randint(0, 64, (2, 6))
    for _ in range(steps):
        loss = m(input_ids=x, labels=x).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
    return opt, x


def _forward(m, x):
    m.eval()
    with torch.no_grad():
        return m(input_ids=x).logits.clone()


# ── (a) slim payload contents + size ──────────────────────────────────────────
def test_slim_payload_drops_frozen_base():
    m = _build_peft()
    _train_a_bit(m)
    full_sd = m.state_dict()
    slim = slim_llm_state_dict(m)

    # Every slim key is a lora tensor (no embed unfrozen here).
    assert all("lora_" in k for k in slim), f"non-lora keys leaked: {[k for k in slim if 'lora_' not in k]}"
    # The frozen base q/k/v/o/mlp weights are present in full but NOT in slim.
    base_only = [k for k in full_sd if "lora_" not in k]
    assert base_only, "expected frozen base keys in the full dict"
    assert not (set(base_only) & set(slim)), "frozen base leaked into slim dict"

    full_bytes = sum(v.numel() * v.element_size() for v in full_sd.values())
    slim_bytes = sum(v.numel() * v.element_size() for v in slim.values())
    assert slim_bytes < full_bytes * 0.2, (
        f"slim ({slim_bytes}B) not << full ({full_bytes}B)"
    )
    assert is_slim_state_dict(slim) is True
    assert is_slim_state_dict(full_sd) is False


def test_slim_payload_keeps_unfrozen_embed_rows():
    """Tagged-mode unfreezes embed/lm_head manually (not via modules_to_save).
    Those rows MUST survive into the slim dict or new tokens can't be emitted."""
    m = _build_peft(tie=False)
    _train_a_bit(m, unfreeze_embed=True)
    slim = slim_llm_state_dict(m)
    embed_keys = [k for k in slim if "embed_tokens" in k or "lm_head" in k]
    assert embed_keys, "unfrozen embed/lm_head rows missing from slim dict"
    # Still slim-classified (embed/lm_head are allowed in slim).
    assert is_slim_state_dict(slim) is True


# ── (b) best.pt save->load round-trip: identical forward ──────────────────────
def test_best_roundtrip_identical_forward(tmp_path):
    m = _build_peft()
    _opt, x = _train_a_bit(m)
    ref = _forward(m, x)

    # Save a slim "best.pt" (no optimizer).
    payload = {
        "epoch": 3,
        "ckpt_format": CKPT_FORMAT_SLIM,
        "llm_state_dict": slim_llm_state_dict(m),
    }
    p = tmp_path / "best.pt"
    torch.save(payload, p)

    # Fresh model (same seed => same frozen base), load slim, compare forward.
    m2 = _build_peft()
    ckpt = torch.load(p, weights_only=False)
    missing, unexpected = load_llm_state_dict(
        m2, ckpt["llm_state_dict"], ckpt_format=ckpt.get("ckpt_format"))
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert missing, "expected frozen-base keys to show up as missing"
    got = _forward(m2, x)
    assert torch.equal(ref, got), "slim round-trip changed the forward output"


def test_best_roundtrip_with_unfrozen_embed(tmp_path):
    m = _build_peft(tie=False)
    _train_a_bit(m, unfreeze_embed=True)
    x = torch.randint(0, 64, (2, 6))
    ref = _forward(m, x)
    p = tmp_path / "best.pt"
    torch.save({"ckpt_format": CKPT_FORMAT_SLIM,
                "llm_state_dict": slim_llm_state_dict(m)}, p)
    m2 = _build_peft(tie=False)
    m2.get_input_embeddings().weight.requires_grad_(True)
    ckpt = torch.load(p, weights_only=False)
    load_llm_state_dict(m2, ckpt["llm_state_dict"], ckpt_format=ckpt["ckpt_format"])
    assert torch.equal(ref, _forward(m2, x))


# ── (c) last.pt has optimizer; best.pt does not; resume tolerates missing ─────
def test_last_has_optimizer_best_does_not(tmp_path):
    m = _build_peft()
    opt, _x = _train_a_bit(m)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)

    def payload(save_optimizer):
        pl = {"epoch": 0, "ckpt_format": CKPT_FORMAT_SLIM,
              "llm_state_dict": slim_llm_state_dict(m)}
        if save_optimizer:
            pl["optimizer_state_dict"] = opt.state_dict()
            pl["scheduler_state_dict"] = sched.state_dict()
        return pl

    last_p = tmp_path / "last.pt"
    best_p = tmp_path / "best.pt"
    torch.save(payload(True), last_p)
    torch.save(payload(False), best_p)

    last = torch.load(last_p, weights_only=False)
    best = torch.load(best_p, weights_only=False)
    assert "optimizer_state_dict" in last and "scheduler_state_dict" in last
    assert "optimizer_state_dict" not in best and "scheduler_state_dict" not in best

    # Resume from last.pt: optimizer/scheduler load cleanly into fresh objects.
    m2 = _build_peft()
    opt2 = torch.optim.AdamW([p for p in m2.parameters() if p.requires_grad], lr=0.5)
    sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=1)
    load_llm_state_dict(m2, last["llm_state_dict"], ckpt_format=last["ckpt_format"])
    opt2.load_state_dict(last["optimizer_state_dict"])      # must not raise
    sched2.load_state_dict(last["scheduler_state_dict"])    # must not raise

    # The "resume tolerates a best.pt with no optimizer" guard, mirrored from
    # train.py: the load path must not KeyError.
    assert "optimizer_state_dict" not in best  # guard branch in train.py handles this


# ── (d) backward compat: fat checkpoint still loads ───────────────────────────
def test_fat_checkpoint_backcompat(tmp_path):
    m = _build_peft()
    _train_a_bit(m)
    x = torch.randint(0, 64, (2, 6))
    ref = _forward(m, x)

    # FAT format: the FULL peft state_dict (frozen base + lora), no ckpt_format tag.
    fat_sd = m.state_dict()
    assert is_slim_state_dict(fat_sd) is False
    p = tmp_path / "fat.pt"
    torch.save({"epoch": 0, "llm_state_dict": fat_sd}, p)  # NOTE: no ckpt_format

    m2 = _build_peft()
    ckpt = torch.load(p, weights_only=False)
    missing, unexpected = load_llm_state_dict(
        m2, ckpt["llm_state_dict"], ckpt_format=ckpt.get("ckpt_format"))
    # Fat load is strict-equivalent: full coverage, nothing missing/unexpected.
    assert not missing and not unexpected, (missing[:3], unexpected[:3])
    assert torch.equal(ref, _forward(m2, x))


def test_legacy_lora_state_dict_alias_loads(tmp_path):
    """Old ckpts used only `lora_state_dict`. The getter must still find it."""
    m = _build_peft()
    _train_a_bit(m)
    x = torch.randint(0, 64, (2, 6))
    ref = _forward(m, x)
    fat_sd = m.state_dict()
    torch.save({"epoch": 0, "lora_state_dict": fat_sd}, tmp_path / "old.pt")
    ckpt = torch.load(tmp_path / "old.pt", weights_only=False)
    llm_sd = ckpt.get("llm_state_dict") or ckpt["lora_state_dict"]
    m2 = _build_peft()
    load_llm_state_dict(m2, llm_sd, ckpt_format=ckpt.get("ckpt_format"))
    assert torch.equal(ref, _forward(m2, x))


# ── (f) DEFECT 2: a slim dict missing a trainable LoRA key must RAISE ──────────
def test_slim_load_missing_lora_key_raises():
    """A truncated/regressed slim dict that drops one lora_B tensor would, under
    strict=False, load that adapter at fresh zero-init with NO error — a silent
    no-op (the base-only `_missing` list hides it). The load path must positively
    assert the model's expected trainable set is fully covered and RAISE if not."""
    m = _build_peft()
    _train_a_bit(m)
    slim = slim_llm_state_dict(m)

    # Sanity: a complete slim dict loads cleanly into a fresh same-arch model.
    m_ok = _build_peft()
    load_llm_state_dict(m_ok, dict(slim), ckpt_format=CKPT_FORMAT_SLIM)  # no raise

    # Drop exactly one lora_B key → the model still EXPECTS it (requires_grad),
    # so the slim dict no longer covers the trainable set.
    dropped = next(k for k in slim if "lora_B" in k)
    broken = {k: v for k, v in slim.items() if k != dropped}

    m2 = _build_peft()
    with pytest.raises(SlimLoadError):
        load_llm_state_dict(m2, broken, ckpt_format=CKPT_FORMAT_SLIM)

    # Also raises on auto-detection (no explicit ckpt_format tag) — a slim dict
    # missing a base key is still classified slim, so the guard must still fire.
    m3 = _build_peft()
    with pytest.raises(SlimLoadError):
        load_llm_state_dict(m3, dict(broken), ckpt_format=None)


# ── (g) DEFECT 3: a trainable non-LoRA, non-embed param stays slim-classified ──
def test_is_slim_classifies_trainable_layernorm():
    """`slim_llm_state_dict` keeps any requires_grad param — including a trainable
    LayerNorm. The detector must agree: such a dict LACKS the frozen base and is
    slim. The old "every key must be lora/embed" rule wrongly called it fat and
    would have routed it to the strict fat path (crashing on the absent base)."""
    m = _build_peft()
    # Unfreeze a transformer LayerNorm (input_layernorm of layer 0) — a trainable
    # non-LoRA, non-embed param.
    ln = m.base_model.model.model.layers[0].input_layernorm
    ln.weight.requires_grad_(True)
    _train_a_bit(m)  # nudge lora + the layernorm off init

    slim = slim_llm_state_dict(m)
    ln_keys = [k for k in slim if "input_layernorm" in k and "layers.0." in k]
    assert ln_keys, "trainable layernorm row missing from slim dict"
    # No frozen-base projection weight present ⇒ classified slim.
    assert is_slim_state_dict(slim) is True
    # And the full dict (with the frozen base) is still classified fat.
    assert is_slim_state_dict(m.state_dict()) is False

    # Round-trips: load the layernorm-bearing slim dict into a fresh model whose
    # layernorm is likewise unfrozen, identical forward.
    x = torch.randint(0, 64, (2, 6))
    ref = _forward(m, x)
    m2 = _build_peft()
    m2.base_model.model.model.layers[0].input_layernorm.weight.requires_grad_(True)
    load_llm_state_dict(m2, slim, ckpt_format=CKPT_FORMAT_SLIM)
    assert torch.equal(ref, _forward(m2, x))


# ── (e) val subset: deterministic + stratified ────────────────────────────────
def test_val_subset_deterministic():
    a = seeded_val_indices(1000, 256, seed=1234)
    b = seeded_val_indices(1000, 256, seed=1234)
    assert a == b, "same seed must give identical subset"
    assert len(a) == 256
    assert len(set(a)) == 256, "no duplicate indices"


def test_val_subset_stratified_proportional():
    # 1000 clips: 100 low, 300 med, 600 high overlap.
    strata = (["low"] * 100) + (["med"] * 300) + (["high"] * 600)
    idx = seeded_val_indices(1000, 256, seed=7, strata=strata)
    assert len(idx) == 256
    counts = {"low": 0, "med": 0, "high": 0}
    for i in idx:
        counts[strata[i]] += 1
    # Every regime represented, roughly proportional (256*0.1≈26, 0.3≈77, 0.6≈154).
    assert counts["low"] > 0 and counts["med"] > 0 and counts["high"] > 0
    assert abs(counts["low"] - 26) <= 8
    assert abs(counts["med"] - 77) <= 12
    assert abs(counts["high"] - 154) <= 12
    # Deterministic across calls.
    assert idx == seeded_val_indices(1000, 256, seed=7, strata=strata)


def test_overlap_bin_edges():
    assert overlap_bin(0.0) == "low"
    assert overlap_bin(0.15) == "low"
    assert overlap_bin(0.16) == "med"
    assert overlap_bin(0.45) == "med"
    assert overlap_bin(0.46) == "high"
    assert overlap_bin(1.0) == "high"


def test_overlap_strata_from_csv_map():
    files = ["a.pt", "b.pt", "c.pt", "d.pt"]
    csv_map = {
        "a.pt": {"overlap_ratio": "0.05"},   # low
        "b.pt": {"overlap_ratio": "0.30"},   # med
        "c.pt": {"overlap_ratio": "0.80"},   # high
        # d.pt missing → "unknown"
    }
    strata = overlap_strata_from_csv_map(files, csv_map)
    assert strata == ["low", "med", "high", "unknown"]

    # stem-key fallback (CSV keyed by stem, not "<stem>.pt")
    csv_stem = {"a": {"overlap_ratio": "0.9"}}
    assert overlap_strata_from_csv_map(["a.pt"], csv_stem) == ["high"]

    # no overlap column / empty map → None (caller falls back to uniform)
    assert overlap_strata_from_csv_map(files, {}) is None
    assert overlap_strata_from_csv_map(
        ["a.pt"], {"a.pt": {"snr_db": "10"}}) is None


def test_overlap_strata_pt_matches_wav_keyed_csv():
    """DEFECT 1 repro: the real val CSV (features_pyannote/dev_cleanf0.csv) keys
    the map by the SOURCE AUDIO name ("<stem>.wav" / "<stem>.flac"), but the
    dataset's files are "<stem>.pt". With extension-blind matching every clip
    fell into "unknown" → saw_ratio False → None → stratification silently never
    engaged (seeded-uniform fallback). After the fix, a .pt file must resolve a
    .wav-keyed (and .flac-keyed) row."""
    # .wav-keyed CSV (Libri2Mix) — the exact case from the bug report.
    csv_wav = {"1089-134686-0000.wav": {"overlap_ratio": "0.05"}}  # low
    assert overlap_strata_from_csv_map(
        ["1089-134686-0000.pt"], csv_wav) == ["low"]

    # .flac-keyed CSV (clean LibriSpeech ingest) resolves too.
    csv_flac = {"x.flac": {"overlap_ratio": "0.80"}}  # high
    assert overlap_strata_from_csv_map(["x.pt"], csv_flac) == ["high"]

    # Mixed extensions across the split all normalize to the same stem key.
    files = ["a.pt", "b.pt", "c.pt"]
    csv_mixed = {
        "a.wav": {"overlap_ratio": "0.05"},    # low
        "b.flac": {"overlap_ratio": "0.30"},   # med
        "c.wav": {"overlap_ratio": "0.90"},    # high
    }
    assert overlap_strata_from_csv_map(files, csv_mixed) == ["low", "med", "high"]


def test_val_subset_size_caps_at_dataset():
    # Asking for more than available returns the whole set, deterministically.
    idx = seeded_val_indices(40, 256, seed=1234)
    assert sorted(idx) == list(range(40))
