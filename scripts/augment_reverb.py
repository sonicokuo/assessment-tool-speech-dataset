#!/usr/bin/env python3
"""
augment_reverb.py — controlled-synthesis REVERB augmentation with RT60 known by construction.

Paper contribution / motivation
--------------------------------
The real Libri2Mix data is near-anechoic: SRMR is high and the reverberation-time
range is narrow, so a quality-description model trained only on it never sees
reverberant rooms and cannot calibrate its reverb claims. This script synthesizes
training data spanning the FULL reverb range by convolving a clean speech clip with
a room impulse response (RIR) generated for a KNOWN target RT60. The trick mirrors
augment_noise.py:

    if we synthesize the RIR for a target RT60 by construction, the reverberation
    time is controlled-by-construction (it is the injection parameter).

How "controlled" the RT60 is depends on the RIR generator:

  * synth (pure-numpy exponential RIR): RT60 is exact-by-construction up to the
    Schroeder-fit approximation; `measure_rt60(synth_exp_rir(rt60)) ≈ rt60` is what
    test_measure_rt60_recovers_synthetic verifies. Dependency-free fallback.
  * pra (pyroomacoustics physical sim): RT60 is the Sabine *target* fed to
    pra.inverse_sabine; the realized RT60 of the simulated room is close but not
    bit-exact (Sabine is an approximation). For exact-GT rigor the caller can
    re-measure with measure_rt60 on the returned RIR; the CLI writes the requested
    target into the CSV.

RT60 ↔ tau derivation (for synth_exp_rir / measure_rt60)
--------------------------------------------------------
The synthetic RIR is white noise n(t) times an AMPLITUDE envelope exp(-t/tau).
Its instantaneous energy is rir(t)^2 = n(t)^2 * exp(-2 t / tau), so the energy
envelope decays as exp(-2 t / tau). In dB the energy decay is

    10*log10(exp(-2 t / tau)) = -(20 / ln 10) * (t / tau).

A 60 dB energy drop is reached at t = RT60:

    (20 / ln 10) * (RT60 / tau) = 60
    => tau = (20 * RT60) / (60 * ln 10)
           = RT60 / (3 * ln 10)
           = RT60 / 6.9078...                  (6.908 ≈ 3 * ln 10 = ln(1000))

So tau = rt60 / 6.908. The Schroeder energy-decay curve of this RIR is then a
straight line in dB, and a linear fit over [-5, -35] dB extrapolated to -60 dB
recovers RT60. measure_rt60 implements exactly that fit.

SRMR / reverb-affected feature handling — DECISION
--------------------------------------------------
Reverberation is a CONVOLUTION, not addition, so unlike noise it genuinely changes
the reverb-sensitive features (SRMR above all, and to a lesser degree SNR, HNR,
shimmer). We must NOT pass those through as if they were unchanged.

CHOICE: emit reverb-affected columns BLANK and require downstream re-extraction.
augment_feature_row clears `srmr` (and the other reverb-sensitive columns listed in
REVERB_AFFECTED_COLUMNS) to "" and stamps the EXACT target into `rt60_s` (a column we
ADD, since the feature CSV has no reverb-time column). The pipeline MUST re-run the
feature extractor on the reverberant wav to refill SRMR. This is cleaner than trying
to model the SRMR shift analytically (which would be an estimate, defeating the
exact-GT premise). The one exactly-known label is rt60_s; SRMR becomes a measured
label after re-extraction.

F0 / speaking-rate / pause columns are PASSED THROUGH. Assumption: linear-convolutive
reverb leaves pitch and timing (pause structure, syllable rate) essentially intact —
the harmonic series and the temporal envelope's gross structure are preserved, even
though tails smear onsets/offsets slightly. This is the standard assumption in reverb
augmentation; it is documented here so a reviewer can challenge it. If a stricter run
is wanted, add those columns to REVERB_AFFECTED_COLUMNS to force their re-extraction.

Clipping is guarded the same way augment_noise does: peak-normalize the reverberant
signal ONLY when |x| > 1, and never amplify a quiet signal.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import fftconvolve

# ── Column names (see src/feature_extractor_mix.py COLUMN_ORDER) ──────────────
FILENAME_COLUMN = "filename"
# The feature CSV has NO reverb-time column, so we ADD this one and stamp the target.
RT60_COLUMN = "rt60_s"
# Columns that reverberation genuinely changes and that must be RE-EXTRACTED from the
# reverberant wav downstream. We blank them rather than pass stale clean values.
REVERB_AFFECTED_COLUMNS = ("srmr",)

# Constant linking RT60 to the exponential amplitude time-constant tau:
#   tau = rt60 / RT60_TAU_CONST,  with RT60_TAU_CONST = 3 * ln(10) = ln(1000) ≈ 6.9078.
RT60_TAU_CONST = 3.0 * np.log(10.0)  # 6.907755...

_EPS = 1e-12


# ──────────────────────────────────────────────────────────────────────────────
# Core RIR / RT60 math (pure numpy — no pyroomacoustics, no IO)
# ──────────────────────────────────────────────────────────────────────────────
def measure_rt60(rir: np.ndarray, sr: int) -> float:
    """
    Measure RT60 (s) from an impulse response via Schroeder backward integration.

    Steps
    -----
    1. Energy decay curve (EDC): integrate the squared IR from the tail backward,
           edc[n] = sum_{k >= n} rir[k]**2 = reverse(cumsum(reverse(rir**2))).
    2. Normalize and convert to dB: edc_db = 10*log10(edc / max(edc)).
    3. Linear-fit edc_db over the [-5, -35] dB region (skip the initial direct
       sound and the noisy floor below -35 dB).
    4. Extrapolate the fitted slope (dB per second) to a full 60 dB drop:
           rt60 = -60 / slope.

    This is PURE NUMPY and is the function the tests verify against synth_exp_rir.
    Returns float('nan') if the IR is too short / degenerate to fit.
    """
    rir = np.asarray(rir, dtype=np.float64).reshape(-1)
    if rir.size < 16 or sr <= 0:
        return float("nan")

    energy = rir ** 2
    # Schroeder backward integration: edc[n] = sum of energy from n to end.
    edc = np.cumsum(energy[::-1])[::-1]
    peak = float(edc[0]) if edc.size else 0.0
    if peak <= _EPS:
        return float("nan")

    edc_db = 10.0 * np.log10(np.maximum(edc, _EPS) / peak)

    # Fit region: between -5 dB and -35 dB (a 30 dB span -> "T30", doubled to T60).
    upper_db, lower_db = -5.0, -35.0
    idx = np.where((edc_db <= upper_db) & (edc_db >= lower_db))[0]
    if idx.size < 2:
        # Decay does not span the full [-5,-35] window (IR too short / too clean).
        # Fall back to the widest monotone-decaying window we have below -5 dB.
        idx = np.where(edc_db <= upper_db)[0]
        if idx.size < 2:
            return float("nan")

    t = idx / float(sr)
    y = edc_db[idx]
    # slope is dB/second (negative). polyfit deg-1 -> [slope, intercept].
    slope, _intercept = np.polyfit(t, y, 1)
    if slope >= 0:
        return float("nan")

    rt60 = -60.0 / slope
    return float(rt60)


def synth_exp_rir(rt60: float, sr: int, length_s: Optional[float] = None) -> np.ndarray:
    """
    Pure-numpy synthetic RIR: white noise * exponential AMPLITUDE decay exp(-t/tau),
    with tau = rt60 / RT60_TAU_CONST so that measure_rt60(synth_exp_rir(rt60)) ≈ rt60.

    See the module docstring for the RT60 ↔ tau derivation. A leading unit spike is
    placed at t=0 so the IR has a clean direct-path onset (the reverberate() trimmer
    relies on a near-zero onset delay for synth RIRs).

    The RIR is deterministic for a given (rt60, sr, length_s): a fixed-seed local RNG
    is used so augmentation is reproducible regardless of any global RNG state.

    Parameters
    ----------
    rt60 : float        target reverberation time in seconds (> 0).
    sr : int            sample rate in Hz.
    length_s : float    IR length in seconds. Default: 1.5*rt60 (enough to span a
                        >60 dB decay, since at t=rt60 the energy is already -60 dB).
    """
    if rt60 <= 0:
        raise ValueError(f"rt60 must be > 0, got {rt60}")
    if length_s is None:
        # 1.5*rt60 spans ~90 dB of energy decay -> the [-5,-35] fit window is well
        # inside the IR. Floor at 0.1 s so very short RT60s still fit a line.
        length_s = max(1.5 * rt60, 0.1)

    n = int(round(length_s * sr))
    n = max(n, 16)
    t = np.arange(n, dtype=np.float64) / float(sr)

    tau = rt60 / RT60_TAU_CONST  # amplitude e-folding time
    env = np.exp(-t / tau)

    # Deterministic white noise (fixed seed) so the RIR is reproducible.
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(n)

    rir = noise * env
    # Clean unit direct-path spike at t=0 (onset delay = 0 samples for synth RIRs).
    rir[0] = 1.0
    return rir


def pra_rir(rt60: float, sr: int, room_dim=(5.0, 4.0, 3.0)) -> np.ndarray:
    """
    Physically-simulated RIR via pyroomacoustics for a target RT60 (seconds).

    Uses pra.inverse_sabine(rt60, room_dim) to get the wall absorption + max_order
    that yield the target RT60 under Sabine's formula, builds a ShoeBox room, places
    one source and one mic, and returns the source→mic RIR.

    pyroomacoustics is imported INSIDE this function so the module imports cleanly
    without it (and the tests never touch it). Raises ImportError if pra is missing.

    Note: the realized RT60 is close to but not exactly the Sabine target (Sabine is
    an approximation). Callers wanting exact GT can re-measure with measure_rt60.
    """
    import pyroomacoustics as pra  # lazy: only the CLI's pra path needs it

    room_dim = list(room_dim)
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(
        room_dim,
        fs=sr,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )
    # Source and mic kept well inside the room and away from walls / each other.
    source_pos = [room_dim[0] * 0.3, room_dim[1] * 0.3, 1.2]
    mic_pos = [room_dim[0] * 0.7, room_dim[1] * 0.6, 1.2]
    room.add_source(source_pos)
    room.add_microphone(mic_pos)
    room.compute_rir()
    rir = np.asarray(room.rir[0][0], dtype=np.float64).reshape(-1)
    return rir


def reverberate(speech: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """
    Convolve `speech` with `rir`, trim back to the ORIGINAL speech length (keeping
    the clip time-aligned), and guard clipping the augment_noise way.

    Alignment
    ---------
    fftconvolve(full) has length len(speech)+len(rir)-1; the first sample of the
    *direct path* lands at the index of the RIR's peak (its onset delay). We trim
    starting at that onset index so the reverberant clip stays aligned with the
    clean clip (rather than slipping forward by the RIR's propagation delay), then
    take exactly len(speech) samples.

    Clipping
    --------
    Peak-normalize only if |out| > 1 (never amplify a quiet signal), exactly as
    augment_noise.mix_at_snr does. This is SNR/structure-preserving (a single global
    scale) and only prevents wav write-time clipping.
    """
    speech = np.asarray(speech, dtype=np.float64).reshape(-1)
    rir = np.asarray(rir, dtype=np.float64).reshape(-1)
    if speech.size == 0:
        raise ValueError("speech signal is empty")
    if rir.size == 0:
        raise ValueError("rir is empty")

    wet = fftconvolve(speech, rir, mode="full")

    # Onset delay = index of the RIR's largest-magnitude tap (the direct path).
    onset = int(np.argmax(np.abs(rir)))
    start = onset
    end = start + speech.size
    if end > wet.size:
        # Pad (can only happen for pathological RIRs); keeps output length exact.
        wet = np.concatenate([wet, np.zeros(end - wet.size, dtype=np.float64)])
    out = wet[start:end]

    # Length guarantee.
    if out.size < speech.size:
        out = np.concatenate([out, np.zeros(speech.size - out.size, dtype=np.float64)])
    out = out[: speech.size]

    # Clipping guard: only normalize DOWN.
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Partial-GT feature row construction (reverb-affected cols blanked for re-extract)
# ──────────────────────────────────────────────────────────────────────────────
def _suffix_filename(filename: str, target_rt60: float) -> str:
    """
    Make the augmented filename unique by inserting `_augR<rt60ms>` before the
    extension, e.g. 'clip.wav' + 0.5 s -> 'clip_augR500.wav'. RT60 is rendered in
    integer milliseconds so distinct grid points (0.3/0.5/0.8/1.2 s) never collide
    and the name is filesystem-safe.
    """
    base, ext = os.path.splitext(filename)
    rt60_ms = int(round(target_rt60 * 1000.0))
    return f"{base}_augR{rt60_ms}{ext}"


def augment_feature_row(clean_row: Dict[str, str], target_rt60: float) -> Dict[str, str]:
    """
    Return a COPY of `clean_row` (a feature-CSV row dict) representing the
    reverb-augmented clip. This is a PARTIAL ground-truth row:

      * `rt60_s`   set to the injected target (string, 3 decimals). Added if absent,
        since the feature CSV has no reverb-time column.
      * `filename` suffixed with `_augR<rt60ms>` so it is unique.
      * every column in REVERB_AFFECTED_COLUMNS (notably `srmr`) BLANKED to "" — it
        must be RE-EXTRACTED from the reverberant wav downstream (convolution changes
        SRMR; we refuse to pass the stale clean value through).
      * all other columns (F0, speaking rate, pauses, ...) passed through UNCHANGED
        under the documented assumption that linear reverb leaves pitch/timing intact.

    Column-agnostic: works on whatever columns the row has (extra trailing columns
    like overlap_segments_vad are preserved).
    """
    row = copy.deepcopy(clean_row)
    row[RT60_COLUMN] = f"{round(float(target_rt60), 3):.3f}"
    for col in REVERB_AFFECTED_COLUMNS:
        if col in row:
            row[col] = ""  # force downstream re-extraction
    if FILENAME_COLUMN in row and row[FILENAME_COLUMN]:
        row[FILENAME_COLUMN] = _suffix_filename(str(row[FILENAME_COLUMN]), target_rt60)
    return row


# ──────────────────────────────────────────────────────────────────────────────
# CLI plumbing
# ──────────────────────────────────────────────────────────────────────────────
def _parse_rt60_grid(s: str) -> List[float]:
    return [float(tok) for tok in s.split(",") if tok.strip() != ""]


def _load_feature_rows(csv_path: str):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return fieldnames, rows


def _make_rir_fn(rir_kind: str):
    """
    Resolve the RIR generator. 'pra' uses pyroomacoustics, falling back to 'synth'
    (with a printed warning) if the import fails. 'synth' always uses the pure-numpy
    RIR. Returns (callable(rt60, sr) -> rir, resolved_kind_str).
    """
    if rir_kind == "synth":
        return (lambda rt60, sr: synth_exp_rir(rt60, sr)), "synth"

    # rir_kind == "pra": probe the import once up front.
    try:
        import pyroomacoustics  # noqa: F401
        return (lambda rt60, sr: pra_rir(rt60, sr)), "pra"
    except Exception as e:  # pragma: no cover - depends on env
        print(f"[WARNING] pyroomacoustics unavailable ({e}); falling back to synth RIR.")
        return (lambda rt60, sr: synth_exp_rir(rt60, sr)), "synth"


def run_cli(args: argparse.Namespace) -> None:
    # soundfile imported lazily (only the CLI needs wav IO), matching augment_noise.
    import soundfile as sf

    rng = np.random.default_rng(args.seed)
    rt60_grid = _parse_rt60_grid(args.rt60_grid)
    if not rt60_grid:
        raise ValueError("empty --rt60_grid")

    fieldnames, rows = _load_feature_rows(args.clean_features_csv)
    # Ensure the reverb-time column exists in the output schema.
    if RT60_COLUMN not in fieldnames:
        fieldnames = fieldnames + [RT60_COLUMN]

    rir_fn, resolved_kind = _make_rir_fn(args.rir)
    print(f"RIR generator: {resolved_kind}")

    os.makedirs(args.out_wav_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)

    if args.limit is not None:
        rows = rows[: args.limit]

    out_rows: List[Dict[str, str]] = []
    n_written = 0
    n_skipped = 0
    rir_cache: Dict[tuple, np.ndarray] = {}

    for clip_idx, clean_row in enumerate(rows):
        clean_name = clean_row.get(FILENAME_COLUMN, "")
        clean_wav_path = os.path.join(args.clean_wav_dir, clean_name)
        if not os.path.isfile(clean_wav_path):
            print(f"  [skip] missing clean wav: {clean_wav_path}")
            n_skipped += 1
            continue

        speech, sr = sf.read(clean_wav_path, dtype="float32")
        if speech.ndim > 1:  # downmix to mono
            speech = speech.mean(axis=1)
        speech = speech.astype(np.float64)

        # Deterministic RT60 selection per clip (without replacement when possible).
        k = min(args.per_clip, len(rt60_grid))
        chosen_idx = rng.choice(len(rt60_grid), size=k, replace=False)

        for rt60_idx in chosen_idx:
            target_rt60 = rt60_grid[int(rt60_idx)]

            # Cache RIRs by (kind, rt60, sr). For 'pra' this is one fixed room per
            # RT60; for 'synth' the RIR is deterministic anyway. Reusing across clips
            # is fine — the RT60 (the GT) is the same regardless of which clip it
            # convolves, and it makes generation much faster.
            cache_key = (resolved_kind, round(target_rt60, 6), sr)
            if cache_key not in rir_cache:
                rir_cache[cache_key] = rir_fn(target_rt60, sr)
            rir = rir_cache[cache_key]

            wet = reverberate(speech, rir)

            new_row = augment_feature_row(clean_row, target_rt60)
            out_name = new_row[FILENAME_COLUMN]
            out_path = os.path.join(args.out_wav_dir, out_name)
            if "filepath" in new_row:
                new_row["filepath"] = out_path
            sf.write(out_path, wet.astype(np.float32), sr)
            out_rows.append(new_row)
            n_written += 1

        if (clip_idx + 1) % 100 == 0:
            print(f"  [{clip_idx + 1}/{len(rows)}] written={n_written} skipped={n_skipped}")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(
        f"Done. Wrote {n_written} reverberant clips ({n_skipped} clips skipped) -> {args.out_csv}\n"
        f"NOTE: SRMR (and other reverb-affected columns) are BLANK — re-run the feature "
        f"extractor on {args.out_wav_dir} to refill them."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Controlled-synthesis REVERB augmentation with RT60 known by construction."
    )
    p.add_argument("--clean_wav_dir", required=True, help="dir of clean source .wav files")
    p.add_argument("--clean_features_csv", required=True, help="feature CSV for the clean clips")
    p.add_argument("--out_wav_dir", required=True, help="output dir for reverberant clips")
    p.add_argument("--out_csv", required=True, help="output (partial-GT) features CSV")
    p.add_argument(
        "--rt60_grid",
        default="0.3,0.5,0.8,1.2",
        help="comma-separated target RT60s in seconds (default: 0.3,0.5,0.8,1.2)",
    )
    p.add_argument("--per_clip", type=int, default=1, help="random RT60 points per clip (default 1)")
    p.add_argument("--limit", type=int, default=None, help="process only the first N clean clips")
    p.add_argument("--seed", type=int, default=1234, help="numpy RNG seed (reproducibility)")
    p.add_argument(
        "--rir",
        choices=["pra", "synth"],
        default="pra",
        help="RIR generator: 'pra' (pyroomacoustics physical sim, falls back to synth "
        "if unavailable) or 'synth' (pure-numpy exponential RIR). Default: pra.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run_cli(args)


if __name__ == "__main__":
    main()
