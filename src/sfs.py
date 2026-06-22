"""Signal Faithfulness Score (SFS) — evaluation metric for speech quality descriptions."""

import re
from dataclasses import dataclass

from feature_tags import (
    FEATURE_TAGS,
    extract_overlap_segments,
    extract_value,
    iter_tagged_spans,
)


# ── Components ──────────────────────────────────────────────
# Signal Faithfulness Score (SFS)
@dataclass
class Claim:
    """A single numerical claim extracted from generated text."""

    feature: str  # e.g., "f0_mean", "snr", "overlap_start"
    value: float  # e.g., 187.0, 28.0, 2.3
    unit: str  # e.g., "Hz", "dB", "s"
    raw_text: str  # the original matched text for debugging


class ClaimParser:
    """Extracts numerical claims from generated speech quality descriptions.

    Handles common patterns:
        "F0 = 187 Hz"           → ("f0_mean", 187.0, "Hz")
        "SNR ≈ 28 dB"          → ("snr", 28.0, "dB")
        "SNR of approximately 28 dB" → ("snr", 28.0, "dB")
        "RT60 < 0.15s"         → ("rt60", 0.15, "s")
        "speaking rate: 7 syl/s" → ("speaking_rate", 7.0, "syl/s")
        "overlap at 2.3-4.1s"  → ("overlap_start", 2.3, "s") + ("overlap_end", 4.1, "s")
        "F1 is 542 Hz"         → ("f1_mean", 542.0, "Hz")
    """

    # Each pattern: (regex, list of (feature_name, group_index_for_value, unit))
    # Group indices are 1-based.
    PATTERNS = [
        # F0 with std dev (must be before base F0 to capture σ): "F0 = 187 Hz (σ = 34 Hz)"
        (r"F0\s*=\s*(\d+\.?\d*)\s*Hz\s*\(?σ\s*=\s*(\d+\.?\d*)\s*Hz", [("f0_mean", 1, "Hz"), ("f0_std", 2, "Hz")]),
        # F0 / pitch (also matches "F0 mean of 96.96 Hz", "mean pitch of 150 Hz",
        # "fundamental frequency mean is 186.69 Hz"). The connector group also
        # accepts the observability builder's assert phrasings
        # "F0 mean can be estimated at X Hz" and "pitch can be measured: the F0
        # mean is …" — the verb forms "(can be )?estimated/measured at" sit where
        # is/of would, so without them the builder's variant-2 assert sentence
        # ("With little overlap, the F0 mean can be estimated at 202.89 Hz.")
        # silently parsed to zero claims (confirmed MISS, builder ~line 285).
        (
            r"(?:F0\s*(?:mean\s*)?|(?:mean\s+)?pitch(?:\s+mean)?|fundamental\s+frequency(?:\s+mean)?)\s*(?:=|≈|~|is|of|(?:can\s+be\s+)?(?:estimated|measured)\s+at)\s*(?:approximately\s+)?(\d+\.?\d*)\s*Hz",
            [("f0_mean", 1, "Hz")],
        ),
        # Combined phrasing the model emits when it states F0 alongside a second
        # feature in one sentence: "The F0 and voice probability are 172.26 Hz and
        # 0.9821 respectively." / "The F0 and speaking rate are 130.16 Hz and 5.221
        # syl/sec." The base F0 pattern above misses these because the "(=|is|of)"
        # connector does not sit directly after "F0" (the words "and X are" intervene).
        # We anchor on the *Hz* unit to disambiguate which of the two listed numbers is
        # F0: only the Hz-denominated value binds to f0_mean, so the unitless second
        # value (voice probability 0.9821) and the syl/sec value are never captured here.
        # The gap "(?:[^.]|\.\d)*?" allows decimal points inside numbers but blocks
        # sentence-ending periods so the match cannot cross a sentence boundary.
        # Requires "F0 and" (not bare "F0"/"F0 mean"/"F0 deviation"), so it never
        # competes with the base f0_mean or the f0_sd "deviation" patterns.
        (
            r"F0\s+and\b(?:[^.]|\.\d)*?\b(?:are|is)\s+(\d+\.?\d*)\s*Hz",
            [("f0_mean", 1, "Hz")],
        ),
        # Formants: F1, F2, F3, F4
        (
            r"(F[1-4])\s*(?:=|≈|is|of)\s*(?:approximately\s+)?(\d+\.?\d*)\s*Hz",
            [("formant", 2, "Hz")],
        ),  # special handling: feature name includes F1/F2/etc
        # SNR (also matches "Signal-to-Noise Ratio (SNR) is 18.54 dB")
        (
            r"(?:Signal-to-Noise\s+Ratio\s*(?:\(SNR\))?\s*|SNR\s*)(?:=|≈|~|is|of)\s*(?:approximately\s+|estimated at\s+)?(\d+\.?\d*)\s*dB",
            [("snr", 1, "dB")],
        ),
        # SNR — leading-number form used by the observability builder:
        # "At 26.15 dB, the signal-to-noise ratio SNR is high." The number
        # PRECEDES the "SNR" mention and the trailing word is a qualitative band
        # ("high"), not a value, so the base pattern above misses it. We anchor
        # on "<num> dB, ... (signal-to-noise ratio|SNR)" with the dB-denominated
        # value bound to snr. The gap "(?:[^.]|\.\d)*?" allows decimals but
        # blocks a sentence boundary so the match can't cross sentences.
        (
            r"(\d+\.?\d*)\s*dB[,]?(?:[^.]|\.\d)*?(?:signal-to-noise\s+ratio|\bSNR\b)",
            [("snr", 1, "dB")],
        ),
        # RT60
        (r"RT60\s*(?:=|≈|~|<|>|is)\s*(?:approximately\s+)?(\d+\.?\d*)\s*s", [("rt60", 1, "s")]),
        # HNR (also matches "Harmonics-to-Noise Ratio (HNR) of 12.59 dB")
        (
            r"(?:Harmonics?-to-Noise\s+Ratio\s*(?:\(HNR\))?\s*|HNR\s*)(?:=|≈|~|is|of)\s*(?:approximately\s+)?(\d+\.?\d*)\s*dB",
            [("hnr", 1, "dB")],
        ),
        # Speaking rate — tightened to ONLY match "speaking rate", not bare "rate",
        # to avoid false matches on "articulation rate" / "pause rate".
        (
            r"speaking\s+rate\s*(?:=|≈|~|is|of|:)\s*(?:approximately\s+)?(\d+\.?\d*)\s*(?:syl(?:lables?)?\s*(?:/\s*|per\s+)s(?:ec(?:ond)?)?)",
            [("speaking_rate", 1, "syl/s")],
        ),
        # Articulation rate — distinct feature from speaking rate
        # (articulation excludes pauses; speaking rate includes them).
        (
            r"articulation\s+rate\s*(?:=|≈|~|is|of|:)\s*(?:approximately\s+)?(\d+\.?\d*)\s*(?:syl(?:lables?)?\s*(?:/\s*|per\s+)s(?:ec(?:ond)?)?)",
            [("articulation_rate", 1, "syl/s")],
        ),
        # Gemma also emits: "X syl/sec for the articulation rate" / "X syl/sec for the speaking rate"
        # inside combined sentences like "Speaking and articulation rates are measured at ...".
        (
            r"(\d+\.?\d*)\s*syl(?:lables?)?\s*(?:/\s*|per\s+)s(?:ec(?:ond)?)?\s+for\s+the\s+articulation\s+rate",
            [("articulation_rate", 1, "syl/s")],
        ),
        (
            r"(\d+\.?\d*)\s*syl(?:lables?)?\s*(?:/\s*|per\s+)s(?:ec(?:ond)?)?\s+for\s+the\s+speaking\s+rate",
            [("speaking_rate", 1, "syl/s")],
        ),
        # Combined phrasing: "The F0 and speaking rate are 130.16 Hz and 5.221 syl/sec."
        # The tightened speaking-rate pattern above misses this because the syl/sec number
        # is not adjacent to "speaking rate" (the F0 value and "Hz and" intervene). We grab
        # the *syl/sec-denominated* number, skipping over the intervening Hz value. The gap
        # "(?:(?!articulation)(?:[^.]|\.\d))*?" allows decimals, blocks sentence periods, and
        # refuses to cross the word "articulation" so a trailing articulation-rate number
        # (a different feature) is never mis-bound to speaking_rate.
        (
            r"speaking\s+rate\b(?:(?!articulation)(?:[^.]|\.\d))*?(\d+\.?\d*)\s*syl(?:lables?)?\s*(?:/\s*|per\s+)s(?:ec(?:ond)?)?",
            [("speaking_rate", 1, "syl/s")],
        ),
        # Duration — "The duration of the speech sample is X s" / "duration is X s"
        (
            r"duration(?:\s+of\s+the\s+(?:speech\s+)?sample)?\s*(?:=|≈|~|is|of)\s*(?:approximately\s+)?(\d+\.?\d*)\s*s(?!yl)",
            [("duration_sec", 1, "s")],
        ),
        # Duration — alternate phrasing "(The) recording is X s long" used by the
        # deterministic builder. The old gemma4 verbalizer used "duration is X s"
        # which the pattern above catches; the new builder phrases it differently
        # and we'd otherwise score zero for duration on every clip.
        (
            r"(?:The\s+)?recording\s+is\s+(\d+\.?\d*)\s*s(?:ec(?:onds?)?)?\s+long",
            [("duration_sec", 1, "s")],
        ),
        # Overlap ratio — "The overlap ratio is 0.7528" (unitless 0-1)
        # Also handles "overlap ratio of the sample is 0.7528"
        (
            r"overlap\s+ratio(?:\s+of\s+the\s+sample)?\s*(?:=|≈|~|is|of)\s*(?:approximately\s+)?(\d+\.?\d*)",
            [("overlap_ratio", 1, "")],
        ),
        # Paraphrase: "high degree of overlap with a ratio of 0.8261",
        # "overlap, with a ratio of 0.73".
        (
            r"overlap[\s,]+with\s+(?:a|an)\s+ratio\s+of\s+(?:approximately\s+)?(\d+\.?\d*)",
            [("overlap_ratio", 1, "")],
        ),
        # F0 standard deviation — "F0 standard deviation SD is X Hz",
        # "F0 SD is X Hz", "F0 standard deviation (SD) is X Hz", and the model's
        # shorthand "F0 deviation is X Hz" ("standard" omitted). "standard" is made
        # optional so bare "deviation" maps to f0_sd; this does NOT leak into f0_mean
        # because the base/combined f0_mean patterns require an "is/of/=" connector or
        # "and" directly after "F0", never the word "deviation".
        (
            r"F0\s+(?:(?:standard\s+)?deviation(?:\s*\(?\s*SD\s*\)?)?|SD)\s*(?:=|≈|~|is|of)\s*(?:approximately\s+)?(\d+\.?\d*)\s*Hz",
            [("f0_sd", 1, "Hz")],
        ),
        # Fallback for split phrasings like "F0 mean is X Hz with a standard deviation SD of Y Hz"
        # or "a standard deviation of Y Hz" — Hz-denominated SD in this corpus is always F0 SD.
        (
            r"standard\s+deviation(?:\s*\(?\s*SD\s*\)?)?\s*(?:=|≈|~|is|of)\s*(?:approximately\s+)?(\d+\.?\d*)\s*Hz",
            [("f0_sd", 1, "Hz")],
        ),
        # Pause count — "The pause count is X" (integer)
        (
            r"pause\s+count\s*(?:=|≈|~|is|of)\s*(?:approximately\s+)?(\d+)",
            [("pause_count", 1, "")],
        ),
        # Paraphrase: "contains a total of 3 pauses", "sample contains 1 pause",
        # "has 2 pauses", "there are 2 pauses in total". Requires a verb
        # (contains/has/are/is) so we don't grab numbers from unrelated spans.
        (
            r"(?:contains?|has|have|with|there\s+(?:are|is))\s+(?:a\s+total\s+of\s+)?(\d+)\s+pauses?\b",
            [("pause_count", 1, "")],
        ),
        # Pause rate per minute — "the pause rate is X per min(ute)"
        (
            r"pause\s+rate\s*(?:=|≈|~|is|of)\s*(?:approximately\s+)?(\d+\.?\d*)\s*per\s+min(?:ute)?",
            [("pause_rate", 1, "per min")],
        ),
        # Spectral tilt
        (
            r"spectral (?:tilt|slope)\s*(?:=|≈|~|is|of)\s*(?:approximately\s+)?(-?\d+\.?\d*)\s*dB/oct(?:ave)?",
            [("spectral_tilt", 1, "dB/oct")],
        ),
        # Jitter — also matches "Jitter local is 2.4784 %" and "jitter local is 1.9686 percent"
        (r"jitter\s*(?:\(?\s*(?:local|rap)\s*\)?\s*)?(?:=|≈|~|is|of|\()\s*(?:approximately\s+)?(\d+\.?\d*)\s*(?:%|percent)", [("jitter", 1, "%")]),
        # Shimmer — also matches "Shimmer of 13.83 %" and "shimmer is 10.94 percent"
        (r"shimmer\s*(?:\(?\s*local\s*\)?\s*)?(?:=|≈|~|is|of|\()\s*(?:approximately\s+)?(\d+\.?\d*)\s*(?:%|percent)", [("shimmer", 1, "%")]),
        # SRMR (reverberation metric, also matches "reverberation score (SRMR) of 9.65", "reverberation score of 10.22")
        (r"(?:SRMR|reverberation\s+score\s*(?:\(SRMR\))?)\s*(?:=|≈|~|is|of)\s*(?:approximately\s+)?(\d+\.?\d*)", [("srmr", 1, "")]),
        # VOT
        (r"VOT\s*(?:=|≈|~|is|of)\s*(?:approximately\s+)?(-?\d+\.?\d*)\s*ms", [("vot", 1, "ms")]),
        # Overlap temporal span — loose enough to match the verbalizer's
        # "Overlap segments are present at 0.5-3.1s" in addition to "overlap at 0.5-3.1s"
        # and "overlapping speech from 0.5 to 3.1s". Only catches the FIRST range;
        # extra comma-separated ranges are picked up by _parse_overlap_segments() below.
        (
            r"overlap(?:ping)?(?:\s+speech|\s+segments?)?(?:\s+(?:are|is)\s+present)?\s*(?:at|from|during|:|,)?\s*(\d+\.?\d*)\s*(?:s|sec)?\s*(?:-|to)\s*(\d+\.?\d*)\s*s",
            [("overlap_start", 1, "s"), ("overlap_end", 2, "s")],
        ),
        # Sample rate
        (r"(?:sampled at|sample rate)\s*(?:=|≈|~|is|of)?\s*(\d+)\s*(?:kHz|Hz)", [("sample_rate", 1, "Hz")]),
    ]

    def parse(self, text: str) -> list[Claim]:
        """Extract all numerical claims from generated text.

        Args:
            text: generated NL description
        Returns:
            list of Claim objects
        """
        claims = []
        text_lower = text  # keep original case for some patterns

        for pattern, extractions in self.PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                for feature, group_idx, unit in extractions:
                    try:
                        value = float(match.group(group_idx))

                        # Special handling for formants: include F1/F2/etc in feature name
                        if feature == "formant":
                            # Find which formant (F1-F4) from the match
                            formant_match = re.search(r"(F[1-4])", match.group(0))
                            if formant_match:
                                feature = f"{formant_match.group(1).lower()}_mean"

                        # Handle kHz → Hz conversion for sample rate
                        if feature == "sample_rate" and "kHz" in match.group(0):
                            value *= 1000
                            unit = "Hz"

                        claims.append(
                            Claim(
                                feature=feature,
                                value=value,
                                unit=unit,
                                raw_text=match.group(0).strip(),
                            )
                        )
                    except (ValueError, IndexError):
                        continue

        # Pick up additional overlap segments beyond the first: the PATTERNS regex only
        # captures one range per match, so multi-segment phrasings like
        #   "Overlap segments are present at 0.5-3.1s, 3.2-4.5s, and 7.4-8.8s."
        # lose the 2nd and 3rd ranges. Scan the overlap-tagged sentence and extract
        # every "X-Ys" range inside it.
        claims.extend(self._parse_extra_overlap_segments(text, already_found=claims))

        # Deduplicate: keep first occurrence of each feature, EXCEPT overlap_start/end
        # which are allowed to repeat (one pair per segment).
        seen = set()
        unique_claims = []
        for c in claims:
            if c.feature in ("overlap_start", "overlap_end"):
                unique_claims.append(c)
            elif c.feature not in seen:
                seen.add(c.feature)
                unique_claims.append(c)

        return unique_claims

    # Capture from "overlap" until a proper sentence-ending period (period followed by
    # whitespace or end of string) — avoids stopping at decimals inside numbers like "0.5".
    _OVERLAP_SENT_RE = re.compile(r"overlap\b.*?(?:\.\s|\.$|$)", re.IGNORECASE | re.DOTALL)
    _RANGE_RE = re.compile(r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*s")

    def _parse_extra_overlap_segments(self, text: str, already_found: list) -> list:
        """Find every 'X-Ys' range inside any overlap-tagged sentence, minus the first one
        (already captured by the main PATTERNS regex).

        Returns a list of extra Claim objects (overlap_start + overlap_end per segment).
        """
        # Track the first (start, end) the main regex found so we don't double-count.
        existing_pairs = set()
        starts = [c.value for c in already_found if c.feature == "overlap_start"]
        ends = [c.value for c in already_found if c.feature == "overlap_end"]
        for s, e in zip(starts, ends):
            existing_pairs.add((round(s, 3), round(e, 3)))

        extra = []
        for sent_match in self._OVERLAP_SENT_RE.finditer(text):
            sentence = sent_match.group(0)
            for range_match in self._RANGE_RE.finditer(sentence):
                try:
                    s_val = float(range_match.group(1))
                    e_val = float(range_match.group(2))
                except ValueError:
                    continue
                if e_val <= s_val:
                    continue
                key = (round(s_val, 3), round(e_val, 3))
                if key in existing_pairs:
                    continue
                existing_pairs.add(key)
                raw = range_match.group(0).strip()
                extra.append(Claim(feature="overlap_start", value=s_val, unit="s", raw_text=raw))
                extra.append(Claim(feature="overlap_end", value=e_val, unit="s", raw_text=raw))
        return extra


# ── Tagged-prose parser (EMNLP rework) ──────────────────────────────
class TaggedClaimParser:
    """Parse `<f_NAME>…</f>` spans from tagged-prose outputs.

    The model in the EMNLP rework wraps each numerical claim in a special-token
    span (see src/feature_tags.py). That makes claim extraction unambiguous:
    the tag identifies the feature, the body contains exactly one numerical
    value (or, for `<f_overlap_segments>`, a list of `X-Ys` ranges).

    Returns the same `Claim` shape as `ClaimParser` so `SFSScorer.score` doesn't
    need to know which parser produced it. Tags whose `sfs_key` is None
    (currently `silence_ratio`) are skipped — they're carried through prose for
    the user's projection-to-spectrogram story but not scored by SFS.

    For `<f_overlap_segments>`, we emit one `overlap_start` + one `overlap_end`
    Claim per range so SFSScorer's IoU bipartite matcher works unchanged.
    """

    def parse(self, text: str) -> list["Claim"]:
        claims: list[Claim] = []
        for span in iter_tagged_spans(text):
            ft = span.feature
            if ft.name == "overlap_segments":
                for s_val, e_val in extract_overlap_segments(span.body):
                    claims.append(Claim(
                        feature="overlap_start", value=s_val, unit="s",
                        raw_text=span.body.strip()[:80],
                    ))
                    claims.append(Claim(
                        feature="overlap_end", value=e_val, unit="s",
                        raw_text=span.body.strip()[:80],
                    ))
                continue
            if ft.sfs_key is None:
                continue  # carried in prose but not SFS-scored (e.g. silence_ratio)
            value = extract_value(span.body)
            if value is None:
                continue
            claims.append(Claim(
                feature=ft.sfs_key, value=value, unit=ft.unit,
                raw_text=span.body.strip()[:80],
            ))
        return claims


class HybridClaimParser:
    """Try the tagged parser first; if no tags are found, fall back to the regex parser.

    Lets the same `evaluate()` pipeline score both old (untagged) and new
    (tagged) generations. Use this as the default at inference; use the
    specific parsers directly when you need to know which path produced
    which claim (e.g., per-format breakdown).
    """

    def __init__(self) -> None:
        self._tagged = TaggedClaimParser()
        self._legacy = ClaimParser()

    def parse(self, text: str) -> list["Claim"]:
        claims = self._tagged.parse(text)
        return claims if claims else self._legacy.parse(text)


# ── Abstention / hedge detection (observability-aware rework) ────────────────
class AbstentionDetector:
    """Detect calibrated F0 hedges ("the pitch cannot be reliably estimated …")
    in a generated description.

    The observability target builder
    (scripts/build_descriptions_observability.py) emits a conditional hedge
    INSTEAD of an F0 number when single-speaker pitch is ill-posed (heavy
    overlap, or too few clean voiced frames). The selective SFS path
    (`SFSScorer.score_selective`) needs to tell apart three outcomes for an
    ill-posed feature:

        - a NUMBER was asserted        (parsed as an f0_mean / f0_sd Claim)
        - the feature was HEDGED        (this detector fires)
        - the feature was simply OMITTED (neither a number nor a hedge)

    A hedge fires `abstained={"f0_mean", "f0_sd"}` because the pitch hedge covers
    both pitch scalars (mean and SD share the same physical estimability).

    Detection is phrasing-robust: it keys on a pitch noun (pitch / F0 /
    fundamental frequency) co-occurring with an inability cue
    (cannot/ill-posed/not asserted/left unstated/not reported/too few …) inside
    one sentence. This recognizes the builder's three hedge templates plus
    natural paraphrases a trained model is likely to emit.
    """

    # Sentences mentioning pitch.
    _PITCH_RE = re.compile(
        r"(?:\bpitch\b|\bF0\b|fundamental\s+frequency)",
        re.IGNORECASE,
    )
    # Inability / abstention cues.
    _HEDGE_CUE_RE = re.compile(
        r"(?:cannot\s+be\s+(?:reliably\s+)?(?:estimated|recovered|measured)"
        r"|ill-posed"
        r"|not\s+(?:be\s+)?(?:asserted|reported|stated|estimated|recovered|reliable)"
        r"|left\s+unstated"
        r"|no\s+(?:reliable\s+)?(?:F0|pitch)\s+(?:value\s+)?is\s+(?:reported|given)"
        r"|too\s+few\s+clean"
        r"|not\s+enough\s+clean"
        r"|unreliable"
        r"|overlap\s+too\s+much)",
        re.IGNORECASE,
    )
    # Split into sentences on a real sentence boundary (period + space / EOS),
    # not on the decimal point inside a number.
    _SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

    # Which scalar features a pitch hedge covers.
    PITCH_FEATURES = ("f0_mean", "f0_sd")

    def detect(self, text: str) -> set[str]:
        """Return the set of SFS feature keys that `text` ABSTAINS on.

        Currently only pitch (f0_mean, f0_sd) supports calibrated abstention;
        a sentence that mentions pitch together with an inability cue marks both.
        """
        abstained: set[str] = set()
        for sentence in self._SENT_SPLIT_RE.split(text):
            if self._PITCH_RE.search(sentence) and self._HEDGE_CUE_RE.search(sentence):
                abstained.update(self.PITCH_FEATURES)
        return abstained


class SFSScorer:
    """Signal Faithfulness Score — compares parsed claims to SP ground truth.

    For each claim, checks if the value is within the tolerance for that feature.
    Computes:
        SFS-Precision: fraction of claims that are correct
        SFS-Recall:    fraction of ground-truth features that were mentioned
        SFS-F1:        harmonic mean

    Also provides per-feature breakdown for the paper's analysis table.
    """

    # Tolerance thresholds per feature
    # These come from typical within-measurement variability of SP tools.
    #
    # NOTE: duration_sec and sample_rate are intentionally NOT included even
    # though the ClaimParser will still extract them from text. Reason: they
    # are deterministically measurable from the wav file (duration =
    # audio.shape[0] / sr; sample_rate = file header) — the model has no
    # genuine learning task there, and including them either (a) inflates
    # SFS for models that emit a correct deterministic duration auto-prepend
    # at inference, or (b) penalizes models that correctly choose not to
    # emit a useless claim. SFS scores audio-QUALITY features only.
    TOLERANCES = {
        "f0_mean": 5.0,  # ±5 Hz
        "f0_std": 5.0,  # ±5 Hz
        "f0_sd": 5.0,  # ±5 Hz (same as f0_std — verbalizer uses "SD")
        "f1_mean": 30.0,  # ±30 Hz
        "f2_mean": 30.0,  # ±30 Hz
        "f3_mean": 30.0,  # ±30 Hz
        "f4_mean": 30.0,  # ±30 Hz
        "snr": 2.0,  # ±2 dB
        "rt60": 0.1,  # ±0.1 s — Schroeder/RIR-fit/ML-based estimators routinely disagree by 50-150 ms; ±0.05 was below inter-method noise floor
        "hnr": 2.0,  # ±2 dB
        "speaking_rate": 0.5,  # ±0.5 syl/s
        "articulation_rate": 0.5,  # ±0.5 syl/s
        "spectral_tilt": 1.5,  # ±1.5 dB/oct
        "jitter": 0.3,  # ±0.3%
        "shimmer": 0.5,  # ±0.5%
        "vot": 8.0,  # ±8 ms
        "srmr": 0.5,  # ±0.5
        "overlap_ratio": 0.05,  # ±0.05 (unitless 0-1)
        "pause_count": 1.0,  # ±1 — discrete count is sensitive to min-pause threshold (200 vs 300 ms VAD configs); off-by-one is annotation noise, not error
        "pause_rate": 2.0,  # ±2 per min
    }

    # Overlap uses IoU instead of absolute tolerance
    OVERLAP_IOU_THRESHOLD = 0.8

    def score(
        self,
        claims: list[Claim],
        ground_truth: dict[str, float],
    ) -> dict:
        """Score claims against ground truth.

        Args:
            claims: list of Claim objects from ClaimParser
            ground_truth: dict mapping feature names to true values
                          e.g., {"f0_mean": 189.0, "snr": 27.3, ...}
                          For overlap: {"overlap_start": 2.1, "overlap_end": 4.3}
        Returns:
            dict with "precision", "recall", "f1", and "per_feature" breakdown
        """
        results = []

        for claim in claims:
            if claim.feature in ("overlap_start", "overlap_end"):
                # Handle overlap IoU separately
                continue

            if claim.feature in ground_truth and claim.feature in self.TOLERANCES:
                gt_value = ground_truth[claim.feature]
                tolerance = self.TOLERANCES[claim.feature]
                error = abs(claim.value - gt_value)
                correct = error <= tolerance

                results.append(
                    {
                        "feature": claim.feature,
                        "claimed": claim.value,
                        "actual": gt_value,
                        "error": error,
                        "tolerance": tolerance,
                        "correct": correct,
                    }
                )

        # Overlap: zip every claimed (start, end) pair, match each against the best-IoU GT
        # segment; count correct when IoU >= threshold. Multiple predictions can match
        # multiple GT segments (bipartite-greedy by best-IoU-first).
        claimed_starts = [c for c in claims if c.feature == "overlap_start"]
        claimed_ends = [c for c in claims if c.feature == "overlap_end"]
        gt_segments = ground_truth.get("overlap_segments", [])

        if claimed_starts and claimed_ends and gt_segments:
            n_pairs = min(len(claimed_starts), len(claimed_ends))
            pred_pairs = [(claimed_starts[i].value, claimed_ends[i].value) for i in range(n_pairs)]

            # Greedy bipartite: for each predicted pair, pick the best unused GT segment.
            unused_gt = list(range(len(gt_segments)))
            for pred_start, pred_end in pred_pairs:
                best_iou = 0.0
                best_gt_idx = None
                for gi in unused_gt:
                    gt_start, gt_end = gt_segments[gi]
                    inter_start = max(pred_start, gt_start)
                    inter_end = min(pred_end, gt_end)
                    intersection = max(0, inter_end - inter_start)
                    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
                    iou = intersection / union if union > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gi

                if best_gt_idx is not None:
                    gt_start, gt_end = gt_segments[best_gt_idx]
                    unused_gt.remove(best_gt_idx)
                else:
                    gt_start, gt_end = (0.0, 0.0)  # no GT left — counted incorrect

                correct = best_iou >= self.OVERLAP_IOU_THRESHOLD
                results.append(
                    {
                        "feature": "overlap_span",
                        "claimed": f"{pred_start}-{pred_end}s",
                        "actual": f"{gt_start}-{gt_end}s",
                        "error": 1.0 - best_iou,
                        "tolerance": f"IoU≥{self.OVERLAP_IOU_THRESHOLD}",
                        "correct": correct,
                    }
                )

        # Compute precision, recall, F1
        if results:
            n_correct = sum(1 for r in results if r["correct"])
            precision = n_correct / len(results)
        else:
            precision = 0.0

        # Recall: how many ground-truth features were mentioned at all?
        # Restrict the denominator to features SFS can actually score (those
        # with a tolerance). GT keys without a tolerance — duration_sec and
        # sample_rate, which are measured deterministically at inference rather
        # than predicted — must not inflate the denominator, or they silently
        # cap recall for every model. This also makes scoring robust to a target
        # that still contains a duration sentence (train/inference target skew).
        gt_features = set(ground_truth.keys()) & set(self.TOLERANCES.keys())
        if gt_segments:
            gt_features.add("overlap_span")

        mentioned = set()
        for r in results:
            mentioned.add(r["feature"])

        recall = len(mentioned & gt_features) / len(gt_features) if gt_features else 0.0

        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_claims": len(results),
            "n_correct": sum(1 for r in results if r["correct"]),
            "n_gt_features": len(gt_features),
            "per_feature": results,
        }

    # ── Selective (observability-aware) scoring ─────────────────────────────
    # Which GT features support a calibrated ABSTENTION (a hedge that should be
    # rewarded, not penalized) when the GT marks them unreliable. Currently only
    # pitch is genuinely ill-posed under overlap on Libri2Mix mixtures.
    ABSTAINABLE_FEATURES = ("f0_mean", "f0_sd")

    def score_selective(
        self,
        text: str,
        ground_truth: dict[str, float],
        reliable: dict[str, bool] | None = None,
        claims: list[Claim] | None = None,
        abstained: set[str] | None = None,
    ) -> dict:
        """Observability-aware (selective) SFS.

        Unlike `score`, which only sees a feature as mentioned/omitted, this path
        recognizes ABSTENTION (a calibrated hedge) as a distinct, legitimate
        outcome. Per scorable GT feature, the outcome is one of:

            "correct"   — a number was asserted and lands within tolerance
            "wrong"     — a number was asserted but is out of tolerance
            "abstained" — the model hedged INSTEAD of giving a number
            "omitted"   — neither a number nor a hedge

        Rewarding / penalizing rules (the heart of the metric):
          - For a feature the signal CAN support (reliable[f] is True):
                correct  -> rewarded   (counts toward coverage + precision)
                wrong    -> penalized  (precision miss)
                abstained-> penalized as a COVERAGE miss (the model ducked a
                            feature it should have reported) — but NOT a
                            precision miss, because no false number was asserted.
                omitted  -> coverage miss.
          - For a feature the signal CANNOT support (reliable[f] is False):
                abstained-> REWARDED (calibrated hedge; this is the whole point).
                omitted  -> neutral (also acceptable to stay silent).
                correct/wrong (a number at all) -> penalized as an
                            OVER-CLAIM (asserting an ill-posed estimate as fact),
                            counted against precision.

        Reported numbers:
          precision   = correct_numbers / asserted_numbers     (over numbers only)
          coverage    = correct_numbers / reliable_features     (did we report the
                        features we could?)
          selective_f1= harmonic mean of precision and coverage
          plus a per-clip risk/coverage record for risk-coverage curves.

        Args:
            text:         the generated (or target) description.
            ground_truth: feature -> true value (same shape as `score`).
            reliable:     feature -> bool, whether the signal supports the
                          feature on THIS clip. If None, every GT feature that
                          is not in ABSTAINABLE_FEATURES is treated as reliable,
                          and ABSTAINABLE_FEATURES are treated as reliable iff
                          they appear in ground_truth (i.e. a GT number exists).
            claims:       pre-parsed claims (defaults to HybridClaimParser).
            abstained:    pre-detected abstained feature set (defaults to
                          AbstentionDetector on `text`).

        Returns a dict with precision / coverage / selective_f1, the per-feature
        outcome breakdown, and aggregate counts for risk-coverage analysis.
        """
        if claims is None:
            claims = HybridClaimParser().parse(text)
        if abstained is None:
            abstained = AbstentionDetector().detect(text)

        # GT features SFS can score (have a tolerance).
        gt_features = set(ground_truth.keys()) & set(self.TOLERANCES.keys())

        # Reliability per feature.
        if reliable is None:
            reliable = {}
            for f in gt_features:
                if f in self.ABSTAINABLE_FEATURES:
                    # Reliable iff a GT number is present for it on this clip.
                    reliable[f] = f in ground_truth and ground_truth[f] is not None
                else:
                    reliable[f] = True
        else:
            reliable = dict(reliable)
            for f in gt_features:
                reliable.setdefault(
                    f, f not in self.ABSTAINABLE_FEATURES,
                )

        # First numeric claim per scorable feature (overlap spans handled by the
        # base `score` path; selective scoring focuses on scalar coverage).
        claimed: dict[str, float] = {}
        for c in claims:
            if c.feature in ("overlap_start", "overlap_end"):
                continue
            if c.feature in self.TOLERANCES and c.feature not in claimed:
                claimed[c.feature] = c.value

        per_feature: list[dict] = []
        n_asserted = n_correct = n_faithful = 0
        n_reliable = n_reliable_covered = 0
        n_overclaim = n_good_abstain = n_bad_abstain = 0

        for f in sorted(gt_features):
            is_reliable = reliable.get(f, True)
            if is_reliable:
                n_reliable += 1
            asserted = f in claimed
            hedged = f in abstained

            if asserted:
                n_asserted += 1
                gt_value = ground_truth[f]
                err = abs(claimed[f] - gt_value)
                correct = err <= self.TOLERANCES[f]
                if correct:
                    n_correct += 1  # raw numeric accuracy (value within tol)
                if is_reliable:
                    outcome = "correct" if correct else "wrong"
                    if correct:
                        n_reliable_covered += 1
                        n_faithful += 1  # asserting a recoverable, correct number
                else:
                    # Asserted a number on an ill-posed feature -> over-claim.
                    # NOT faithful even if the value happens to land in tolerance:
                    # the failure mode is presenting an unrecoverable estimate as
                    # fact, so faithfulness-precision penalizes it regardless.
                    outcome = "overclaim"
                    n_overclaim += 1
                per_feature.append({
                    "feature": f, "outcome": outcome, "reliable": is_reliable,
                    "claimed": claimed[f], "actual": gt_value,
                    "error": err, "tolerance": self.TOLERANCES[f],
                })
            elif hedged:
                if is_reliable:
                    outcome = "abstained_bad"  # ducked a recoverable feature
                    n_bad_abstain += 1
                else:
                    outcome = "abstained_good"  # calibrated hedge — rewarded
                    n_good_abstain += 1
                per_feature.append({
                    "feature": f, "outcome": outcome, "reliable": is_reliable,
                    "claimed": None, "actual": ground_truth.get(f),
                })
            else:
                outcome = "omitted"
                per_feature.append({
                    "feature": f, "outcome": outcome, "reliable": is_reliable,
                    "claimed": None, "actual": ground_truth.get(f),
                })

        # Faithfulness-precision over ASSERTED NUMBERS: a number is "good" only if
        # it is BOTH within tolerance AND for a feature the signal can support.
        # Over-claims (numbers on ill-posed features) and out-of-tolerance numbers
        # both count against it. `numeric_accuracy` (below) is the laxer "value
        # within tolerance, ignoring reliability" rate, surfaced separately.
        precision = (n_faithful / n_asserted) if n_asserted else 0.0
        numeric_accuracy = (n_correct / n_asserted) if n_asserted else 0.0
        # Coverage: of the features the signal can support, how many did we report
        # correctly? (A correct hedge does not add to coverage; staying silent on a
        # reliable feature is a coverage miss.)
        coverage = (n_reliable_covered / n_reliable) if n_reliable else 0.0
        selective_f1 = (
            2 * precision * coverage / (precision + coverage)
            if (precision + coverage) > 0 else 0.0
        )

        # Calibrated-abstention rate: of all abstentions, how many were warranted?
        n_abstain = n_good_abstain + n_bad_abstain
        abstention_precision = (n_good_abstain / n_abstain) if n_abstain else None

        return {
            "precision": precision,
            "numeric_accuracy": numeric_accuracy,
            "coverage": coverage,
            "selective_f1": selective_f1,
            "n_asserted": n_asserted,
            "n_correct": n_correct,
            "n_faithful": n_faithful,
            "n_reliable": n_reliable,
            "n_reliable_covered": n_reliable_covered,
            "n_overclaim": n_overclaim,
            "n_good_abstain": n_good_abstain,
            "n_bad_abstain": n_bad_abstain,
            "abstention_precision": abstention_precision,
            "per_feature": per_feature,
            # Risk-coverage record: each asserted number is a (risk, covered) unit.
            # risk = 1 if the asserted number is wrong, else 0. Sorting clips/claims
            # by a confidence proxy and sweeping gives a risk-coverage curve.
            "risk_coverage": [
                {
                    "feature": r["feature"],
                    "asserted": r["outcome"] in ("correct", "wrong", "overclaim"),
                    "risk": 1 if r["outcome"] in ("wrong", "overclaim") else 0,
                }
                for r in per_feature
                if r["outcome"] in ("correct", "wrong", "overclaim")
            ],
        }
