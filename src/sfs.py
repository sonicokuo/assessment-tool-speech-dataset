"""Signal Faithfulness Score (SFS) — evaluation metric for speech quality descriptions."""

import re
from dataclasses import dataclass


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
        # "fundamental frequency mean is 186.69 Hz")
        (
            r"(?:F0\s*(?:mean\s*)?|(?:mean\s+)?pitch(?:\s+mean)?|fundamental\s+frequency(?:\s+mean)?)\s*(?:=|≈|~|is|of)\s*(?:approximately\s+)?(\d+\.?\d*)\s*Hz",
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
        # RT60
        (r"RT60\s*(?:=|≈|~|<|>|is)\s*(?:approximately\s+)?(\d+\.?\d*)\s*s", [("rt60", 1, "s")]),
        # HNR (also matches "Harmonics-to-Noise Ratio (HNR) of 12.59 dB")
        (
            r"(?:Harmonics?-to-Noise\s+Ratio\s*(?:\(HNR\))?\s*|HNR\s*)(?:=|≈|~|is|of)\s*(?:approximately\s+)?(\d+\.?\d*)\s*dB",
            [("hnr", 1, "dB")],
        ),
        # Speaking rate (matches "5.875 syl/s", "4.934 syllables per second")
        (
            r"(?:speaking rate|rate)\s*(?:=|≈|~|is|of|:)\s*(?:approximately\s+)?(\d+\.?\d*)\s*(?:syl(?:lables?)?\s*(?:/\s*|per\s+)s(?:ec(?:ond)?)?)",
            [("speaking_rate", 1, "syl/s")],
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
        # Overlap temporal span: "overlap at 2.3-4.1s" or "overlapping speech from 2.3 to 4.1s"
        (
            r"overlap(?:ping)?\s*(?:speech\s+)?(?:at|from|during)\s*(\d+\.?\d*)\s*(?:s|sec)?\s*(?:-|to)\s*(\d+\.?\d*)\s*s",
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

        # Deduplicate: keep first occurrence of each feature
        seen = set()
        unique_claims = []
        for c in claims:
            if c.feature not in seen:
                seen.add(c.feature)
                unique_claims.append(c)

        return unique_claims


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
    # These come from typical within-measurement variability of SP tools
    TOLERANCES = {
        "f0_mean": 5.0,  # ±5 Hz
        "f0_std": 5.0,  # ±5 Hz
        "f1_mean": 30.0,  # ±30 Hz
        "f2_mean": 30.0,  # ±30 Hz
        "f3_mean": 30.0,  # ±30 Hz
        "f4_mean": 30.0,  # ±30 Hz
        "snr": 2.0,  # ±2 dB
        "rt60": 0.05,  # ±0.05 s
        "hnr": 2.0,  # ±2 dB
        "speaking_rate": 0.5,  # ±0.5 syl/s
        "spectral_tilt": 1.5,  # ±1.5 dB/oct
        "jitter": 0.3,  # ±0.3%
        "shimmer": 0.5,  # ±0.5%
        "vot": 8.0,  # ±8 ms
        "srmr": 0.5,  # ±0.5
        "sample_rate": 0.0,  # exact match (it's an integer)
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

        # Handle overlap IoU if both start and end are claimed
        claimed_starts = [c for c in claims if c.feature == "overlap_start"]
        claimed_ends = [c for c in claims if c.feature == "overlap_end"]
        gt_segments = ground_truth.get("overlap_segments", [])

        if claimed_starts and claimed_ends and gt_segments:
            pred_start = claimed_starts[0].value
            pred_end = claimed_ends[0].value

            # Match against best GT segment by IoU
            best_iou = 0.0
            best_gt = gt_segments[0]
            for gt_start, gt_end in gt_segments:
                inter_start = max(pred_start, gt_start)
                inter_end = min(pred_end, gt_end)
                intersection = max(0, inter_end - inter_start)
                union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
                iou = intersection / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_gt = (gt_start, gt_end)

            correct = best_iou >= self.OVERLAP_IOU_THRESHOLD

            results.append(
                {
                    "feature": "overlap_span",
                    "claimed": f"{pred_start}-{pred_end}s",
                    "actual": f"{best_gt[0]}-{best_gt[1]}s",
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
        gt_features = set(ground_truth.keys()) - {"overlap_segments"}
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
