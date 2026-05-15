#!/usr/bin/env python3
"""Post-process a descriptions.partN.json to fix known issues.

DETERMINISTIC FIXES applied per entry:
  B1   ensure each description ends with a period.
  A7   strip leaked 'Overlap context: ...' prompt echo.
  C1   swap reversed <r>X-Ys</r> where Y < X.
  C2   drop <r> ranges where BOTH endpoints exceed the clip duration
       stated in the leading 'The recording is X s long.' sentence.
       (Band-aid; the source overlap_segments column should also be
       fixed upstream in the feature extractor.)
  A4   remove unmatched </sec> closing tags (stack-based pruning).
  A5   remove unmatched </r> closing tags.
  Ax   remove unmatched </f> closing tags.
  A1   if no <sec_overlap> but there is an orphan <f_overlap_segments>
       or bare 'overlap segments are present at <r>...</r>' prose,
       wrap it inside <sec_overlap>...</sec>.

REPORTED but NOT auto-fixed:
  A3   orphan <f_*> outside any <sec_*> (other than overlap_segments,
       which A1 wraps) -> needs regeneration.
  A6   empty descriptions -> needs regeneration.

Reusable: run on Part 1 now, then Parts 2/3 when they finish.

Usage:
    python scripts/fix_descriptions.py --input descriptions.part1.json
    python scripts/fix_descriptions.py --input X.json --output Y.json
    python scripts/fix_descriptions.py --input X.json --in-place
    python scripts/fix_descriptions.py --input X.json --audit-only
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

# ---- regexes ----

# Feature -> canonical section. Used by merge_orphan_f_into_section to move
# orphan <f_*> tags into their owning section when present in the text.
_FEATURE_TO_SECTION = {
    "snr": "noise", "hnr": "noise",
    "srmr": "reverb",
    "f0_mean": "pitch", "f0_sd": "pitch",
    "speaking_rate": "tempo", "articulation_rate": "tempo",
    "pause_count": "pauses", "pause_rate": "pauses",
    "overlap_ratio": "overlap", "overlap_segments": "overlap",
}

_RANGE_RE = re.compile(r'<r>([\d.]+)-([\d.]+)s</r>')
_DURATION_RE = re.compile(r'^The recording is (\d+(?:\.\d+)?) s long\.')
_OVERLAP_PROSE_RE = re.compile(
    r'overlap segments\s+(?:are\s+|is\s+)?present\s+at\s+'
    r'(?:<r>[\d.]+-[\d.]+s</r>(?:\s*,?\s*(?:and\s+)?)?)+',
    re.IGNORECASE,
)
_PROMPT_ECHO_RE = re.compile(r'\s*Overlap context:\s*[^<.]*\.?', re.IGNORECASE)
_SEC_OPEN_RE = re.compile(r'<sec_\w+>')
_F_OPEN_RE = re.compile(r'<f_\w+>')
_F_OVERLAP_SEG_RE = re.compile(r'<f_overlap_segments>.*?</f>', re.DOTALL)


# ---- individual fixes (each is pure: str -> str) ----

def swap_reversed_ranges(text):
    """C1: <r>X-Ys</r> with Y < X -> <r>Y-Xs</r>."""
    def swap(m):
        a, b = float(m.group(1)), float(m.group(2))
        return f"<r>{m.group(2)}-{m.group(1)}s</r>" if b < a else m.group(0)
    return _RANGE_RE.sub(swap, text)


def drop_oob_ranges(text):
    """C2: drop <r>X-Ys</r> where BOTH endpoints exceed clip duration.
    Cleans up dangling commas/'and' between dropped tokens, and trims an
    empty 'overlap segments are present at .' phrase if all ranges go.
    """
    dm = _DURATION_RE.match(text)
    if not dm:
        return text
    dur = float(dm.group(1))
    marker = '\x00OOB\x00'

    def mark(m):
        a, b = float(m.group(1)), float(m.group(2))
        return marker if (a > dur + 0.01 and b > dur + 0.01) else m.group(0)
    text = _RANGE_RE.sub(mark, text)

    # iterative cleanup of separators around dropped markers
    cleanups = [
        (rf',\s*and\s+{marker}', ''),
        (rf',\s*{marker}', ''),
        (rf'{marker}\s*,\s*', ''),
        (rf'{marker}\s+and\s+', ''),
        (rf'and\s+{marker}', ''),
        (rf'\s*{marker}\s*', ''),
    ]
    prev = None
    while prev != text:
        prev = text
        for pat, rep in cleanups:
            text = re.sub(pat, rep, text)

    # If the "overlap segments ... present at" phrase has nothing after it
    # (no <r> tag follows), drop the dangling phrase. The negative lookahead
    # leaves the phrase alone whenever a <r> tag is the next non-whitespace.
    text = re.sub(
        r'overlap segments\s+(?:are\s+|is\s+)?present\s+at\s*\.?(?!\s*<r>)',
        '',
        text,
    )
    text = re.sub(r'<f_overlap_segments>\s*</f>', '', text)
    text = re.sub(r'<sec_overlap>\s*</sec>', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s+\.', '.', text)
    text = re.sub(r',\s*\.', '.', text)
    text = re.sub(r',\s*</', '</', text)
    text = re.sub(r'\.{2,}', '.', text)
    return text.rstrip()


def strip_prompt_echo(text):
    """A7: strip 'Overlap context: ...' echoed from prompt template."""
    return _PROMPT_ECHO_RE.sub('', text).strip()


def remove_unmatched_closings(text):
    """A4/A5/Ax: drop </sec>, </f>, </r> tokens with no matching opener."""
    out = []
    sec_depth = f_depth = r_depth = 0
    i, n = 0, len(text)
    while i < n:
        if text.startswith('<sec_', i):
            j = text.find('>', i)
            if j > 0:
                out.append(text[i:j+1]); sec_depth += 1; i = j + 1; continue
        if text.startswith('</sec>', i):
            if sec_depth > 0:
                out.append('</sec>'); sec_depth -= 1
            i += 6; continue
        if text.startswith('<f_', i):
            j = text.find('>', i)
            if j > 0:
                out.append(text[i:j+1]); f_depth += 1; i = j + 1; continue
        if text.startswith('</f>', i):
            if f_depth > 0:
                out.append('</f>'); f_depth -= 1
            i += 4; continue
        if text.startswith('<r>', i):
            out.append('<r>'); r_depth += 1; i += 3; continue
        if text.startswith('</r>', i):
            if r_depth > 0:
                out.append('</r>'); r_depth -= 1
            i += 4; continue
        out.append(text[i]); i += 1
    return ''.join(out)


def wrap_bare_overlap_section(text):
    """A1: if <sec_overlap> is absent, wrap bare overlap content.
    Handles two forms:
      (1) orphan <f_overlap_segments>...</f>  -> wrap in <sec_overlap>.
      (2) bare 'overlap segments are present at <r>...</r>' prose -> wrap
          both in <sec_overlap><f_overlap_segments>...</f></sec>.
    """
    if '<sec_overlap>' in text:
        return text
    m = _F_OVERLAP_SEG_RE.search(text)
    if m:
        return text[:m.start()] + '<sec_overlap>' + m.group(0) + '</sec>' + text[m.end():]
    m = _OVERLAP_PROSE_RE.search(text)
    if m:
        bare = m.group(0)
        wrapped = f"<sec_overlap><f_overlap_segments>{bare}</f></sec>"
        return text[:m.start()] + wrapped + text[m.end():]
    return text


def _section_spans(text):
    """Return list of (start, end, sec_name) for each <sec_NAME>...</sec> block."""
    spans = []
    i = 0
    while i < len(text):
        sm = re.search(r'<sec_(\w+)>', text[i:])
        if not sm:
            break
        start = i + sm.start()
        cm = re.search(r'</sec>', text[start:])
        if not cm:
            break
        end = start + cm.end()
        spans.append((start, end, sm.group(1)))
        i = end
    return spans


def merge_orphan_f_into_section(text):
    """A3 (when target section exists): move each orphan <f_X>Y</f> into the
    canonical <sec_*> for that feature, inserted at the start of the section
    content. The dangling connective prose before the now-merged orphan is
    not cleaned (it remains a small cosmetic artifact)."""
    # iterate until no more merges happen
    for _ in range(20):
        spans = _section_spans(text)
        if not spans:
            return text
        sec_by_name = {n: (s, e) for s, e, n in spans}
        moved = False
        for fm in re.finditer(r'<f_(\w+)>.*?</f>', text, re.DOTALL):
            if any(s <= fm.start() < e for s, e, _ in spans):
                continue
            target = _FEATURE_TO_SECTION.get(fm.group(1))
            if not target or target not in sec_by_name:
                continue
            fbody = fm.group(0)
            without = text[:fm.start()] + text[fm.end():]
            tm = re.search(rf'<sec_{target}>', without)
            if not tm:
                continue
            ip = tm.end()
            text = without[:ip] + fbody + ' and ' + without[ip:]
            moved = True
            break
        if not moved:
            return text
    return text


def strip_orphan_r_tags(text):
    """A9: drop <r>...</r> markers that sit outside any <sec_*> block.
    These are typically duplicates that the model paraphrased into prose
    while also keeping the canonical instance inside <sec_overlap>."""
    spans = _section_spans(text)
    out, last = [], 0
    for rm in re.finditer(r'<r>[^<]*</r>', text):
        if not any(s <= rm.start() < e for s, e, _ in spans):
            out.append(text[last:rm.start()])
            last = rm.end()
    out.append(text[last:])
    return ''.join(out)


def ensure_terminal_period(text):
    """B1: append '.' if missing."""
    t = text.rstrip()
    return t + '.' if t and not t.endswith('.') else t


def strip_orphan_overlap_artifacts(text):
    """A8: if there is no <sec_overlap> AND no <r> tag in the text but the
    region after the last </sec> mentions overlap / F0 / formant, that text
    is an orphaned reference to overlap content that no longer exists
    (typically: drop_oob_ranges removed the only range, leaving a dangling
    'Finally, there are. F0 and formant estimates are unreliable during
    overlap windows.' fragment). Truncate at the last </sec>.
    """
    if '<sec_overlap>' in text or '<r>' in text:
        return text
    last_close = text.rfind('</sec>')
    if last_close < 0:
        return text
    after = text[last_close + len('</sec>'):]
    if ('overlap' in after.lower()
            or 'F0' in after
            or 'formant' in after.lower()):
        return text[:last_close + len('</sec>')] + '.'
    return text


# ---- composite ----

def fix_all(text):
    if not text:
        return text
    text = strip_prompt_echo(text)
    text = swap_reversed_ranges(text)
    text = drop_oob_ranges(text)
    text = remove_unmatched_closings(text)
    text = wrap_bare_overlap_section(text)
    text = merge_orphan_f_into_section(text)
    text = strip_orphan_r_tags(text)
    text = strip_orphan_overlap_artifacts(text)
    text = ensure_terminal_period(text)
    return text


# ---- audit ----

def audit(text):
    """Return a list of issue codes present in this description."""
    issues = []
    if not text:
        return ['A6_empty']
    if 'Overlap context:' in text:
        issues.append('A7_prompt_echo')
    for a, b in _RANGE_RE.findall(text):
        if float(b) < float(a):
            issues.append('C1_reversed_range'); break
    dm = _DURATION_RE.match(text)
    if dm:
        d = float(dm.group(1))
        for a, b in _RANGE_RE.findall(text):
            if float(a) > d + 0.01 and float(b) > d + 0.01:
                issues.append('C2_range_beyond_duration'); break
    if '<sec_overlap>' not in text and _OVERLAP_PROSE_RE.search(text):
        issues.append('A1_bare_overlap')
    if not text.rstrip().endswith('.'):
        issues.append('B1_no_terminal_period')
    if len(_SEC_OPEN_RE.findall(text)) != len(re.findall(r'</sec>', text)):
        issues.append('A4_unbalanced_sec')
    if len(_F_OPEN_RE.findall(text)) != len(re.findall(r'</f>', text)):
        issues.append('Ax_unbalanced_f')
    if len(re.findall(r'<r>', text)) != len(re.findall(r'</r>', text)):
        issues.append('A5_unbalanced_r')
    # orphan <f_*> outside any <sec_*>
    sec_spans = []
    i = 0
    while i < len(text):
        sm = re.search(r'<sec_\w+>', text[i:])
        if not sm: break
        start = i + sm.start()
        cm = re.search(r'</sec>', text[start:])
        if not cm: break
        end = start + cm.end()
        sec_spans.append((start, end))
        i = end
    for fm in _F_OPEN_RE.finditer(text):
        if not any(s <= fm.start() < e for s, e in sec_spans):
            issues.append('A3_orphan_f_outside_section'); break
    for rm in re.finditer(r'<r>', text):
        if not any(s <= rm.start() < e for s, e in sec_spans):
            issues.append('A9_orphan_r_outside_section'); break
    # A8: orphan overlap-trailing artifacts (no <sec_overlap>, no <r>, but
    # prose after the last </sec> mentions overlap/F0/formant).
    if '<sec_overlap>' not in text and '<r>' not in text:
        last_close = text.rfind('</sec>')
        if last_close >= 0:
            after = text[last_close + len('</sec>'):]
            if ('overlap' in after.lower()
                    or 'F0' in after
                    or 'formant' in after.lower()):
                issues.append('A8_orphan_overlap_trailing')
    return issues


# ---- CLI ----

def _summarise(label, counter, n):
    print(f"\n{label}:")
    if not counter:
        print("  (clean)"); return
    print(f"  {'issue':40s} {'count':>7s} {'pct':>6s}")
    for k in sorted(counter):
        c = counter[k]
        print(f"  {k:40s} {c:7d} {c*100/n:5.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    ap.add_argument('--input', required=True, help='descriptions JSON to read')
    ap.add_argument('--output', default=None, help='where to write fixed JSON')
    ap.add_argument('--in-place', action='store_true', help='overwrite --input')
    ap.add_argument('--audit-only', action='store_true', help='only report; do not fix')
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text())
    n = len(data)
    print(f"input: {args.input}  ({n} entries)")

    pre = Counter()
    for txt in data.values():
        for issue in audit(txt):
            pre[issue] += 1
    _summarise("issues in source", pre, n)

    if args.audit_only:
        return

    fixed = {}
    n_changed = 0
    for k, t in data.items():
        new = fix_all(t)
        if new != t:
            n_changed += 1
        fixed[k] = new

    if args.in_place:
        out_path = args.input
    elif args.output:
        out_path = args.output
    else:
        out_path = args.input.replace('.json', '.fixed.json')
    Path(out_path).write_text(json.dumps(fixed, ensure_ascii=False, indent=2))

    post = Counter()
    for txt in fixed.values():
        for issue in audit(txt):
            post[issue] += 1

    print(f"\ncomposite fix changed {n_changed}/{n} entries")
    print(f"written: {out_path}")

    print(f"\n{'issue':40s} {'pre':>7s} {'post':>7s} {'delta':>7s}")
    print("-" * 70)
    for k in sorted(set(pre) | set(post)):
        p, q = pre.get(k, 0), post.get(k, 0)
        print(f"  {k:38s} {p:7d} {q:7d} {p - q:+7d}")


if __name__ == '__main__':
    main()
