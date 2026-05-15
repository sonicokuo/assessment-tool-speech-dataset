#!/usr/bin/env python3
"""Tests for scripts/fix_descriptions.py.

Run:  python scripts/test_fix_descriptions.py
Exit: 0 if all pass, 1 if any fail.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_descriptions import (  # noqa: E402
    swap_reversed_ranges,
    drop_oob_ranges,
    strip_prompt_echo,
    remove_unmatched_closings,
    wrap_bare_overlap_section,
    merge_orphan_f_into_section,
    strip_orphan_r_tags,
    strip_orphan_overlap_artifacts,
    ensure_terminal_period,
    fix_all,
    audit,
)

failed = []


def expect(name, got, want):
    if got == want:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}")
        print(f"       want: {want!r}")
        print(f"       got : {got!r}")
        failed.append(name)


def expect_true(name, got):
    expect(name, bool(got), True)


def section(label):
    print(f"\n== {label} ==")


def main():
    section("swap_reversed_ranges (C1)")
    expect("swap when end < start",
           swap_reversed_ranges("<r>20.0-7.5s</r>"),
           "<r>7.5-20.0s</r>")
    expect("untouched when normal",
           swap_reversed_ranges("<r>2.5-7.5s</r>"),
           "<r>2.5-7.5s</r>")
    expect("untouched when equal endpoints",
           swap_reversed_ranges("<r>5.0-5.0s</r>"),
           "<r>5.0-5.0s</r>")
    expect("mixed list: only reversed swapped",
           swap_reversed_ranges("<r>2.5-7.5s</r>, <r>20.0-7.5s</r>"),
           "<r>2.5-7.5s</r>, <r>7.5-20.0s</r>")
    expect("no ranges: untouched",
           swap_reversed_ranges("hello world"),
           "hello world")

    section("drop_oob_ranges (C2)")
    expect("drops fully-beyond range",
           drop_oob_ranges("The recording is 3.92 s long. <r>5.0-7.5s</r>."),
           "The recording is 3.92 s long.")
    expect("keeps range within duration",
           drop_oob_ranges("The recording is 10.0 s long. <r>2.0-4.0s</r>"),
           "The recording is 10.0 s long. <r>2.0-4.0s</r>")
    expect("keeps half-in range",
           drop_oob_ranges("The recording is 4.0 s long. <r>3.5-5.0s</r>"),
           "The recording is 4.0 s long. <r>3.5-5.0s</r>")
    expect("drops middle range in list",
           drop_oob_ranges("The recording is 4.0 s long. <r>1-2s</r>, <r>5-6s</r>, and <r>3-4s</r>"),
           "The recording is 4.0 s long. <r>1-2s</r>, and <r>3-4s</r>")
    expect("drops all three: leaves bare prose with trailing-period cleaned",
           drop_oob_ranges(
               "The recording is 3.0 s long. overlap segments are present at "
               "<r>5-6s</r>, <r>7-8s</r>, and <r>9-10s</r>."),
           "The recording is 3.0 s long.")
    expect("no duration line: untouched",
           drop_oob_ranges("<r>5-7s</r>"),
           "<r>5-7s</r>")

    section("strip_prompt_echo (A7)")
    expect("strips short echo at end",
           strip_prompt_echo("done. Overlap context: 2"),
           "done.")
    expect("strips 'None' form",
           strip_prompt_echo("done. Overlap context: None"),
           "done.")
    expect("no echo: passthrough",
           strip_prompt_echo("done."),
           "done.")
    expect("empty: passthrough",
           strip_prompt_echo(""),
           "")

    section("remove_unmatched_closings (A4/A5/Ax)")
    expect("drop extra </sec>",
           remove_unmatched_closings("<sec_a>x</sec></sec>"),
           "<sec_a>x</sec>")
    expect("drop extra </r>",
           remove_unmatched_closings("<r>1</r></r>"),
           "<r>1</r>")
    expect("drop extra </f>",
           remove_unmatched_closings("<f_a>x</f></f>"),
           "<f_a>x</f>")
    expect("balanced untouched",
           remove_unmatched_closings("<sec_a>x</sec>"),
           "<sec_a>x</sec>")
    expect("nested balanced untouched",
           remove_unmatched_closings("<sec_a><f_x>1</f></sec>"),
           "<sec_a><f_x>1</f></sec>")
    expect("plain prose untouched",
           remove_unmatched_closings("hello world"),
           "hello world")

    section("wrap_bare_overlap_section (A1)")
    text1 = "a. Finally, there are overlap segments present at <r>1.0-2.0s</r>. F0..."
    got1 = wrap_bare_overlap_section(text1)
    expect_true("wraps: contains <sec_overlap>", '<sec_overlap>' in got1)
    expect_true("wraps: contains <f_overlap_segments>", '<f_overlap_segments>' in got1)
    expect_true("wraps: range preserved", '<r>1.0-2.0s</r>' in got1)
    expect_true("wraps: original prose preserved before/after",
                got1.startswith('a. Finally, there are ') and got1.endswith('. F0...'))
    expect("no double-wrap when <sec_overlap> already present",
           wrap_bare_overlap_section(
               "<sec_overlap>x</sec> overlap segments present at <r>1-2s</r>"),
           "<sec_overlap>x</sec> overlap segments present at <r>1-2s</r>")
    expect("no overlap content: untouched",
           wrap_bare_overlap_section("<sec_noise>x</sec>"),
           "<sec_noise>x</sec>")
    got2 = wrap_bare_overlap_section(
        "a. <f_overlap_segments>overlap segments present at <r>1-2s</r></f>. b")
    expect_true("orphan <f_overlap_segments> wrapped in <sec_overlap>",
                '<sec_overlap><f_overlap_segments>' in got2 and '</f></sec>' in got2)

    section("merge_orphan_f_into_section (A3 when target exists)")
    got = merge_orphan_f_into_section(
        "x <f_overlap_ratio>R</f> and "
        "<sec_overlap><f_overlap_segments>S</f></sec>"
    )
    expect_true("orphan <f_overlap_ratio> moved into <sec_overlap>",
                '<sec_overlap><f_overlap_ratio>R</f>' in got)
    expect_true("inner <f_overlap_segments> still present", '<f_overlap_segments>S</f>' in got)
    expect_true("merged section closes correctly", got.count('</sec>') == got.count('<sec_'))
    expect("idempotent when no orphan",
           merge_orphan_f_into_section(
               "<sec_overlap><f_overlap_ratio>R</f></sec>"),
           "<sec_overlap><f_overlap_ratio>R</f></sec>")
    expect("untouched when target section absent",
           merge_orphan_f_into_section("x <f_overlap_ratio>R</f>"),
           "x <f_overlap_ratio>R</f>")
    # snr orphan merged into sec_noise
    got2 = merge_orphan_f_into_section(
        "a <f_snr>S</f> b <sec_noise><f_other>X</f></sec> c"
    )
    expect_true("snr orphan merged into sec_noise",
                '<sec_noise><f_snr>S</f>' in got2)

    section("strip_orphan_r_tags (A9)")
    expect("strips orphan <r> outside any <sec>",
           strip_orphan_r_tags("prose <r>1-2s</r> more prose"),
           "prose  more prose")
    expect("keeps <r> that is inside <sec_*>",
           strip_orphan_r_tags("<sec_a><r>1-2s</r></sec>"),
           "<sec_a><r>1-2s</r></sec>")
    expect("mixed: keeps inside, strips outside",
           strip_orphan_r_tags("<r>0-1s</r> <sec_a><r>1-2s</r></sec>"),
           " <sec_a><r>1-2s</r></sec>")

    section("strip_orphan_overlap_artifacts (A8)")
    expect("strips trailing F0 sentence when no overlap content",
           strip_orphan_overlap_artifacts(
               "x<sec_pauses>y</sec>. Finally, there are F0 and formant estimates are unreliable during overlap windows."),
           "x<sec_pauses>y</sec>.")
    expect("untouched when <sec_overlap> present",
           strip_orphan_overlap_artifacts(
               "<sec_overlap>x</sec>. F0 and formant estimates are unreliable during overlap windows."),
           "<sec_overlap>x</sec>. F0 and formant estimates are unreliable during overlap windows.")
    expect("untouched when <r> present somewhere",
           strip_orphan_overlap_artifacts(
               "<sec_x><r>1-2s</r></sec>. F0 and formant estimates are unreliable during overlap windows."),
           "<sec_x><r>1-2s</r></sec>. F0 and formant estimates are unreliable during overlap windows.")
    expect("untouched when no </sec>",
           strip_orphan_overlap_artifacts("plain text"),
           "plain text")
    expect("untouched when trailing is innocuous",
           strip_orphan_overlap_artifacts("x<sec_a>y</sec>. plain ending."),
           "x<sec_a>y</sec>. plain ending.")

    section("ensure_terminal_period (B1)")
    expect("appends . when missing", ensure_terminal_period("a"), "a.")
    expect("idempotent when present", ensure_terminal_period("a."), "a.")
    expect("trims trailing whitespace then adds .",
           ensure_terminal_period("a   "), "a.")
    expect("empty stays empty",
           ensure_terminal_period(""), "")
    expect("just whitespace stays empty",
           ensure_terminal_period("   "), "")

    section("fix_all (integration)")
    ugly = (
        "The recording is 14.7 s long. "
        "<sec_noise><f_snr>SNR is 24.13 dB</f></sec>. "
        "Finally, there are overlap segments present at <r>20.0-7.5s</r>"
    )
    g = fix_all(ugly)
    expect_true("reversed range swapped",  "<r>7.5-20.0s</r>" in g)
    expect_true("wrapped in <sec_overlap>", "<sec_overlap>" in g)
    expect_true("ends with period",        g.endswith('.'))
    expect_true("no 'Overlap context:' residue", 'Overlap context:' not in g)

    # already-clean stays clean
    clean = (
        "The recording is 5.0 s long. "
        "<sec_noise><f_snr>SNR is 13 dB</f></sec>. "
        "<sec_overlap><f_overlap_segments>overlap segments are present at "
        "<r>1.0-2.0s</r></f></sec>."
    )
    expect("idempotent on clean input", fix_all(clean), clean)

    # prompt echo + bare overlap together
    mix = (
        "The recording is 6.0 s long. <sec_noise><f_snr>x</f></sec>. "
        "done. Overlap context: 2. "
        "Finally, there are overlap segments present at <r>2-3s</r>."
    )
    gm = fix_all(mix)
    expect_true("mix: no 'Overlap context:' residue", 'Overlap context:' not in gm)
    expect_true("mix: <sec_overlap> wrapped",          '<sec_overlap>' in gm)
    expect_true("mix: ends with .",                     gm.endswith('.'))

    # range fully beyond duration: dropped, prose cleaned
    oob = "The recording is 3.92 s long. <sec_noise><f_snr>x</f></sec>. " \
          "Finally, there are overlap segments present at <r>5.0-7.5s</r>."
    go = fix_all(oob)
    expect_true("OOB range removed", "<r>5.0-7.5s</r>" not in go)
    expect_true("ends with period",  go.endswith('.'))

    # unmatched </sec> gets stripped
    bad = "The recording is 5.0 s long. <sec_noise>x</sec></sec>."
    gb = fix_all(bad)
    expect("extra </sec> stripped",
           gb.count('</sec>'), gb.count('<sec_'))

    section("audit")
    expect("audit clean: []", audit(clean), [])
    expect("audit empty: A6 only", audit(""), ['A6_empty'])
    # reversed range (20 > 7.5) + no terminal period + the <r> is orphan
    a = audit("The recording is 10 s long. <r>20.0-7.5s</r>")
    expect("audit reversed + B1 + A9",
           sorted(a),
           sorted(['C1_reversed_range', 'B1_no_terminal_period',
                   'A9_orphan_r_outside_section']))
    # range fully beyond duration + no terminal period + the <r> is orphan
    a2 = audit("The recording is 4 s long. <r>5.0-7.0s</r>")
    expect("audit beyond + B1 + A9",
           sorted(a2),
           sorted(['C2_range_beyond_duration', 'B1_no_terminal_period',
                   'A9_orphan_r_outside_section']))
    # bare overlap content, no <sec_overlap>
    a3 = audit("done. overlap segments are present at <r>1-2s</r>.")
    expect_true("audit detects A1_bare_overlap", 'A1_bare_overlap' in a3)

    print()
    if failed:
        print(f"=== {len(failed)} FAILED ===")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    print("=== all passed ===")


if __name__ == '__main__':
    main()
