#!/usr/bin/env python3
"""Tests for fix_overlap_csv.py — focused on the pure helpers that don't need
Silero loaded (overlap_for_pair / merge_adjacent). The full CSV-roundtrip
test is integration-level and requires actual wav stems; this file is the
fast unit suite.

Run:
    python scripts/test_fix_overlap_csv.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fix_overlap_csv import merge_adjacent  # noqa: E402

PASS = FAIL = 0


def expect(label, got, want):
    global PASS, FAIL
    ok = got == want
    PASS += int(ok); FAIL += int(not ok)
    print(f"  {'ok  ' if ok else 'FAIL'} {label}")
    if not ok:
        print(f"       want={want!r}")
        print(f"       got ={got!r}")


def section(label):
    print(f"\n== {label} ==")


def main() -> int:
    print("================ TESTS ================")

    section("merge_adjacent")
    expect("empty -> empty",          merge_adjacent([], 0), [])
    expect("single -> unchanged",     merge_adjacent([(0, 10)], 0), [(0, 10)])
    expect("non-adjacent untouched",  merge_adjacent([(0, 10), (20, 30)], 0),
                                       [(0, 10), (20, 30)])
    expect("touching at boundary -> merged with gap=0",
           merge_adjacent([(0, 10), (10, 20)], 0), [(0, 20)])
    expect("close enough -> merged with gap=5",
           merge_adjacent([(0, 10), (13, 20)], 5), [(0, 20)])
    expect("just past gap -> NOT merged",
           merge_adjacent([(0, 10), (16, 20)], 5), [(0, 10), (16, 20)])
    expect("three-way merge cascades",
           merge_adjacent([(0, 10), (10, 20), (20, 30)], 0), [(0, 30)])
    expect("overlapping pair merged (end picked correctly)",
           merge_adjacent([(0, 15), (10, 20)], 0), [(0, 20)])

    print()
    print(f"================ {PASS} passed, {FAIL} failed ================")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
