"""Emit a shields.io endpoint JSON describing the test-suite size.

Reads a JUnit XML report produced by ``pytest --junitxml`` and prints the
shields.io "endpoint" payload to stdout. CI publishes that payload to the
orphan ``badges`` branch, and the README badge reads it from there.

WHY THIS EXISTS
    The README used to carry a hard-coded ``tests-393_passing`` badge. Every
    feature PR either churned it or, more often, silently let it go stale — it
    read 393 while the suite was at 471. A count that lies is worse than no
    count, so the number now comes from the run that actually executed the
    tests, and no human ever edits it.

WHY THE COUNT EXCLUDES SKIPS
    The suite deliberately skips optional tests when their extras or network
    are unavailable (``[pii]``, ``[redteam]``, live-model and MCP contract
    tests). Counting those as "passing" would overstate what CI verified, so
    the badge reports executed passes only and notes the skips separately.

Usage:
    python .github/scripts/make_tests_badge.py junit.xml
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET


def read_counts(xml_path: str) -> tuple[int, int, int, int]:
    """Return (tests, failures, errors, skipped) from a JUnit XML report.

    pytest writes a <testsuites> root wrapping one <testsuite>, but some
    versions emit <testsuite> as the root directly. Handle both rather than
    assume, and sum across suites so a future matrix build still adds up.
    """
    root = ET.parse(xml_path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    if not suites:
        raise ValueError(f"no <testsuite> element found in {xml_path}")

    def total(attr: str) -> int:
        return sum(int(s.get(attr, 0) or 0) for s in suites)

    return total("tests"), total("failures"), total("errors"), total("skipped")


def build_badge(tests: int, failures: int, errors: int, skipped: int) -> dict:
    """Build the shields.io endpoint payload.

    Colour encodes the honest state: green only when nothing failed. A red
    badge on a green-looking README is the point — it should be impossible to
    ship a passing-looking badge over a failing suite.
    """
    passed = tests - failures - errors - skipped
    broken = failures + errors

    if broken:
        message = f"{broken} failing"
        color = "red"
    elif skipped:
        message = f"{passed} passing, {skipped} skipped"
        color = "brightgreen"
    else:
        message = f"{passed} passing"
        color = "brightgreen"

    return {
        "schemaVersion": 1,
        "label": "tests",
        "message": message,
        "color": color,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    tests, failures, errors, skipped = read_counts(argv[1])
    print(json.dumps(build_badge(tests, failures, errors, skipped)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
