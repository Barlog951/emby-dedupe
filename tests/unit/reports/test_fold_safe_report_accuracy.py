"""A report must state what actually happened to each duplicate.

Regression for 2026-08-09. Two defects made a correct run *look* wrong, both found by
reading a real report against the disk:

1. ``--fold-safe-delete`` runs AFTER the report was rendered, so a duplicate the guard
   refused and fold-safe then removed over SSH was frozen in the report as still present.
   Live case: `My Mother's Wedding` showed "Pending" for a file already deleted.
2. ``skipped_unsafe`` had no label of its own, so a deliberate, protective guard refusal
   rendered as the same generic "Pending" a dry run produces — indistinguishable from
   "nothing has happened yet".

The guard itself was never wrong in either case; only the reporting of it was.
"""

from __future__ import annotations

from unittest.mock import patch

from emby_dedupe.cli.app import _apply_fold_safe_results_to_decisions
from emby_dedupe.reports.html import format_html_report

BASE = "https://emby.example.com:443"


def _decision(status: str) -> list[dict]:
    """One keep/delete group whose delete item carries ``status``."""
    return [
        {
            "keep": {
                "id": "keep1",
                "name": "My Mother's Wedding",
                "serverid": "s1",
                "quality_description": {
                    "video": {"codec": "h264", "resolution": "1080p"},
                    "audio": {"codec": "eac3", "channels": 6, "languages": ["cze"]},
                    "date_added": "2026-08-08",
                    "path": "/Movies/DiViX/MMW (2025) - 1080p WEB-DL/MMW (2023) - 1080p CZ.mkv",
                },
            },
            "delete": [
                {
                    "id": "del1",
                    "name": "My Mother's Wedding",
                    "path": "/Movies/DiViX/MMW (2025) - 1080p WEB-DL/MMW (2025) - 1080p WEB-DL.mkv",
                    "deletion_result": {"status": status, "error": "co-located keeper"},
                    "quality_description": {
                        "video": {"codec": "hevc", "resolution": "1080p"},
                        "audio": {"codec": "eac3", "channels": 6, "languages": ["eng"]},
                        "date_added": "2025-10-28",
                        "path": "/Movies/DiViX/MMW (2025) - 1080p WEB-DL/MMW (2025) - 1080p WEB-DL.mkv",
                    },
                }
            ],
        }
    ]


def _render(decisions: list[dict]) -> str:
    """Render the real template with poster fetching disabled."""
    with patch("emby_dedupe.reports.images.inline_poster_urls", side_effect=lambda urls, *a, **k: {}):
        return format_html_report(BASE, decisions)


def test_guard_refusal_is_not_reported_as_pending():
    """Defect 2: a protective refusal must not read like a dry run's 'nothing happened'."""
    html = _render(_decision("skipped_unsafe"))

    assert "Blocked by safety guard" in html
    assert ">Pending<" not in html


def test_dry_run_still_reports_pending():
    """The 'Pending' label must survive for the case it actually describes."""
    html = _render(_decision("not_attempted"))

    assert ">Pending<" in html
    assert "Blocked by safety guard" not in html


def test_fold_safe_removal_overrides_the_guard_verdict():
    """Defect 1: a file fold-safe actually removed must not still read as blocked."""
    decisions = _decision("skipped_unsafe")
    results = [{"id": "del1", "delete": decisions[0]["delete"][0]["path"], "status": "removed"}]

    _apply_fold_safe_results_to_decisions(decisions, results)

    assert decisions[0]["delete"][0]["deletion_result"]["status"] == "fold_safe_removed"
    html = _render(decisions)
    assert "fold-safe" in html
    assert "Blocked by safety guard" not in html


def test_only_a_real_removal_clears_the_refusal():
    """needs_review / skipped / failed leave the file on disk — the refusal must stand.

    Silently upgrading these to "deleted" would be the dangerous direction of this fix:
    the report would claim a file is gone while it is still occupying the folder.
    """
    for status in ("needs_review", "skipped", "failed", "would_remove"):
        decisions = _decision("skipped_unsafe")
        results = [{"id": "del1", "delete": decisions[0]["delete"][0]["path"], "status": status}]

        _apply_fold_safe_results_to_decisions(decisions, results)

        assert decisions[0]["delete"][0]["deletion_result"]["status"] == "skipped_unsafe", (
            f"{status!r} must not be reported as removed — the file is still there"
        )


def test_results_match_by_path_when_the_id_is_missing():
    """Fold-safe plans built from a path-only item must still be matched back."""
    decisions = _decision("skipped_unsafe")
    del decisions[0]["delete"][0]["id"]
    path = decisions[0]["delete"][0]["path"]

    _apply_fold_safe_results_to_decisions(decisions, [{"delete": path, "status": "removed"}])

    assert decisions[0]["delete"][0]["deletion_result"]["status"] == "fold_safe_removed"


def test_unrelated_items_are_left_untouched():
    """A result for a different file must never rewrite this item's verdict."""
    decisions = _decision("skipped_unsafe")

    _apply_fold_safe_results_to_decisions(
        decisions, [{"id": "somebody-else", "delete": "/other/file.mkv", "status": "removed"}]
    )

    assert decisions[0]["delete"][0]["deletion_result"]["status"] == "skipped_unsafe"


def test_empty_inputs_are_safe():
    _apply_fold_safe_results_to_decisions([], [])
    _apply_fold_safe_results_to_decisions(None, [])
    _apply_fold_safe_results_to_decisions(_decision("skipped_unsafe"), [])


def test_fold_safe_runs_before_the_report_is_generated():
    """Ordering canary for defect 1.

    The status fix only helps if fold-safe has already run when the report renders.
    That ordering lives in the call sequence of ``dedupe_cmd``, so no amount of
    unit-testing the helpers can catch a refactor that moves reporting back in front —
    this asserts the source order directly.
    """
    import inspect

    from emby_dedupe.cli import app

    body = inspect.getsource(app.dedupe_cmd)
    fold_safe_at = body.index("_run_fold_safe_delete(")
    report_at = body.index("_generate_reports(")

    assert fold_safe_at < report_at, (
        "_generate_reports() must run AFTER _run_fold_safe_delete(), otherwise the report "
        "records the guard's 'blocked' verdict for duplicates fold-safe then removed."
    )
