"""Series cleanup must actually be able to delete a series.

Regression for 2026-08-18. `is_cleanup_delete_safe` refused **every** series ever
submitted to it — 4 of 4 (50.94 GB) on the 2026-08-18 run, and every run before. Series
cleanup had never once deleted anything, and the reports said only "Skipped" with no
reason, so it looked like there was simply nothing to do.

Cause: for a media FILE, ``dirname(path)`` is the folder Emby may fold-delete. For a
SERIES the path is already a folder, so ``dirname`` climbs one level too far — to the
library container (``/Movies/Serials/--- UKONCENE ---``). That container's direct children
are folders, never media files, so the direct-children survivor test found nothing every
time and fell through to "dedicated folder → refuse".

Ground truth (probe on emby-gpu, throwaway library, 2026-08-18): deleting a Series item
removes exactly that series' own folder; a sibling series and the container both survived.
"""

from emby_dedupe.api.deletion_guard import is_cleanup_delete_safe

UK = "/Movies/Serials/--- UKONCENE ---"
SERIES = f"{UK}/A Series of Unfortunate Events"

# A real library: every known path is an episode FILE nested under <Show>/<Season>/.
# Nothing is ever a direct child of the container — that is the whole bug.
KNOWN = [
    f"{SERIES}/S01/A Series of Unfortunate Events S01E01 - 1080p.mkv",
    f"{SERIES}/S01/A Series of Unfortunate Events S01E02 - 1080p.mkv",
    f"{UK}/Some Other Show/S01/Some Other Show S01E01 - 1080p.mkv",
    f"{UK}/Third Show/S02/Third Show S02E05 - 1080p.mkv",
]


def test_series_delete_is_allowed():
    """The exact 2026-08-18 false positive: this refused, making series cleanup inert."""
    safe, reason = is_cleanup_delete_safe(
        SERIES, KNOWN, [SERIES], delete_is_container=True
    )

    assert safe, f"series cleanup is inert again: {reason}"
    assert "only this folder" in reason


def test_every_skipped_series_from_the_live_run_is_now_deletable():
    """All 4 series the 2026-08-18 run refused (50.94 GB)."""
    live = [
        f"{UK}/A Series of Unfortunate Events",
        f"{UK}/The Salisbury Poisonings",
        f"{UK}/Hit and Run",
        "/Movies/Serials/--- PREBIEHAJUCE ---/Two Sides of the Abyss",
    ]
    refused = [p for p in live
               if not is_cleanup_delete_safe(p, KNOWN, live, delete_is_container=True)[0]]

    assert not refused, f"still refused: {refused}"


def test_a_library_root_is_never_deletable():
    """The catastrophic shape: a mis-built candidate handing us a library root.

    Allowing this would delete an entire library, so it must be refused no matter what.
    """
    # "/" is absent here on purpose: it normalises to "" and takes the no-path branch
    # below, not the depth check.
    for root in ("/Movies", "/Movies/Serials", UK):
        safe, _ = is_cleanup_delete_safe(root, KNOWN, [root], delete_is_container=True)
        assert not safe, f"{root!r} must never be deletable as a container"


def test_absent_path_is_refused_even_for_a_container():
    """Superseded 2026-08-18 -> 2026-08-20.

    This originally pinned "absent path returns safe" as a pre-existing contract, with a
    note that it let a candidate with no ``path`` bypass the guard entirely. Two days
    later that exact bypass fired on a live run, so the contract was inverted rather than
    documented: no usable path now means refuse, on both the movie and container branches.
    """
    assert is_cleanup_delete_safe("", KNOWN, [], delete_is_container=True)[0] is False
    assert is_cleanup_delete_safe(None, KNOWN, [], delete_is_container=True)[0] is False


def test_movies_keep_the_file_based_rules():
    """The container branch must not leak into the movie path.

    A movie co-located with survivors in a per-title folder is still a fold trap; that is
    the Victoria's Secret / Marty Supreme protection and it must be untouched.
    """
    d = "/Movies/Dokumenty/Some Doc (2019)"
    delete = f"{d}/Some Doc (2019) - 1080p.mkv"
    known = [delete, f"{d}/Some Doc (2019) - extras.mkv"]

    safe, _ = is_cleanup_delete_safe(delete, known, [delete])

    assert not safe, "movie fold-trap protection regressed"


def test_container_flag_defaults_off():
    """Callers that don't opt in keep the exact previous behaviour."""
    assert is_cleanup_delete_safe(SERIES, KNOWN, [SERIES])[0] is False


def test_cli_passes_the_container_flag_for_series_only():
    """Wiring canary. The guard fix is inert unless the CLI opts series in.

    Asserts the source directly: the flag is derived from the run label and handed to the
    guard. A unit test of the guard alone cannot catch the caller forgetting to pass it —
    which is precisely the failure that would silently keep series cleanup dead.
    """
    import inspect

    from emby_dedupe.cli import cleanup

    src = inspect.getsource(cleanup._perform_deletions)

    assert 'is_series = label == "series"' in src
    assert "delete_is_container=is_series" in src


def test_unknown_delete_path_is_refused_not_waved_through():
    """Regression for the 2026-08-20 silent bypass.

    Emby returned no ``Path`` for 3 of 15 delete items (the report rendered "unknown").
    The guard's old "no delete path -> nothing to reason about -> safe" turned those into
    unguarded deletes: no fold check, no warning, no log. They happened to sit in a season
    folder (file-only, harmless), but the same item in a per-title movie folder next to its
    keeper is exactly the Marty Supreme data loss.
    """
    from emby_dedupe.api.deletion_guard import is_delete_safe

    keeper = "/Movies/4K/Some Movie (2024)/Some Movie (2024) - 2160p.mkv"

    for missing in (None, "", "   "):
        safe, reason = is_delete_safe(keeper, missing, [keeper])
        assert not safe, f"delete path {missing!r} must be refused, not waved through"
        assert "unknown" in reason

    # cleanup path shares the helper, so it must refuse too
    safe, _ = is_cleanup_delete_safe(None, KNOWN, [])
    assert not safe
