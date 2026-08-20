"""Tests for the deletion safety guard (emby_dedupe/api/deletion_guard.py).

The guard refuses any Emby delete that would fold-delete a folder containing the
keeper — the exact failure that destroyed 19 keepers in the DV Profile-5 incident.
Scenarios below mirror the real 44-item baseline.
"""
from unittest.mock import MagicMock, patch

from emby_dedupe.api.deduplication import process_deletion_and_generate_report
from emby_dedupe.api.deletion_guard import (
    collect_known_paths,
    is_delete_safe,
)

# --- shorthands for the real layouts ------------------------------------------------
MOVIE_P5 = "/Movies/4K/Alexander (2025) - DoVi SK/Alexander (2025) - DoVi SK.mkv"
MOVIE_KEEP_SIBLING = "/Movies/4K/Alexander (2025) - DoVi SK [HDR10]/Alexander (2025) - DoVi SK [HDR10].mkv"
MOVIE_KEEP_COLOCATED = "/Movies/4K/Alexander (2025) - DoVi SK/Alexander (2025) - DoVi SK [HDR10].mkv"

EP_SUBFOLDER_P5 = "/Movies/Serials/Reacher/S03/Reacher S03E02 - DoVi/Reacher S03E02 - DoVi.mkv"
EP_KEEP_IN_SEASON = "/Movies/Serials/Reacher/S03/Reacher S03E02 - DoVi [HDR10].mkv"
EP_KEEP_COLOCATED = "/Movies/Serials/Reacher/S03/Reacher S03E02 - DoVi/Reacher S03E02 - DoVi [HDR10].mkv"

EP_LOOSE_P5 = "/Movies/Serials/Reacher/S03/Reacher S03E01 - DoVi.mkv"
EP_LOOSE_KEEP = "/Movies/Serials/Reacher/S03/Reacher S03E01 - DoVi [HDR10].mkv"
EP_SIBLING = "/Movies/Serials/Reacher/S03/Reacher S03E03 - DoVi [HDR10].mkv"


# --- the dangerous cases the guard MUST refuse (the 19 lost) -------------------------
def test_movie_keeper_colocated_in_dedicated_folder_is_unsafe():
    safe, reason = is_delete_safe(MOVIE_KEEP_COLOCATED, MOVIE_P5, [MOVIE_KEEP_COLOCATED, MOVIE_P5])
    assert safe is False
    assert "dedicated folder" in reason


def test_episode_keeper_colocated_in_subfolder_is_unsafe():
    paths = [EP_KEEP_COLOCATED, EP_SUBFOLDER_P5]
    safe, reason = is_delete_safe(EP_KEEP_COLOCATED, EP_SUBFOLDER_P5, paths)
    assert safe is False
    assert "dedicated folder" in reason


# --- the post-converter-fix safe layouts (keeper in a different folder) --------------
def test_movie_keeper_in_sibling_folder_is_safe():
    safe, reason = is_delete_safe(MOVIE_KEEP_SIBLING, MOVIE_P5, [MOVIE_KEEP_SIBLING, MOVIE_P5])
    assert safe is True
    assert "different folder" in reason


def test_episode_keeper_in_season_folder_p5_in_subfolder_is_safe():
    paths = [EP_KEEP_IN_SEASON, EP_SUBFOLDER_P5]
    safe, reason = is_delete_safe(EP_KEEP_IN_SEASON, EP_SUBFOLDER_P5, paths)
    assert safe is True
    assert "different folder" in reason


# --- the loose case: file-only delete, keeper survives, must be ALLOWED --------------
def test_loose_pair_in_shared_season_folder_is_safe():
    # other episodes present in the same season folder → Emby deletes only the file
    known = [EP_LOOSE_KEEP, EP_LOOSE_P5, EP_SIBLING]
    safe, reason = is_delete_safe(EP_LOOSE_KEEP, EP_LOOSE_P5, known)
    assert safe is True
    assert "shared folder" in reason


def test_loose_singleton_without_known_siblings_is_conservatively_refused():
    # only the pair itself is known in the folder → can't prove it's shared → refuse (safe default)
    known = [EP_LOOSE_KEEP, EP_LOOSE_P5]
    safe, _ = is_delete_safe(EP_LOOSE_KEEP, EP_LOOSE_P5, known)
    assert safe is False


# --- degenerate inputs ---------------------------------------------------------------
def test_missing_paths_do_not_crash():
    # An unknown DELETE path is REFUSED (changed 2026-08-20). Without it the guard cannot
    # evaluate any of its rules, so allowing was the guard silently switching itself off —
    # 3 of 15 deletes went through unguarded on the 2026-08-20 run because Emby returned
    # no Path. Refusing costs one deferred delete; allowing costs a keeper.
    assert is_delete_safe("", "", [])[0] is False
    assert is_delete_safe(MOVIE_KEEP_SIBLING, None, [])[0] is False
    # An unknown KEEPER path still allows: there is no identified keeper to protect, and
    # the delete path is known so the caller's other rules still applied.
    assert is_delete_safe(None, MOVIE_P5, [])[0] is True


def test_windows_style_paths_normalised():
    keep = r"C:\Media\Movie\Movie [HDR10].mkv"   # same folder as delete
    delete = r"C:\Media\Movie\Movie.mkv"
    safe, _ = is_delete_safe(keep, delete, [keep, delete])
    assert safe is False


# --- the fix: full library visibility must keep the refusal AND lift false positives ---
def test_dedicated_folder_still_refused_with_full_library_visibility():
    """SAFETY (the data-loss direction): even when ``known_paths`` is the *whole* library,
    a keeper co-located alone with the delete in a dedicated folder must STILL be refused.
    Extra paths in OTHER folders (and nested under the dedicated one) must never be
    mistaken for same-folder siblings → no under-refusal."""
    full_library = [
        MOVIE_KEEP_COLOCATED, MOVIE_P5,                 # the dedicated pair
        MOVIE_KEEP_SIBLING,                             # a DIFFERENT folder
        EP_LOOSE_P5, EP_LOOSE_KEEP, EP_SIBLING,         # an unrelated season folder
        "/Movies/4K/Alexander (2025) - DoVi SK/extras/featurette.mkv",  # NESTED, must not count
    ]
    safe, reason = is_delete_safe(MOVIE_KEEP_COLOCATED, MOVIE_P5, full_library)
    assert safe is False
    assert "dedicated folder" in reason


def test_single_duplicate_season_folder_is_safe_with_library_siblings():
    """THE FIX (Dutton S01E07 / Proud S01E03): a loose duplicate pair is the only
    duplicated item in its season folder, so the decision paths alone make the folder
    look dedicated. With the season's other episodes visible (full library paths) the
    guard correctly sees a shared folder → file-only delete → safe."""
    known = [EP_LOOSE_KEEP, EP_LOOSE_P5, EP_SIBLING]  # EP_SIBLING only known via library paths
    safe, reason = is_delete_safe(EP_LOOSE_KEEP, EP_LOOSE_P5, known)
    assert safe is True
    assert "shared folder" in reason


# --- backstop: co-deleted siblings must not vouch a folder "shared" (Marty Supreme) ---
def test_marty_supreme_per_title_folder_refused_regardless_of_siblings():
    """The Marty Supreme data loss (2026-07-01): keeper (REMUX) co-located in a per-title
    folder `Marty Supreme (2025)/` with a P5 + HDR10. Emby fold-deletes the whole directory
    on ANY delete (log-proven), so every co-located delete must be refused — with OR
    without delete_paths, because the per-title trap check is unconditional."""
    keeper = "/Movies/4K/Marty Supreme (2025)/Marty Supreme (2025) - 2160p Remux DoVi CZ.mkv"
    p5 = "/Movies/4K/Marty Supreme (2025)/Marty Supreme (2025) - 2160p WEB-DL DoVi CZ.mkv"
    hdr10 = "/Movies/4K/Marty Supreme (2025)/Marty Supreme (2025) - 2160p WEB-DL DoVi CZ [HDR10].mkv"
    known = [keeper, p5, hdr10]
    safe, reason = is_delete_safe(keeper, p5, known, delete_paths=[p5, hdr10])
    assert safe is False
    assert "dedicated folder" in reason
    assert is_delete_safe(keeper, hdr10, known, delete_paths=[p5, hdr10])[0] is False
    # per-title trap is unconditional — refused even with no delete/run context at all
    assert is_delete_safe(keeper, p5, known)[0] is False


def test_backstop_refuses_non_per_title_folder_when_only_sibling_is_co_deleted():
    """Backstop for a NON-per-title container (folder not named for the movie): keeper
    co-located with a delete, and the only other sibling is ALSO a delete this run → no
    survivor remains → can't prove "shared" → refuse. Without the delete-set info the old
    guard let the co-deleted sibling vouch "shared" and wrongly allowed it."""
    keeper = "/Movies/4K/Collection/Movie A - 2160p.mkv"
    delete = "/Movies/4K/Collection/Movie A - 1080p.mkv"
    codel = "/Movies/4K/Collection/Movie B - 1080p.mkv"  # also being deleted → not a survivor
    known = [keeper, delete, codel]
    assert is_delete_safe(keeper, delete, known, delete_paths=[delete, codel])[0] is False
    assert is_delete_safe(keeper, delete, known)[0] is True  # legacy over-allow the backstop closes


def test_surviving_non_deleted_sibling_still_marks_folder_shared():
    """A neighbour that is NOT being deleted still proves a shared folder → file-only
    delete allowed — BUT only in a non-per-title container (a season folder here, so the
    per-title trap check doesn't fire). The backstop only excludes co-deleted siblings."""
    keeper = "/Movies/Serials/Show/S01/Show S01E01 - 2160p.mkv"
    delete = "/Movies/Serials/Show/S01/Show S01E01 - 1080p.mkv"
    survivor = "/Movies/Serials/Show/S01/Show S01E02 - 2160p.mkv"  # not being deleted
    known = [keeper, delete, survivor]
    safe, reason = is_delete_safe(keeper, delete, known, delete_paths=[delete])
    assert safe is True
    assert "shared folder" in reason


def test_per_title_movie_folder_refused_even_with_surviving_sibling():
    """The Marty class, fully closed: Emby fold-deletes a per-title movie folder on ANY
    delete regardless of what survives in it (log-proven 2026-07-01). A keeper co-located
    in a folder NAMED for the movie must be refused even when a non-deleted sibling is
    present — the surviving-sibling 'shared' test must NOT apply to per-title folders."""
    keeper = "/Movies/4K/Movie (2020)/Movie (2020) - 2160p REMUX.mkv"   # kept
    delete = "/Movies/4K/Movie (2020)/Movie (2020) - 1080p.mkv"          # deleted
    survivor = "/Movies/4K/Movie (2020)/Movie (2020) - 720p.mkv"         # NOT deleted this run
    safe, reason = is_delete_safe(keeper, delete, [keeper, delete, survivor], delete_paths=[delete])
    assert safe is False
    assert "dedicated folder" in reason


# --- collect_known_paths -------------------------------------------------------------
def test_collect_delete_paths_gathers_only_deletes():
    from emby_dedupe.api.deletion_guard import collect_delete_paths
    decisions = [
        {"keep": {"path": "/a/keep.mkv"}, "delete": [{"path": "/a/d1.mkv"}, {"path": "/a/d2.mkv"}]},
        {"keep": {"path": "/b/keep2.mkv"}, "delete": [{"path": None}]},
    ]
    assert set(collect_delete_paths(decisions)) == {"/a/d1.mkv", "/a/d2.mkv"}


def test_collect_known_paths_gathers_keep_and_delete():
    decisions = [
        {"keep": {"path": "/a/keep1.mkv"}, "delete": [{"path": "/a/del1.mkv"}, {"path": "/a/del2.mkv"}]},
        {"keep": {"path": "/b/keep2.mkv"}, "delete": [{"path": None}]},
        {"keep": {}, "delete": []},
    ]
    paths = collect_known_paths(decisions)
    assert set(paths) == {"/a/keep1.mkv", "/a/del1.mkv", "/a/del2.mkv", "/b/keep2.mkv"}


# --- integration: the guard must stop the actual Emby delete call (the incident) -----
def test_process_deletion_skips_colocated_delete_but_runs_safe_one():
    """Regression for the DV-P5 data loss: a co-located keeper must NOT be deleted,
    while a duplicate whose keeper lives in a different folder still is."""
    decisions = [
        {  # UNSAFE: keeper co-located with the P5 in one dedicated movie folder
            "keep": {"id": "kA", "name": "A", "path": "/Movies/4K/Movie/Movie [HDR10].mkv"},
            "delete": [{"id": "dA", "name": "A p5", "path": "/Movies/4K/Movie/Movie.mkv"}],
        },
        {  # SAFE: keeper in a sibling folder, P5 in its own folder
            "keep": {"id": "kB", "name": "B", "path": "/Movies/4K/Film [HDR10]/Film [HDR10].mkv"},
            "delete": [{"id": "dB", "name": "B p5", "path": "/Movies/4K/Film/Film.mkv"}],
        },
    ]
    with patch("emby_dedupe.api.deduplication.delete_item") as mock_del, \
         patch("emby_dedupe.reports.markdown.format_markdown_table", return_value="RPT"), \
         patch("emby_dedupe.api.deduplication.tqdm"):
        mock_del.return_value = {"id": "dB", "status": "success", "error": None}
        result = process_deletion_and_generate_report(
            MagicMock(), "http://emby", decisions, True, "u", "p", "k"
        )

    assert result == "RPT"
    # delete_item was called for the SAFE duplicate only (positional item_id == args[2])
    called_ids = [call.args[2] for call in mock_del.call_args_list]
    assert called_ids == ["dB"]
    # the co-located one was refused by the guard, not deleted
    assert decisions[0]["delete"][0]["deletion_result"]["status"] == "skipped_unsafe"
    assert decisions[1]["delete"][0]["deletion_result"]["status"] == "success"


def _single_duplicate_season_decisions():
    """A loose duplicate pair that is the ONLY duplicated item in its season folder —
    the Dutton/Proud shape that the decision paths alone make look 'dedicated'."""
    return [
        {
            "keep": {"id": "k", "name": "ep", "path": EP_LOOSE_KEEP},
            "delete": [{"id": "d", "name": "ep dup", "path": EP_LOOSE_P5}],
        },
    ]


def test_process_deletion_over_refuses_single_duplicate_folder_without_library_paths():
    """Regression guard for the bug: WITHOUT full library visibility the loose pair's
    folder looks dedicated and the safe file-only delete is wrongly skipped."""
    decisions = _single_duplicate_season_decisions()
    with patch("emby_dedupe.api.deduplication.delete_item") as mock_del, \
         patch("emby_dedupe.reports.markdown.format_markdown_table", return_value="RPT"), \
         patch("emby_dedupe.api.deduplication.tqdm"):
        process_deletion_and_generate_report(
            MagicMock(), "http://emby", decisions, True, "u", "p", "k"
        )
    mock_del.assert_not_called()
    assert decisions[0]["delete"][0]["deletion_result"]["status"] == "skipped_unsafe"


def test_process_deletion_library_paths_lift_single_duplicate_over_refusal():
    """THE FIX end-to-end: passing the season's other episodes (library_paths) makes the
    guard see a shared folder, so the previously-blocked safe delete now runs."""
    decisions = _single_duplicate_season_decisions()
    library_paths = [EP_LOOSE_KEEP, EP_LOOSE_P5, EP_SIBLING]  # EP_SIBLING = the neighbour
    with patch("emby_dedupe.api.deduplication.delete_item") as mock_del, \
         patch("emby_dedupe.reports.markdown.format_markdown_table", return_value="RPT"), \
         patch("emby_dedupe.api.deduplication.tqdm"):
        mock_del.return_value = {"id": "d", "status": "success", "error": None}
        process_deletion_and_generate_report(
            MagicMock(), "http://emby", decisions, True, "u", "p", "k",
            None, library_paths,
        )
    assert [call.args[2] for call in mock_del.call_args_list] == ["d"]
    assert decisions[0]["delete"][0]["deletion_result"]["status"] == "success"


def test_process_deletion_backstop_protects_multi_version_movie_folder():
    """End-to-end backstop for the Marty Supreme loss: a REMUX keeper + P5 + HDR10 all in
    one movie folder, plus a 1080p in its own folder. The two co-located deletes must be
    REFUSED (each other's only sibling is also a delete → folder is effectively dedicated →
    Emby would fold-delete the keeper); the isolated 1080p in a different folder still deletes."""
    folder = "/Movies/4K/Marty Supreme (2025)"
    keeper = f"{folder}/Marty Supreme (2025) - 2160p Remux DoVi CZ.mkv"
    p5 = f"{folder}/Marty Supreme (2025) - 2160p WEB-DL DoVi CZ.mkv"
    hdr10 = f"{folder}/Marty Supreme (2025) - 2160p WEB-DL DoVi CZ [HDR10].mkv"
    hd1080 = "/Movies/HD/Marty Supreme (2025) - 1080p WEB-DL CZ/Marty Supreme (2025) - 1080p WEB-DL CZ.mkv"
    decisions = [{
        "keep": {"id": "k", "name": "Marty", "path": keeper},
        "delete": [
            {"id": "p5", "name": "p5", "path": p5},
            {"id": "hdr", "name": "hdr10", "path": hdr10},
            {"id": "hd", "name": "1080p", "path": hd1080},
        ],
    }]
    with patch("emby_dedupe.api.deduplication.delete_item") as mock_del, \
         patch("emby_dedupe.reports.markdown.format_markdown_table", return_value="RPT"), \
         patch("emby_dedupe.api.deduplication.tqdm"):
        mock_del.return_value = {"id": "hd", "status": "success", "error": None}
        process_deletion_and_generate_report(
            MagicMock(), "http://emby", decisions, True, "u", "p", "k"
        )
    status = {d["id"]: d["deletion_result"]["status"] for d in decisions[0]["delete"]}
    assert status["p5"] == "skipped_unsafe"   # co-located; only sibling (HDR10) is co-deleted
    assert status["hdr"] == "skipped_unsafe"  # co-located; only sibling (P5) is co-deleted
    assert status["hd"] == "success"          # isolated in its own folder → safe
    assert [call.args[2] for call in mock_del.call_args_list] == ["hd"]


def test_converter_and_guard_agree_a_per_title_folder_is_a_trap(tmp_path):
    """Cross-module drift canary. The converter (`dv_common._owns_folder`) and the guard
    (`is_delete_safe`) encode the SAME per-title rule (`stem.startswith(folder)`) in two
    separate files — the converter decides where to place the HDR10, the guard decides
    whether to clear a delete. If one drifts, the converter places into / the guard clears
    a folder the other treats as safe → another Marty-class fold-delete. This fails loudly
    if they ever disagree on the exact Marty layout (a comment can't; a test can)."""
    import importlib.util
    from pathlib import Path

    dv_path = Path(__file__).resolve().parents[3] / "scripts" / "dv" / "dv_common.py"
    spec = importlib.util.spec_from_file_location("dv_common_canary", dv_path)
    dvc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dvc)

    folder = tmp_path / "4K" / "Marty Supreme (2025)"
    folder.mkdir(parents=True)
    p5 = folder / "Marty Supreme (2025) - 2160p WEB-DL DoVi CZ.mkv"
    keeper = folder / "Marty Supreme (2025) - 2160p Remux DoVi CZ.mkv"
    p5.touch()
    keeper.touch()

    # converter: this per-title folder is "owned" → HDR10 must go to a NEW folder, not loose here
    assert dvc._owns_folder(str(p5)) is True
    # guard: keeper co-located in that same per-title folder → refuse the delete
    safe, reason = is_delete_safe(str(keeper), str(p5), [str(keeper), str(p5)])
    assert safe is False
    assert "dedicated folder" in reason


class TestCleanupDeleteGuard:
    """The cleanup path had NO fold-delete guard: cli/cleanup.py called delete_item()
    directly, so deleting a movie that shares its folder with keepers could fold-delete
    the whole directory. Live case 2026-07-28: Victoria's Secret 2014 sat loose in a
    folder holding 13 editions, 9 of them keepers.
    """

    VS = "/Movies/Dokumenty/The Victoria's Secret Fashion Show - 720p x264"

    def test_loose_file_in_shared_folder_is_file_only_delete(self):
        from emby_dedupe.api.deletion_guard import is_cleanup_delete_safe

        delete = f"{self.VS}/The Victoria's Secret Fashion Show (2014) - 720p x264.mp4"
        known = [
            delete,
            f"{self.VS}/The Victoria's Secret Fashion Show (2005) - 720p x264.mp4",
            f"{self.VS}/The Victoria's Secret Fashion Show (2010) - 720p x264.mkv",
            f"{self.VS}/The Victoria's Secret Fashion Show 2017/x.mkv",
        ]
        safe, reason = is_cleanup_delete_safe(delete, known, [delete])
        assert safe, reason
        assert "file only" in reason

    def test_per_title_folder_with_a_survivor_is_refused(self):
        """A folder named for the title is fold-deleted wholesale — never safe."""
        from emby_dedupe.api.deletion_guard import is_cleanup_delete_safe

        d = "/Movies/Dokumenty/Some Doc (2019)"
        delete = f"{d}/Some Doc (2019) - 1080p.mkv"
        known = [delete, f"{d}/Some Doc (2019) - extras.mkv"]
        safe, reason = is_cleanup_delete_safe(delete, known, [delete])
        assert not safe
        assert "fold-delete" in reason or "dedicated folder" in reason

    def test_nested_survivor_blocks_the_delete(self):
        """A survivor in a SUBfolder doesn't stop Emby fold-deleting the parent."""
        from emby_dedupe.api.deletion_guard import is_cleanup_delete_safe

        d = "/Movies/Dokumenty/Container"
        delete = f"{d}/loose.mkv"
        known = [delete, f"{d}/Sub/keeper.mkv"]
        safe, _ = is_cleanup_delete_safe(delete, known, [delete])
        assert not safe

    def test_folder_where_everything_is_being_deleted_is_safe(self):
        from emby_dedupe.api.deletion_guard import is_cleanup_delete_safe

        d = "/Movies/Dokumenty/Doomed"
        a, b = f"{d}/a.mkv", f"{d}/b.mkv"
        safe, reason = is_cleanup_delete_safe(a, [a, b], [a, b])
        assert safe
        assert "survives" in reason

    def test_sibling_that_is_also_a_delete_target_does_not_vouch_shared(self):
        """Two co-located deletes must not vouch for each other while a keeper sits there."""
        from emby_dedupe.api.deletion_guard import is_cleanup_delete_safe

        d = "/Movies/Dokumenty/Mixed"
        keep = f"{d}/Mixed/keeper.mkv"          # nested keeper
        a, b = f"{d}/a.mkv", f"{d}/b.mkv"
        safe, _ = is_cleanup_delete_safe(a, [a, b, keep], [a, b])
        assert not safe

    def test_missing_path_is_refused(self):
        """Changed 2026-08-20: an unusable path is refused, not waved through.

        It used to return safe, which meant a candidate whose Path Emby did not return
        was deleted with no fold check at all.
        """
        from emby_dedupe.api.deletion_guard import is_cleanup_delete_safe

        assert is_cleanup_delete_safe(None, [], [])[0] is False
        assert is_cleanup_delete_safe("", ["/x/y.mkv"], [])[0] is False
        assert is_cleanup_delete_safe("unknown", ["/x/y.mkv"], [])[0] is False
