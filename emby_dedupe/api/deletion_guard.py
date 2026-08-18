"""Safety guard for Emby item deletion — the seatbelt against fold-delete data loss.

Emby's ``DELETE /Items/{id}`` removes the item's media file, AND if that file owns a
dedicated folder (a movie folder, or a per-episode subfolder) it removes the **whole
folder** too. When a *keeper* (e.g. a clean HDR10 copy) was co-located in that folder,
deleting the duplicate destroyed the keeper as well. (Proven live 2026-06-25 — see the
``reference_emby_delete_folder_rule`` memory; this is what wiped 19 keepers in the DV
Profile-5 incident.)

This module decides, from paths alone, whether deleting an item could destroy its
keeper. It NEVER deletes anything and NEVER touches the filesystem — it just lets the
deletion code *decline an unsafe Emby call*. The converter places new keepers in safe
locations; this guard is the backstop that guarantees the keeper survives even if a
co-located pair slips through (old files, manual ops, a converter miss).
"""
import logging
import posixpath

logger = logging.getLogger(__name__)


def _norm(path: str | None) -> str:
    """Normalise a path for comparison (forward slashes, no trailing slash)."""
    if not path:
        return ""
    return path.replace("\\", "/").rstrip("/")


def _under(path: str | None, folder: str | None) -> bool:
    """True if the file at ``path`` lives directly in or under ``folder``."""
    path, folder = _norm(path), _norm(folder)
    if not path or not folder:
        return False
    return path == folder or path.startswith(folder + "/")


def is_delete_safe(
    keeper_path: str | None, delete_path: str | None, known_paths, delete_paths=None
) -> tuple[bool, str]:
    """Decide whether deleting the item at ``delete_path`` via Emby can destroy the
    keeper at ``keeper_path``.

    Args:
        keeper_path: filesystem path of the item being KEPT for this duplicate group.
        delete_path: filesystem path of the item about to be deleted.
        known_paths: every other media path the run can see — ideally *every* media path
            in the library (see ``fetch_all_media_paths``), not just the duplicated items,
            so a folder's non-duplicate neighbours are visible. That is what tells a
            *dedicated* folder (Emby fold-deletes it) from a *shared* one (Emby deletes
            only the file). Decision paths alone make a single-duplicate folder look
            dedicated and over-refuse (safe, but blocks valid deletes).
        delete_paths: every path being deleted in THIS run. A same-folder sibling that is
            itself a delete target does not prove the folder is shared — after the run it
            is gone too — so it must not count as a surviving neighbour. Without this, a
            movie folder holding a keeper + two co-located deletes (each the other's
            "sibling") looks shared and the delete is wrongly allowed → Emby fold-deletes
            the whole folder and destroys the keeper (proven live 2026-07-01, Marty
            Supreme). Omit (None) to treat all ``known_paths`` as survivors (legacy).

    Returns:
        (safe: bool, reason: str). ``safe`` False means: refuse the Emby delete,
        because it would fold-delete a folder containing the keeper.
    """
    dp, kp = _norm(delete_path), _norm(keeper_path)
    if not dp:
        return True, "no delete path — nothing to reason about"
    if not kp:
        return True, "no keeper path to protect"

    ddir = posixpath.dirname(dp)
    if not _under(kp, ddir):
        return True, f"keeper is in a different folder than {ddir!r}"

    # PER-TITLE FOLDER: the folder is named for the movie/episode being deleted (the
    # file stem starts with the folder name). Emby fold-deletes the ENTIRE directory on
    # ANY item delete here, regardless of what else survives in it — proven live
    # 2026-07-01: deleting Marty's HDR10 logged "Deleting directory /Movies/4K/Marty
    # Supreme (2025)" while the REMUX + P5 were STILL present. With the keeper inside such
    # a folder there is NO safe delete, so refuse outright without consulting siblings.
    # This is the SAME rule the converter uses (dv_common._owns_folder) — keep in sync:
    # both must treat a per-title folder as a trap, or one will place/clear what the other
    # then destroys.
    folder = posixpath.basename(ddir)
    delete_stem = posixpath.splitext(posixpath.basename(dp))[0]
    if folder and delete_stem.startswith(folder):
        return (
            False,
            f"keeper co-located in a dedicated folder named for this title ({ddir!r}); "
            "Emby fold-deletes the whole directory on any delete and destroys the keeper",
        )

    # Not a per-title folder (a season/collection/category container). Deleting here is
    # safe ONLY if Emby removes just the file — i.e. the folder is shared with other
    # distinct media that SURVIVES the run. Two refinements, both only ever over-refuse:
    #   * DIRECT children only — a media item nested in a SUBfolder does not stop Emby
    #     fold-deleting ddir, so it must not count as "shared".
    #   * SURVIVING children only — a sibling that is itself being deleted this run is
    #     gone afterwards, so it cannot keep the folder alive; exclude the run's other
    #     delete targets. (Without this, a movie folder with keeper + 2 co-located
    #     deletes mutually vouch each other "shared" and the keeper is fold-deleted.)
    delete_set = {_norm(p) for p in (delete_paths or [])}
    others = [
        p for p in known_paths
        if _norm(p) not in (dp, kp)
        and _norm(p) not in delete_set
        and posixpath.dirname(_norm(p)) == ddir
    ]
    if others:
        return True, f"shared folder ({len(others)} surviving item(s)) → Emby deletes file only"

    return (
        False,
        f"keeper co-located in a dedicated folder ({ddir!r}); Emby would fold-delete it "
        "and destroy the keeper",
    )


def collect_known_paths(decisions) -> list:
    """Gather every keep + delete media path across all decision groups.

    This is the *minimal* path set (duplicated items only). For complete folder
    visibility the caller unions in every library path (``fetch_all_media_paths``);
    decision paths alone can leave a single-duplicate folder looking "dedicated" and
    make the guard over-refuse a safe file-only delete.
    """
    paths = []
    for decision in decisions or []:
        keep = decision.get("keep") or {}
        if keep.get("path"):
            paths.append(keep["path"])
        for item in decision.get("delete", []) or []:
            if item.get("path"):
                paths.append(item["path"])
    return paths


def collect_delete_paths(decisions) -> list:
    """Gather every path being DELETED across all decision groups.

    Passed to :func:`is_delete_safe` so a same-folder sibling that is itself a delete
    target does not falsely vouch that the folder is "shared" (it won't survive the run).
    """
    paths = []
    for decision in decisions or []:
        for item in decision.get("delete", []) or []:
            if item.get("path"):
                paths.append(item["path"])
    return paths


# A series folder must sit at least this deep to be deletable. Guards against a
# mis-constructed candidate handing us a library root (e.g. "/Movies/Serials") and
# green-lighting removal of an ENTIRE library. Real series live at
# /Movies/Serials/--- UKONCENE ---/<Show> (5 components), so 4 is generous while still
# rejecting the catastrophic shapes.
_MIN_CONTAINER_DEPTH = 4


def _is_container_delete_safe(dp: str) -> tuple[bool, str]:
    """Decide whether deleting a FOLDER-backed item (a Series) is safe.

    PROVEN LIVE 2026-08-18 on emby-gpu with a throwaway library: deleting a Series item
    removes **exactly that series' own folder** and nothing else — a sibling series in the
    same container, and the container itself, were both untouched. So the only media at
    risk is what lives inside the series folder, which is precisely what the run intends
    to delete.

    The file-based reasoning in :func:`is_cleanup_delete_safe` cannot express that. For a
    file, ``dirname(path)`` is the folder Emby may fold-delete; for a series folder it
    climbs one level too far — to the LIBRARY container
    (e.g. ``/Movies/Serials/--- UKONCENE ---``). That container's direct children are
    folders, never media files, so the direct-children survivor test found nothing every
    single time and refused **every** series: 4 of 4 on the 2026-08-18 run (50.94 GB), and
    every run before it. Series cleanup had never once been able to delete anything.
    """
    depth = len([part for part in dp.split("/") if part])
    if depth < _MIN_CONTAINER_DEPTH:
        return (
            False,
            f"refusing to delete {dp!r}: too shallow to be a series folder "
            f"({depth} path components, need >= {_MIN_CONTAINER_DEPTH}) — this looks like "
            "a library root, and deleting it would remove the whole library",
        )
    return (
        True,
        f"series folder ({dp!r}) — Emby removes only this folder; siblings in the same "
        "container are unaffected (proven live 2026-08-18)",
    )


def is_cleanup_delete_safe(
    delete_path: str | None, known_paths, delete_paths=None,
    delete_is_container: bool = False,
) -> tuple[bool, str]:
    """Decide whether deleting ``delete_path`` can destroy media the run is KEEPING.

    The cleanup path has no per-group "keeper" the way dedup does — the thing to protect
    is *anything that survives the run* in the same folder. Emby fold-deletes an item's
    owning directory, so removing one movie can take unrelated titles with it when they
    share that directory. Hit live on 2026-07-28: a Victoria's Secret edition sat loose in
    a folder holding 13 editions, 9 of them keepers.

    Every survivor under the delete's directory is checked with the SAME rules dedup uses
    (:func:`is_delete_safe`) — per-title-folder trap, direct-children-only, and
    surviving-children-only — so both paths stay consistent by construction.

    Args:
        delete_path: filesystem path of the item about to be deleted.
        known_paths: every media path in the library (``fetch_all_media_paths``). Without
            full visibility a shared folder can look dedicated and the guard over-refuses.
        delete_paths: every path being deleted in THIS run, so a same-folder sibling that
            is itself a target does not count as a survivor.
        delete_is_container: True when ``delete_path`` is a FOLDER Emby owns as one item
            (a Series) rather than a media file. The file-based reasoning below is simply
            wrong for those — see :func:`_is_container_delete_safe`.

    Returns:
        (safe, reason). ``safe`` False means refuse the Emby delete — it would fold-delete
        a directory containing media that is meant to survive.
    """
    dp = _norm(delete_path)
    if not dp:
        return True, "no delete path — nothing to reason about"

    if delete_is_container:
        return _is_container_delete_safe(dp)

    ddir = posixpath.dirname(dp)
    delete_set = {_norm(p) for p in (delete_paths or [])}
    survivors = [
        p for p in known_paths
        if _norm(p) != dp and _norm(p) not in delete_set and _under(_norm(p), ddir)
    ]
    if not survivors:
        return True, "nothing in this folder survives the run — fold-delete harms nothing"

    # Delegate to the dedup guard once per survivor; any single unsafe verdict refuses.
    for survivor in survivors:
        safe, reason = is_delete_safe(survivor, delete_path, known_paths, delete_paths)
        if not safe:
            return False, reason
    return True, f"shared folder ({len(survivors)} surviving item(s)) → Emby deletes file only"
