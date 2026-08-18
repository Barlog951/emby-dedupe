"""
Library cleanup command — identify and optionally remove unwatched stale media.

Handles both movies and TV series. Libraries are auto-probed for content type
and the appropriate pipeline is run for each.

Movies pass through a 7-layer filter pipeline (age, exclusion, play/interest,
actors, franchise, path, rating decay).

Series pass through a 5-layer filter pipeline (staleness, exclusion,
play/favorites, path, rating decay). Staleness is measured as years since
the last episode was added to Emby.

Both use a dynamic rating decay model: older items must have a higher
community rating to be protected.
"""

from __future__ import annotations

import json

import httpx
from tqdm import tqdm

import emby_dedupe.api.client as _client_mod
from emby_dedupe.api.cleanup_pipeline import (
    _probe_library_content,
    _resolve_primary_user_id,
    _run_cleanup_pipeline,
    _run_series_cleanup_pipeline,
)
from emby_dedupe.api.client import (
    check_emby_connection,
    delete_item,
    fetch_all_media_paths,
    handle_host_and_port,
    logout,
)
from emby_dedupe.api.deletion_guard import is_cleanup_delete_safe
from emby_dedupe.api.search import get_all_library_ids, get_library_ids_by_name
from emby_dedupe.models.cleanup import (
    _DEFAULT_PROTECT_PATH,
    CleanupCandidate,
    CleanupConfig,
    SeriesCleanupCandidate,
)
from emby_dedupe.reports.cleanup import (
    _format_cleanup_report_console,
    _format_cleanup_report_json,
    _generate_cleanup_html_report,
    _save_cleanup_html_report,
)
from emby_dedupe.utils.http import make_http_request
from emby_dedupe.utils.logging import logger

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_cleanup_args(
    host: str | None,
    api_key: str | None,
    libraries: list[str],
    all_libraries: bool,
    doit: bool,
    username: str | None,
    password: str | None,
) -> None:
    """Validate cleanup command arguments.

    Custom validation (not validate_required_arguments) to support --all-libraries
    and the nuanced username requirement (DA fix #4, #11).

    Args:
        host: Emby server URL.
        api_key: Emby API key.
        libraries: List of library names.
        all_libraries: If True, libraries list may be empty.
        doit: If True, deletions will be performed.
        username: Emby username (required for --doit, recommended always).
        password: Emby password (required for --doit).

    Raises:
        SystemExit: If required arguments are missing.
    """
    import sys

    errors = []
    if not host:
        errors.append("--host / DEDUPE_EMBY_HOST is required.")
    if not api_key:
        errors.append("--api-key / DEDUPE_EMBY_API_KEY is required.")
    if not libraries and not all_libraries:
        errors.append("Specify at least one --library / -l or use --all-libraries.")
    if doit and not username:
        errors.append("--username is required when --doit is set (needed for DELETE auth).")
    if doit and not password:
        errors.append("--password is required when --doit is set (needed for DELETE auth).")

    if errors:
        for err in errors:
            logger.error(err)
        sys.exit(1)

    if not doit and not username:
        logger.warning(
            "No --username provided; favorite-actor protection will use first Emby user. "
            "Recommend: --username Barlog for accurate results."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_excluded_ids(raw: str) -> set[str]:
    """Parse comma-separated provider IDs into a set.

    Args:
        raw: Comma-separated string of provider IDs.

    Returns:
        Set of stripped, non-empty provider ID strings.
    """
    return {s.strip() for s in raw.split(",") if s.strip()} if raw else set()


def _normalize_protect_paths(raw: str | list | tuple) -> list[str]:
    """Normalize protect_paths input to a list of non-empty path strings.

    Args:
        raw: Protect paths as a comma-string, list, or tuple.

    Returns:
        List of non-empty path strings, defaulting to [_DEFAULT_PROTECT_PATH].
    """
    if isinstance(raw, str):
        paths = [p.strip() for p in raw.split(",") if p.strip()]
    elif isinstance(raw, (list, tuple)):
        paths = list(raw)
    else:
        paths = []
    return paths or [_DEFAULT_PROTECT_PATH]


def _probe_and_split_libraries(
    client: httpx.Client,
    base_url: str,
    library_ids: list[str],
    lib_id_to_name: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Probe libraries to split into movie-containing and series-containing lists.

    Args:
        client: Configured httpx client with auth headers.
        base_url: Emby server base URL.
        library_ids: Library IDs to probe.
        lib_id_to_name: Mapping of library ID to display name.

    Returns:
        Tuple of (movie_lib_ids, series_lib_ids).
    """
    movie_lib_ids: list[str] = []
    series_lib_ids: list[str] = []

    for lib_id in library_ids:
        movie_count, series_count = _probe_library_content(client, base_url, lib_id)
        lib_name = lib_id_to_name.get(lib_id, lib_id)
        if movie_count > 0:
            movie_lib_ids.append(lib_id)
        if series_count > 0:
            series_lib_ids.append(lib_id)
        logger.info(f"Library '{lib_name}': {movie_count} movies, {series_count} series")

    return movie_lib_ids, series_lib_ids


def _perform_deletions(
    client: httpx.Client,
    base_url: str,
    candidates: list,
    username: str | None,
    password: str | None,
    api_key: str | None,
    label: str = "movies",
) -> None:
    """Delete cleanup candidates from Emby and record results.

    Args:
        client: Configured httpx client with auth headers.
        base_url: Emby server base URL.
        candidates: List of CleanupCandidate or SeriesCleanupCandidate objects.
        username: Emby username for DELETE auth.
        password: Emby password for DELETE auth.
        api_key: Emby API key.
        label: Label for progress bar (e.g. "movies" or "series").

    Raises:
        ValueError: If username, password, or api_key is missing (deletions
            require full auth; enforced earlier by _validate_cleanup_args).
    """
    if username is None or password is None or api_key is None:
        raise ValueError("username, password and api_key are required for deletions")

    # Emby fold-deletes an item's owning directory, so a candidate sharing its folder with
    # media the run is KEEPING can take that media with it. Fetch every library path once
    # and gate each delete on the same guard the dedup path uses.
    known_paths = fetch_all_media_paths(client, base_url)
    delete_paths = [c.path for c in candidates if getattr(c, "path", None)]
    # A series candidate's path is a FOLDER Emby owns as one item, not a media file, so it
    # needs the container rule rather than the file one (which refused 100% of series).
    is_series = label == "series"

    logger.info(f"Deleting {len(candidates)} {label} candidates...")
    with tqdm(candidates, desc=f"Deleting {label}", unit=label.rstrip("s")) as progress:
        for candidate in progress:
            progress.set_postfix_str(candidate.name[:40])
            safe, reason = is_cleanup_delete_safe(
                getattr(candidate, "path", None), known_paths, delete_paths,
                delete_is_container=is_series,
            )
            if not safe:
                candidate.deletion_result = {"status": "skipped_unsafe", "error": reason}
                logger.error(
                    "SAFETY GUARD blocked cleanup deletion of %r — %s (path=%r). "
                    "Item kept; resolve the layout manually.",
                    candidate.name, reason, getattr(candidate, "path", None),
                )
                continue
            result = delete_item(
                client, base_url, candidate.item_id,
                username=username, password=password, api_key=api_key,
            )
            candidate.deletion_result = result
            logger.info(f"Deleted {candidate.name}: {result.get('status', 'unknown')}")


def _resolve_library_ids(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    libraries: list[str],
    all_libraries: bool,
) -> tuple[list[str], dict[str, str]]:
    """Resolve library IDs and build a name mapping.

    Args:
        client: Configured httpx client with auth headers.
        base_url: Emby server base URL.
        api_key: Emby API key.
        libraries: List of library names (empty if all_libraries is True).
        all_libraries: If True, use all libraries.

    Returns:
        Tuple of (library_ids, lib_id_to_name).
    """
    all_lib_infos = get_all_library_ids(client, base_url, api_key)
    lib_id_to_name: dict[str, str] = {
        lib["id"]: lib["name"] for lib in all_lib_infos if lib.get("id") and lib.get("name")
    }

    if all_libraries:
        library_ids = list(lib_id_to_name.keys())
    else:
        library_ids = get_library_ids_by_name(client, base_url, api_key, libraries)

    return library_ids, lib_id_to_name


_EMPTY_MOVIE_STATS: dict = {
    "total_analyzed": 0, "age_filtered": 0, "excluded_filtered": 0,
    "play_protected": 0, "interest_protected": 0, "actor_protected": 0,
    "franchise_protected": 0, "path_protected": 0, "rating_protected": 0,
    "final_candidates": 0,
}


def _output_report(
    output_format: str,
    candidates: list[CleanupCandidate],
    protection_stats: dict,
    config: CleanupConfig,
    series_candidates: list[SeriesCleanupCandidate] | None,
    series_stats: dict | None,
    movie_near_miss: list[CleanupCandidate] | None = None,
    series_near_miss: list[SeriesCleanupCandidate] | None = None,
) -> None:
    """Output cleanup report in the requested format (console or JSON).

    Args:
        output_format: "json" or "console".
        candidates: Movie candidates.
        protection_stats: Movie filter stage counts.
        config: Cleanup configuration.
        series_candidates: Series candidates (None if no series scanned).
        series_stats: Series filter stage counts.
        movie_near_miss: Movies protected only by rating.
        series_near_miss: Series protected only by rating.
    """
    series_for_report = series_candidates if series_stats else None
    if output_format == "json":
        report_data = _format_cleanup_report_json(
            candidates, protection_stats, config,
            series_candidates=series_for_report, series_stats=series_stats,
        )
        print(json.dumps(report_data, indent=2, default=str))
    else:
        _format_cleanup_report_console(
            candidates, protection_stats, config,
            series_candidates=series_for_report, series_stats=series_stats,
            movie_near_miss=movie_near_miss, series_near_miss=series_near_miss,
        )


def _execute_cleanup(
    client: httpx.Client,
    base_url: str,
    config: CleanupConfig,
    api_key: str,
    libraries: list[str],
    all_libraries: bool,
    username: str | None,
    password: str | None,
    output_format: str,
    html_report: bool,
    html_only: bool,
    no_open: bool,
    doit: bool,
) -> None:
    """Execute the full cleanup workflow after connection is established.

    Args:
        client: Configured httpx client with auth headers.
        base_url: Emby server base URL.
        config: Cleanup configuration.
        api_key: Emby API key.
        libraries: Library names to scan.
        all_libraries: If True, scan all libraries.
        username: Emby username.
        password: Emby password.
        output_format: "json" or "console".
        html_report: Whether to generate HTML report.
        html_only: HTML-only mode (no console, no browser).
        no_open: Whether to suppress browser auto-open.
        doit: Whether to perform actual deletions.
    """
    server_id = make_http_request(client, "GET", f"{base_url}/System/Info").json().get("Id", "")

    primary_user_id = _resolve_primary_user_id(client, base_url, username)
    if not primary_user_id:
        logger.error("Cannot resolve a valid Emby user ID; aborting cleanup.")
        return

    library_ids, lib_id_to_name = _resolve_library_ids(
        client, base_url, api_key, libraries, all_libraries
    )
    if not library_ids:
        logger.error("No library IDs resolved. Check library names and permissions.")
        return

    movie_lib_ids, series_lib_ids = _probe_and_split_libraries(
        client, base_url, library_ids, lib_id_to_name
    )

    # Run pipelines
    candidates, protection_stats, movie_near_miss = _run_cleanup_pipeline(
        client, base_url, config, movie_lib_ids, primary_user_id,
        lib_id_to_name=lib_id_to_name,
    ) if movie_lib_ids else ([], dict(_EMPTY_MOVIE_STATS), [])

    series_candidates, series_stats, series_near_miss = _run_series_cleanup_pipeline(
        client, base_url, config, series_lib_ids, primary_user_id,
        lib_id_to_name=lib_id_to_name,
    ) if series_lib_ids else ([], None, [])

    _output_report(
        output_format, candidates, protection_stats, config,
        series_candidates, series_stats,
        movie_near_miss=movie_near_miss, series_near_miss=series_near_miss,
    )

    if doit and candidates:
        _perform_deletions(client, base_url, candidates, username, password, api_key, "movies")
    if doit and series_candidates:
        _perform_deletions(client, base_url, series_candidates, username, password, api_key, "series")

    if html_report or html_only:
        series_for_report = series_candidates if series_stats else None
        html_content = _generate_cleanup_html_report(
            base_url, candidates, protection_stats, config, doit,
            server_id=server_id, api_key=api_key,
            series_candidates=series_for_report, series_stats=series_stats,
            movie_near_miss=movie_near_miss, series_near_miss=series_near_miss,
        )
        report_path = _save_cleanup_html_report(html_content, no_open=no_open)
        print(f"\nHTML report: {report_path}")


def run_cleanup_command(args) -> None:
    """Entry point for the cleanup subcommand.

    Orchestrates: validation -> connection -> pipeline -> report -> optional
    deletion -> HTML report -> logout.

    Args:
        args: Namespace with cleanup arguments (from argparse.Namespace or typer ctx).
    """
    host = getattr(args, "host", None)
    port = getattr(args, "port", None)
    api_key = getattr(args, "api_key", None)
    libraries = getattr(args, "library", []) or []
    all_libraries = getattr(args, "all_libraries", False)
    doit = getattr(args, "doit", False)
    username = getattr(args, "username", None)
    password = getattr(args, "password", None)

    excluded_provider_ids = _parse_excluded_ids(getattr(args, "exclude_ids", "") or "")
    protect_paths = _normalize_protect_paths(
        getattr(args, "protect_paths", [_DEFAULT_PROTECT_PATH])
    )

    _validate_cleanup_args(host, api_key, libraries, all_libraries, doit, username, password)
    # _validate_cleanup_args exits on missing host/api_key; narrow for type checking
    assert host is not None and api_key is not None

    base_url, resolved_port = handle_host_and_port(host, port)
    if resolved_port not in (80, 443):
        base_url = f"{base_url}:{resolved_port}"

    config = CleanupConfig(
        min_age_years=getattr(args, "min_age_years", 3),
        protect_paths=protect_paths,
        base_rating=getattr(args, "base_rating", 6.0),
        decay_step=getattr(args, "decay_step", 0.5),
        max_rating=getattr(args, "max_rating", 8.0),
        excluded_provider_ids=excluded_provider_ids,
        near_miss_count=getattr(args, "near_miss_count", 5),
    )

    client = httpx.Client(headers={"X-Emby-Token": api_key}, timeout=120)

    try:
        check_emby_connection(client, f"{base_url}/System/Info")
        _execute_cleanup(
            client, base_url, config, api_key, libraries, all_libraries,
            username, password,
            output_format=getattr(args, "format", "console"),
            html_report=getattr(args, "html_report", False),
            html_only=getattr(args, "html_only", False),
            no_open=getattr(args, "no_open", False),
            doit=doit,
        )
    except Exception as e:
        logger.error(f"Cleanup command failed: {e}")
        raise
    finally:
        if _client_mod.auth_state.token_for_delete and doit:
            logout(client, base_url, _client_mod.auth_state.token_for_delete)
        client.close()
