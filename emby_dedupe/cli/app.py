"""
Typer CLI application for emby-dedupe.

This module is the central typer-based CLI entry point.  Shared options (host,
port, api-key, library, verbosity) are declared once in the @app.callback() and
passed to all subcommands via ctx.obj (AppConfig dataclass).

The business logic remains in the original cli/*.py modules; this file is a thin
routing layer so that existing internal functions and their tests are undisturbed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import typer
from dotenv import load_dotenv

# Load .env BEFORE any typer option resolution: click reads envvar= values while
# parsing parameters, so this must happen at import time (the console-script entry
# point imports this module directly — nothing else runs first).
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# CRITICAL: invoke_without_command=True ensures @app.callback() fires BEFORE
# subcommands — without it ctx.obj is never set when a subcommand runs.
app = typer.Typer(
    name="emby-dedupe",
    no_args_is_help=True,
    invoke_without_command=True,
    help="Emby duplicate media manager and genre tool.",
)

genres_app = typer.Typer(name="genres", help="Genre audit and management.")
app.add_typer(genres_app, name="genres")

descriptions_app = typer.Typer(
    name="descriptions", help="Overview/description management."
)
app.add_typer(descriptions_app, name="descriptions")

# Shared help strings used across multiple genre subcommands
_LOCK_OPT = "--lock/--no-lock"
_LOCK_HELP = "Lock genres after update."
_ALL_LIBS_HELP = "Scan all Emby libraries."
_ITEM_IDS_HELP = "Comma-separated Emby item IDs to process (skips full library scan)."
_DOIT_HELP = "Apply changes (dry-run by default)."
_TMDB_KEY_HELP = "TMDB API key."


@dataclass
class AppConfig:
    host: str | None = None
    port: int | None = None
    api_key: str | None = None
    libraries: list[str] = field(default_factory=list)
    verbosity: int = 0
    lock: bool = True
    doit: bool = False


@app.callback()
def common(
    ctx: typer.Context,
    host: str | None = typer.Option(
        None, "--host", "-H", envvar="DEDUPE_EMBY_HOST", help="Emby server URL."
    ),
    port: int | None = typer.Option(
        None, "--port", "-p", envvar="DEDUPE_EMBY_PORT", help="Emby server port."
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", "-a", envvar="DEDUPE_EMBY_API_KEY", help="Emby API key."
    ),
    library: list[str] | None = typer.Option(
        None,
        "--library",
        "-l",
        # NO envvar= here: click whitespace-splits list envvars, which destroys the
        # documented comma format (DEDUPE_EMBY_LIBRARY="Movies,TV Shows" would become
        # ['Movies,TV', 'Shows']). The env fallback is read manually below instead.
        help="Library name (repeatable; env: DEDUPE_EMBY_LIBRARY, comma-separated).",
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Verbosity (-v, -vv, -vvv)."
    ),
    lock: bool = typer.Option(
        True, _LOCK_OPT, envvar="DEDUPE_LOCK", help="Lock genres after normalization."
    ),
    doit: bool = typer.Option(
        False, "--doit/--no-doit", envvar="DEDUPE_DOIT", help="Execute changes (dry-run by default)."
    ),
) -> None:
    """Emby duplicate media manager and genre tool."""
    ctx.ensure_object(dict)
    ctx.obj = AppConfig(
        host=host,
        port=port,
        api_key=api_key,
        libraries=list(library) if library else _libraries_from_env(),
        verbosity=verbose,
        lock=lock,
        doit=doit,
    )


def _libraries_from_env() -> list[str]:
    """Read DEDUPE_EMBY_LIBRARY as a comma-separated list (README-documented format).

    Handled manually instead of typer ``envvar=`` because click whitespace-splits
    list envvars — "Movies,TV Shows" must resolve to ["Movies", "TV Shows"].
    """
    raw = os.environ.get("DEDUPE_EMBY_LIBRARY", "")
    return [part.strip() for part in raw.split(",") if part.strip()]


# ---------------------------------------------------------------------------
# dedupe subcommand
# ---------------------------------------------------------------------------

def _apply_fold_safe_results_to_decisions(decisions, results) -> None:
    """Write each fold-safe outcome back onto its decision item's ``deletion_result``.

    The guard sets ``skipped_unsafe`` when it refuses the Emby delete; fold-safe may then
    remove the file over SSH. Only a real removal changes the verdict — ``needs_review``,
    ``skipped`` and ``failed`` all leave the guard's refusal standing, which is accurate:
    the file is still there. Matched by item id, falling back to the delete path for items
    whose id is missing.
    """
    by_id = {r.get("id"): r for r in results if r.get("id")}
    by_path = {r.get("delete"): r for r in results if r.get("delete")}
    for decision in decisions or []:
        for item in decision.get("delete", []) or []:
            result = by_id.get(item.get("id")) or by_path.get(item.get("path"))
            if not result or result.get("status") != "removed":
                continue
            item["deletion_result"] = {
                "id": item.get("id"),
                "status": "fold_safe_removed",
                "error": None,
            }


def _run_fold_safe_delete(client, base_url, decisions, doit, ssh_host) -> None:
    """Remove guard-refused co-located duplicates by file-deleting them on the media host.

    Previews (verify-only) when ``doit`` is False; actually removes when True. After any
    real removal, best-effort triggers an Emby library refresh so the now-missing items
    drop from the DB.
    """
    from emby_dedupe.api.fold_safe_delete import (
        execute_fold_safe_deletes,
        plan_fold_safe_deletes,
    )
    from emby_dedupe.utils.logging import logger

    plans = plan_fold_safe_deletes(decisions)
    if not plans:
        logger.info("Fold-safe delete: no guard-refused duplicates to remove.")
        return

    if not ssh_host:
        logger.error(
            "Fold-safe delete: no media host configured — %d guard-refused duplicate(s) left "
            "in place. Pass --fold-safe-host user@host (or set DEDUPE_FOLD_SAFE_HOST).",
            len(plans),
        )
        return

    review_count = sum(1 for p in plans if p.get("status") == "needs_review")
    logger.info(
        "Fold-safe delete: %d guard-refused duplicate(s) to %s on %s (%d held for review)",
        len(plans) - review_count, "remove" if doit else "preview", ssh_host, review_count,
    )
    results = execute_fold_safe_deletes(plans, ssh_host=ssh_host, doit=doit)

    # Record each outcome back onto the decision item so the HTML/markdown report shows
    # what actually happened. Without this the report keeps the guard's "blocked" verdict
    # for a file that fold-safe removed seconds later — permanently mis-recording the run.
    _apply_fold_safe_results_to_decisions(decisions, results)

    for r in results:
        logger.info("  [%s] %s", r["status"], r["delete"])

    held = [r for r in results if r["status"] == "needs_review"]
    if held:
        logger.warning(
            "Fold-safe delete: %d item(s) NOT removed — failed the same-content sanity "
            "check (see review_reason above); resolve them manually.", len(held),
        )

    removed = sum(1 for r in results if r["status"] == "removed")
    if removed:
        try:
            client.post(f"{base_url}/Library/Refresh")
            logger.info(
                "Fold-safe delete: removed %d file(s); triggered Emby library refresh "
                "to drop the stale items.", removed,
            )
        except Exception as e:  # noqa: BLE001 — refresh is best-effort
            logger.warning(
                "Fold-safe delete: removed %d file(s), but the Emby refresh call failed "
                "(%s). Emby will drop the stale items on its next scheduled scan.",
                removed, e,
            )


@app.command("dedupe")
def dedupe_cmd(
    ctx: typer.Context,
    username: str | None = typer.Option(
        None, "--username", envvar="DEDUPE_EMBY_USERNAME", help="Emby username for auth."
    ),
    password: str | None = typer.Option(
        None, "--password", envvar="DEDUPE_EMBY_PASSWORD", help="Emby password for auth."
    ),
    lang_prio: str | None = typer.Option(
        None,
        "--lang-prio",
        envvar="DEDUPE_LANG_PRIO",
        help="Comma-separated language priority (e.g. 'sk,cs,en').",
    ),
    exclude_ids: str | None = typer.Option(
        None,
        "--exclude-ids",
        envvar="DEDUPE_EXCLUDE_IDS",
        help="Comma-separated provider IDs to exclude.",
    ),
    html_report: bool = typer.Option(
        False, "--html-report", envvar="DEDUPE_HTML_REPORT", help="Generate HTML report."
    ),
    html_only: bool = typer.Option(
        False, "--html-only", envvar="DEDUPE_HTML_ONLY", help="HTML report only, no terminal output."
    ),
    no_open: bool = typer.Option(
        False, "--no-open", help="Don't open HTML report in browser."
    ),
    fold_safe_delete: bool = typer.Option(
        False,
        "--fold-safe-delete",
        "-F",
        envvar="DEDUPE_FOLD_SAFE_DELETE",
        help=(
            "After the run, remove duplicates the safety guard refused (keeper co-located "
            "in a folder Emby would fold-delete) by deleting the single FILE on the media "
            "host over SSH — never via the Emby API, never the directory. Previews unless "
            "combined with --doit."
        ),
    ),
    fold_safe_host: str = typer.Option(
        "",
        "--fold-safe-host",
        envvar="DEDUPE_FOLD_SAFE_HOST",
        help=(
            "user@host of the media host for --fold-safe-delete SSH file removal. "
            "Required when --fold-safe-delete is used (no default — it runs rm on that host)."
        ),
    ),
) -> None:
    """Find and optionally remove duplicate media items."""
    import json
    from argparse import Namespace

    import httpx

    import emby_dedupe.api.client as _client_mod
    from emby_dedupe.api.client import handle_host_and_port, logout
    from emby_dedupe.cli.arguments import validate_required_arguments
    from emby_dedupe.cli.main import (
        _connect_and_fetch_libraries,
        _generate_reports,
        _resolve_configuration,
        _run_deduplication_pipeline,
    )
    from emby_dedupe.utils.exceptions import EmbyServerConnectionError

    config: AppConfig = ctx.obj

    # Build a Namespace that _resolve_configuration understands
    args = Namespace(
        verbosity=config.verbosity,
        host=config.host,
        port=config.port,
        api_key=config.api_key,
        library=config.libraries or None,
        doit=config.doit,
        lang_prio=lang_prio,
        exclude_ids=exclude_ids,
        username=username,
        password=password,
        html_report=html_report,
        html_only=html_only,
        no_open=no_open,
    )

    from emby_dedupe.utils.logging import logger
    resolved = _resolve_configuration(args)

    validate_required_arguments(
        resolved.host, resolved.api_key, resolved.library, resolved.doit,
        resolved.username, resolved.password,
    )
    # validate_required_arguments exits the process when host/api_key are missing.
    assert resolved.host is not None and resolved.api_key is not None

    validated_host, validated_port = handle_host_and_port(resolved.host, resolved.port)

    try:
        base_url = f"{validated_host}:{validated_port}"
        client = httpx.Client(headers={"X-Emby-Token": resolved.api_key})

        all_provider_tables = _connect_and_fetch_libraries(client, base_url, resolved.library)

        decisions, exclusion_metadata, _pipeline_markdown = _run_deduplication_pipeline(
            client, base_url, all_provider_tables, resolved.excluded_ids,
            resolved.lang_priorities, resolved.api_key, resolved.doit,
            resolved.username, resolved.password,
        )

        # Fold-safe runs BEFORE the report: it removes guard-refused duplicates over SSH
        # and records the outcome on the decisions. Reporting first would freeze the
        # guard's "blocked" verdict into the report for files that were then removed.
        # (The pipeline's markdown is discarded for the same reason — _generate_reports
        # re-renders it from the final decisions.)
        if fold_safe_delete:
            _run_fold_safe_delete(client, base_url, decisions, resolved.doit, fold_safe_host)

        _generate_reports(
            base_url, decisions, exclusion_metadata, resolved.excluded_ids,
            resolved.lang_priorities,
            resolved.html_report, resolved.html_only, resolved.no_open,
        )

    except EmbyServerConnectionError as e:
        logger.error(str(e))
        raise typer.Exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON: {str(e)}")
        raise typer.Exit(1)
    except httpx.TimeoutException as e:
        logger.error(f"HTTP request timed out: {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise typer.Exit(1)
    finally:
        if _client_mod.auth_state.token_for_delete and resolved.doit:
            logout(client, base_url, _client_mod.auth_state.token_for_delete)


# ---------------------------------------------------------------------------
# check subcommand
# ---------------------------------------------------------------------------

@app.command("check")
def check_cmd(
    ctx: typer.Context,  # NOSONAR — typer CLI requires one param per CLI option; cannot reduce
    name: str | None = typer.Option(None, "--name", help="Media name to search for."),
    year: int | None = typer.Option(None, "--year", help="Release year (movies)."),
    imdb: str | None = typer.Option(None, "--imdb", help="IMDB ID (e.g. tt1375666)."),
    tmdb: str | None = typer.Option(None, "--tmdb", help="TMDB ID."),
    tvdb: str | None = typer.Option(None, "--tvdb", help="TVDB ID."),
    season: int | None = typer.Option(None, "--season", help="Season number."),
    episode: int | None = typer.Option(None, "--episode", help="Episode number."),
    resolution: str | None = typer.Option(None, "--resolution", help="Resolution (2160p, 1080p …)."),
    codec: str | None = typer.Option(None, "--codec", help="Video codec (x265, x264 …)."),
    hdr: str | None = typer.Option(None, "--hdr", help="HDR type (HDR, DV, SDR …)."),
    audio: str | None = typer.Option(None, "--audio", help="Audio type (Atmos, DTS-HD …)."),
    audio_lang: str | None = typer.Option(None, "--audio-lang", help="Comma-separated audio languages."),
    size_mb: int | None = typer.Option(None, "--size-mb", help="File size in MB."),
    bitrate_kbps: int | None = typer.Option(None, "--bitrate-kbps", help="Video bitrate in kbps."),
    simple: bool = typer.Option(False, "--simple", help="Simple output: 'download' or 'skip'."),
    exit_code: bool = typer.Option(False, "--exit-code", help="Exit code only: 0=download, 1=skip."),
    all_libraries: bool = typer.Option(False, "--all-libraries", help="Search all libraries."),
    lang_prio: str | None = typer.Option(None, "--lang-prio", envvar="DEDUPE_LANG_PRIO"),
    exclude_ids: str | None = typer.Option(None, "--exclude-ids", envvar="DEDUPE_EXCLUDE_IDS"),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="Use cached library data."),
) -> None:
    """Check whether media should be downloaded based on existing library."""
    from argparse import Namespace

    from emby_dedupe.cli.check import run_check

    config: AppConfig = ctx.obj

    args = Namespace(
        host=config.host,
        api_key=config.api_key,
        library=config.libraries or None,
        verbosity=config.verbosity,
        name=name,
        year=year,
        imdb=imdb,
        tmdb=tmdb,
        tvdb=tvdb,
        season=season,
        episode=episode,
        resolution=resolution,
        codec=codec,
        hdr=hdr,
        audio=audio,
        audio_lang=audio_lang,
        size_mb=size_mb,
        bitrate_kbps=bitrate_kbps,
        simple=simple,
        exit_code=exit_code,
        all_libraries=all_libraries,
        lang_prio=lang_prio,
        exclude_ids=exclude_ids,
        cache=cache,
    )

    result = run_check(args)
    raise typer.Exit(result)


# ---------------------------------------------------------------------------
# missing-episodes subcommand
# ---------------------------------------------------------------------------

@app.command("missing-episodes")
def missing_episodes_cmd(
    ctx: typer.Context,
    format: str = typer.Option(
        "console",
        "--format",
        help="Output format: console, html, json, structured_json.",
    ),
    output: str | None = typer.Option(
        None, "--output", help="Output file path for JSON formats."
    ),
    username: str | None = typer.Option(
        None, "--username", envvar="DEDUPE_EMBY_USERNAME", help="Emby username."
    ),
    password: str | None = typer.Option(
        None, "--password", envvar="DEDUPE_EMBY_PASSWORD", help="Emby password."
    ),
    html_report: bool = typer.Option(False, "--html-report", envvar="DEDUPE_HTML_REPORT"),
    html_only: bool = typer.Option(False, "--html-only", envvar="DEDUPE_HTML_ONLY"),
) -> None:
    """Search for missing episodes in TV series libraries."""
    from argparse import Namespace

    from emby_dedupe.cli.missing_episodes import run_missing_episodes_command

    config: AppConfig = ctx.obj

    args = Namespace(
        host=config.host,
        port=config.port,
        api_key=config.api_key,
        library=config.libraries or None,
        verbosity=config.verbosity,
        format=format,
        output=output,
        username=username,
        password=password,
        html_report=html_report,
        html_only=html_only,
    )

    run_missing_episodes_command(args)


# ---------------------------------------------------------------------------
# cleanup subcommand
# ---------------------------------------------------------------------------

@app.command("cleanup")
def cleanup_cmd(
    ctx: typer.Context,
    username: str | None = typer.Option(
        None, "--username", envvar="DEDUPE_EMBY_USERNAME",
        help="Emby username (recommended for accurate actor protection; required with --doit)."
    ),
    password: str | None = typer.Option(
        None, "--password", envvar="DEDUPE_EMBY_PASSWORD",
        help="Emby password (required with --doit)."
    ),
    min_age_years: int = typer.Option(3, "--min-age-years", help="Minimum age in years to be eligible for cleanup."),
    protect_path: list[str] | None = typer.Option(
        None, "--protect-path", help="Path substring to protect (repeatable, default: /Dokumenty/)."
    ),
    base_rating: float = typer.Option(6.0, "--base-rating", help="Rating threshold at minimum age."),
    decay_step: float = typer.Option(0.5, "--decay-step", help="Rating increase per year over minimum age."),
    max_rating: float = typer.Option(8.0, "--max-rating", help="Maximum rating threshold cap."),
    exclude_ids: str | None = typer.Option(
        None, "--exclude-ids", envvar="DEDUPE_EXCLUDE_IDS",
        help="Comma-separated provider IDs to always protect (IMDB tt*, TMDB IDs)."
    ),
    all_libraries: bool = typer.Option(False, "--all-libraries", help="Scan all libraries (skips -l requirement)."),
    output_json: bool = typer.Option(False, "--json", help="Output report as JSON instead of console table."),
    html_report: bool = typer.Option(False, "--html-report", envvar="DEDUPE_HTML_REPORT", help="Generate HTML report (use --html-only for no browser)."),
    html_only: bool = typer.Option(False, "--html-only", envvar="DEDUPE_HTML_ONLY", help="HTML report only, skip console output and don't open browser."),
) -> None:
    """Identify dead movies nobody watches for library hygiene (dynamic rating decay model)."""
    from argparse import Namespace

    from emby_dedupe.cli.cleanup import run_cleanup_command

    config: AppConfig = ctx.obj

    args = Namespace(
        host=config.host,
        port=config.port,
        api_key=config.api_key,
        library=config.libraries or [],
        verbosity=config.verbosity,
        doit=config.doit,
        username=username,
        password=password,
        min_age_years=min_age_years,
        protect_paths=protect_path or ["/Dokumenty/"],
        base_rating=base_rating,
        decay_step=decay_step,
        max_rating=max_rating,
        exclude_ids=exclude_ids,
        all_libraries=all_libraries,
        format="json" if output_json else "console",
        html_report=html_report,
        html_only=html_only,
        no_open=html_only,
        near_miss_count=5,
    )

    run_cleanup_command(args)


# ---------------------------------------------------------------------------
# genres subcommands
# ---------------------------------------------------------------------------

@genres_app.callback(invoke_without_command=True)
def genres_callback(ctx: typer.Context) -> None:
    """Genre audit and management."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@genres_app.command("audit")
def genres_audit(
    ctx: typer.Context,
    suggest: bool = typer.Option(
        False, "--suggest", help="Flag non-canonical genres and suggest mappings."
    ),
    output_json: str | None = typer.Option(
        None, "--output-json", help="Save audit results as JSON to this path."
    ),
    all_libraries: bool = typer.Option(False, "--all-libraries", help=_ALL_LIBS_HELP),
    item_ids: str | None = typer.Option(None, "--item-ids", help=_ITEM_IDS_HELP),
) -> None:
    """Audit genre health across libraries (read-only)."""
    _run_genres_subcommand(
        ctx,
        action="audit",
        all_libraries=all_libraries,
        item_ids=item_ids,
        suggest=suggest,
        output_json=output_json,
    )


@genres_app.command("normalize")
def genres_normalize(
    ctx: typer.Context,
    doit: bool = typer.Option(False, "--doit", help="Apply normalization (dry-run by default)."),
    lock: bool = typer.Option(True, _LOCK_OPT, help=_LOCK_HELP),
    repair_dupes: bool = typer.Option(
        False, "--repair-dupes", help="Also fix duplicate genres caused by normalization collisions."
    ),
    all_libraries: bool = typer.Option(False, "--all-libraries", help=_ALL_LIBS_HELP),
    item_ids: str | None = typer.Option(None, "--item-ids", help=_ITEM_IDS_HELP),
) -> None:
    """Fix variant genre names (Sci-Fi→Science Fiction, dada→Comedy …)."""
    _run_genres_subcommand(
        ctx,
        action="normalize",
        doit=doit,
        lock=lock,
        repair_dupes=repair_dupes,
        all_libraries=all_libraries,
        item_ids=item_ids,
    )


@genres_app.command("process")
def genres_process(
    ctx: typer.Context,
    doit: bool = typer.Option(False, "--doit", help=_DOIT_HELP),
    lock: bool = typer.Option(True, _LOCK_OPT, help=_LOCK_HELP),
    validate: bool = typer.Option(
        False, "--validate", help="Compare existing genres against TMDB/OMDb and add missing ones."
    ),
    tmdb_api_key: str | None = typer.Option(
        None, "--tmdb-api-key", envvar="DEDUPE_TMDB_API_KEY", help=_TMDB_KEY_HELP
    ),
    item_ids: str = typer.Option(..., "--item-ids", help="Comma-separated Emby item IDs (required)."),
) -> None:
    """Normalize + fix in a single pass (webhook listener mode).

    Fetches items once and runs both normalize and fix, avoiding double-fetch.
    Requires --item-ids.
    """
    _run_genres_subcommand(
        ctx,
        action="process",
        doit=doit,
        lock=lock,
        validate=validate,
        tmdb_api_key=tmdb_api_key,
        item_ids=item_ids,
    )


@genres_app.command("fix")
def genres_fix(
    ctx: typer.Context,
    doit: bool = typer.Option(False, "--doit", help=_DOIT_HELP),
    lock: bool = typer.Option(True, _LOCK_OPT, help=_LOCK_HELP),
    gaps_only: bool = typer.Option(False, "--gaps-only", help="Only process items with no genres."),
    validate: bool = typer.Option(
        False, "--validate", help="Compare existing genres against TMDB/OMDb and add missing ones."
    ),
    tmdb_api_key: str | None = typer.Option(
        None, "--tmdb-api-key", envvar="DEDUPE_TMDB_API_KEY", help=_TMDB_KEY_HELP
    ),
    all_libraries: bool = typer.Option(False, "--all-libraries", help=_ALL_LIBS_HELP),
    item_ids: str | None = typer.Option(None, "--item-ids", help=_ITEM_IDS_HELP),
) -> None:
    """Fetch genres from TMDB/OMDb and fill gaps or validate existing genres."""
    _run_genres_subcommand(
        ctx,
        action="fix",
        doit=doit,
        lock=lock,
        gaps_only=gaps_only,
        validate=validate,
        tmdb_api_key=tmdb_api_key,
        all_libraries=all_libraries,
        item_ids=item_ids,
    )


def _run_genres_subcommand(ctx: typer.Context, **kwargs) -> None:
    """Shared dispatcher: build an argparse-like Namespace and call run_genres_command."""
    from argparse import Namespace

    from emby_dedupe.cli.genres import run_genres_command

    config: AppConfig = ctx.obj if ctx.obj else AppConfig()

    # Merge AppConfig fields with subcommand-specific kwargs.
    # Subcommand kwargs take precedence over AppConfig when both are present.
    args = Namespace(
        host=config.host,
        port=config.port,
        api_key=config.api_key,
        library=config.libraries or [],
        verbosity=config.verbosity,
        # subcommand-specific fields (with sensible defaults)
        action=kwargs.get("action", "audit"),
        doit=kwargs.get("doit", False) or config.doit,
        lock=kwargs.get("lock", config.lock),
        repair_dupes=kwargs.get("repair_dupes", False),
        suggest=kwargs.get("suggest", False),
        output_json=kwargs.get("output_json", None),
        all_libraries=kwargs.get("all_libraries", False),
        item_ids=kwargs.get("item_ids", None),
        gaps_only=kwargs.get("gaps_only", False),
        validate=kwargs.get("validate", False),
        tmdb_api_key=kwargs.get("tmdb_api_key", None),
    )

    run_genres_command(args)


# ---------------------------------------------------------------------------
# descriptions subcommands
# ---------------------------------------------------------------------------

@descriptions_app.callback(invoke_without_command=True)
def descriptions_callback(ctx: typer.Context) -> None:
    """Overview/description management."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@descriptions_app.command("fill")
def descriptions_fill(
    ctx: typer.Context,
    doit: bool = typer.Option(False, "--doit", help=_DOIT_HELP),
    lock: bool = typer.Option(
        True, _LOCK_OPT, help="Lock Overview after update to prevent agent revert."
    ),
    overview_langs: str | None = typer.Option(
        None,
        "--overview-langs",
        help="Comma-separated BCP47 codes in priority order. Default: sk-SK,cs-CZ",
    ),
    update_title: bool = typer.Option(
        False,
        "--update-title",
        help=(
            "Also fix Name. Policy: keep if it matches TMDB EN/CZ/SK title; "
            "otherwise replace with EN. OriginalTitle is never touched."
        ),
    ),
    tmdb_api_key: str | None = typer.Option(
        None, "--tmdb-api-key", envvar="DEDUPE_TMDB_API_KEY", help=_TMDB_KEY_HELP
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Cap the number of items processed (useful for dry-run sampling)."
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache",
        help="Bypass the on-disk TMDB cache and re-query every item (force refresh).",
    ),
    cache_ttl_days: int | None = typer.Option(
        None, "--cache-ttl-days",
        help="How many days a cached TMDB entry stays fresh (default 30).",
    ),
    all_libraries: bool = typer.Option(False, "--all-libraries", help=_ALL_LIBS_HELP),
    item_ids: str | None = typer.Option(None, "--item-ids", help=_ITEM_IDS_HELP),
) -> None:
    """Replace English Overviews with SK/CZ from TMDB (configurable chain).

    With --update-title, also replaces non-EN/CZ/SK titles with English.

    A persistent cache at ~/.cache/emby-dedupe/description-cache.json stores
    TMDB results (including 'no data' negatives) — re-runs on the same library
    skip cached items entirely, finishing in minutes instead of hours.
    """
    from argparse import Namespace

    from emby_dedupe.cli.descriptions import run_descriptions_command

    config: AppConfig = ctx.obj if ctx.obj else AppConfig()
    args = Namespace(
        host=config.host,
        port=config.port,
        api_key=config.api_key,
        library=config.libraries or [],
        verbosity=config.verbosity,
        doit=doit or config.doit,
        lock=lock,
        overview_langs=overview_langs,
        update_title=update_title,
        tmdb_api_key=tmdb_api_key,
        limit=limit,
        no_cache=no_cache,
        cache_ttl_days=cache_ttl_days,
        all_libraries=all_libraries,
        item_ids=item_ids,
    )
    run_descriptions_command(args)


def main() -> None:
    """Entry point for the emby-dedupe CLI."""
    app()
