"""
Main command-line interface for the Emby Dedupe tool.
"""

import logging
import sys
from dataclasses import dataclass, field

from emby_dedupe.api.client import (
    check_emby_connection,
    fetch_all_media_paths,
    fetch_and_process_media_items,
    get_library_id,
)
from emby_dedupe.api.deduplication import (
    identify_duplicates,
    process_deletion_and_generate_report,
    process_duplicate_groups,
    rationalize_duplicates,
)
from emby_dedupe.cli.arguments import (
    get_env_variable,
)
from emby_dedupe.reports.html import generate_html_report
from emby_dedupe.reports.markdown import format_markdown_table, output_report_to_stdout
from emby_dedupe.utils.constants import (
    ENV_DEDUPE_EMBY_PASSWORD,
    ENV_DEDUPE_EMBY_USERNAME,
    ENV_DEDUPE_LOGGING,
    LANGUAGE_NORMALIZATION_MAP,
)
from emby_dedupe.utils.file_ops import dump_object_to_file
from emby_dedupe.utils.logging import logger, set_logging_level


def _parse_language_priorities(lang_prio_str: str) -> list:
    """
    Parse and normalize language priority string.

    Args:
        lang_prio_str: Comma-separated language priority string.

    Returns:
        List of normalized language codes.
    """
    lang_priorities = []
    if lang_prio_str:
        # Create normalized language priority list treating Slovak/Czech variants as equivalent
        lang_mapping = LANGUAGE_NORMALIZATION_MAP

        raw_langs = [lang.strip().lower() for lang in lang_prio_str.split(',') if lang.strip()]
        seen_langs = set()

        for lang in raw_langs:
            # Normalize Slovak/Czech variants, keep others as-is
            normalized_lang = lang_mapping.get(lang, lang)

            # Only add if we haven't seen this normalized language before
            if normalized_lang not in seen_langs:
                lang_priorities.append(normalized_lang)
                seen_langs.add(normalized_lang)

        logger.info(f"Language priorities set: {', '.join(lang_priorities)} (Slovak/Czech variants normalized)")
        if raw_langs != [lang_mapping.get(lang, lang) for lang in raw_langs]:
            logger.debug(f"Original input: {', '.join(raw_langs)}")
    else:
        logger.debug("No language priorities specified, using default quality-based evaluation")

    return lang_priorities


def _parse_excluded_ids(exclude_ids_str: str) -> list:
    """
    Parse excluded IDs from comma-separated string.

    Args:
        exclude_ids_str: Comma-separated ID string.

    Returns:
        List of excluded ID strings.
    """
    excluded_ids = []
    if exclude_ids_str:
        excluded_ids = [id.strip() for id in exclude_ids_str.split(',') if id.strip()]
        logger.info(f"Excluding provider IDs from deduplication: {', '.join(excluded_ids)}")
    else:
        logger.debug("No provider IDs excluded from deduplication")
    return excluded_ids


def _resolve_auth_credentials(args, doit):
    """
    Resolve authentication credentials if doit is enabled.

    Args:
        args: Parsed argument namespace.
        doit: Whether to actually delete items.

    Returns:
        Tuple of (username, password).
    """
    if doit:
        username = args.username or get_env_variable(ENV_DEDUPE_EMBY_USERNAME)
        password = args.password or get_env_variable(ENV_DEDUPE_EMBY_PASSWORD)
        return username, password
    return None, None


@dataclass(frozen=True)
class ResolvedConfig:
    """Fully-resolved dedupe run configuration (replaces the old positional 12-tuple)."""

    host: str | None
    port: int | None
    api_key: str | None
    library: list = field(default_factory=list)
    doit: bool = False
    lang_priorities: list = field(default_factory=list)
    excluded_ids: list = field(default_factory=list)
    username: str | None = None
    password: str | None = None
    html_report: bool = False
    html_only: bool = False
    no_open: bool = False


def _resolve_configuration(args) -> ResolvedConfig:
    """Finish resolving a dedupe run's configuration from parsed arguments.

    Environment variables are resolved by typer's ``envvar=`` declarations BEFORE this
    runs (single resolution layer — the old duplicate env re-read produced false
    "CLI overrides env" warnings). This only performs the steps typer cannot:
    DEDUPE_LOGGING log-level setup, language/exclude-id parsing, auth gating on
    ``doit``, and the html_report |= html_only derivation.

    Args:
        args: Parsed argument namespace (values already CLI>env resolved).

    Returns:
        ResolvedConfig: immutable resolved configuration.
    """
    set_logging_level(args.verbosity, get_env_variable(ENV_DEDUPE_LOGGING))

    lang_priorities = _parse_language_priorities(args.lang_prio)
    excluded_ids = _parse_excluded_ids(args.exclude_ids)
    username, password = _resolve_auth_credentials(args, args.doit)

    return ResolvedConfig(
        host=args.host,
        port=args.port or None,
        api_key=args.api_key,
        library=args.library or [],
        doit=args.doit,
        lang_priorities=lang_priorities,
        excluded_ids=excluded_ids,
        username=username,
        password=password,
        html_report=args.html_report or args.html_only,
        html_only=args.html_only,
        no_open=getattr(args, "no_open", False),
    )


def _build_report_metadata(excluded_ids, lang_priorities, exclusion_metadata) -> dict:
    """Build the report metadata dict (single construction for pipeline + reports)."""
    return {
        "excluded_ids": excluded_ids if excluded_ids else [],
        "language_priorities": lang_priorities if lang_priorities else [],
        "excluded_groups_count": exclusion_metadata.get("excluded_groups_count", 0),
        "excluded_titles": exclusion_metadata.get("excluded_titles", {}),
    }


def _connect_and_fetch_libraries(client, base_url, library):
    """
    Connect to Emby and fetch provider tables from all libraries.

    Args:
        client: HTTP client.
        base_url: Emby server base URL.
        library: List of library names.

    Returns:
        Combined provider tables dict.
    """
    connection_url = f"{base_url}/System/Info"
    if not check_emby_connection(client, connection_url):
        logger.error(f"Unable to connect to the Emby server at {base_url}.")
        sys.exit(1)

    all_provider_tables = {"imdb": {}, "tvdb": {}, "tmdb": {}, "series_episode": {}}

    # Process each library
    for library_name in library:
        logger.debug(f"Processing library: {library_name}")

        library_id = get_library_id(client, base_url, library_name)
        if library_id is None:
            logger.error(f"Unable to find library '{library_name}'. Skipping.")
            continue

        provider_tables = fetch_and_process_media_items(client, base_url, library_id, library_name)

        for provider in ["imdb", "tvdb", "tmdb", "series_episode"]:
            for provider_id, items in provider_tables[provider].items():
                if provider_id not in all_provider_tables[provider]:
                    all_provider_tables[provider][provider_id] = []
                all_provider_tables[provider][provider_id].extend(items)

    if all(not table for table in all_provider_tables.values()):
        logger.error("No media items found in any of the specified libraries.")
        sys.exit(1)

    dump_object_to_file(
        all_provider_tables, "testing/provider_tables"
    ) if logger.isEnabledFor(logging.DEBUG) else None

    return all_provider_tables


def _run_deduplication_pipeline(client, base_url, all_provider_tables, excluded_ids,
                                lang_priorities, api_key, doit, username, password):
    """
    Run the deduplication pipeline: identify, rationalize, process.

    Args:
        client: HTTP client.
        base_url: Emby server base URL.
        all_provider_tables: Provider tables dictionary.
        excluded_ids: IDs to exclude.
        lang_priorities: Language priorities list.
        api_key: API key.
        doit: Whether to actually delete.
        username: Username for auth.
        password: Password for auth.

    Returns:
        Tuple of (decisions, exclusion_metadata, markdown_report).
    """
    duplicates = identify_duplicates(all_provider_tables, excluded_ids)

    dump_object_to_file(duplicates, "testing/duplicates") if logger.isEnabledFor(
        logging.DEBUG
    ) else None

    duplicates = rationalize_duplicates(duplicates)

    dump_object_to_file(duplicates, "testing/aggregate") if logger.isEnabledFor(
        logging.DEBUG
    ) else None

    decisions, exclusion_metadata = process_duplicate_groups(
        client, base_url, duplicates, api_key, lang_priorities, excluded_ids
    )

    dump_object_to_file(decisions, "testing/decisions") if logger.isEnabledFor(
        logging.DEBUG
    ) else None

    logger.debug(f"Processing {len(decisions)} decisions for markdown report generation")

    # Create metadata dictionary for report generation
    report_metadata = _build_report_metadata(excluded_ids, lang_priorities, exclusion_metadata)

    # Give the deletion safety guard full folder visibility by fetching every library
    # media path (not just the duplicated ones), so it stops over-refusing safe deletes
    # in single-duplicate folders. Only needed when something will actually be deleted;
    # skip the extra fetch for no-op runs.
    has_deletes = any(decision.get("delete") for decision in decisions)
    library_paths = None
    if has_deletes:
        library_paths = fetch_all_media_paths(client, base_url)
        logger.info(
            f"Fetched {len(library_paths)} library paths for the deletion safety guard."
        )

    markdown_report = process_deletion_and_generate_report(
        client, base_url, decisions, doit, username, password, api_key,
        report_metadata, library_paths,
    )

    dump_object_to_file(decisions, "testing/deletions") if logger.isEnabledFor(
        logging.DEBUG
    ) else None
    dump_object_to_file(markdown_report, "testing/report") if logger.isEnabledFor(
        logging.DEBUG
    ) else None

    return decisions, exclusion_metadata, markdown_report


def _generate_reports(base_url, decisions, exclusion_metadata, excluded_ids,
                     lang_priorities, html_report, html_only, no_open):
    """
    Generate and output reports (markdown and/or HTML).

    The markdown report is rendered HERE rather than taken as an argument. The pipeline
    renders one during the deletion pass, but that happens before the fold-safe pass can
    remove any guard-refused duplicate — passing it in would print a file as blocked when
    it was actually removed. Rendering from the final ``decisions`` keeps the console
    output and the HTML in agreement.

    Args:
        base_url: Emby server base URL.
        decisions: Deduplication decisions (final — after any fold-safe removals).
        exclusion_metadata: Metadata about exclusions.
        excluded_ids: IDs that were excluded.
        lang_priorities: Language priorities list.
        html_report: Whether to generate HTML report.
        html_only: Whether to only output HTML (skip console).
        no_open: Whether to skip opening browser.
    """
    # Create metadata dictionary for report generation
    report_metadata = _build_report_metadata(excluded_ids, lang_priorities, exclusion_metadata)

    # Pure function of `decisions` — no I/O, safe to (re-)render at this point.
    markdown_report = format_markdown_table(base_url, decisions, report_metadata)

    if html_report:
        try:
            logger.debug(f"Generating HTML report with {len(decisions)} decisions total")

            html_report_path = generate_html_report(base_url, decisions, report_metadata)

            if not no_open:
                try:
                    import webbrowser
                    print(f"Opening HTML report in browser: {html_report_path}")
                    webbrowser.open(f"file://{html_report_path}")
                except Exception as e:
                    logger.warning(f"Could not open browser: {e}")
                    print(f"HTML report generated at: {html_report_path}")
            else:
                print(f"HTML report generated at: {html_report_path}")
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            if html_only:
                logger.error("HTML-only mode requested but HTML report generation failed")
                sys.exit(1)
            logger.info("Continuing with console report output")

    if not html_only:
        output_report_to_stdout(markdown_report)
