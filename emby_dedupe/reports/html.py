"""
HTML report generation for the Emby Dedupe tool.
"""

import os
import tempfile
import time
from typing import Any

from tqdm import tqdm

from emby_dedupe.reports.common import calculate_report_statistics
from emby_dedupe.utils.logging import logger


def _validate_decisions(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Filter decisions to only include valid ones with proper keep and delete items.

    Args:
        decisions: List of decision objects.

    Returns:
        List of valid decision objects.
    """
    valid_decisions = []

    for decision in decisions:
        # Regular validation for decisions
        if not decision.get("keep"):
            logger.debug("Skipping decision with no 'keep' item")
            continue
        if "id" not in decision.get("keep", {}):
            logger.debug(f"Skipping decision with no ID in keep item: {decision.get('keep')}")
            continue
        if not decision.get("delete"):
            logger.debug("Skipping decision with no items to delete")
            continue
        valid_decisions.append(decision)

    logger.debug(f"Found {len(valid_decisions)} valid decisions out of {len(decisions)} total")
    return valid_decisions


def _detect_language_priority_usage(valid_decisions: list[dict[str, Any]]) -> tuple[bool, bool, list | None]:
    """
    Detect if language prioritization was used and if it changed any decisions.

    Args:
        valid_decisions: List of valid decision objects.

    Returns:
        Tuple of (language_priorities_used, language_priorities_changed_selection, language_priorities_list).
    """
    language_priorities_used = False
    language_priorities_changed_selection = False
    language_priorities_list = None

    for decision in valid_decisions:
        if "keep" in decision and decision["keep"].get("selected_by_language_priority", False):
            language_priorities_used = True
            language_priorities_list = decision["keep"].get("language_priority_list", [])

            # Check if language priority changed the decision
            if decision["keep"].get("changed_by_language_priority", False):
                language_priorities_changed_selection = True
                break
        elif "keep" in decision and "language_priority_list" in decision["keep"] and decision["keep"]["language_priority_list"]:
            language_priorities_list = decision["keep"]["language_priority_list"]

    return language_priorities_used, language_priorities_changed_selection, language_priorities_list


def _ensure_quality_fields(quality_desc: dict[str, Any]) -> None:
    """
    Ensure quality description has proper audio and video fields with defaults.
    Modifies quality_desc in place.

    Args:
        quality_desc: Quality description dictionary to validate and populate.
    """
    if not quality_desc:
        return

    # Ensure video section exists
    if "video" not in quality_desc:
        quality_desc["video"] = {"codec": "unknown", "resolution": "unknown"}

    # Ensure audio section exists
    if "audio" not in quality_desc:
        quality_desc["audio"] = {"codec": "unknown", "channels": "unknown", "languages": ["unknown"]}
    elif "languages" not in quality_desc["audio"]:
        quality_desc["audio"]["languages"] = ["unknown"]
    else:
        languages = quality_desc["audio"].get("languages")
        if languages is None or callable(languages) or not isinstance(languages, (list, tuple, set)):
            quality_desc["audio"]["languages"] = ["unknown"]


def _create_language_priority_message(keep_item: dict[str, Any]) -> str:
    """
    Create a human-readable message about language priority selection.

    Args:
        keep_item: The item being kept.

    Returns:
        Language priority message string (empty if not applicable).
    """
    selected_by_language = keep_item.get('selected_by_language_priority', False)
    changed_by_language = keep_item.get('changed_by_language_priority', False)
    priority_language = keep_item.get('priority_language_used', None)
    language_priority_list = keep_item.get('language_priority_list', [])

    if selected_by_language and priority_language and changed_by_language:
        return f"This file was selected because it has '{priority_language}' audio track, overriding quality-based selection (priority languages: {', '.join(language_priority_list)})"
    elif selected_by_language and priority_language and not changed_by_language:
        return f"This file has '{priority_language}' audio track and also has the best quality (priorities: {', '.join(language_priority_list)})"
    elif language_priority_list:
        return f"Language priorities ({', '.join(language_priority_list)}) were considered but didn't change selection"

    return ""


def _process_delete_item(item: dict[str, Any], base_url: str, keep_serverid: str) -> dict[str, Any]:
    """
    Process a single delete item into template-friendly format.

    Args:
        item: Delete item to process.
        base_url: Emby server base URL.
        keep_serverid: Server ID from the kept item.

    Returns:
        Processed delete item dictionary.
    """
    deletion_status = item.get('deletion_result', {})
    status = deletion_status.get('status', 'not_attempted')

    # "Pending" is the fallback for a dry run (nothing was attempted). Every status the
    # run actually produced gets its own label — a guard refusal in particular must NOT
    # read as "Pending", which looks like "nothing happened yet" when the truth is
    # "deliberately refused to protect the keeper".
    status_class = "status-pending"
    status_text = "Pending"
    if status == "success":
        status_class = "status-success"
        status_text = "Deleted"
    elif status == "failed":
        status_class = "status-error"
        status_text = "Failed"
    elif status == "skipped_unsafe":
        status_class = "status-blocked"
        status_text = "Blocked by safety guard (both files kept)"
    elif status == "fold_safe_removed":
        status_class = "status-success"
        status_text = "Deleted (file-only, fold-safe)"

    # Create URL for this item
    item_url = f"{base_url}/web/index.html#!/item?id={item['id']}&serverId={keep_serverid}"

    # Ensure audio and video fields are proper
    item_quality_desc = item.get("quality_description", {})
    _ensure_quality_fields(item_quality_desc)

    return {
        "id": item["id"],
        "name": item["name"],
        "image_url": item.get('image_url', f'{base_url}/web/assets/img/media.png'),
        "quality_description": item_quality_desc,
        "url": item_url,
        "status_class": status_class,
        "status_text": status_text,
        "error": deletion_status.get('error'),
        "is_episode": item.get("is_episode", False),
        "series_name": item.get("series_name", ""),
        "season_number": item.get("season_number", ""),
        "episode_number": item.get("episode_number", ""),
        "deletion_result": deletion_status,
        "provider_id": item.get("provider_id", "")
    }


def _item_path(item: dict[str, Any]) -> str:
    """The item's filesystem path for display, preferring the authoritative field.

    ``quality_description`` is built per-item from the details fetch and can come back
    empty (no codec, no size, no date, no path) when Emby returns a sparse record. Reading
    the path only from there then printed "unknown" for an item whose real path was known
    all along — the top-level ``path`` is the field the deletion guard actually uses.

    That mismatch is not cosmetic: on 2026-08-20 a report showing "unknown" was read as
    evidence that deletes had run with the guard bypassed, when the guard had in fact seen
    a valid path and evaluated normally. The report is the audit artifact, so it must show
    the same path the guard reasoned about.
    """
    return item.get("quality_description", {}).get("path") or item.get("path") or "unknown"


def _process_decision_group(decision: dict[str, Any], base_url: str) -> dict[str, Any]:
    """
    Process a single decision group into template-friendly format.

    Args:
        decision: Decision object containing keep and delete items.
        base_url: Emby server base URL.

    Returns:
        Processed group dictionary ready for template rendering.
    """
    keep_item = decision["keep"]
    delete_items = decision["delete"]

    # Create item URLs
    keep_url = f"{base_url}/web/index.html#!/item?id={keep_item['id']}&serverId={keep_item['serverid']}"

    # Find the newest and oldest items based on the date_added field
    all_items = [keep_item] + delete_items

    # Sort by date_added to find newest and oldest
    def date_sort_key(item: dict[str, Any]) -> str:
        return item['quality_description'].get('date_added', '0000-00-00')

    try:
        # Try to sort items by date. Safety net for potential None values or sorting errors.
        sorted_items = sorted(all_items, key=date_sort_key, reverse=True)
        newest_item = sorted_items[0] if sorted_items else keep_item
        oldest_item = sorted_items[-1] if len(sorted_items) > 1 else keep_item
    except (TypeError, ValueError) as e:
        logger.warning(f"Error sorting items by date: {e}")
        newest_item = keep_item
        oldest_item = keep_item

    # Process delete items
    processed_delete_items = []
    has_deleted_items = False

    for item in delete_items:
        processed_item = _process_delete_item(item, base_url, keep_item['serverid'])
        if processed_item["status_text"] == "Deleted":
            has_deleted_items = True
        processed_delete_items.append(processed_item)

    # Create language priority message
    language_priority_message = _create_language_priority_message(keep_item)

    # Ensure keep_item's quality_description has proper audio and video fields
    keep_quality_desc = keep_item.get("quality_description", {})
    _ensure_quality_fields(keep_quality_desc)

    # Extract language priority info
    selected_by_language = keep_item.get('selected_by_language_priority', False)
    changed_by_language = keep_item.get('changed_by_language_priority', False)
    priority_language = keep_item.get('priority_language_used', None)

    return {
        "keep": keep_item,
        "keep_url": keep_url,
        "delete": processed_delete_items,
        "has_deleted_items": has_deleted_items,
        "newest_date_added": newest_item['quality_description'].get('date_added', 'unknown'),
        "newest_path": _item_path(newest_item),
        "newest_status": "✓ This file is being kept" if newest_item['id'] == keep_item['id'] else "⚠️ This newer file is being deleted!",
        "oldest_date_added": oldest_item['quality_description'].get('date_added', 'unknown'),
        "oldest_path": _item_path(oldest_item),
        "oldest_status": "✓ This file is being kept despite being older" if oldest_item['id'] == keep_item['id'] else "",
        "selected_by_language": selected_by_language,
        "changed_by_language_priority": changed_by_language,
        "priority_language": priority_language,
        "language_priority_message": language_priority_message
    }


def _prepare_template_data(base_url: str, stats: dict[str, Any], language_priorities_used: bool,
                           language_priorities_changed_selection: bool, language_priorities_list: Any,
                           metadata: dict[str, Any]) -> dict[str, Any]:
    """Prepare template data structure with stats and metadata."""
    excluded_ids = metadata.get("excluded_ids", [])

    return {
        # Server info
        "base_url": base_url,

        # Stats data
        "total_groups": stats["total_groups"],
        "total_items_to_keep": stats["total_items_to_keep"],
        "total_items_to_delete": stats["total_items_to_delete"],
        "deleted_items": stats["deleted_items"],
        "failed_deletions": stats["failed_deletions"],
        "skipped_deletions": stats["skipped_deletions"],
        "formatted_size_to_keep": stats["formatted_size_to_keep"],
        "formatted_size_to_delete": stats["formatted_size_to_delete"],
        "percentage_saved": f"{stats['percentage_saved']:.1f}",

        # Language priority info
        "language_priorities_used": language_priorities_used,
        "language_priorities_changed_selection": language_priorities_changed_selection,
        "language_priorities_list": language_priorities_list if isinstance(language_priorities_list, (list, tuple)) else [],

        # Exclusion info
        "excluded_ids": excluded_ids,
        "has_excluded_ids": len(excluded_ids) > 0,
        "excluded_titles": metadata.get("excluded_titles", {}),
        "excluded_groups_count": metadata.get("excluded_groups_count", 0),
        "debug_metadata": metadata,

        # Groups data
        "duplicate_groups": []
    }


def _process_decisions_to_groups(valid_decisions: list[dict[str, Any]], base_url: str,
                                  template_data: dict[str, Any]) -> None:
    """Process decisions into template-friendly group format."""
    with tqdm(total=len(valid_decisions), desc="Preparing report data", unit="item") as progress_bar:
        for decision in valid_decisions:
            try:
                group_data = _process_decision_group(decision, base_url)
                template_data["duplicate_groups"].append(group_data)
                progress_bar.update(1)
            except Exception as e:
                logger.error(f"Error formatting decision group: {e}")
                continue


def _inline_group_posters(template_data: dict[str, Any]) -> None:
    """Replace every poster URL in the report with an embedded ``data:`` URI, in place.

    The scan builds image URLs with ``?api_key=<live key>`` so the browser can load
    them. Writing that into the report would put a working credential in a file that
    gets opened and shared, so the posters are fetched here (key sent as a header) and
    embedded instead. On failure the URL is kept but stripped of the key.
    """
    from emby_dedupe.reports.images import inline_images_in_place

    # Walks the entire context, not just duplicate_groups: the excluded-media section
    # renders posters too, and enumerating sections by hand let it leak the live key.
    inline_images_in_place(template_data)


def _log_rendering_error_details(template_data: dict[str, Any]) -> None:
    """Log detailed error information when template rendering fails."""
    logger.error("Template data structure:")
    logger.error(f"Number of duplicate groups: {len(template_data.get('duplicate_groups', []))}")

    # Debug the first group structure if available
    if not template_data.get('duplicate_groups'):
        return

    first_group = template_data['duplicate_groups'][0]
    logger.error(f"First group keep item structure: {type(first_group.get('keep'))}")

    if 'keep' not in first_group:
        return

    keep_item = first_group['keep']
    quality_desc = keep_item.get('quality_description', {})
    logger.error(f"Quality description type: {type(quality_desc)}")
    logger.error(f"Quality description keys: {list(quality_desc.keys()) if isinstance(quality_desc, dict) else 'Not a dict'}")

    if isinstance(quality_desc, dict) and 'video' in quality_desc:
        logger.error(f"Video section type: {type(quality_desc['video'])}")
        logger.error(f"Video section content: {quality_desc['video']}")


def format_html_report(base_url: str, decisions: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> str:
    """
    Formats decisions into a beautiful HTML report using Jinja2 templates.

    Args:
        base_url (str): The base URL of the Emby server.
        decisions (list): List of decision objects containing items to keep and delete.
        metadata (dict, optional): Additional metadata such as excluded IDs and language priorities.

    Returns:
        str: A fully formed HTML document as a string.
    """
    # jinja2 is a hard runtime dependency (pyproject) — import it plainly.
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    # Calculate statistics
    stats = calculate_report_statistics(decisions)

    # Filter decisions to only valid ones
    valid_decisions = _validate_decisions(decisions)

    # Check if language prioritization was used and if it changed any decisions
    language_priorities_used, language_priorities_changed_selection, language_priorities_list = _detect_language_priority_usage(valid_decisions)

    # Handle metadata if provided
    if metadata is None:
        metadata = {}

    # Prepare template data structure
    template_data = _prepare_template_data(
        base_url, stats, language_priorities_used,
        language_priorities_changed_selection, language_priorities_list, metadata
    )

    # Process each decision group into a template-friendly format
    _process_decisions_to_groups(valid_decisions, base_url, template_data)

    # Embed the posters and drop the API key: the scan builds image URLs carrying
    # ?api_key=..., which would write a live credential into this file.
    _inline_group_posters(template_data)

    # Set up Jinja2 environment
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(os.path.dirname(script_dir), "templates")

    env = Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=select_autoescape(['html', 'xml'])
    )

    # Load and render the template
    template = env.get_template('report.html')
    try:
        return template.render(**template_data)
    except Exception as e:
        logger.error(f"Template rendering error: {e}")
        _log_rendering_error_details(template_data)
        raise


def generate_html_report(base_url: str, decisions: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> str:
    """
    Generates an HTML report and saves it to a temporary file that can be opened in a browser.

    Args:
        base_url (str): The base URL of the Emby server.
        decisions (list): List of decision objects containing items to keep and delete.
        metadata (dict, optional): Additional metadata such as excluded IDs and language priorities.

    Returns:
        str: Path to the generated HTML file.
    """
    # Get package directory to locate static files
    import shutil

    # Generate the HTML report content
    html_content = format_html_report(base_url, decisions, metadata)

    # Create a temporary file with the HTML content
    # In a directory that will be accessible from a web browser
    temp_dir = tempfile.gettempdir()
    report_timestamp = int(time.time())
    temp_filename = f"emby_dedupe_report_{report_timestamp}.html"
    temp_path = os.path.join(temp_dir, temp_filename)

    # Copy the CSS file to the same directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(script_dir)
    css_src_path = os.path.join(pkg_dir, "static", "css", "report.css")
    css_dest_path = os.path.join(temp_dir, "report.css")

    try:
        # Copy the CSS file
        shutil.copy2(css_src_path, css_dest_path)
        logger.debug(f"CSS file copied to: {css_dest_path}")
    except OSError as e:
        # If copying fails, log the error but continue without the CSS
        logger.error(f"Failed to copy CSS file: {e}")

    # Write the HTML file
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"HTML report generated at: {temp_path}")
    return temp_path
