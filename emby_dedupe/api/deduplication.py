"""
Core deduplication logic for identifying and processing duplicate media items.
"""

import logging
import os
import re
from typing import Any

import httpx
from tqdm import tqdm

from emby_dedupe.api.client import delete_item, fetch_items_details
from emby_dedupe.api.deletion_guard import (
    collect_delete_paths,
    collect_known_paths,
    is_delete_safe,
)
from emby_dedupe.api.metadata import get_image_url, rate_media_items
from emby_dedupe.models.disjoint_set import DisjointSet
from emby_dedupe.reports import markdown
from emby_dedupe.utils.constants import LANGUAGE_NORMALIZATION_MAP
from emby_dedupe.utils.formatting import format_file_size
from emby_dedupe.utils.logging import logger
from emby_dedupe.utils.providers import PROVIDER_PRIORITY, normalize_provider_ids


def _extract_episode_key_from_path(filename: str) -> tuple[str | None, str | None]:
    """
    Extract season and episode numbers from a filename using common naming patterns.

    Supports multiple TV show naming conventions:
    - S01E01, s01e01 (standard)
    - 1x01 (alternate)
    - s01.e01 (dot separator)
    - s01_e01 (underscore separator)
    - 101, 102 (3-digit format: season 1 episode 1, season 1 episode 2)

    Args:
        filename: The filename to extract episode info from

    Returns:
        Tuple of (season, episode) as normalized strings, or (None, None) if no match
    """
    # Try standard patterns first (S01E01, s01e01)
    ep_match = re.search(r'[Ss](\d+)[Ee](\d+)', filename)

    # Try alternative common patterns if standard doesn't match
    if not ep_match:
        # 1x01 format
        ep_match = re.search(r'(\d+)[xX](\d+)', filename)
    if not ep_match:
        # s01.e01 format
        ep_match = re.search(r'[sS](\d+)\.?[eE](\d+)', filename)
    if not ep_match:
        # s01_e01 format
        ep_match = re.search(r'[sS](\d+)_[eE](\d+)', filename)
    if not ep_match:
        # 3-digit format like 101, 102 (season 1 episode 1, season 1 episode 2)
        # Only match if it's exactly 3 digits to avoid false positives
        ep_match = re.search(r'(?<!\d)([1-9])(\d{2})(?!\d)', filename)

    if ep_match:
        season, episode = ep_match.groups()
        # Normalize by removing leading zeros
        return str(int(season)), str(int(episode))

    return None, None


def _build_exclusion_map(excluded_ids) -> dict:
    """
    Parse excluded provider IDs into provider-type-keyed map.

    Args:
        excluded_ids: List of provider IDs to exclude

    Returns:
        Dictionary with keys 'imdb', 'tmdb', 'tvdb' containing filtered ID lists
    """
    excluded_ids = excluded_ids or []
    return {
        "imdb": [id.lower() for id in excluded_ids if id.lower().startswith("tt")],
        "tmdb": [id for id in excluded_ids if id.isdigit()],
        "tvdb": [id for id in excluded_ids if id.isdigit()]
    }


def _check_group_exclusion(items_details, exclusion_map) -> tuple[bool, str | None, dict | None]:
    """
    Check if any item in group has a provider ID in the exclusion list.

    Args:
        items_details: List of item detail dictionaries
        exclusion_map: Dictionary mapping provider types to excluded ID lists

    Returns:
        Tuple of (should_exclude, excluded_provider_id, excluded_item)
    """
    for item in items_details:
        provider_ids = item.get("ProviderIds", {})

        if provider_ids:
            # Use case-insensitive lookup (Emby API returns inconsistent casing)
            provider_ids_lower = normalize_provider_ids(provider_ids)
            imdb_id = provider_ids_lower.get("imdb", "").lower()
            tmdb_id = provider_ids_lower.get("tmdb", "")
            tvdb_id = provider_ids_lower.get("tvdb", "")

            # Check if this item should be excluded
            if imdb_id and imdb_id in exclusion_map["imdb"]:
                return True, imdb_id, item
            elif tmdb_id and tmdb_id in exclusion_map["tmdb"]:
                return True, tmdb_id, item
            elif tvdb_id and tvdb_id in exclusion_map["tvdb"]:
                return True, tvdb_id, item

    return False, None, None


def _format_file_size(size_bytes: int) -> str:
    """Format file size in bytes to human-readable string (unknown/zero → "Unknown")."""
    return format_file_size(size_bytes, zero_label="Unknown")


def _determine_resolution(width: int, height: int) -> str:
    """Determine resolution string from width and height."""
    if not (width and height):
        return "Unknown"

    if height >= 2160:
        return "4K"
    elif height >= 1080:
        return "1080p"
    elif height >= 720:
        return "720p"
    elif height >= 480:
        return "480p"
    else:
        return f"{width}x{height}"


def _extract_video_info(video_stream: dict) -> dict:
    """Extract video codec, resolution from video stream."""
    width = video_stream.get("Width", 0)
    height = video_stream.get("Height", 0)

    return {
        "codec": video_stream.get("Codec", "Unknown"),
        "resolution": _determine_resolution(width, height),
        "width": width,
        "height": height
    }


def _extract_audio_info(audio_streams: list) -> dict:
    """Extract audio codec, channels, languages from audio streams."""
    if not audio_streams:
        return {}

    audio_info = {
        "codec": audio_streams[0].get("Codec", "Unknown"),
        "channels": f"{audio_streams[0].get('Channels', 0)} ch",
        "languages": []
    }

    for stream in audio_streams:
        lang = stream.get("Language", "")
        if lang and lang not in audio_info["languages"]:
            audio_info["languages"].append(lang)

    return audio_info


def _extract_media_info(media_streams: list) -> dict:
    """Extract video and audio info from media streams."""
    media_info = {}

    video_stream: dict = next((s for s in media_streams if s.get("Type") == "Video"), {})
    if video_stream:
        media_info["video"] = _extract_video_info(video_stream)

    audio_streams = [s for s in media_streams if s.get("Type") == "Audio"]
    if audio_streams:
        media_info["audio"] = _extract_audio_info(audio_streams)

    return media_info


def _build_image_url(item_id: str, image_tags: dict, base_url: str, api_key: str) -> str:
    """Build image URL for primary image."""
    if not (item_id and image_tags and "Primary" in image_tags):
        return ""

    primary_tag = image_tags["Primary"]
    image_url = f"{base_url}/Items/{item_id}/Images/Primary?tag={primary_tag}&quality=90&maxHeight=300"
    if api_key:
        image_url += f"&api_key={api_key}"

    return image_url


def _format_title(item: dict) -> str:
    """Format title including series name if present."""
    title = item.get("Name", "Unknown")
    series_name = item.get("SeriesName", "")
    if series_name:
        return f"{series_name} - {title}"
    return title


def _extract_excluded_item_info(item, base_url, api_key) -> dict:
    """
    Extract comprehensive metadata for an excluded item (for reporting).

    Args:
        item: Excluded item dictionary
        base_url: Emby server base URL
        api_key: API key for image URLs

    Returns:
        Dictionary with complete item metadata
    """
    title = _format_title(item)
    item_id = item.get("Id")
    image_url = _build_image_url(item_id, item.get("ImageTags", {}), base_url, api_key)

    size_bytes = item.get("Size", 0)
    size_formatted = _format_file_size(size_bytes)
    media_info = _extract_media_info(item.get("MediaStreams", []))

    return {
        "id": item_id or "",
        "title": title,
        "year": item.get("ProductionYear", ""),
        "overview": item.get("Overview", ""),
        "path": item.get("Path", ""),
        "image_url": image_url,
        "size": size_bytes,
        "size_formatted": size_formatted,
        "media_info": media_info,
        "provider_ids": item.get("ProviderIds", {}),
        "server_id": item.get("ServerId", "")
    }


def _enrich_keep_item(keep_item, items_details, base_url, api_key) -> None:
    """
    Add image URL, group name, and episode metadata to keep item (in-place).

    Args:
        keep_item: The item to keep (mutated in-place)
        items_details: List of all item details
        base_url: Emby server base URL
        api_key: API key for image URLs
    """
    keep_details = next((item for item in items_details if item.get("Id") == keep_item["id"]), None)
    if keep_details:
        # Add image URL
        image_url = get_image_url(
            base_url,
            keep_details.get("Id", ""),
            keep_details.get("ImageTags", {}),
            keep_item.get("serverid", ""),
            api_key
        )
        keep_item["image_url"] = image_url
        keep_item["name"] = keep_details.get("Name", "Unknown Group")

        # Add episode metadata if TV episode
        keep_item["is_episode"] = "SeriesName" in keep_details
        if keep_item["is_episode"]:
            keep_item["series_name"] = keep_details.get("SeriesName", "")
            keep_item["season_number"] = keep_details.get("ParentIndexNumber", "")
            keep_item["episode_number"] = keep_details.get("IndexNumber", "")


def _extract_primary_provider_id(provider_ids: dict) -> str | None:
    """
    Extract primary provider ID with priority: IMDB > TMDB > TVDB.

    Args:
        provider_ids: Dictionary of provider IDs (case-insensitive)

    Returns:
        Primary provider ID or None if none found
    """
    provider_ids_lower = normalize_provider_ids(provider_ids)

    for provider in PROVIDER_PRIORITY:  # IMDB > TMDB > TVDB
        if provider in provider_ids_lower:
            return provider_ids_lower[provider]

    return None


def _enrich_delete_item(delete_item, items_details, base_url, keep_serverid, api_key) -> None:
    """
    Add image URL, provider IDs, and episode metadata to delete item (in-place).

    Args:
        delete_item: The item to delete (mutated in-place)
        items_details: List of all item details
        base_url: Emby server base URL
        keep_serverid: Server ID of the kept item
        api_key: API key for image URLs
    """
    delete_details = next((item for item in items_details if item.get("Id") == delete_item["id"]), None)
    if delete_details:
        # Add image URL
        image_url = get_image_url(
            base_url,
            delete_details.get("Id", ""),
            delete_details.get("ImageTags", {}),
            keep_serverid,
            api_key
        )
        delete_item["image_url"] = image_url

        # Extract provider IDs for fallback URLs
        if "ProviderIds" in delete_details:
            provider_ids = delete_details.get("ProviderIds", {})
            delete_item["provider_ids"] = provider_ids

            # Extract primary provider ID (IMDB > TMDB > TVDB priority)
            primary_id = _extract_primary_provider_id(provider_ids)
            if primary_id:
                delete_item["provider_id"] = primary_id

        # Add episode metadata if TV episode
        delete_item["is_episode"] = "SeriesName" in delete_details
        if delete_item["is_episode"]:
            delete_item["series_name"] = delete_details.get("SeriesName", "")
            delete_item["season_number"] = delete_details.get("ParentIndexNumber", "")
            delete_item["episode_number"] = delete_details.get("IndexNumber", "")


def _collect_items_metadata(media_items_by_provider) -> dict:
    """
    Build dictionary of item_id to item_data from provider tables.

    Args:
        media_items_by_provider: Dictionary containing media items grouped by provider

    Returns:
        Dictionary mapping item IDs to their full metadata
    """
    all_items_dict = {}
    for provider, id_table in media_items_by_provider.items():
        if provider == "library_name":
            continue
        # Flatten nested iteration
        for items in id_table.values():
            for item in items:
                if isinstance(item, dict) and "id" in item:
                    all_items_dict[item["id"]] = item

    return all_items_dict


def _group_by_disjoint_root(ds) -> dict:
    """
    Create groups dictionary from disjoint set parent map.

    Args:
        ds: DisjointSet with parent mappings

    Returns:
        Dictionary mapping root IDs to sets of grouped item IDs
    """
    groups = {}
    with tqdm(total=len(ds.parent), desc="Grouping duplicates", unit="item") as grouping_progress:
        for item in ds.parent:
            root = ds.find(item)

            if root not in groups:
                groups[root] = {item}
            else:
                groups[root].add(item)

            grouping_progress.update(1)

    return groups


def _verify_movie_group(items, all_items_dict) -> tuple[bool, set]:
    """
    Check if group contains only movies (not TV episodes) and collect provider IDs.

    Args:
        items: Set of item IDs to check
        all_items_dict: Dictionary of item metadata

    Returns:
        Tuple of (is_movie_group boolean, set of provider IDs)
    """
    is_movie_group = True
    movie_providers = set()

    for item_id in items:
        item_data = all_items_dict.get(item_id, {})

        if item_data.get("is_episode", False):
            is_movie_group = False
            break

        provider_id = item_data.get("provider_id", "")
        if provider_id:
            movie_providers.add(provider_id)

    return is_movie_group, movie_providers


def _create_series_key(item_data) -> str:
    """
    Create series grouping key from item metadata.

    Generates key like "SeriesName|S1E1" or "SeriesName|S1E1|PATH_S1E1" with path verification.

    Args:
        item_data: Item metadata dictionary

    Returns:
        Series grouping key string
    """
    series_name = item_data.get("series_name", "")
    season_num = item_data.get("season_number", "")
    episode_num = item_data.get("episode_number", "")

    if not series_name:
        return "NON_SERIES"

    if season_num and episode_num:
        norm_season = str(int(season_num)) if season_num else ""
        norm_episode = str(int(episode_num)) if episode_num else ""
        series_key = f"{series_name}|S{norm_season}E{norm_episode}"

        # Add path-based verification
        path = item_data.get("path", "")
        if path:
            filename = os.path.basename(path)
            path_season, path_episode = _extract_episode_key_from_path(filename)
            if path_season and path_episode:
                series_key = f"{series_key}|PATH_S{path_season}E{path_episode}"

        return series_key
    else:
        return series_name


def _verify_tv_series_group(items, all_items_dict) -> dict:
    """
    Group TV items by series/season/episode with path verification.

    Creates series groups with keys like: "SeriesName|S1E1|PATH_S1E1"
    Uses path extraction for extra verification to prevent false grouping.

    Args:
        items: Set of item IDs to group
        all_items_dict: Dictionary of item metadata

    Returns:
        Dictionary mapping series keys to sets of item IDs
    """
    series_groups: dict = {}
    for item_id in items:
        item_data = all_items_dict.get(item_id, {})

        # Create series key using helper
        series_key = _create_series_key(item_data)

        if series_key not in series_groups:
            series_groups[series_key] = set()
        series_groups[series_key].add(item_id)

    return series_groups


def _classify_item_by_episode_path(item) -> tuple[str | None, bool]:
    """
    Classify a single item by extracting episode info from path.

    Args:
        item: Item dictionary with Path and SeriesName

    Returns:
        Tuple of (path_key or None, is_non_episode bool)
    """
    item_path = item.get("Path", "")
    series_name = item.get("SeriesName", "")

    if series_name and item_path:
        filename = os.path.basename(item_path)
        path_season, path_episode = _extract_episode_key_from_path(filename)

        if path_season and path_episode:
            path_key = f"{series_name}|S{path_season}E{path_episode}"

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Item {item.get('Id', 'unknown')} - Path-extracted episode: S{path_season}E{path_episode} - {filename}")

            return path_key, False

    return None, True


def _group_items_by_episode_path(all_items_details) -> tuple[list, bool]:
    """
    Group TV items by series/season/episode pattern extracted from file path.

    For TV series, extracts episode info from filenames and groups items with matching
    series/season/episode. If multiple episode groups are found (false grouping), returns
    only the largest group plus non-episode items.

    Args:
        all_items_details: List of media item details with Path and SeriesName

    Returns:
        Tuple of (filtered_items list, is_movie_group bool)
    """
    episode_path_groups: dict = {}
    non_episode_items = []

    for item in all_items_details:
        path_key, is_non_episode = _classify_item_by_episode_path(item)

        if is_non_episode:
            non_episode_items.append(item)
        else:
            if path_key not in episode_path_groups:
                episode_path_groups[path_key] = []
            episode_path_groups[path_key].append(item)

    # Check if we have multiple episode groups - if so, this is likely a false grouping
    if len(episode_path_groups) > 1:
        largest_group = max(episode_path_groups.values(), key=len)
        filtered_items = largest_group + non_episode_items

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Found {len(episode_path_groups)} different episodes in this group - separating them")
            logger.debug(f"Proceeding with {len(filtered_items)} items after episode-based filtering")

        return filtered_items, False

    # Return all items if single episode group or no episodes
    all_items = non_episode_items
    for group in episode_path_groups.values():
        all_items.extend(group)

    is_movie_group = not any(item.get("SeriesName") for item in all_items)
    return all_items if all_items else all_items_details, is_movie_group


def _deduplicate_by_path(items_details) -> list:
    """
    Remove items with duplicate file paths (keep the first item per unique path).

    Movies and TV share one strategy — the earlier movie/TV split was behaviourally
    identical apart from a debug line. Pathless items are skipped.

    Args:
        items_details: List of items to deduplicate

    Returns:
        List of items with unique paths
    """
    unique_items = []
    seen_paths: dict = {}

    for item in items_details:
        item_path = item.get("Path")
        item_id = item.get("Id", "unknown")
        if not item_path:
            continue

        if item_path in seen_paths:
            logger.debug(f"Skipping item with duplicate path: {item_path} (ID: {item_id})")
            logger.debug(f"Already seen with ID: {seen_paths[item_path]}")
            continue

        unique_items.append(item)
        seen_paths[item_path] = item_id

    return unique_items


def _calculate_language_scores(rated_items, lang_priorities) -> None:
    """
    Calculate language priority scores for all items (mutates in-place).

    Adds three fields to each item:
    - lang_priority: Numeric score (0=highest, 9999=no match)
    - has_priority_lang: Boolean
    - priority_language: Highest priority language code or None

    Args:
        rated_items: List of rated items to score
        lang_priorities: List of language codes in priority order
    """
    lang_mapping = LANGUAGE_NORMALIZATION_MAP

    for item in rated_items:
        languages = item.get("quality_description", {}).get("audio", {}).get("languages", [])
        languages = [lang.lower() for lang in languages if lang and lang != "unknown"]
        normalized_languages = [lang_mapping.get(lang, lang) for lang in languages]

        lang_score = 9999
        highest_prio_lang = None
        for lang in normalized_languages:
            if lang in lang_priorities:
                priority_pos = lang_priorities.index(lang)
                if priority_pos < lang_score:
                    lang_score = priority_pos
                    highest_prio_lang = lang
                logger.debug(f"Item {item['id']} has priority language '{lang}' (priority {priority_pos})")

        item["lang_priority"] = lang_score
        item["has_priority_lang"] = lang_score < 9999
        item["priority_language"] = highest_prio_lang


def _get_clean_languages(item) -> list:
    """Extract and clean language list from item."""
    languages = item.get("quality_description", {}).get("audio", {}).get("languages", [])
    return [lang for lang in languages if lang and lang != "unknown"]


def _log_override_decision(override: bool, is_single_lang: bool, best_quality_item: dict, best_quality_langs: list, best_lang_item: dict, best_lang_langs: list, quality_ratio: float) -> None:
    """Log the override decision with appropriate message."""
    if override:
        if is_single_lang:
            logger.info(f"Quality override (single-lang): Keeping multi-language item {best_quality_item['id']} " +
                      f"(languages: {best_quality_langs}, quality: {best_quality_item['rating']:.1f}) " +
                      f"over single-language higher-priority item {best_lang_item['id']} " +
                      f"(language: {best_lang_langs}, quality: {best_lang_item['rating']:.1f})")
        else:
            scenario = "both-priority-lang" if best_quality_item.get("has_priority_lang") else "no-priority-lang"
            logger.info(f"Quality override ({scenario}): Keeping better quality item {best_quality_item['id']} " +
                      f"(languages: {best_quality_langs}, quality: {best_quality_item['rating']:.1f}, ratio: {quality_ratio:.2f}x) " +
                      f"over priority language item {best_lang_item['id']} " +
                      f"(languages: {best_lang_langs}, quality: {best_lang_item['rating']:.1f})")


def _add_selection_metadata(top_item, default_top_item, lang_priorities) -> None:
    """Add selection metadata to the chosen item (in-place)."""
    decision_changed = default_top_item and top_item["id"] != default_top_item["id"]

    if top_item.get("has_priority_lang"):
        prio_lang = top_item.get("priority_language")

        # Log selection reason
        if decision_changed:
            logger.info(f"Language priority changed selection: Selected item {top_item['id']} (language '{prio_lang}') instead of item {default_top_item['id']} (higher quality)")
        else:
            logger.debug(f"Selected item {top_item['id']} based on priority language '{prio_lang}' and quality rating {top_item['rating']}")

        # Set metadata fields
        top_item["selected_by_language_priority"] = True
        top_item["changed_by_language_priority"] = decision_changed
        top_item["priority_language_used"] = prio_lang
        top_item["language_priority_list"] = lang_priorities
    else:
        logger.info(f"No items have priority languages. Selected item {top_item['id']} based on quality rating {top_item['rating']}")
        top_item["selected_by_language_priority"] = False
        top_item["changed_by_language_priority"] = False
        top_item["language_priority_list"] = lang_priorities


def _apply_smart_override_and_sort(rated_items, lang_priorities, default_top_item) -> None:
    """
    Apply smart language override logic and sort items (mutates in-place).

    Implements smart override using should_quality_override_language(), which
    evaluates all three scenarios (single-vs-multi-lang, no-priority-lang,
    both-priority-lang) against the OVERRIDE_RATIO_* thresholds in
    utils/constants.py.

    Adds selection metadata to top item after sorting.

    Args:
        rated_items: List of rated items with language scores (sorted in-place)
        lang_priorities: List of language priority codes
        default_top_item: Top item by quality alone (for comparison)
    """
    from emby_dedupe.utils.constants import should_quality_override_language

    best_quality_item = max(rated_items, key=lambda x: x["rating"])
    best_lang_items = [item for item in rated_items if item["has_priority_lang"]]

    if best_lang_items:
        best_lang_item = min(best_lang_items, key=lambda x: (x["lang_priority"], -x["rating"]))

        if best_quality_item["id"] != best_lang_item["id"]:

            best_quality_langs = _get_clean_languages(best_quality_item)
            best_lang_langs = _get_clean_languages(best_lang_item)

            quality_ratio = best_quality_item["rating"] / best_lang_item["rating"] if best_lang_item["rating"] > 0 else float('inf')

            logger.debug("Smart language priority check: " +
                       f"Best quality: ID {best_quality_item['id']}, langs: {best_quality_langs}, rating: {best_quality_item['rating']:.1f} | " +
                       f"Best lang: ID {best_lang_item['id']}, langs: {best_lang_langs}, rating: {best_lang_item['rating']:.1f} | " +
                       f"Quality ratio: {quality_ratio:.2f}")

            # Use shared smart override function
            is_single_lang_scenario = len(best_lang_langs) == 1 and len(best_quality_langs) >= 2
            override = should_quality_override_language(
                quality_ratio=quality_ratio,
                lang_item_has_priority_lang=best_lang_item["has_priority_lang"],
                quality_item_has_priority_lang=best_quality_item["has_priority_lang"],
                is_single_lang_scenario=is_single_lang_scenario
            )

            # Log and apply override decision
            _log_override_decision(override, is_single_lang_scenario, best_quality_item, best_quality_langs, best_lang_item, best_lang_langs, quality_ratio)

            if override:
                rated_items.sort(key=lambda x: -x["rating"])
            else:
                rated_items.sort(key=lambda x: (not x["has_priority_lang"], x["lang_priority"], -x["rating"]))
        else:
            rated_items.sort(key=lambda x: (not x["has_priority_lang"], x["lang_priority"], -x["rating"]))
    else:
        rated_items.sort(key=lambda x: -x["rating"])

    # Add selection metadata to top item
    _add_selection_metadata(rated_items[0], default_top_item, lang_priorities)


def _initialize_disjoint_set_and_calculate_total(media_items_by_provider) -> tuple[DisjointSet, int]:
    """
    Initialize DisjointSet and calculate total number of items to process.

    Args:
        media_items_by_provider: Dictionary containing media items grouped by provider

    Returns:
        Tuple of (DisjointSet instance, total item count)
    """
    ds = DisjointSet()

    total_items = sum(
        len(items)
        for provider_dict in media_items_by_provider.values()
        if isinstance(provider_dict, dict)  # Skip non-dict values like 'library_name'
        for items in provider_dict.values()
    )

    return ds, total_items


def _classify_single_item(item, ds) -> tuple[str | None, bool]:
    """
    Classify a single item as TV episode or movie.

    Args:
        item: Media item to classify
        ds: DisjointSet to initialize item in

    Returns:
        Tuple of (episode_key or None, is_movie bool)
    """
    item_id = item["id"] if isinstance(item, dict) else item

    if item_id not in ds.parent:
        ds.parent[item_id] = item_id

    # Check if it's a TV episode
    if isinstance(item, dict) and item.get("is_episode", False):
        series_name = item.get("series_name", "")
        season_num = item.get("season_number")
        episode_num = item.get("episode_number")

        if season_num is not None and episode_num is not None:
            item_provider_id = item.get("provider_id", "unknown")
            episode_key = f"{item_provider_id}|{series_name}|S{season_num}E{episode_num}"
            return episode_key, False

    return None, True


def _classify_items_by_type(items, ds) -> tuple[dict, list]:
    """
    Classify items into TV episode groups and movie items.

    TV episodes are grouped by provider_id|series|season|episode key.
    Movies and items without season/episode info are returned as a list.

    Args:
        items: List of media items to classify
        ds: DisjointSet to ensure all items are initialized

    Returns:
        Tuple of (tv_episode_groups dict, movie_items list)
    """
    tv_episode_groups: dict = {}
    movie_items = []

    for item in items:
        episode_key, is_movie = _classify_single_item(item, ds)

        if is_movie:
            movie_items.append(item)
        else:
            if episode_key not in tv_episode_groups:
                tv_episode_groups[episode_key] = []
            tv_episode_groups[episode_key].append(item)

    return tv_episode_groups, movie_items


def _union_episode_groups(ds, tv_episode_groups) -> int:
    """
    Union all items within each TV episode group.

    Args:
        ds: DisjointSet to perform unions on
        tv_episode_groups: Dictionary of episode keys to item lists

    Returns:
        Count of union operations performed
    """
    update_count = 0

    if logger.isEnabledFor(logging.DEBUG):
        for key, group in tv_episode_groups.items():
            if len(group) > 1:
                logger.debug(f"Found episode group: {key} with {len(group)} items")

    for episode_group in tv_episode_groups.values():
        if len(episode_group) > 1:
            # Union all items in this specific episode group
            first_item = episode_group[0]
            for other_item in episode_group[1:]:
                ds.union(first_item, other_item)
                update_count += 1

    return update_count


def _union_movie_groups(ds, movie_items) -> int:
    """
    Group movies by provider ID and union within each group.

    Args:
        ds: DisjointSet to perform unions on
        movie_items: List of movie items to group

    Returns:
        Count of union operations performed
    """
    update_count = 0

    # Group movies by provider ID
    movie_groups_by_provider: dict = {}
    for item in movie_items:
        item_provider_id = item.get("provider_id", "unknown")

        if item_provider_id not in movie_groups_by_provider:
            movie_groups_by_provider[item_provider_id] = []
        movie_groups_by_provider[item_provider_id].append(item)

    if logger.isEnabledFor(logging.DEBUG):
        for provider_id, group in movie_groups_by_provider.items():
            if len(group) > 1:
                logger.debug(f"Found movie group with provider ID: {provider_id} ({len(group)} items)")

    for movie_group in movie_groups_by_provider.values():
        if len(movie_group) > 1:
            first_item = movie_group[0]
            for other_item in movie_group[1:]:
                ds.union(first_item, other_item)
                update_count += 1

    return update_count


def identify_duplicates(provider_tables: dict, excluded_ids: list = None) -> dict:
    """
    Identifies duplicates by looking for provider IDs with multiple associated media item IDs.
    Skips any provider IDs that are in the excluded_ids list.

    Args:
        provider_tables (dict): The table of provider IDs and associated media item IDs.
        excluded_ids (list, optional): List of provider IDs to exclude from deduplication.

    Returns:
        dict: A dictionary of provider IDs and list of duplicate media item IDs.
    """
    from emby_dedupe.utils.logging import logger

    excluded_ids = excluded_ids or []
    excluded_count = 0
    duplicates: dict = {}

    for provider, id_table in provider_tables.items():
        duplicates[provider] = {}
        for pid, items in id_table.items():
            # Skip this provider ID if it's in the excluded list
            if pid in excluded_ids:
                if len(items) > 1:
                    excluded_count += 1
                    logger.debug(f"Skipping excluded provider ID {pid} with {len(items)} items")
                continue

            # Only include items with more than one entry (duplicates)
            if len(items) > 1:
                duplicates[provider][pid] = items

    if excluded_count > 0:
        logger.info(f"Excluded {excluded_count} provider IDs with multiple items from deduplication")

    return duplicates


def _initialize_items_in_disjoint_set(ds, provider_items, items_progress):
    """
    Initialize all items in the disjoint set structure.

    Args:
        ds: DisjointSet instance
        provider_items: Dictionary mapping provider_id to list of items
        items_progress: tqdm progress bar

    Returns:
        None (modifies ds in place)
    """
    for provider_id, items in provider_items.items():
        for item in items:
            item_id = item["id"] if isinstance(item, dict) else item
            if item_id not in ds.parent:
                ds.parent[item_id] = item_id
            items_progress.update(1)


def _process_provider_items(ds, provider_items):
    """
    Classify and union all related items for a provider (mutates ``ds`` in place).

    Args:
        ds: DisjointSet instance
        provider_items: Dictionary mapping provider_id to list of items

    Returns:
        Total number of union (merge) operations performed.
    """
    merge_count = 0
    for items in provider_items.values():
        tv_episode_groups, movie_items = _classify_items_by_type(items, ds)
        merge_count += _union_episode_groups(ds, tv_episode_groups)
        merge_count += _union_movie_groups(ds, movie_items)

    return merge_count


def build_disjoint_set(media_items_by_provider):
    """
    Builds a disjoint set structure to efficiently group related media items.
    Handles TV series episodes by grouping only episodes from the same season and episode number.

    Args:
        media_items_by_provider: Dictionary containing media items grouped by provider.

    Returns:
        DisjointSet: The constructed disjoint set.
    """
    # Initialize disjoint set and calculate total items
    ds, total_items = _initialize_disjoint_set_and_calculate_total(media_items_by_provider)
    logger.debug(f"Building sets: processing {total_items} total items")

    merge_count = 0
    with tqdm(total=total_items, desc="Building sets", unit="item") as items_progress:
        for provider in media_items_by_provider:
            if provider == "library_name":
                continue

            provider_items = media_items_by_provider[provider]

            # Initialize all items in disjoint set first, then union related ones.
            _initialize_items_in_disjoint_set(ds, provider_items, items_progress)
            merge_count += _process_provider_items(ds, provider_items)

    logger.debug(f"Building sets complete: {merge_count} merges across {total_items} items")

    return ds


def _verify_and_categorize_group(root: str, items: list[str], all_items_dict: dict[str, Any],
                                  verified_groups: dict[str, list[str]]) -> None:
    """Verify group type (movie or TV) and categorize appropriately."""
    if len(items) <= 1:
        return  # Skip single-item groups

    # Check if this is a movie group
    is_movie_group, movie_providers = _verify_movie_group(items, all_items_dict)

    if is_movie_group:
        # Movies are already grouped by provider ID - trust that grouping
        verified_groups[root] = items

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Movie group verification - group with {len(items)} items and provider IDs: {', '.join(movie_providers)}")
    else:
        # For TV shows, use strict series/season/episode verification
        series_groups = _verify_tv_series_group(items, all_items_dict)

        # Only add groups that have more than one item
        for series_name, series_items in series_groups.items():
            if len(series_items) > 1:
                series_root = next(iter(series_items))  # Use first item as root
                verified_groups[series_root] = series_items

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Series group verification - {series_name} with {len(series_items)} items")


def rationalize_duplicates(media_items_by_provider):
    """
    Rationalizes duplicates by grouping related items using disjoint sets.

    Args:
        media_items_by_provider: Dictionary containing media items grouped by provider.

    Returns:
        list: A list of lists, where each inner list contains IDs of duplicate items.
    """
    # Step 1: Collect all items with their metadata
    all_items_dict = _collect_items_metadata(media_items_by_provider)

    # Step 2: Build the disjoint set
    ds = build_disjoint_set(media_items_by_provider)

    # Step 3: Group items by their disjoint set root
    groups = _group_by_disjoint_root(ds)

    # Step 4: Verify groups (movies vs TV series)
    verified_groups = {}

    for root, items in groups.items():
        _verify_and_categorize_group(root, items, all_items_dict, verified_groups)

    # Step 5: Assemble final rationalized list
    rationalized_list = [list(group) for group in verified_groups.values() if len(group) > 1]

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Original groups: {len([g for g in groups.values() if len(g) > 1])}, Verified groups: {len(rationalized_list)}")

    return rationalized_list




def determine_items_to_delete(duplicate_ids: list, all_items_details: list, lang_priorities: list = None) -> dict:
    """
    Determines the best quality media item to keep and marks the rest for deletion.

    The criteria for the best quality is based on resolution, audio channels, bitrate, file size,
    and optionally language preferences. Also filters out false positives where multiple items
    have the same file path.

    Args:
        duplicate_ids (list): A list of IDs of potentially duplicate media items.
        all_items_details (list): Detailed media items information, including MediaStreams.
        lang_priorities (list, optional): List of language codes in order of priority.

    Returns:
        dict: A dictionary containing details of the item to keep and the ones to delete.
    """
    from emby_dedupe.utils.logging import logger

    # Step 1: Group items by episode path, handle multi-episode false groupings
    filtered_items, _ = _group_items_by_episode_path(all_items_details)

    # Step 2: Deduplicate by path (keep first item per unique path)
    unique_path_items = _deduplicate_by_path(filtered_items)

    # If filtering left only 0 or 1 items, this isn't a real duplicate group
    if len(unique_path_items) <= 1:
        logger.debug(f"After filtering duplicates, no true duplicates remain in group with IDs: {duplicate_ids}")
        return {"keep": {}, "delete": []}

    # Step 3: Rate items by quality
    rated_items = rate_media_items(unique_path_items)

    if not rated_items:
        logger.debug(f"No rated items found for duplicate IDs: {duplicate_ids}.")
        return {"keep": {}, "delete": []}

    # Get default top choice (quality-only)
    quality_sorted_items = sorted(rated_items, key=lambda x: x["rating"], reverse=True)
    default_top_item = quality_sorted_items[0] if quality_sorted_items else None

    # Step 4: Apply language priority if specified
    if lang_priorities and len(lang_priorities) > 0:
        logger.debug(f"Applying language prioritization: {lang_priorities}")

        # Calculate language scores for all items
        _calculate_language_scores(rated_items, lang_priorities)

        # Apply smart override and sort
        _apply_smart_override_and_sort(rated_items, lang_priorities, default_top_item)
    else:
        # No language priorities - sort by quality only
        rated_items.sort(key=lambda x: x["rating"], reverse=True)
        logger.debug(f"Selected item {rated_items[0]['id']} based on quality rating {rated_items[0]['rating']}")

        # Mark that language prioritization was not used
        rated_items[0]["selected_by_language_priority"] = False
        rated_items[0]["changed_by_language_priority"] = False

    # The first item in the sorted list is the one to keep; the rest are duplicates
    item_to_keep = rated_items[0]
    items_to_delete = rated_items[1:]

    return {"keep": item_to_keep, "delete": items_to_delete}




def _enrich_and_add_decision(decision: dict, items_details: list, base_url: str, api_key: str, decisions: list) -> None:
    """
    Enrich decision with metadata and add to decisions list.

    Args:
        decision: Decision dictionary with keep/delete items.
        items_details: List of all item details.
        base_url: Emby server base URL.
        api_key: API key for image URLs.
        decisions: List to append decision to.
    """
    if not decision.get("keep") or not decision.get("delete"):
        return

    keep_item = decision["keep"]

    # Enrich keep item with image and metadata
    _enrich_keep_item(keep_item, items_details, base_url, api_key)

    # Set group name from keep item
    decision["name"] = keep_item.get("name", "Unknown Group")

    # Track episode metadata for reporting
    decision["is_episode"] = keep_item.get("is_episode", False)
    if decision["is_episode"]:
        decision["series_name"] = keep_item.get("series_name", "")

    # Enrich each delete item with image and provider metadata
    for item_to_delete in decision["delete"]:
        _enrich_delete_item(item_to_delete, items_details, base_url, keep_item.get("serverid", ""), api_key)

    decisions.append(decision)


def process_duplicate_groups(
    client: httpx.Client, base_url: str, duplicate_groups: list, api_key: str = None,
    lang_priorities: list = None, excluded_ids: list = None
) -> tuple[list, dict]:
    """
    Processes each group of duplicate items to identify the item to keep and the ones to delete.

    Args:
        client (httpx.Client): The httpx client configured for the Emby server communication.
        base_url (str): Base URL of the Emby server.
        duplicate_groups (list): A list of lists, where each inner list contains IDs of potentially duplicate items.
        api_key (str, optional): API key for authentication with image URLs.
        lang_priorities (list, optional): List of language codes in order of priority.
        excluded_ids (list, optional): List of provider IDs to exclude from deduplication.

    Returns:
        Tuple of (decisions list, exclusion_metadata dict)
    """
    # Build exclusion map from excluded IDs
    exclusion_map = _build_exclusion_map(excluded_ids)

    # Track statistics
    excluded_groups_count = 0
    excluded_titles = {}
    decisions: list = []

    with tqdm(total=len(duplicate_groups), desc="Processing duplicate groups", unit="group") as progress_bar:
        for group in duplicate_groups:
            # Fetch details for all items in the group
            items_details = fetch_items_details(client, base_url, group)

            if not items_details:
                progress_bar.update(1)
                continue

            # Check if group should be excluded
            should_exclude, excluded_provider_id, excluded_item = _check_group_exclusion(items_details, exclusion_map)

            if should_exclude and excluded_item:
                # Extract and store excluded item info for reporting
                excluded_groups_count += 1
                excluded_info = _extract_excluded_item_info(excluded_item, base_url, api_key)
                excluded_titles[excluded_provider_id] = excluded_info

                logger.debug(f"Skipping group with excluded provider ID {excluded_provider_id}: {excluded_info['title']}. Items: {[i.get('Id') for i in items_details]}")
                progress_bar.update(1)
                continue

            # Process the group normally if no exclusions apply
            try:
                decision = determine_items_to_delete(group, items_details, lang_priorities)
                if decision:
                    _enrich_and_add_decision(decision, items_details, base_url, api_key or "", decisions)
            except Exception as e:
                logger.error(f"Error processing group: {e}")
                logger.debug(f"Group details: {group}")
                continue

            progress_bar.update(1)

    logger.debug(f"Processed {len(duplicate_groups)} duplicate groups: {len(decisions)} decisions, {excluded_groups_count} excluded groups")

    # Return decisions with exclusion metadata
    exclusion_metadata = {
        "excluded_groups_count": excluded_groups_count,
        "excluded_titles": excluded_titles
    }

    return decisions, exclusion_metadata


def _mark_guard_refused(item: dict, keeper_path: str | None, reason: str) -> None:
    """Tag a delete item the safety guard refused, so the fold-safe deletion pass can find
    it later. Records the keeper + delete paths and the reason. Set on BOTH the --doit and
    dry-run guard paths, so the marker reflects the guard's actual decision in either mode.
    See :mod:`emby_dedupe.api.fold_safe_delete`.
    """
    item["fold_safe_candidate"] = {
        "keeper": keeper_path,
        "delete": item.get("path"),
        "reason": reason,
    }


def _warn_unsafe_deletions(
    decisions: list, known_paths: list, delete_paths: list = None
) -> None:
    """Dry-run visibility: log any deletion the safety guard would refuse under
    ``--doit`` (keeper co-located in a folder Emby would fold-delete), so the user
    sees it before running for real.
    """
    unsafe = 0
    for decision in decisions:
        keeper_path = (decision.get("keep") or {}).get("path")
        for item in decision.get("delete", []):
            safe, reason = is_delete_safe(
                keeper_path, item.get("path"), known_paths, delete_paths
            )
            if not safe:
                unsafe += 1
                _mark_guard_refused(item, keeper_path, reason)
                logger.warning(
                    "SAFETY GUARD would SKIP deletion of id=%s under --doit — %s "
                    "(delete=%r keeper=%r).",
                    item["id"], reason, item.get("path"), keeper_path,
                )
    if unsafe:
        logger.warning(
            "%d deletion(s) would be SKIPPED by the safety guard to protect co-located "
            "keepers. Fix the file layout (move the keeper out of the duplicate's folder) "
            "before --doit if you want them removed.", unsafe,
        )


def _mark_items_as_not_attempted(decisions: list, progress_bar) -> None:
    """
    Mark all items in decisions as not attempted (for dry-run mode).

    Args:
        decisions: List of decision groups
        progress_bar: tqdm progress bar to update
    """
    for decision in decisions:
        for item in decision.get("delete", []):
            deletion_status = {
                "id": item["id"],
                "status": "not_attempted",
                "error": None,
            }
            item["deletion_result"] = deletion_status
            progress_bar.update(1)


def _resolve_image_url_for_deleted_item(item: dict, item_group: dict | None) -> str:
    """
    Resolve the best image URL for a deleted item.

    Tries in order:
    1. Image URL from the kept item in the same group
    2. TMDB fallback using provider_id
    3. TMDB fallback using provider_ids dictionary
    4. Returns empty string if no URL found

    Args:
        item: The item being deleted
        item_group: The decision group containing the item (or None)

    Returns:
        Resolved image URL string
    """
    # Try to use the kept item's image URL
    if item_group and "keep" in item_group and "image_url" in item_group["keep"]:
        kept_image_url = item_group["keep"]["image_url"]
        logger.debug(f"Using kept item's image URL for deleted item {item.get('name')}: {kept_image_url}")
        return kept_image_url

    # Fallback to TMDB if we have a numeric provider_id
    if "provider_id" in item and item["provider_id"].isdigit():
        provider_id = item["provider_id"]
        fallback_url = f"https://image.tmdb.org/t/p/w300/{provider_id}.jpg"
        logger.debug(f"Using TMDB fallback image URL for {item.get('name')}: {fallback_url}")
        return fallback_url

    # If we have provider_ids dictionary, try TMDB (case-insensitive)
    if "provider_ids" in item and item["provider_ids"]:
        pids_lower = normalize_provider_ids(item["provider_ids"])
        if "tmdb" in pids_lower:
            tmdb_id = pids_lower["tmdb"]
            fallback_url = f"https://image.tmdb.org/t/p/w300/{tmdb_id}.jpg"
            logger.debug(f"Using TMDB fallback image URL from provider_ids for {item.get('name')}: {fallback_url}")
            return fallback_url

    return ""


def _extract_original_item_data(item: dict) -> dict:
    """
    Extract and preserve original item data before deletion.

    Args:
        item: Item dictionary to extract data from

    Returns:
        Dictionary containing preserved item data
    """
    return {
        "id": item.get("id", ""),
        "name": item.get("name", ""),
        "image_url": item.get("image_url", ""),
        "quality_description": item.get("quality_description", {}),
        "is_episode": item.get("is_episode", False),
        "series_name": item.get("series_name", ""),
        "season_number": item.get("season_number", ""),
        "episode_number": item.get("episode_number", "")
    }


def _execute_one_deletion(
    client, base_url, item, keeper_path, decision, known_paths, delete_paths,
    username, password, api_key, progress_bar,
) -> None:
    """Delete a single item unless the safety guard refuses it, then restore the item's
    display data (lost during deletion) and advance the progress bar. Extracted from the
    deletion loop to keep the caller's cognitive complexity in check. ``decision`` is the
    enclosing keep/delete group the caller is iterating — used to resolve the deleted
    item's display image from its kept sibling. Only reached on the doit path (the
    not-doit branch returns earlier via _mark_items_as_not_attempted).
    """
    progress_bar.set_description(f"Deleting ID: {item['id']}")

    # Store the original item data before deletion, incl. a resolved image URL.
    original_item_data = _extract_original_item_data(item)
    try:
        resolved_url = _resolve_image_url_for_deleted_item(item, decision)
        if resolved_url:
            original_item_data["image_url"] = resolved_url
    except Exception as e:
        logger.warning(f"Error setting image URL for deleted item: {e}")

    # SAFETY GUARD: never issue an Emby delete that would fold-delete a folder holding
    # the keeper (the data-loss bug). Refuse and skip — keep both files.
    safe, reason = is_delete_safe(keeper_path, item.get("path"), known_paths, delete_paths)
    if not safe:
        # Pre-format into ONE f-string with no positional args, so logging never runs
        # ``msg % args`` (immune to stray % in paths or an arg-count mismatch — this line
        # crashed twice before via stale bytecode).
        # WARNING, not ERROR: refusing is the guard working as designed, and nothing has
        # failed. Logging it at ERROR made correct, protective behaviour look like a fault.
        # The wording must not promise "both files kept" either — with --fold-safe-delete
        # the duplicate is removed file-only moments later, which made this line false.
        logger.warning(
            f"SAFETY GUARD blocked the Emby delete of id={item['id']} — {reason} "
            f"(delete={item.get('path')!r} keeper={keeper_path!r}). "
            "Keeper protected. The duplicate is still on disk: --fold-safe-delete removes "
            "it file-only, otherwise resolve the layout manually."
        )
        item["deletion_result"] = {
            "id": item["id"], "status": "skipped_unsafe", "error": reason,
        }
        _mark_guard_refused(item, keeper_path, reason)
    else:
        item["deletion_result"] = delete_item(
            client, base_url, item["id"], username, password, api_key
        )

    # Restore all the original data that might have been lost during deletion.
    for key, value in original_item_data.items():
        item[key] = value
    progress_bar.update(1)


def process_deletion_and_generate_report(
    client: httpx.Client,
    base_url: str,
    decisions: list,
    doit: bool,
    username: str,
    password: str,
    api_key: str,
    metadata: dict = None,
    library_paths: list = None,
) -> str:
    """
    Processes deletions based on the decisions and generates a markdown report.
    This function includes a progress bar reporting for deletions.

    Args:
        client (httpx.Client): The httpx client configured for the Emby server communication.
        base_url (str): The base URL of the Emby server.
        decisions (list): A list of decision objects containing items to keep and delete.
        doit (bool): If True, the deletion process will be attempted; otherwise, it is only simulated.
        username (str): The username for authentication.
        password (str): The password for authentication.
        api_key (str): The API key for non-DELETE requests.
        metadata (dict, optional): Additional metadata such as excluded IDs and language priorities.
        library_paths (list, optional): Every media path in the library, used to give the
            deletion safety guard true folder visibility (so it tells a dedicated folder
            from a shared one). When omitted, the guard sees only the decision paths and
            may over-refuse safe deletes (the Dutton/Proud false positives). Defaults to None.

    Returns:
        str: The generated markdown report.
    """
    # Calculate total deletions
    total_deletions = sum(len(decision.get("delete", [])) for decision in decisions)

    # Index every keep+delete path so the safety guard can tell a dedicated folder
    # (Emby fold-deletes it) from a shared one (Emby deletes only the file). The decision
    # paths alone miss non-duplicate neighbours (e.g. the other episodes in a season
    # folder), making a single-duplicate folder look "dedicated" and over-refusing a safe
    # file-only delete. Union in every library path to give the guard real folder
    # visibility. Both sources are Emby's verbatim ``Path`` field, so identical items
    # produce byte-identical strings that dedupe exactly (a mismatch could mask the
    # keeper → an under-refusal, which is why the union must never normalise paths).
    known_paths = collect_known_paths(decisions)
    if library_paths:
        known_paths = list(set(known_paths) | set(library_paths))

    # Every path being removed this run. A sibling that is itself being deleted must not
    # vouch that a folder is "shared" (it won't survive) — this is what makes the guard
    # refuse a delete co-located with the keeper when the only neighbours are also going
    # away (the Marty Supreme multi-version-folder fold-delete).
    delete_paths = collect_delete_paths(decisions)

    if not doit:
        deletion_progress_bar = tqdm(
            total=total_deletions, desc="Skipping deletion", unit="item"
        )
        _warn_unsafe_deletions(decisions, known_paths, delete_paths)
        _mark_items_as_not_attempted(decisions, deletion_progress_bar)
        deletion_progress_bar.close()
        return markdown.format_markdown_table(base_url, decisions, metadata)

    deletion_progress_bar = tqdm(
        total=total_deletions, desc="Deleting items", unit="item", dynamic_ncols=True
    )

    for decision in decisions:
        keeper_path = (decision.get("keep") or {}).get("path")
        for item in decision.get("delete", []):
            _execute_one_deletion(
                client, base_url, item, keeper_path, decision, known_paths,
                delete_paths, username, password, api_key, deletion_progress_bar,
            )

    deletion_progress_bar.close()
    return markdown.format_markdown_table(base_url, decisions, metadata)
