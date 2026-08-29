"""
Media search functions for finding items in Emby libraries.

Provides search capabilities for:
- Searching by name (fuzzy and exact)
- Searching by provider IDs (IMDB, TMDB, TVDB)
- Searching for TV episodes by series/season/episode
"""

import re
from typing import Any

import httpx

from emby_dedupe.utils.http import make_http_request
from emby_dedupe.utils.logging import logger
from emby_dedupe.utils.providers import iter_provider_ids

# Standard fields to fetch when searching for media items
SEARCH_FIELDS = "ProviderIds,Path,MediaStreams,DateCreated,DateModified,PremiereDate,ProductionYear,Tags,Overview,ParentId,SeriesName,ParentIndexNumber,IndexNumber,RunTimeTicks"


def normalize_title(title: str) -> str:
    """Normalize a title for comparison.

    Args:
        title: The title to normalize.

    Returns:
        Normalized title (lowercase, no special chars, single spaces).
    """
    # Convert to lowercase
    normalized = title.lower()
    # Remove special characters except spaces
    normalized = re.sub(r'[^\w\s]', '', normalized)
    # Replace multiple spaces with single space
    normalized = re.sub(r'\s+', ' ', normalized)
    # Strip leading/trailing whitespace
    normalized = normalized.strip()
    return normalized


def titles_match(title1: str, title2: str, fuzzy: bool = True) -> bool:
    """Check if two titles match.

    Args:
        title1: First title.
        title2: Second title.
        fuzzy: If True, use fuzzy matching. If False, require exact match.

    Returns:
        True if titles match.
    """
    norm1 = normalize_title(title1)
    norm2 = normalize_title(title2)

    if norm1 == norm2:
        return True

    if not fuzzy:
        return False

    # Check if one title contains the other (for subtitle handling)
    if norm1 in norm2 or norm2 in norm1:
        return True

    return False


def search_by_name(
    client: httpx.Client,
    host: str,
    api_key: str,
    name: str,
    year: int | None = None,
    media_type: str | None = None,
    library_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Search for media items by name.

    Args:
        client: HTTP client.
        host: Emby server URL.
        api_key: Emby API key.
        name: Media name to search for.
        year: Optional release year to filter by.
        media_type: Optional media type ('Movie', 'Series', 'Episode').
        library_ids: Optional list of library IDs to search in.

    Returns:
        List of matching media items.
    """
    # Build search query
    params = {
        "api_key": api_key,
        "SearchTerm": name,
        "Recursive": "true",
        "Fields": SEARCH_FIELDS,
    }

    if media_type:
        params["IncludeItemTypes"] = media_type

    # Note: Emby API doesn't support comma-separated ParentId in search
    # For multiple libraries, we'll search all and filter results
    # Only use ParentId for single library searches
    if library_ids and len(library_ids) == 1:
        params["ParentId"] = library_ids[0]

    if year:
        params["Years"] = str(year)

    url = f"{host}/Items"
    try:
        response = make_http_request(client, "GET", url, params=params)
        data = response.json()
        items = data.get("Items", [])

        # Filter by name similarity
        matching_items = []
        for item in items:
            item_name = item.get("Name", "")
            if titles_match(name, item_name):
                matching_items.append(item)

        logger.debug(f"Found {len(matching_items)} items matching '{name}'")
        return matching_items

    except httpx.HTTPError as e:
        logger.error(f"Error searching for '{name}': {e}")
        return []


def search_by_provider_id(
    client: httpx.Client,
    host: str,
    api_key: str,
    provider_id: str,
    provider_type: str = "Imdb",
    library_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Search for media items by provider ID.

    Args:
        client: HTTP client.
        host: Emby server URL.
        api_key: Emby API key.
        provider_id: Provider ID (e.g., 'tt1375666' for IMDB).
        provider_type: Provider type ('Imdb', 'Tmdb', 'Tvdb').
        library_ids: Optional list of library IDs to search in.

    Returns:
        List of matching media items.
    """
    # Normalize provider type
    provider_type_map = {
        'imdb': 'Imdb',
        'tmdb': 'Tmdb',
        'tvdb': 'Tvdb',
    }
    provider_type = provider_type_map.get(provider_type.lower(), provider_type)

    params = {
        "api_key": api_key,
        f"Any{provider_type}Id": provider_id,
        "Recursive": "true",
        "Fields": SEARCH_FIELDS,
    }

    # Note: Emby API doesn't support comma-separated ParentId in search
    # For multiple libraries, we'll search all and filter results
    # Only use ParentId for single library searches
    if library_ids and len(library_ids) == 1:
        params["ParentId"] = library_ids[0]

    url = f"{host}/Items"
    try:
        response = make_http_request(client, "GET", url, params=params)
        data = response.json()
        items = data.get("Items", [])
        logger.debug(f"Found {len(items)} items with {provider_type} ID '{provider_id}'")
        return items

    except httpx.HTTPError as e:
        logger.error(f"Error searching for {provider_type} ID '{provider_id}': {e}")
        return []


# A fuzzy (containment) title match is allowed to disagree with the caller by at most
# this many years — premiere vs. production year routinely differ by one.
_FUZZY_YEAR_TOLERANCE = 1


def _provider_id_conflicts(
    series: dict[str, Any], imdb: str | None, tmdb: str | None, tvdb: str | None
) -> bool:
    """True when the series carries a DIFFERENT id for a provider the caller supplied.

    The caller's ids were already looked up and did not resolve, so a candidate
    that carries another id for the same provider is provably a different show.
    """
    pids = series.get("ProviderIds") or {}
    for provider, wanted in iter_provider_ids(imdb, tmdb, tvdb):
        have = pids.get(provider.capitalize()) or pids.get(provider) or pids.get(provider.upper())
        if have and str(have).lower() != str(wanted).lower():
            return True
    return False


def select_series_candidate(
    series_name: str,
    series_items: list[dict[str, Any]],
    *,
    year: int | None = None,
    imdb: str | None = None,
    tmdb: str | None = None,
    tvdb: str | None = None,
) -> dict[str, Any] | None:
    """Pick the Emby series that really is ``series_name`` from a SearchTerm result.

    Emby's SearchTerm is a substring search and ``titles_match`` accepts containment,
    so "Malcolm in the Middle" happily matched the unrelated show "The Middle" and
    149 of 151 episodes of a season pack were dropped as "duplicates" (2026-08-29).

    Rules (deterministic, cheapest first):
      1. A candidate whose provider id CONFLICTS with a supplied id is never a match.
      2. An exact normalized title match wins over a containment match.
      3. A containment-only match is rejected when both years are known and differ by
         more than ``_FUZZY_YEAR_TOLERANCE``.
    """
    exact = None
    fuzzy = None
    for series in series_items:
        candidate_name = series.get("Name", "")
        if not titles_match(series_name, candidate_name):
            continue
        if _provider_id_conflicts(series, imdb, tmdb, tvdb):
            logger.debug(
                f"Rejecting series '{candidate_name}' for '{series_name}': provider id conflict "
                f"({series.get('ProviderIds')})"
            )
            continue
        if titles_match(series_name, candidate_name, fuzzy=False):
            if exact is None:
                exact = series
            continue
        candidate_year = series.get("ProductionYear")
        if (
            year is not None
            and candidate_year is not None
            and abs(int(candidate_year) - int(year)) > _FUZZY_YEAR_TOLERANCE
        ):
            logger.debug(
                f"Rejecting fuzzy series match '{candidate_name}' ({candidate_year}) for "
                f"'{series_name}' ({year}): year conflict"
            )
            continue
        if fuzzy is None:
            fuzzy = series
    return exact or fuzzy


def search_tv_episode(
    client: httpx.Client,
    host: str,
    api_key: str,
    series_name: str,
    season: int,
    episode: int,
    library_ids: list[str] | None = None,
    *,
    year: int | None = None,
    imdb: str | None = None,
    tmdb: str | None = None,
    tvdb: str | None = None,
) -> list[dict[str, Any]]:
    """Search for a TV episode by series, season, and episode number.

    Args:
        client: HTTP client.
        host: Emby server URL.
        api_key: Emby API key.
        series_name: Name of the TV series.
        season: Season number.
        episode: Episode number.
        library_ids: Optional list of library IDs to search in.
        year: Series year known to the caller; a containment-only title match that
            disagrees with it is rejected (see ``select_series_candidate``).
        imdb/tmdb/tvdb: Provider ids known to the caller; a candidate carrying a
            different id for the same provider is rejected.

    Returns:
        List of matching episodes.
    """
    # First, search for the series
    params = {
        "api_key": api_key,
        "SearchTerm": series_name,
        "IncludeItemTypes": "Series",
        "Recursive": "true",
        "Fields": "ProviderIds,ProductionYear",
    }

    # Note: Emby API doesn't support comma-separated ParentId in search
    # For multiple libraries, we'll search all and filter results
    # Only use ParentId for single library searches
    if library_ids and len(library_ids) == 1:
        params["ParentId"] = library_ids[0]

    url = f"{host}/Items"
    try:
        response = make_http_request(client, "GET", url, params=params)
        data = response.json()
        series_items = data.get("Items", [])

        # Find matching series (exact title preferred; provider-id / year conflicts rejected)
        matching_series = select_series_candidate(
            series_name, series_items, year=year, imdb=imdb, tmdb=tmdb, tvdb=tvdb
        )

        if not matching_series:
            logger.debug(f"Series '{series_name}' not found")
            return []

        # Search for the episode within the series
        series_id = matching_series["Id"]
        episode_params = {
            "api_key": api_key,
            "ParentId": series_id,
            "IncludeItemTypes": "Episode",
            "Recursive": "true",
            "Fields": SEARCH_FIELDS,
        }

        response = make_http_request(client, "GET", url, params=episode_params)
        data = response.json()
        episodes = data.get("Items", [])

        # Filter to specific season/episode
        matching_episodes = []
        for ep in episodes:
            ep_season = ep.get("ParentIndexNumber")
            ep_number = ep.get("IndexNumber")

            if ep_season == season and ep_number == episode:
                # Add series name for consistency
                ep["SeriesName"] = matching_series.get("Name")
                matching_episodes.append(ep)

        logger.debug(f"Found {len(matching_episodes)} episodes matching S{season:02d}E{episode:02d}")
        return matching_episodes

    except httpx.HTTPError as e:
        logger.error(f"Error searching for episode: {e}")
        return []


def get_all_library_ids(
    client: httpx.Client,
    host: str,
    api_key: str,
) -> list[dict[str, str]]:
    """Get all library IDs from the Emby server.

    Args:
        client: HTTP client.
        host: Emby server URL.
        api_key: Emby API key.

    Returns:
        List of dicts with 'id' and 'name' keys.
    """
    url = f"{host}/Library/VirtualFolders"
    params = {"api_key": api_key}

    try:
        response = make_http_request(client, "GET", url, params=params)
        data = response.json()

        libraries = []
        for folder in data:
            libraries.append({
                "id": folder.get("ItemId"),
                "name": folder.get("Name"),
            })

        logger.debug(f"Found {len(libraries)} libraries")
        return libraries

    except httpx.HTTPError as e:
        logger.error(f"Error getting library IDs: {e}")
        return []


def get_library_ids_by_name(
    client: httpx.Client,
    host: str,
    api_key: str,
    library_names: list[str],
) -> list[str]:
    """Get library IDs by name.

    Args:
        client: HTTP client.
        host: Emby server URL.
        api_key: Emby API key.
        library_names: List of library names to find.

    Returns:
        List of library IDs.
    """
    all_libraries = get_all_library_ids(client, host, api_key)

    library_ids = []
    for name in library_names:
        for lib in all_libraries:
            if lib["name"].lower() == name.lower():
                library_ids.append(lib["id"])
                break
        else:
            logger.warning(f"Library '{name}' not found")

    return library_ids


def _search_provider_id_across_libraries(
    client: httpx.Client,
    host: str,
    api_key: str,
    provider_id: str,
    provider_type: str,
    library_ids: list[str] | None,
) -> list[dict[str, Any]]:
    """Search for provider ID across multiple libraries (one at a time to avoid timeouts).

    Args:
        client: HTTP client.
        host: Emby server URL.
        api_key: Emby API key.
        provider_id: Provider ID.
        provider_type: Provider type ('Imdb', 'Tmdb', 'Tvdb').
        library_ids: List of library IDs to search.

    Returns:
        Combined list of matching items from all libraries.
    """
    if not library_ids:
        # Search all libraries at once (might timeout but we'll try)
        try:
            return search_by_provider_id(client, host, api_key, provider_id, provider_type, None)
        except Exception as e:
            logger.debug(f"Provider ID search across all libraries failed: {e}")
            return []

    # Search each library individually (fast, no timeouts)
    all_results = []
    for lib_id in library_ids:
        try:
            results = search_by_provider_id(client, host, api_key, provider_id, provider_type, [lib_id])
            all_results.extend(results)
        except Exception as e:
            logger.debug(f"Provider ID search in library {lib_id} failed: {e}")
            continue

    return all_results


def _try_provider_id_searches(
    client: httpx.Client,
    host: str,
    api_key: str,
    imdb: str | None,
    tmdb: str | None,
    tvdb: str | None,
    library_ids: list[str] | None,
) -> list[dict[str, Any]] | None:
    """Try searching by provider IDs (IMDB, TMDB, TVDB).

    Args:
        client: HTTP client.
        host: Emby server URL.
        api_key: Emby API key.
        imdb: IMDB ID.
        tmdb: TMDB ID.
        tvdb: TVDB ID.
        library_ids: Library IDs to search in.

    Returns:
        Results if found, None otherwise.
    """
    for provider, pid in iter_provider_ids(imdb, tmdb, tvdb):
        results = _search_provider_id_across_libraries(
            client, host, api_key, pid, provider.capitalize(), library_ids
        )
        if results:
            return results

    return None


def _search_by_name_with_type(
    client: httpx.Client,
    host: str,
    api_key: str,
    name: str,
    year: int | None,
    season: int | None,
    episode: int | None,
    library_ids: list[str] | None,
    *,
    imdb: str | None = None,
    tmdb: str | None = None,
    tvdb: str | None = None,
) -> list[dict[str, Any]]:
    """Search by name with appropriate media type.

    Args:
        client: HTTP client.
        host: Emby server URL.
        api_key: Emby API key.
        name: Media name.
        year: Release year.
        season: Season number (for TV).
        episode: Episode number (for TV).
        library_ids: Library IDs to search in.

    Returns:
        List of matching media items.
    """
    if season is not None and episode is not None:
        # TV episode search
        return search_tv_episode(
            client, host, api_key, name, season, episode, library_ids,
            year=year, imdb=imdb, tmdb=tmdb, tvdb=tvdb,
        )
    elif season is not None:
        # Search for series, then filter by season
        return search_by_name(client, host, api_key, name, year, "Series", library_ids)
    else:
        # Movie or general search
        return search_by_name(client, host, api_key, name, year, None, library_ids)


def search_media(
    client: httpx.Client,
    host: str,
    api_key: str,
    name: str | None = None,
    year: int | None = None,
    imdb: str | None = None,
    tmdb: str | None = None,
    tvdb: str | None = None,
    season: int | None = None,
    episode: int | None = None,
    library_names: list[str] | None = None,
    skip_provider_search: bool = False,
) -> list[dict[str, Any]]:
    """Search for media using various criteria.

    Args:
        client: HTTP client.
        host: Emby server URL.
        api_key: Emby API key.
        name: Media name.
        year: Release year.
        imdb: IMDB ID.
        tmdb: TMDB ID.
        tvdb: TVDB ID.
        season: Season number (for TV episodes).
        episode: Episode number (for TV episodes).
        library_names: Library names to search in. None = all libraries.

    Returns:
        List of matching media items.
    """
    # Get library IDs if specified
    library_ids = None
    if library_names:
        library_ids = get_library_ids_by_name(client, host, api_key, library_names)
        if not library_ids:
            logger.warning("No valid library IDs found")
            return []

    # Search by provider ID first (most accurate)
    if not skip_provider_search:
        results = _try_provider_id_searches(
            client, host, api_key, imdb, tmdb, tvdb, library_ids
        )
        if results:
            return results

    # Search by name if no provider ID results
    if name:
        return _search_by_name_with_type(
            client, host, api_key, name, year, season, episode, library_ids,
            imdb=imdb, tmdb=tmdb, tvdb=tvdb,
        )

    return []
