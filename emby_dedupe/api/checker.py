"""
EmbyChecker - Main Python library interface for checking media quality.

Usage:
    from emby_dedupe.api.checker import EmbyChecker

    # From config file
    checker = EmbyChecker.from_config()

    # Manual configuration
    checker = EmbyChecker(
        host="https://emby.example.com",
        api_key="your-api-key",
        libraries=["Movies", "TV Shows"],
        lang_priorities=["sk", "cs", "en"],
    )

    # Check a movie
    result = checker.check(name="Inception", year=2010, resolution="2160p")
    if result.should_download:
        logger.info("Download it!")

    # Simple boolean check
    should_dl = checker.should_download("Inception", year=2010, resolution="2160p")
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import httpx

from emby_dedupe.api.client import (
    check_emby_connection,
    fetch_and_process_media_items,
    fetch_items_details,
    get_library_id,
)
from emby_dedupe.api.quality_compare import (
    ComparisonResult,
    ExistingQuality,
    MediaQualityFields,
    ProposedQuality,
    compare_quality,
)
from emby_dedupe.api.search import SEARCH_FIELDS, search_media
from emby_dedupe.utils.config import Config, ensure_cache_dir, get_config_path
from emby_dedupe.utils.exceptions import EmbyConfigError, EmbyConfigMissingError
from emby_dedupe.utils.http import make_http_request
from emby_dedupe.utils.json_cache import load_json_cache, save_json_cache
from emby_dedupe.utils.logging import logger
from emby_dedupe.utils.providers import iter_provider_ids


@dataclass
class CheckConfig(MediaQualityFields):
    """Configuration for media quality check.

    Bundles all parameters needed for check() and should_download() methods.
    The media-quality descriptor fields are inherited from MediaQualityFields;
    only the identification fields are declared here.
    """

    name: str | None = None
    year: int | None = None
    imdb: str | None = None
    tmdb: str | None = None
    tvdb: str | None = None
    season: int | None = None
    episode: int | None = None


@dataclass
class EpisodeCheckResult:
    """One episode's check within a :meth:`EmbyChecker.check_episodes` call."""

    season: int
    episode: int
    result: ComparisonResult


@dataclass
class EpisodeSetResult:
    """Aggregate outcome of checking a set of TV episodes.

    Returned by :meth:`EmbyChecker.check_episodes`. ``should_download`` is True when
    any requested episode is missing OR a proposed upgrade beats the existing copy.
    """

    episodes_checked: int
    episodes_found: int
    all_same_or_better: bool
    episodes_to_download: dict[int, list[int]]
    first_existing: ExistingQuality | None = None
    results: list[EpisodeCheckResult] = field(default_factory=list)

    @property
    def should_download(self) -> bool:
        """True if the set is worth downloading (missing episodes or an upgrade).

        An empty set (nothing requested) is not worth downloading → False.
        """
        return not (
            self.all_same_or_better
            and self.episodes_found == self.episodes_checked
        )


def _iter_episode_pairs(
    episodes: Mapping[int, Sequence[int]],
) -> "Iterator[tuple[int, int]]":
    """Yield every ``(season, episode)`` pair, flattening the season→episodes mapping."""
    for season in episodes:
        for episode in episodes[season]:
            yield season, episode


def _episode_config(
    base: "CheckConfig",
    season: int,
    episode: int,
    episode_sizes: Mapping[tuple[int, int], int] | None,
) -> "CheckConfig":
    """Return ``base`` with this episode's season/episode (and per-episode size, if given)."""
    overrides: dict[str, Any] = {"season": season, "episode": episode}
    if episode_sizes and (season, episode) in episode_sizes:
        overrides["size_mb"] = episode_sizes[(season, episode)]
    return replace(base, **overrides)


class EmbyChecker:
    """Main interface for checking if media should be downloaded."""

    def __init__(
        self,
        host: str | None = None,
        api_key: str | None = None,
        libraries: list[str] | None = None,
        lang_priorities: list[str] | None = None,
        exclude_ids: list[str] | None = None,
        use_cache: bool = True,
        cache_ttl_minutes: int = 10,
        config: Config | None = None,
    ):
        """Initialize EmbyChecker.

        Args:
            host: Emby server URL.
            api_key: Emby API key.
            libraries: List of libraries to search. None = all libraries.
            lang_priorities: Language priority list (e.g., ['sk', 'cs', 'en']).
            exclude_ids: Provider IDs to exclude from checking.
            use_cache: Whether to cache library data.
            cache_ttl_minutes: Cache TTL in minutes.
            config: Optional Config object to use instead of individual params.
        """
        if config:
            self.host = config.host
            self.api_key = config.api_key
            self.libraries = config.libraries
            self.lang_priorities = config.lang_priorities
            self.exclude_ids = config.exclude_ids or []
            self.use_cache = config.cache_enabled
            self.cache_ttl_minutes = config.cache_ttl_minutes
        else:
            self.host = host
            self.api_key = api_key
            self.libraries = libraries
            self.lang_priorities = lang_priorities
            self.exclude_ids = exclude_ids or []
            self.use_cache = use_cache
            self.cache_ttl_minutes = cache_ttl_minutes

        self._client: httpx.Client | None = None
        self._cache_dir: Path | None = None
        self._provider_tables: dict | None = None  # Cached provider ID tables

    @classmethod
    def from_config(cls, **overrides) -> EmbyChecker:
        """Create EmbyChecker from config file.

        Args:
            **overrides: Values to override from config file.

        Returns:
            EmbyChecker instance.

        Raises:
            EmbyConfigMissingError: The config file does not exist and the overrides
                did not supply host + api_key. (Also a FileNotFoundError.)
            EmbyConfigError: The config file exists but is missing host or api_key.
                (Also a ValueError.)
        """
        config = Config.from_config_file(**overrides)
        # Fail HERE, loudly, instead of building a checker that only errors on the
        # first check() with a message that never mentions the config file.
        if not config.host or not config.api_key:
            config_path = get_config_path()
            missing = [k for k in ("host", "api_key") if not getattr(config, k)]
            if not config_path.exists():
                raise EmbyConfigMissingError(
                    f"Emby config file not found: {config_path}. Create it with 'host' "
                    f"and 'api_key' keys, or pass them as overrides to from_config()."
                )
            raise EmbyConfigError(
                f"Emby config file {config_path} is incomplete: missing "
                f"{', '.join(missing)}."
            )
        return cls(config=config)

    def _ensure_config(self) -> tuple[str, str]:
        """Ensure host and api_key are configured and return them."""
        if not self.host or not self.api_key:
            msg = "EmbyChecker requires host and api_key to be configured"
            raise EmbyConfigError(msg)
        return self.host, self.api_key

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            # Use shorter timeouts for faster fallback to name search
            # connect=10s, read=15s, write=10s, pool=10s
            timeout = httpx.Timeout(
                connect=10.0,
                read=15.0,  # Provider ID searches can be slow, but not too long
                write=10.0,
                pool=10.0
            )
            # Add API key to headers for authenticated requests
            headers = {"X-Emby-Token": self.api_key} if self.api_key else {}
            self._client = httpx.Client(timeout=timeout, headers=headers)
        return self._client

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path for a cache file."""
        if self._cache_dir is None:
            self._cache_dir = ensure_cache_dir()
        return self._cache_dir / f"{cache_key}.json"

    def _get_provider_tables_cache_path(self) -> Path:
        """Get path for provider tables cache."""
        if self._cache_dir is None:
            self._cache_dir = ensure_cache_dir()
        return self._cache_dir / "provider_tables.json"

    def _cache_read(self, path: Path, payload_key: str, default: Any) -> Any:
        """Read a TTL-checked payload from a timestamped JSON cache.

        The single read path behind every cache getter (built on ``utils.json_cache``,
        which has no TTL logic — this method owns the timestamp check). Returns None
        when caching is disabled, the file is missing/corrupt, or the entry has expired;
        otherwise the payload stored under ``payload_key`` (``default`` if absent).
        """
        if not self.use_cache:
            return None
        data = load_json_cache(path, label="cache")
        if not data:
            return None
        ttl_seconds = self.cache_ttl_minutes * 60
        if time.time() - data.get("timestamp", 0) > ttl_seconds:
            logger.debug(f"Cache expired: {path.name}")
            return None
        return data.get(payload_key, default)

    def _cache_write(self, path: Path, payload_key: str, data: Any) -> None:
        """Write ``data`` under ``payload_key`` to a timestamped JSON cache.

        The single write path behind every cache setter — atomic (sibling ``.tmp`` +
        rename) via ``utils.json_cache.save_json_cache``. No-op when caching is disabled.
        Written compact (``indent=None``) — these are machine-only caches and the
        provider-ID tables for a large library are big enough that ``indent=2`` would
        roughly double the file; this matches the pre-refactor compact writes.
        """
        if not self.use_cache:
            return
        save_json_cache(
            path, {"timestamp": time.time(), payload_key: data}, label="cache", compact=True
        )

    def _load_provider_tables(self) -> dict | None:
        """Load cached provider ID tables (TTL-checked); None if absent/expired."""
        tables = self._cache_read(self._get_provider_tables_cache_path(), "tables", None)
        if tables is not None:
            logger.info("Using cached provider ID tables for instant IMDB lookups")
        return tables

    def _save_provider_tables(self, tables: dict) -> None:
        """Save provider ID tables to cache."""
        self._cache_write(self._get_provider_tables_cache_path(), "tables", tables)
        if self.use_cache:
            logger.info("Saved provider ID tables to cache")

    def _get_library_names(self, client) -> list:
        """Get list of library names to process."""
        if self.libraries:
            return self.libraries

        from emby_dedupe.api.search import get_all_library_ids
        host, api_key = self._ensure_config()
        all_libs = get_all_library_ids(client, host, api_key)
        return [lib["name"] for lib in all_libs]

    def _merge_provider_tables(self, all_tables: dict, tables: dict) -> None:
        """Merge library tables into all_tables (in-place)."""
        for provider in ["imdb", "tvdb", "tmdb", "series_episode"]:
            for pid, items in tables[provider].items():
                if pid not in all_tables[provider]:
                    all_tables[provider][pid] = []
                all_tables[provider][pid].extend(items)

    def _build_provider_tables(self) -> dict:
        """Build provider ID tables from configured libraries.

        Returns:
            dict: Provider tables with 'imdb', 'tvdb', 'tmdb' keys.
        """
        logger.info("Building provider ID index from libraries (this may take 30-60s)...")

        client = self._get_client()
        host, _ = self._ensure_config()
        all_tables: dict[str, dict] = {"imdb": {}, "tvdb": {}, "tmdb": {}, "series_episode": {}}

        library_names = self._get_library_names(client)

        # Fetch from each library and build tables
        for lib_name in library_names:
            logger.info(f"  Fetching items from library: {lib_name}")
            try:
                lib_id = get_library_id(client, host, lib_name)
                if not lib_id:
                    logger.warning(f"  Library '{lib_name}' not found, skipping")
                    continue

                tables = fetch_and_process_media_items(client, host, lib_id, lib_name)
                self._merge_provider_tables(all_tables, tables)

            except Exception as e:
                logger.warning(f"  Error fetching library '{lib_name}': {e}")
                continue

        # Count total provider IDs
        total_imdb = len(all_tables["imdb"])
        total_tmdb = len(all_tables["tmdb"])
        total_tvdb = len(all_tables["tvdb"])
        total_se = len(all_tables["series_episode"])
        logger.info(f"Provider ID index built: {total_imdb} IMDB, {total_tmdb} TMDB, {total_tvdb} TVDB, {total_se} series-episode groups")

        return all_tables

    def _get_provider_tables(self) -> dict:
        """Get provider ID tables (from cache or build new)."""
        if self._provider_tables is not None:
            return self._provider_tables

        # Try to load from cache
        self._provider_tables = self._load_provider_tables()

        if self._provider_tables is None:
            # Build new tables
            self._provider_tables = self._build_provider_tables()
            # Save to cache
            self._save_provider_tables(self._provider_tables)

        return self._provider_tables

    def _lookup_by_provider_id(self, provider_id: str, provider_type: str = "imdb") -> list[dict]:
        """Fast lookup by provider ID using cached tables.

        Args:
            provider_id: Provider ID (e.g., 'tt1375666').
            provider_type: Provider type ('imdb', 'tmdb', 'tvdb').

        Returns:
            List of item IDs with this provider ID.
        """
        tables = self._get_provider_tables()
        provider_table = tables.get(provider_type.lower(), {})

        # Case-insensitive lookup
        item_ids = provider_table.get(provider_id.lower(), [])

        if not item_ids:
            return []

        # Extract just the IDs (tables store dicts with 'id' key)
        if isinstance(item_ids[0], dict):
            item_ids = [item['id'] for item in item_ids]

        # Fetch full item details
        client = self._get_client()
        host, _ = self._ensure_config()
        items = fetch_items_details(client, host, item_ids)
        return items

    def _get_from_cache(self, cache_key: str) -> list[dict] | None:
        """Get data from cache if valid (TTL-checked); None on miss/expiry."""
        items = self._cache_read(self._get_cache_path(cache_key), "items", [])
        if items is not None:
            logger.debug(f"Cache hit for {cache_key}")
        return items

    def _save_to_cache(self, cache_key: str, items: list[dict]) -> None:
        """Save data to cache."""
        self._cache_write(self._get_cache_path(cache_key), "items", items)
        if self.use_cache:
            logger.debug(f"Saved to cache: {cache_key}")

    def _make_cache_key(self, **params) -> str:
        """Generate a cache key from search parameters."""
        # Create a deterministic hash of the parameters
        key_data = json.dumps(params, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()[:16]

    def validate(self) -> list[str]:
        """Validate configuration.

        Returns:
            List of validation errors. Empty if valid.
        """
        errors = []
        if not self.host:
            errors.append("host is required")
        if not self.api_key:
            errors.append("api_key is required")
        return errors

    def rebuild_index(self) -> None:
        """Rebuild the provider ID index from scratch.

        This will:
        1. Clear cached provider tables
        2. Fetch all items from configured libraries
        3. Build new provider ID index
        4. Save to cache

        Use this when you want to refresh the index with newly added content.
        """
        logger.info("Rebuilding provider ID index...")

        # Clear cached tables
        self._provider_tables = None
        cache_path = self._get_provider_tables_cache_path()
        if cache_path.exists():
            cache_path.unlink()
            logger.debug("Cleared cached provider tables")

        # Build new tables (will automatically cache)
        self._get_provider_tables()

        logger.info("Provider ID index rebuilt successfully")

    def _lookup_by_any_provider_id(self, imdb: str | None, tmdb: str | None, tvdb: str | None) -> list | None:
        """Try to lookup items by provider ID (IMDB > TMDB > TVDB priority)."""
        for provider, pid in iter_provider_ids(imdb, tmdb, tvdb):
            logger.debug(f"Looking up {provider.upper()} ID: {pid}")
            items = self._lookup_by_provider_id(pid, provider)
            if items:
                logger.debug(f"Found {len(items)} items via {provider.upper()} lookup")
                return items

        return None

    def _find_validated_series(
        self, client: httpx.Client, host: str, pid: str, ptype: str,
    ) -> dict | None:
        """Search Emby for a series matching the given provider ID.

        Emby may ignore the AnyXxxId filter for Series items and return
        ALL series. This method validates that the returned series actually
        has the expected provider ID.

        Returns:
            Matching series dict, or None if not found.
        """
        url = f"{host}/Items"
        params = {
            f"Any{ptype}Id": pid,
            "IncludeItemTypes": "Series",
            "Recursive": "true",
            "Fields": "ProviderIds",
        }

        response = make_http_request(client, "GET", url, params=params)
        series_items = response.json().get("Items", [])

        for candidate in series_items:
            candidate_pids = candidate.get("ProviderIds", {})
            if candidate_pids.get(ptype, "").lower() == pid.lower():
                return candidate

        if series_items:
            logger.debug(
                f"No series with {ptype} ID {pid} found "
                f"(API returned {len(series_items)} unrelated series)"
            )
        return None

    def _fetch_episode_from_series(
        self, client: httpx.Client, host: str, series: dict,
        season: int, episode: int, ptype: str,
    ) -> list[dict]:
        """Fetch a specific episode from a known series.

        Returns:
            List of matching episodes (may be empty).
        """
        series_id = series["Id"]
        series_name = series.get("Name", "Unknown")

        url = f"{host}/Items"
        ep_params = {
            "ParentId": series_id,
            "IncludeItemTypes": "Episode",
            "Recursive": "true",
            "Fields": SEARCH_FIELDS,
        }

        response = make_http_request(client, "GET", url, params=ep_params)
        episodes = response.json().get("Items", [])

        matching = [
            ep for ep in episodes
            if ep.get("ParentIndexNumber") == season
            and ep.get("IndexNumber") == episode
        ]

        for ep in matching:
            if not ep.get("SeriesName"):
                ep["SeriesName"] = series_name

        label = f"S{season:02d}E{episode:02d}"
        if matching:
            logger.debug(f"Found {label} in series '{series_name}' via {ptype} provider ID fallback")
        else:
            logger.debug(f"{label} not found in series '{series_name}'")

        return matching

    def _lookup_episode_via_series(
        self,
        imdb: str | None,
        tmdb: str | None,
        tvdb: str | None,
        season: int,
        episode: int,
    ) -> list[dict] | None:
        """Find episode by searching for the parent series via provider ID.

        The cached provider tables only index non-folder items (episodes).
        Individual episodes often don't carry the series-level IMDB ID in their
        own ProviderIds — that ID lives on the Series item. This method queries
        the Emby API directly to find the series, then looks up the episode.

        Args:
            imdb: IMDB ID (series-level).
            tmdb: TMDB ID (series-level).
            tvdb: TVDB ID (series-level).
            season: Season number.
            episode: Episode number.

        Returns:
            List of matching episodes if series found (may be empty if episode
            doesn't exist in that series). None if no series found via any
            provider ID.
        """
        client = self._get_client()
        host, _ = self._ensure_config()

        for provider, pid in iter_provider_ids(imdb, tmdb, tvdb):
            ptype = provider.capitalize()  # Emby API param form: Imdb/Tmdb/Tvdb
            logger.debug(f"Series fallback: searching for series via {ptype} ID: {pid}")

            try:
                series = self._find_validated_series(client, host, pid, ptype)
                if series is None:
                    continue

                logger.debug(
                    f"Found series '{series.get('Name', 'Unknown')}' "
                    f"(ID: {series['Id']}) via {ptype} ID"
                )

                # Series was found — return result even if empty (episode doesn't exist)
                return self._fetch_episode_from_series(
                    client, host, series, season, episode, ptype
                )

            except Exception as e:
                logger.debug(f"Series provider ID lookup failed ({ptype}): {e}")
                continue

        return None  # No series found via any provider ID

    def _search_by_name(
        self,
        name: str,
        year: int | None,
        season: int | None,
        episode: int | None,
        imdb: str | None = None,
        tmdb: str | None = None,
        tvdb: str | None = None,
    ) -> list:
        """Search for existing media by name with caching.

        The provider ids are NOT searched again here (that already failed) — they are
        handed down so a name hit that carries a conflicting id can be rejected.
        """
        logger.debug(f"Provider ID not found or not provided, searching by name: {name}")

        cache_key = self._make_cache_key(
            name=name, year=year, season=season, episode=episode, libraries=self.libraries,
            imdb=imdb, tmdb=tmdb, tvdb=tvdb,
        )
        existing_items = self._get_from_cache(cache_key)

        if existing_items is None:
            client = self._get_client()
            host, api_key = self._ensure_config()
            existing_items = search_media(
                client=client,
                host=host,
                api_key=api_key,
                name=name,
                year=year,
                imdb=imdb,
                tmdb=tmdb,
                tvdb=tvdb,
                season=season,
                episode=episode,
                library_names=self.libraries,
                skip_provider_search=True,
            )
            self._save_to_cache(cache_key, existing_items)

        return existing_items

    def check(
        self,
        config: CheckConfig | None = None,
        **kwargs
    ) -> ComparisonResult:
        """Check if media should be downloaded.

        Args:
            config: CheckConfig object with all parameters (preferred).
            **kwargs: Individual parameters (name, year, imdb, tmdb, tvdb, season,
                episode, resolution, codec, hdr, audio, audio_languages, size_mb,
                bitrate_kbps, path, source_quality_tier).

        Returns:
            ComparisonResult with recommendation.

        Raises:
            EmbyConfigError: host/api_key are not configured. (Also a ValueError,
                so existing ``except ValueError`` handlers keep working.)
            TypeError: an unknown keyword argument was passed (previously such typos
                were silently ignored; CheckConfig now rejects them).
        """
        # Single source of parameter values: build a CheckConfig from kwargs when one
        # was not supplied. Unknown kwargs now raise TypeError instead of being
        # silently dropped (the old hand-copied kwargs.get() footgun).
        config = config if config is not None else CheckConfig(**kwargs)

        # Validate configuration
        errors = self.validate()
        if errors:
            raise EmbyConfigError(f"Invalid configuration: {', '.join(errors)}")

        # Check if provider ID is excluded
        for provider_id in [config.imdb, config.tmdb, config.tvdb]:
            if provider_id and provider_id in self.exclude_ids:
                logger.info(f"Skipping excluded provider ID: {provider_id}")
                return ComparisonResult(
                    recommendation="skip",
                    reason="excluded_id",
                    status="excluded",
                )

        # Create proposed quality object from the shared media-quality fields
        # (single field list — MediaQualityFields is the one declaration).
        proposed = ProposedQuality(
            name=config.name,
            **{f.name: getattr(config, f.name) for f in fields(MediaQualityFields)},
        )

        # Try provider ID lookup first (instant with cached tables)
        existing_items = self._lookup_by_any_provider_id(
            config.imdb, config.tmdb, config.tvdb
        )

        # Fallback for TV episodes: cached tables only index episodes (IsFolder=False),
        # so series-level provider IDs (e.g., IMDB on the Series item) won't be found.
        # Search the Emby API directly for the series by provider ID, then find the episode.
        if not existing_items and config.season is not None and config.episode is not None:
            series_result = self._lookup_episode_via_series(
                config.imdb, config.tmdb, config.tvdb, config.season, config.episode
            )
            if series_result is not None:
                existing_items = series_result

        # Fall back to name search only if no provider-based result found
        if existing_items is None and config.name:
            existing_items = self._search_by_name(
                config.name, config.year, config.season, config.episode,
                imdb=config.imdb, tmdb=config.tmdb, tvdb=config.tvdb,
            )
        if not existing_items:
            existing_items = []

        # Compare quality
        return compare_quality(proposed, existing_items, self.lang_priorities)

    def should_download(
        self,
        config: CheckConfig | None = None,
        **kwargs
    ) -> bool:
        """Check if media should be downloaded (simple boolean interface).

        Returns True if:
        - Media doesn't exist in Emby (not found)
        - Proposed quality is better than existing

        Returns False if:
        - Existing media is same or better quality
        - Provider ID is excluded

        Args:
            config: CheckConfig object with all parameters (preferred).
            **kwargs: Individual parameters (same as check()).

        Returns:
            True if should download, False otherwise.
        """
        return self.check(config=config, **kwargs).should_download

    def check_episodes(
        self,
        episodes: Mapping[int, Sequence[int]],
        config: CheckConfig | None = None,
        *,
        episode_sizes: Mapping[tuple[int, int], int] | None = None,
        on_episode: Callable[[EpisodeCheckResult], None] | None = None,
        **kwargs,
    ) -> EpisodeSetResult:
        """Check a set of TV episodes and aggregate the per-episode decisions.

        This is the batch API for episode ranges / season packs / multi-season packs:
        it runs :meth:`check` for every ``(season, episode)`` and folds the results
        into one :class:`EpisodeSetResult` (found count, per-season download list,
        whether every existing episode is same-or-better). It owns the loop so
        consumers no longer hand-roll it.

        Args:
            episodes: Mapping of season number to the episode numbers to check,
                e.g. ``{1: [8, 9, 10]}`` or ``{1: [...], 2: [...]}``.
            config: Base quality descriptor (resolution/codec/size_mb/…) shared by
                every episode. Its ``season``/``episode`` are ignored (set per item).
            episode_sizes: Optional per-episode size override in MB, keyed by
                ``(season, episode)`` — lets each episode be sized independently.
            on_episode: Optional callback invoked with each EpisodeCheckResult as it
                completes (for progress/diagnostic logging), preserving per-episode
                visibility a plain loop would give.
            **kwargs: Base quality fields, as an alternative to ``config``.

        Returns:
            EpisodeSetResult with the aggregated decision.
        """
        base = config if config is not None else CheckConfig(**kwargs)

        episodes_checked = 0
        episodes_found = 0
        all_same_or_better = True
        first_existing: ExistingQuality | None = None
        to_download: dict[int, list[int]] = {}
        results: list[EpisodeCheckResult] = []

        for season, episode in _iter_episode_pairs(episodes):
            episodes_checked += 1
            item_config = _episode_config(base, season, episode, episode_sizes)

            result = self.check(config=item_config)
            episode_result = EpisodeCheckResult(season, episode, result)
            results.append(episode_result)

            if result.existing is not None:
                episodes_found += 1
                if first_existing is None:
                    first_existing = result.existing

            if result.should_download:
                all_same_or_better = False
                to_download.setdefault(season, []).append(episode)

            if on_episode is not None:
                on_episode(episode_result)

        return EpisodeSetResult(
            episodes_checked=episodes_checked,
            episodes_found=episodes_found,
            all_same_or_better=all_same_or_better,
            episodes_to_download=to_download,
            first_existing=first_existing,
            results=results,
        )

    def check_batch(
        self,
        items: list[dict[str, Any]],
    ) -> list[ComparisonResult]:
        """Check multiple items at once.

        .. deprecated::
            Use :meth:`check` in a loop for arbitrary items, or
            :meth:`check_episodes` for TV episode sets (which aggregates the
            per-episode decisions). ``check_batch`` is a bare loop kept only for
            backward compatibility and may be removed in a future major release.

        Args:
            items: List of dicts with check parameters.

        Returns:
            List of ComparisonResults.
        """
        results = []
        for item in items:
            result = self.check(**item)
            results.append(result)
        return results

    def test_connection(self) -> bool:
        """Verify the Emby server is reachable and the API key is valid.

        A lightweight probe (``GET /System/Info``) — unlike a fabricated ``check()``
        it performs a real network round-trip every call (no result cache), so it
        cannot pass against a server that is actually down.

        Returns:
            True if the server responds successfully.

        Raises:
            EmbyConfigError: host/api_key are not configured.
            EmbyServerConnectionError: the server is unreachable or rejected the key.
        """
        host, _ = self._ensure_config()
        return check_emby_connection(self._get_client(), f"{host}/System/Info")

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> EmbyChecker:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
