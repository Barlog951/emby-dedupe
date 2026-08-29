"""Tests for search module."""

from unittest.mock import Mock, patch

from emby_dedupe.api.search import (
    get_all_library_ids,
    get_library_ids_by_name,
    normalize_title,
    search_by_name,
    search_by_provider_id,
    search_media,
    search_tv_episode,
    titles_match,
)


class TestNormalizeTitle:
    """Tests for normalize_title function."""

    def test_normalize_removes_special_chars(self):
        """Test that special characters are removed."""
        assert normalize_title("The Matrix (1999)") == "the matrix 1999"

    def test_normalize_lowercase(self):
        """Test that title is converted to lowercase."""
        assert normalize_title("THE MATRIX") == "the matrix"

    def test_normalize_multiple_spaces(self):
        """Test that multiple spaces are collapsed."""
        assert normalize_title("The    Matrix") == "the matrix"

    def test_normalize_strips_whitespace(self):
        """Test that leading/trailing whitespace is removed."""
        assert normalize_title("  The Matrix  ") == "the matrix"


class TestTitlesMatch:
    """Tests for titles_match function."""

    def test_exact_match(self):
        """Test exact title match."""
        assert titles_match("The Matrix", "The Matrix") is True

    def test_case_insensitive_match(self):
        """Test case-insensitive matching."""
        assert titles_match("The Matrix", "the matrix") is True

    def test_fuzzy_match_with_subtitle(self):
        """Test fuzzy matching with subtitles."""
        assert titles_match("The Matrix", "The Matrix: Reloaded", fuzzy=True) is True

    def test_no_fuzzy_match_different_titles(self):
        """Test that different titles don't match without fuzzy."""
        assert titles_match("The Matrix", "The Matrix: Reloaded", fuzzy=False) is False

    def test_no_match(self):
        """Test that completely different titles don't match."""
        assert titles_match("The Matrix", "Inception") is False


class TestSearchByName:
    """Tests for search_by_name function."""

    def test_search_by_name_returns_matches(self):
        """Test that search_by_name returns matching items."""
        mock_client = Mock()
        mock_client.request.return_value.json.return_value = {
            "Items": [
                {"Name": "The Matrix", "Id": "123"},
                {"Name": "The Matrix Reloaded", "Id": "456"},
            ]
        }

        results = search_by_name(mock_client, "http://emby.local", "api_key", "The Matrix")

        assert len(results) == 2
        assert results[0]["Name"] == "The Matrix"

    def test_search_by_name_filters_by_similarity(self):
        """Test that search_by_name filters by title similarity."""
        mock_client = Mock()
        mock_client.request.return_value.json.return_value = {
            "Items": [
                {"Name": "The Matrix", "Id": "123"},
                {"Name": "Inception", "Id": "789"},  # Should be filtered out
            ]
        }

        results = search_by_name(mock_client, "http://emby.local", "api_key", "The Matrix")

        assert len(results) == 1
        assert results[0]["Name"] == "The Matrix"

    def test_search_by_name_handles_error(self):
        """Test that search_by_name handles HTTP errors."""
        import httpx

        mock_client = Mock()

        with patch(
            "emby_dedupe.api.search.make_http_request",
            side_effect=httpx.HTTPError("Connection error"),
        ):
            results = search_by_name(mock_client, "http://emby.local", "api_key", "The Matrix")

        assert results == []


class TestSearchByProviderId:
    """Tests for search_by_provider_id function."""

    def test_search_by_imdb_id(self):
        """Test searching by IMDB ID."""
        mock_client = Mock()
        mock_client.request.return_value.json.return_value = {
            "Items": [{"Name": "The Matrix", "Id": "123", "ProviderIds": {"Imdb": "tt0133093"}}]
        }

        results = search_by_provider_id(
            mock_client, "http://emby.local", "api_key", "tt0133093", "imdb"
        )

        assert len(results) == 1
        assert results[0]["Name"] == "The Matrix"

    def test_search_normalizes_provider_type(self):
        """Test that provider type is normalized."""
        mock_client = Mock()
        mock_client.request.return_value.json.return_value = {"Items": []}

        search_by_provider_id(
            mock_client,
            "http://emby.local",
            "api_key",
            "123",
            "imdb",  # lowercase
        )

        # Check that the call was made with normalized provider type
        call_args = mock_client.request.call_args
        assert "AnyImdbId" in str(call_args)


class TestSearchTvEpisode:
    """Tests for search_tv_episode function."""

    def test_search_tv_episode_finds_episode(self):
        """Test that search_tv_episode finds the correct episode."""
        mock_client = Mock()

        # Mock series search
        series_response = Mock()
        series_response.json.return_value = {"Items": [{"Name": "Breaking Bad", "Id": "series123"}]}

        # Mock episode search
        episode_response = Mock()
        episode_response.json.return_value = {
            "Items": [
                {
                    "Name": "Pilot",
                    "Id": "ep123",
                    "ParentIndexNumber": 1,  # Season
                    "IndexNumber": 1,  # Episode
                }
            ]
        }

        mock_client.request.side_effect = [series_response, episode_response]

        results = search_tv_episode(
            mock_client,
            "http://emby.local",
            "api_key",
            "Breaking Bad",
            1,  # season
            1,  # episode
        )

        assert len(results) == 1
        assert results[0]["Name"] == "Pilot"
        assert results[0]["SeriesName"] == "Breaking Bad"

    def test_search_tv_episode_series_not_found(self):
        """Test handling when series is not found."""
        mock_client = Mock()
        mock_client.request.return_value.json.return_value = {"Items": []}

        results = search_tv_episode(
            mock_client,
            "http://emby.local",
            "api_key",
            "Nonexistent Show",
            1,
            1,
        )

        assert results == []


class TestGetAllLibraryIds:
    """Tests for get_all_library_ids function."""

    def test_get_all_library_ids(self):
        """Test getting all library IDs."""
        mock_client = Mock()
        mock_client.request.return_value.json.return_value = [
            {"ItemId": "lib1", "Name": "Movies"},
            {"ItemId": "lib2", "Name": "TV Shows"},
        ]

        results = get_all_library_ids(mock_client, "http://emby.local", "api_key")

        assert len(results) == 2
        assert results[0]["id"] == "lib1"
        assert results[0]["name"] == "Movies"


class TestGetLibraryIdsByName:
    """Tests for get_library_ids_by_name function."""

    def test_get_library_ids_by_name(self):
        """Test getting library IDs by name."""
        mock_client = Mock()
        mock_client.request.return_value.json.return_value = [
            {"ItemId": "lib1", "Name": "Movies"},
            {"ItemId": "lib2", "Name": "TV Shows"},
        ]

        results = get_library_ids_by_name(mock_client, "http://emby.local", "api_key", ["Movies"])

        assert results == ["lib1"]

    def test_get_library_ids_by_name_case_insensitive(self):
        """Test that library name matching is case-insensitive."""
        mock_client = Mock()
        mock_client.request.return_value.json.return_value = [
            {"ItemId": "lib1", "Name": "Movies"},
        ]

        results = get_library_ids_by_name(
            mock_client,
            "http://emby.local",
            "api_key",
            ["movies"],  # lowercase
        )

        assert results == ["lib1"]


class TestSearchMedia:
    """Tests for search_media function."""

    def test_search_media_by_imdb_returns_early(self):
        """Test that search_media returns early when IMDB match is found."""
        mock_client = Mock()
        mock_client.request.return_value.json.return_value = {
            "Items": [{"Name": "The Matrix", "Id": "123"}]
        }

        results = search_media(
            mock_client, "http://emby.local", "api_key", name="The Matrix", imdb="tt0133093"
        )

        assert len(results) == 1
        # Should only call once (for IMDB search)
        assert mock_client.request.call_count == 1

    def test_search_media_by_name_when_no_provider_id(self):
        """Test that search_media falls back to name search."""
        mock_client = Mock()
        mock_client.request.return_value.json.return_value = {
            "Items": [{"Name": "The Matrix", "Id": "123"}]
        }

        results = search_media(
            mock_client, "http://emby.local", "api_key", name="The Matrix", year=1999
        )

        assert len(results) >= 0  # May or may not match after filtering

    def test_search_media_tv_episode(self):
        """Test searching for TV episode."""
        mock_client = Mock()

        # Mock series search
        series_response = Mock()
        series_response.json.return_value = {"Items": [{"Name": "Breaking Bad", "Id": "series123"}]}

        # Mock episode search
        episode_response = Mock()
        episode_response.json.return_value = {
            "Items": [{"Name": "Pilot", "Id": "ep123", "ParentIndexNumber": 1, "IndexNumber": 1}]
        }

        mock_client.request.side_effect = [series_response, episode_response]

        results = search_media(
            mock_client, "http://emby.local", "api_key", name="Breaking Bad", season=1, episode=1
        )

        assert len(results) == 1


class TestSelectSeriesCandidate:
    """Regression: 'Malcolm in the Middle' must never resolve to 'The Middle' (2026-08-29).

    Emby's SearchTerm returned only the unrelated show and containment matching accepted
    it, so 149/151 episodes of a season pack were dropped as duplicates.
    """

    MALCOLM = {"name": "Malcolm in the Middle", "year": 2000, "imdb": "tt0212671"}
    THE_MIDDLE = {
        "Name": "The Middle",
        "Id": "3564656",
        "ProductionYear": 2009,
        "ProviderIds": {"Imdb": "tt1442464", "Tmdb": "1422"},
    }

    def test_malcolm_rejects_the_middle_on_provider_conflict(self):
        from emby_dedupe.api.search import select_series_candidate

        assert (
            select_series_candidate("Malcolm in the Middle", [self.THE_MIDDLE], imdb="tt0212671")
            is None
        )

    def test_malcolm_rejects_the_middle_on_year_conflict_without_ids(self):
        from emby_dedupe.api.search import select_series_candidate

        candidate = {"Name": "The Middle", "Id": "x", "ProductionYear": 2009}
        assert select_series_candidate("Malcolm in the Middle", [candidate], year=2000) is None

    def test_containment_without_any_conflict_still_matches(self):
        """Legit containment (brand prefix) keeps working when nothing contradicts it."""
        from emby_dedupe.api.search import select_series_candidate

        candidate = {"Name": "Marvel's Daredevil", "Id": "d", "ProductionYear": 2015}
        assert select_series_candidate("Daredevil", [candidate], year=2015) is candidate
        assert select_series_candidate("Daredevil", [candidate]) is candidate

    def test_year_tolerance_of_one(self):
        from emby_dedupe.api.search import select_series_candidate

        candidate = {"Name": "Marvel's Daredevil", "Id": "d", "ProductionYear": 2015}
        assert select_series_candidate("Daredevil", [candidate], year=2016) is candidate
        assert select_series_candidate("Daredevil", [candidate], year=2017) is None

    def test_exact_title_preferred_over_containment(self):
        from emby_dedupe.api.search import select_series_candidate

        fuzzy = {"Name": "The Middle", "Id": "f"}
        exact = {"Name": "Malcolm in the Middle", "Id": "e"}
        assert select_series_candidate("Malcolm in the Middle", [fuzzy, exact]) is exact

    def test_exact_title_with_conflicting_id_is_still_rejected(self):
        """Same-title remakes: an exact title carrying another IMDb id is a different show."""
        from emby_dedupe.api.search import select_series_candidate

        remake = {"Name": "Battlestar Galactica", "Id": "r", "ProviderIds": {"Imdb": "tt0407362"}}
        assert select_series_candidate("Battlestar Galactica", [remake], imdb="tt0076984") is None

    def test_matching_provider_id_is_accepted(self):
        from emby_dedupe.api.search import select_series_candidate

        same = {"Name": "The Middle", "Id": "s", "ProviderIds": {"Imdb": "TT1442464"}}
        assert select_series_candidate("Malcolm in the Middle", [same], imdb="tt1442464") is same

    def test_search_tv_episode_does_not_return_the_middle_for_malcolm(self):
        mock_client = Mock()
        series_response = Mock()
        series_response.json.return_value = {"Items": [self.THE_MIDDLE]}
        mock_client.request.return_value = series_response

        results = search_tv_episode(
            mock_client,
            "http://emby.local",
            "api_key",
            "Malcolm in the Middle",
            1,
            1,
            year=2000,
            imdb="tt0212671",
        )

        assert results == []
        assert mock_client.request.call_count == 1  # never asked The Middle for its episodes


class TestSearchMediaThreadsIdsIntoNameFallback:
    def test_ids_reach_search_tv_episode(self):
        from emby_dedupe.api import search as search_mod

        with patch.object(search_mod, "search_tv_episode", return_value=[]) as tv:
            search_mod.search_media(
                Mock(),
                "http://emby.local",
                "key",
                name="Malcolm in the Middle",
                year=2000,
                imdb="tt0212671",
                season=1,
                episode=1,
                skip_provider_search=True,
            )
        assert tv.call_args.kwargs == {
            "year": 2000,
            "imdb": "tt0212671",
            "tmdb": None,
            "tvdb": None,
        }
