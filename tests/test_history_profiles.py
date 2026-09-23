from datetime import datetime
from types import SimpleNamespace

import pytest

from plex_auto_languages import history_profiles as history_module
from plex_auto_languages import plex_server as plex_server_module
from plex_auto_languages.constants import EventType
from plex_auto_languages.history_profiles import (
    HistoryProfiles, HistorySample, PROFILE_TTL, UserProfile, preferred_subtitle_class,
)
from plex_auto_languages.plex_server import PlexServer
from plex_auto_languages.track_changes import TrackChanges
from tests.fakes import FakeAudioStream, FakeEpisode, FakeMovie, FakePart, FakeSubtitleStream
from tests.test_cache_refresh import _cache

HEB = ("he", False)
ENG = ("en", False)
UNTAGGED = (None, False)  # external .srt without a language in its filename


def _sample(available, chosen):
    return HistorySample(available=frozenset(available), chosen=chosen)


def test_class_chosen_whenever_offered_is_preferred():
    samples = [_sample({HEB, ENG}, HEB)] * 3 + [_sample({ENG}, None)] * 5
    assert preferred_subtitle_class(samples) == HEB


def test_too_few_choices_is_no_preference():
    assert preferred_subtitle_class([_sample({HEB}, HEB)] * 2) is None


def test_class_chosen_only_sometimes_is_no_preference():
    samples = [_sample({HEB, ENG}, HEB)] * 3 + [_sample({HEB, ENG}, ENG)] * 3
    assert preferred_subtitle_class(samples) is None


def test_subtitles_off_is_never_a_preference():
    assert preferred_subtitle_class([_sample({HEB}, None)] * 10) is None


def test_untagged_external_subtitle_can_be_the_preference():
    samples = [_sample({ENG, UNTAGGED}, UNTAGGED)] * 4
    assert preferred_subtitle_class(samples) == UNTAGGED


def _item_with_subs(cls, key, chosen_code, codes):
    subs = [FakeSubtitleStream(language_code=code, selected=(code == chosen_code)) for code in codes]
    return cls(key=key, parts=[FakePart(subtitle_streams=subs)])


def _movie(key, chosen_code, codes):
    return _item_with_subs(FakeMovie, key, chosen_code, codes)


class _Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 23, 12, 0)

    def __call__(self):
        return self.now


@pytest.fixture
def fakes_are_plex_types(monkeypatch):
    monkeypatch.setattr(history_module, "Episode", FakeEpisode)
    monkeypatch.setattr(history_module, "Movie", FakeMovie)
    monkeypatch.setattr(plex_server_module, "Episode", FakeEpisode)
    monkeypatch.setattr(plex_server_module, "Movie", FakeMovie)


def _profiles(plays, items, clock=None):
    connects = []
    history_calls = []
    plex = SimpleNamespace(history_for_user=lambda user_id, n: (history_calls.append(user_id), plays)[1])
    user_plex = SimpleNamespace(fetch_item=lambda rating_key: items.get(rating_key))

    def connect():
        connects.append(1)
        return user_plex

    return HistoryProfiles(plex, clock or _Clock()), connect, connects, history_calls


def _play(kind, rating_key, show_key=None):
    # Like plexapi history entries: grandparentKey only, no grandparentRatingKey.
    grandparent_key = f"/library/metadata/{show_key}" if show_key is not None else None
    return SimpleNamespace(type=kind, ratingKey=rating_key, grandparentKey=grandparent_key)


def test_profile_learns_played_shows_and_most_recent_reference(fakes_are_plex_types):
    items = {i: _movie(f"/library/metadata/{i}", "he", ["en", "he"]) for i in (1, 2, 3)}
    plays = [_play("movie", 1), _play("episode", 50, show_key=7), _play("movie", 2), _play("movie", 3),
             _play("track", 99)]
    profiles, connect, _, _ = _profiles(plays, items)

    profile = profiles.get("5", connect)

    assert profile.played_shows == {"7"}
    assert profile.subtitle_reference is items[1]


def test_profile_is_cached_until_it_expires(fakes_are_plex_types):
    clock = _Clock()
    profiles, connect, connects, history_calls = _profiles([], {}, clock)

    profiles.get("5", connect)
    profiles.get("5", connect)
    assert history_calls == ["5"]

    clock.now += PROFILE_TTL
    profiles.get("5", connect)
    assert history_calls == ["5", "5"]


def test_note_played_counts_a_show_before_the_next_rebuild(fakes_are_plex_types):
    profiles, connect, _, _ = _profiles([], {})
    profiles.get("5", connect)

    profiles.note_played("5", 42)

    assert "42" in profiles.get("5", connect).played_shows


def test_history_reference_only_switches_to_a_matching_subtitle():
    reference = _movie("/library/metadata/1", "he", ["en", "he"])
    reference.parts[0].audio_streams = [FakeAudioStream(language_code="fr", selected=True)]

    eng_sub, heb_sub = FakeSubtitleStream(language_code="en", selected=True), FakeSubtitleStream(language_code="he")
    audio = [FakeAudioStream(language_code="en", selected=True), FakeAudioStream(language_code="fr")]
    target_part = FakePart(audio_streams=audio, subtitle_streams=[eng_sub, heb_sub])
    target = FakeMovie(key="/library/metadata/2", parts=[target_part])

    changes = TrackChanges("alice", reference, EventType.NEW_EPISODE, from_history=True)
    changes.compute([target])
    changes.apply()

    assert target_part.selected_subtitle_stream is heb_sub
    assert target_part.selected_audio_stream is None  # audio is never touched from history


def test_history_reference_never_clears_subtitles():
    reference = _movie("/library/metadata/1", "he", ["he"])
    target_part = FakePart(subtitle_streams=[FakeSubtitleStream(language_code="en", selected=True)])
    target = FakeEpisode(parts=[target_part])

    changes = TrackChanges("alice", reference, EventType.NEW_EPISODE, from_history=True)
    changes.compute([target])

    assert not changes.has_changes


class _UserPlex:
    def __init__(self, calls):
        self._calls = calls

    def fetch_item(self, item_id):
        self._calls.append(("fetch", item_id))
        if "movie" in str(item_id):
            return FakeMovie()
        return FakeEpisode(key=item_id, show_key=42)

    def get_show_reference(self, show):
        self._calls.append(("show_reference",))
        return FakeEpisode(key="/library/metadata/ref", show_key=42)


def _server(monkeypatch, profiles_by_user, item):
    calls, applied = [], []
    monkeypatch.setattr(plex_server_module, "NewOrUpdatedTrackChanges", lambda *_: SimpleNamespace(
        change_track_for_user=lambda username, reference, user_item, from_history:
            applied.append((username, reference.key, from_history)),
        has_changes=False, _track_changes=[], _episode=None))
    plex = object.__new__(PlexServer)
    plex.fetch_item = lambda item_id: item
    plex.get_all_user_ids = lambda: list(profiles_by_user)

    def instance(user_id):
        calls.append(("connect", user_id))
        return _UserPlex(calls)

    plex.get_plex_instance_of_user = instance
    plex.get_user_by_id = lambda user_id: SimpleNamespace(name=f"user{user_id}")
    plex.history_profiles = SimpleNamespace(get=lambda user_id, connect: profiles_by_user[user_id])
    return plex, calls, applied


HISTORY_REF = FakeMovie(key="/library/metadata/900")


def test_each_user_gets_show_reference_history_reference_or_nothing(monkeypatch, fakes_are_plex_types):
    profiles = {
        "watcher": UserProfile(played_shows={"42"}),
        "subtitle_user": UserProfile(subtitle_reference=HISTORY_REF),
        "stranger": UserProfile(),
    }
    plex, calls, applied = _server(monkeypatch, profiles, FakeEpisode(key="/library/metadata/1", show_key=42))

    plex.process_new_or_updated_item("/library/metadata/1", EventType.NEW_EPISODE, True)

    assert sorted(applied) == [
        ("usersubtitle_user", "/library/metadata/900", True),
        ("userwatcher", "/library/metadata/ref", False),
    ]
    assert not [c for c in calls if c[-1] == "stranger"]  # skipped without a single request


def test_movies_use_the_history_reference(monkeypatch, fakes_are_plex_types):
    profiles = {"watcher": UserProfile(played_shows={"42"}), "subtitle_user": UserProfile(subtitle_reference=HISTORY_REF)}
    plex, _, applied = _server(monkeypatch, profiles, FakeMovie(key="/library/metadata/movie"))

    plex.process_new_or_updated_item("/library/metadata/movie", EventType.NEW_EPISODE, True)

    assert applied == [("usersubtitle_user", "/library/metadata/900", True)]


def test_reference_memo_looks_up_show_reference_once_per_user_and_show(monkeypatch, fakes_are_plex_types):
    profiles = {"a": UserProfile(played_shows={"42"}), "b": UserProfile(played_shows={"42"})}
    plex, calls, _ = _server(monkeypatch, profiles, FakeEpisode(show_key=42))

    memo = {}
    for key in ("/library/metadata/1", "/library/metadata/2", "/library/metadata/3"):
        plex.process_new_or_updated_item(key, EventType.NEW_EPISODE, True, memo)

    assert calls.count(("show_reference",)) == 2


def test_refresh_does_not_report_long_existing_items_as_added():
    # First refresh after movies became covered: an old movie is new to the cache
    # but was not added since the last refresh, so it is a baseline, not "added".
    old = FakeMovie(key="/library/metadata/10", added_at=datetime(2026, 1, 1))
    new = FakeMovie(key="/library/metadata/11", added_at=datetime(2026, 9, 23, 12, 30))
    undated = FakeMovie(key="/library/metadata/12", added_at=None)
    cache = _cache({"/library/metadata/1": ["/part/a"]}, [old, new, undated])
    cache._last_refresh = datetime(2026, 9, 23, 12, 0)

    added = cache.refresh_library_cache()

    assert [(ref.key, ref.is_movie, ref.labels_key) for ref in added] == [
        ("/library/metadata/11", True, "/library/metadata/11")]
    assert "/library/metadata/10" in cache.episode_parts
