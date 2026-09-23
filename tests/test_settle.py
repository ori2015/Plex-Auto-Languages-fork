import json
import sqlite3
from datetime import datetime, timedelta
from threading import Lock, RLock
from types import SimpleNamespace

from plex_auto_languages import plex_server as plex_server_module
from plex_auto_languages.constants import EventType
from plex_auto_languages.plex_server import PlexServer, SETTLE_QUIET_PERIOD
from plex_auto_languages.plex_server_cache import PlexServerCache, parts_changed
from tests.fakes import FakeEpisode, FakeMedia, FakePart
from tests.test_cache_refresh import _cache

KEY = "/library/metadata/1"


def _episode(key, part_keys):
    episode = FakeEpisode(key=key)
    episode.media = [FakeMedia([FakePart(key=k) for k in part_keys])]
    return episode


def test_parts_changed():
    raw = "/library/parts/10/1790161850/file.mkv"
    assert not parts_changed([raw], [raw])
    # Same part written to again: Plex bumps the mtime segment.
    assert parts_changed([raw], ["/library/parts/10/1790161999/file.mkv"])
    # Replaced file: new part id.
    assert parts_changed([raw], ["/library/parts/11/1790161850/file.mkv"])
    assert parts_changed([raw], [raw, "/library/parts/12/1790161850/file.mkv"])
    # An mtime missing on one side is unknown, not a change.
    assert not parts_changed(["/library/parts/10/file.mkv"], [raw])


def test_refresh_settles_mtime_change_instead_of_returning_it():
    cache = _cache(
        {KEY: ["/library/parts/10/1790161850/file.mkv"]},
        [_episode(KEY, ["/library/parts/10/1790161999/file.mkv"])],
    )

    assert cache.refresh_library_cache() == []
    assert list(cache.settling) == [KEY]


def test_refresh_adopts_stripped_keys_without_settling():
    cache = _cache(
        {KEY: ["/library/parts/10/file.mkv"]},
        [_episode(KEY, ["/library/parts/10/1790161999/file.mkv"])],
    )

    cache.refresh_library_cache()

    assert cache.settling == {}
    assert cache.episode_parts[KEY] == ["/library/parts/10/1790161999/file.mkv"]


def test_every_change_restarts_the_settling_clock():
    cache = _cache({KEY: ["/library/parts/10/1/file.mkv"]}, [])

    assert cache.note_episode_parts(_episode(KEY, ["/library/parts/10/2/file.mkv"])) is True
    cache.settling[KEY] = datetime.now() - timedelta(hours=1)
    assert cache.settled_episode_keys(SETTLE_QUIET_PERIOD) == [KEY]

    assert cache.note_episode_parts(_episode(KEY, ["/library/parts/10/3/file.mkv"])) is True
    assert cache.settled_episode_keys(SETTLE_QUIET_PERIOD) == []


def test_unchanged_parts_do_not_settle():
    cache = _cache({KEY: ["/library/parts/10/1/file.mkv"]}, [])

    assert cache.note_episode_parts(_episode(KEY, ["/library/parts/10/1/file.mkv"])) is False
    assert cache.settling == {}


def test_settle_releases_only_unchanged_media():
    cache = _cache({KEY: ["/library/parts/10/2/file.mkv"]}, [])
    cache.settling[KEY] = datetime.now() - timedelta(hours=1)

    assert cache.settle(_episode(KEY, ["/library/parts/10/3/file.mkv"])) is False
    assert KEY in cache.settling and cache.settled_episode_keys(SETTLE_QUIET_PERIOD) == []

    assert cache.settle(_episode(KEY, ["/library/parts/10/3/file.mkv"])) is True
    assert cache.settling == {}


def _db_cache(db_path):
    cache = object.__new__(PlexServerCache)
    cache._db_path = db_path
    cache._lock = RLock()
    cache._save_in_progress = False
    cache._save_pending = False
    cache._last_refresh = datetime.fromtimestamp(0)
    cache.newly_added = {}
    cache.settling = {}
    cache.episode_parts = {}
    return cache


def test_settling_survives_save_and_load(tmp_path):
    db_path = str(tmp_path / "cache.sqlite3")
    since = datetime(2026, 9, 23, 14, 0)
    cache = _db_cache(db_path)
    cache._initialize_database()
    cache.episode_parts = {KEY: ["/library/parts/10/1/file.mkv"]}
    cache.settling = {KEY: since}
    cache.save(force=True)

    loaded = _db_cache(db_path)
    assert loaded._load_from_database() is True
    assert loaded.settling == {KEY: since}
    assert loaded.episode_parts == {KEY: ["/library/parts/10/1/file.mkv"]}


def test_initialize_adds_settling_column_to_existing_database(tmp_path):
    db_path = str(tmp_path / "cache.sqlite3")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE episodes (episode_key TEXT PRIMARY KEY, newly_added_at TEXT NULL, "
            "newly_updated_at TEXT NULL, part_keys_json TEXT NOT NULL DEFAULT '[]')"
        )
        conn.execute("INSERT INTO episodes (episode_key, part_keys_json) VALUES (?, ?)",
                     (KEY, json.dumps(["/library/parts/10/1/file.mkv"])))

    cache = _db_cache(db_path)
    cache._initialize_database()

    assert cache._load_from_database() is True
    assert cache.settling == {}


def _settle_plex(monkeypatch, current_parts, library="TV Shows"):
    processed = []
    plex = object.__new__(PlexServer)
    plex.config = {"ignore_libraries": ["Kids"], "ignore_filepatterns": [""]}
    plex._settle_lock = Lock()
    plex.cache = _cache({KEY: ["/library/parts/10/2/file.mkv"]}, [])
    plex.cache.settling[KEY] = datetime.now() - SETTLE_QUIET_PERIOD - timedelta(seconds=1)
    monkeypatch.setattr(plex_server_module, "Episode", FakeEpisode)
    episode = _episode(KEY, current_parts or [])
    episode.index = 1
    episode.librarySectionTitle = library
    plex.fetch_item = lambda key: episode if current_parts is not None else None
    plex.should_ignore_show_by_key = lambda *_: False
    plex.process_new_or_updated_episode = lambda *args: processed.append(args[:3])
    return plex, processed


def test_settled_episode_is_processed_once(monkeypatch):
    plex, processed = _settle_plex(monkeypatch, ["/library/parts/10/2/file.mkv"])

    plex.process_settled_episodes()
    plex.process_settled_episodes()

    assert processed == [(KEY, EventType.UPDATED_EPISODE, False)]
    assert plex.cache.settling == {}


def test_episode_changed_since_last_seen_keeps_settling(monkeypatch):
    plex, processed = _settle_plex(monkeypatch, ["/library/parts/10/3/file.mkv"])

    plex.process_settled_episodes()

    assert processed == []
    assert KEY in plex.cache.settling


def test_deleted_episode_stops_settling(monkeypatch):
    plex, processed = _settle_plex(monkeypatch, None)

    plex.process_settled_episodes()

    assert processed == [] and plex.cache.settling == {}


def test_ignored_library_stops_settling_without_processing(monkeypatch):
    plex, processed = _settle_plex(monkeypatch, ["/library/parts/10/2/file.mkv"], library="Kids")

    plex.process_settled_episodes()

    assert processed == [] and plex.cache.settling == {}


class _UserPlex:
    def __init__(self, calls):
        self._calls = calls

    def fetch_item(self, item_id):
        return FakeEpisode(key=item_id, show_key=42)

    def get_last_watched_or_first_episode(self, show):
        self._calls.append(show)
        return FakeEpisode(key="/library/metadata/ref", show_key=42)


def _process_batch(monkeypatch, reference_memo):
    calls = []
    monkeypatch.setattr(plex_server_module, "NewOrUpdatedTrackChanges", lambda *_: SimpleNamespace(
        change_track_for_user=lambda *_: None, has_changes=False, _track_changes=[], _episode=None))
    plex = object.__new__(PlexServer)
    plex.get_all_user_ids = lambda: ["1", "2"]
    plex.get_plex_instance_of_user = lambda user_id: _UserPlex(calls)
    plex.get_user_by_id = lambda user_id: SimpleNamespace(name=f"user{user_id}")

    for key in ("/library/metadata/1", "/library/metadata/2", "/library/metadata/3"):
        plex.process_new_or_updated_episode(key, EventType.NEW_EPISODE, True, reference_memo)
    return calls


def test_reference_memo_looks_up_reference_once_per_user_and_show(monkeypatch):
    assert len(_process_batch(monkeypatch, {})) == 2


def test_without_memo_reference_is_looked_up_per_episode(monkeypatch):
    assert len(_process_batch(monkeypatch, None)) == 6
