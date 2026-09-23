import json
import sqlite3
from datetime import datetime
from threading import RLock
from types import SimpleNamespace

from plex_auto_languages import plex_server as plex_server_module
from plex_auto_languages.constants import EventType
from plex_auto_languages.plex_server import PlexServer
from plex_auto_languages.plex_server_cache import PlexServerCache, part_identity
from tests.fakes import FakeEpisode, FakeMedia, FakePart
from tests.test_cache_refresh import _cache


def _episode(key, part_keys):
    episode = FakeEpisode(key=key)
    episode.media = [FakeMedia([FakePart(key=k) for k in part_keys])]
    return episode


def test_part_identity_strips_mtime_and_is_idempotent():
    raw = "/library/parts/98644/1790161850/file.mkv"
    assert part_identity(raw) == "/library/parts/98644/file.mkv"
    assert part_identity(part_identity(raw)) == "/library/parts/98644/file.mkv"


def test_part_identity_leaves_unknown_shapes_untouched():
    for key in ("/part/a", "/library/parts/1/file.mkv", "/library/parts/1/abc/file.mkv"):
        assert part_identity(key) == key


def test_refresh_ignores_mtime_only_change():
    # A file still being written into the library keeps its part id; only the
    # mtime in the key moves. That must not count as an updated episode.
    cache = _cache(
        {"/library/metadata/1": ["/library/parts/10/file.mkv"]},
        [_episode("/library/metadata/1", ["/library/parts/10/1790161999/file.mkv"])],
    )

    added, updated = cache.refresh_library_cache()

    assert (added, updated) == ([], [])
    assert cache.episode_parts["/library/metadata/1"] == ["/library/parts/10/file.mkv"]


def test_refresh_detects_replaced_part():
    cache = _cache(
        {"/library/metadata/1": ["/library/parts/10/file.mkv"]},
        [_episode("/library/metadata/1", ["/library/parts/11/1790161999/file.mkv"])],
    )

    _, updated = cache.refresh_library_cache()

    assert [ref.key for ref in updated] == ["/library/metadata/1"]


def test_did_episode_parts_change_ignores_mtime_only_change():
    cache = _cache({"/library/metadata/1": ["/library/parts/10/file.mkv"]}, [])

    touched = _episode("/library/metadata/1", ["/library/parts/10/1790162000/file.mkv"])
    replaced = _episode("/library/metadata/1", ["/library/parts/11/1790162000/file.mkv"])

    assert cache.did_episode_parts_change(touched) is False
    assert cache.did_episode_parts_change(replaced) is True


def test_load_normalizes_raw_keys_from_older_caches(tmp_path):
    # Caches written before part_identity hold raw keys. Without normalizing on
    # load, the first refresh after upgrading would flag the whole library.
    db_path = str(tmp_path / "cache.sqlite3")
    cache = object.__new__(PlexServerCache)
    cache._db_path = db_path
    cache._lock = RLock()
    cache._initialize_database()
    with cache._connect() as conn:
        conn.execute(
            "INSERT INTO episodes (episode_key, part_keys_json) VALUES (?, ?)",
            ("/library/metadata/1", json.dumps(["/library/parts/10/1790161850/file.mkv"])),
        )
        conn.commit()

    assert cache._load_from_database() is True
    assert cache.episode_parts == {"/library/metadata/1": ["/library/parts/10/file.mkv"]}


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
