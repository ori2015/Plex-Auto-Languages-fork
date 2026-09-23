from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable, Optional

from plexapi.video import Episode, Movie

from plex_auto_languages.track_changes import TrackChanges
from plex_auto_languages.utils.logger import get_logger

if TYPE_CHECKING:
    from plex_auto_languages.plex_server import PlexServer, UnprivilegedPlexServer

logger = get_logger()

# A subtitle "class" is what a preference is expressed in: (languageCode, forced).
# languageCode may be None - on this server that is overwhelmingly an external
# Hebrew .srt whose filename carries no language tag - and the matcher pairs
# None with None, so such a preference carries over like any other language.
SubtitleClass = tuple[Optional[str], bool]

PROFILE_TTL = timedelta(hours=24)
# How many distinct titles (shows or movies) of a user's history to sample,
# most recent first. Each costs one metadata request, once per PROFILE_TTL.
SAMPLED_TITLES = 20
# Plays read from history to know which shows a user has played.
HISTORY_PLAYS = 2000
# A class becomes the user's preference only when chosen at least this many
# times, and in at least this share of the sampled titles that offered it.
MIN_CHOICES = 3
MIN_SHARE = 0.6


@dataclass(frozen=True)
class HistorySample:
    """One sampled title: which subtitle classes it offered and which one the user had selected."""

    available: frozenset
    chosen: Optional[SubtitleClass]


def subtitle_class(stream) -> SubtitleClass:
    return stream.languageCode, TrackChanges.is_forced_subtitle(stream)


def show_key_of(play) -> Optional[str]:
    """
    The show rating key of an episode play, as a string.

    History entries carry grandparentKey ("/library/metadata/123") but not
    grandparentRatingKey, so the key is parsed from the path.
    """
    key = getattr(play, "grandparentRatingKey", None)
    if key is None and getattr(play, "grandparentKey", None):
        key = play.grandparentKey.rsplit("/", 1)[-1]
    return str(key) if key is not None else None


def preferred_subtitle_class(samples: list[HistorySample]) -> Optional[SubtitleClass]:
    """
    The subtitle class a user reliably picks when it is on offer, or None.

    A class is judged only against the titles that offered it, so a user who
    always takes Hebrew when there is one, and nothing otherwise, still has a
    Hebrew preference. "Subtitles off" is never a preference: without a learned
    class the user's items are left as Plex selects them.
    """
    chosen = Counter(sample.chosen for sample in samples if sample.chosen is not None)
    offered = Counter(cls for sample in samples for cls in sample.available)
    candidates = [
        (count, count / offered[cls], cls)
        for cls, count in chosen.items()
        if count >= MIN_CHOICES and count / offered[cls] >= MIN_SHARE
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate[:2])[2]


@dataclass
class UserProfile:
    """What a user's watch history says, refreshed every PROFILE_TTL."""

    played_shows: set = field(default_factory=set)
    # A real item from the user's history with the preferred subtitle selected.
    # It is used as a TrackChanges reference, so matching reuses the same
    # scoring as show-based propagation.
    subtitle_reference: Optional[object] = None
    expires_at: datetime = field(default_factory=datetime.now)


class HistoryProfiles:
    """Per-user profiles learned from Plex watch history, cached in memory for a day."""

    def __init__(self, plex: PlexServer, clock: Callable[[], datetime] = datetime.now):
        self._plex = plex
        self._clock = clock
        self._lock = threading.Lock()
        self._user_locks: dict = {}
        self._profiles: dict = {}

    def get(self, user_id, connect: Callable[[], Optional[UnprivilegedPlexServer]]) -> UserProfile:
        """
        The user's profile, building it from history when missing or expired.

        connect() returns the user's server instance. It is only called to build a
        profile, since creating one costs requests of its own.
        """
        user_id = str(user_id)
        with self._lock:
            user_lock = self._user_locks.setdefault(user_id, threading.Lock())
        with user_lock:
            profile = self._profiles.get(user_id)
            if profile is None or profile.expires_at <= self._clock():
                profile = self._build(user_id, connect)
                with self._lock:
                    self._profiles[user_id] = profile
            return profile

    def note_played(self, user_id, show_key) -> None:
        """Record a play seen live, so a show started today counts before the next rebuild."""
        with self._lock:
            profile = self._profiles.get(str(user_id))
            if profile is not None and show_key is not None:
                profile.played_shows.add(str(show_key))

    def _build(self, user_id: str, connect: Callable[[], Optional[UnprivilegedPlexServer]]) -> UserProfile:
        plays = self._plex.history_for_user(user_id, HISTORY_PLAYS)
        user_plex = connect()
        played_shows = {show_key_of(p) for p in plays if p.type == "episode"} - {None}

        latest_per_title = {}
        for play in plays:  # most recent first
            if play.type == "episode":
                title_key = f"show:{show_key_of(play)}"
            elif play.type == "movie":
                title_key = f"movie:{play.ratingKey}"
            else:
                continue
            latest_per_title.setdefault(title_key, play)

        samples, sampled_items = [], []
        for play in list(latest_per_title.values())[:SAMPLED_TITLES] if user_plex is not None else []:
            # Fetched as the user: selected streams are per user.
            item = user_plex.fetch_item(play.ratingKey)
            if item is None or not isinstance(item, (Episode, Movie)):
                continue
            streams = item.subtitleStreams()
            if not streams:
                continue  # offered no choice, says nothing about a preference
            selected = next((s for s in streams if s.selected), None)
            samples.append(HistorySample(
                available=frozenset(subtitle_class(s) for s in streams),
                chosen=subtitle_class(selected) if selected is not None else None,
            ))
            sampled_items.append(item)

        preferred = preferred_subtitle_class(samples)
        reference = None
        if preferred is not None:
            reference = next(item for sample, item in zip(samples, sampled_items) if sample.chosen == preferred)
        logger.debug(f"[History] User {user_id}: {len(played_shows)} played show(s), {len(samples)} informative "
                     f"sample(s), preferred subtitles: {preferred}")
        return UserProfile(played_shows=played_shows, subtitle_reference=reference,
                           expires_at=self._clock() + PROFILE_TTL)
