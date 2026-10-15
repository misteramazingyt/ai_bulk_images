from .base import Collector, Frame
from .bluesky import BlueskyCollector
from .diffusiondb import DiffusionDBCollector
from .factcheck import FactCheckCollector
from .reddit import RedditCollector
from .wikimedia import WikimediaCollector

REGISTRY: dict[str, type[Collector]] = {
    c.name: c for c in (DiffusionDBCollector, RedditCollector, BlueskyCollector, WikimediaCollector, FactCheckCollector)
}

__all__ = ["Collector", "Frame", "REGISTRY"]
