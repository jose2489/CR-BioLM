# Distribution Map Module — public API
from .ficha import DistributionFicha, EntityRef, ElevationRange
from .parser import build_ficha
from .gazetteer import normalize_text, lookup

__all__ = [
    "DistributionFicha",
    "EntityRef",
    "ElevationRange",
    "build_ficha",
    "normalize_text",
    "lookup",
]
