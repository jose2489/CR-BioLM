"""
habitat_map.py — DEPRECATED shim.

All functionality has moved to utils/distribution_map/.
This file exists only for backward compatibility during the transition.
"""
import math
import warnings

warnings.warn(
    "utils.map_gen.habitat_map is deprecated. "
    "Use utils.distribution_map.generate_distribution_map instead.",
    DeprecationWarning,
    stacklevel=2,
)

from utils.distribution_map import build_ficha, generate_distribution_map
from utils.distribution_map.ficha import ElevationRange
from utils.distribution_map.gazetteer import normalize_text as _normalize


def generate_habitat_map(
    species_name: str,
    geographic_notes: str,
    elevation_min,
    elevation_max,
    presencias_gdf=None,
    output_path="outputs/habitat_map.png",
    dem_path=None,
    elev_outlier_min=None,
    elev_outlier_max=None,
):
    """Compatibility wrapper — delegates to generate_distribution_map."""
    ficha = build_ficha(
        habitat_raw="",
        geographic_notes=str(geographic_notes or ""),
        species=species_name,
    )
    if not ficha.elevation.has_data() and elevation_min is not None and elevation_max is not None:
        try:
            if not (math.isnan(float(elevation_min)) or math.isnan(float(elevation_max))):
                ficha = ficha.__class__(
                    **{**ficha.to_dict(),
                       "elevation": ElevationRange(
                           min_m=float(elevation_min),
                           max_m=float(elevation_max),
                           outlier_min_m=elev_outlier_min,
                           outlier_max_m=elev_outlier_max,
                       )}
                )
        except Exception:
            pass
    return generate_distribution_map(
        ficha=ficha,
        output_path=output_path,
        presencias_gdf=presencias_gdf,
    )
