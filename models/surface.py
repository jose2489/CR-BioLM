"""
Superficie de idoneidad del SDM: proyecta el modelo entrenado sobre la malla
climática y la dibuja.

Esta era la pieza ausente. El Random Forest se entrenaba, se evaluaba y se
explicaba con SHAP/LIME, pero su PREDICCIÓN ESPACIAL —el producto propio de un
modelo de distribución— nunca se generaba. El tier T3 describía al LLM una
"Imagen 2: mapa predictivo climático RF" que no existía: el archivo al que
apuntaba lo escribía ``plot_spatial_overlap``, que dibuja puntos GBIF y un
polígono experto (siempre None), no una predicción.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio


def predict_surface(model, raster_paths: dict, feature_names: list[str]):
    """
    Proyecta ``model`` sobre la malla de los rasters recortados.

    ``feature_names`` DEBE venir en el mismo orden con el que se entrenó el
    modelo (``RandomForestSDM.feature_names``); de lo contrario las columnas se
    mezclan silenciosamente y la predicción queda sin sentido.

    Devuelve (idoneidad, transform, crs) donde idoneidad es un arreglo float32
    de la forma de la malla, con NaN fuera de la zona con datos.
    """
    faltantes = [v for v in feature_names if v not in raster_paths]
    if faltantes:
        raise ValueError(f"Faltan rasters para las variables del modelo: {faltantes}")

    capas, transform, crs, forma = [], None, None, None
    for v in feature_names:
        with rasterio.open(raster_paths[v]) as s:
            a = s.read(1).astype("float32")
            nod = s.nodata if s.nodata is not None else -3.4e38
            if transform is None:
                transform, crs, forma = s.transform, s.crs, a.shape
            elif a.shape != forma:
                raise ValueError(
                    f"Malla inconsistente en {v}: {a.shape} vs {forma}. "
                    f"Todos los rasters deben venir del mismo recorte.")
        a[(a == nod) | (a < -1e30)] = np.nan
        capas.append(a)

    cubo = np.stack(capas)                                  # (n_vars, H, W)
    valido = np.all(np.isfinite(cubo), axis=0)              # celdas con dato completo
    if not valido.any():
        raise ValueError("No hay celdas con datos completos en todas las variables.")

    X = cubo[:, valido].T                                   # (n_celdas, n_vars)
    proba = model.predict_proba(X)[:, 1]

    idoneidad = np.full(forma, np.nan, dtype="float32")
    idoneidad[valido] = proba
    return idoneidad, transform, crs


def plot_suitability(idoneidad, transform, output_path, *, species_name="",
                     presencias_gdf=None, boundary_gdf=None, auc=None):
    """
    Dibuja la superficie de idoneidad. Mismo encuadre que el mapa de hábitat para
    que ambas imágenes sean comparables cuando se le pasan juntas al modelo.
    """
    h, w = idoneidad.shape
    izq, arr = transform.c, transform.f
    der, aba = izq + transform.a * w, arr + transform.e * h

    fig, ax = plt.subplots(figsize=(9, 9), dpi=140)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    if boundary_gdf is not None and not boundary_gdf.empty:
        boundary_gdf.plot(ax=ax, facecolor="#2e3440", edgecolor="#4c566a",
                          linewidth=0.6, alpha=0.85, zorder=1)

    im = ax.imshow(idoneidad, extent=[izq, der, aba, arr], origin="upper",
                   cmap="magma", vmin=0.0, vmax=1.0, alpha=0.92,
                   interpolation="nearest", zorder=2)

    if boundary_gdf is not None and not boundary_gdf.empty:
        boundary_gdf.plot(ax=ax, facecolor="none", edgecolor="#9ca3af",
                          linewidth=0.8, zorder=3)

    n_pts = 0
    if presencias_gdf is not None and not presencias_gdf.empty:
        n_pts = len(presencias_gdf)
        ax.scatter(presencias_gdf.geometry.x, presencias_gdf.geometry.y,
                   s=14, c="#22d3ee", edgecolors="white", linewidths=0.4,
                   alpha=0.9, zorder=4, label=f"Presencias GBIF (n={n_pts})")
        ax.legend(loc="lower left", fontsize=7.5, facecolor="#1f2937",
                  labelcolor="white", framealpha=0.9, edgecolor="#374151")

    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cb.set_label("Idoneidad climática predicha (0-1)", color="#e5e7eb", fontsize=8)
    cb.ax.tick_params(colors="#9ca3af", labelsize=7)
    cb.outline.set_edgecolor("#374151")

    titulo = f"Idoneidad climática — Random Forest\n{species_name}".strip()
    if auc is not None:
        titulo += f"   (AUC bloques espaciales: {auc:.3f})"
    ax.set_title(titulo, color="white", fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Zonas más CLARAS = mayor idoneidad climática predicha",
                  color="#9ca3af", fontsize=8)
    ax.tick_params(colors="#6b7280", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#374151")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Superficie de idoneidad RF guardada en: {output_path}")
    return output_path


def build_and_plot(model, raster_paths, feature_names, output_path, **kw):
    """Conveniencia: proyecta y dibuja en un paso."""
    idoneidad, transform, _crs = predict_surface(model, raster_paths, feature_names)
    return plot_suitability(idoneidad, transform, output_path, **kw)
