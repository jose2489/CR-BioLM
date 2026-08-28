"""Generate Maps_demo_V2_new for the 45 new species from seleccion_manual_especies_tomos_varios.txt"""
import os
from utils.distribution_map.parser import build_ficha
from utils.distribution_map.renderer import generate_distribution_map
from data.gbif_extractor import GBIFExtractor
from data.expert_maps import ExpertMapLoader
import config

NEW_SPECIES = [
    ("Elleanthus hymenophorus",    "Bosque muy húmedo, pluvial y nuboso, 200-1700(-2400) m",   "vert. Carib. y cerca de la División Continental, Cords. de Guanacaste, de Tilarán y Central, ambas verts. Cord. de Talamanca, vert. Pac., región de Puriscal (Z.P. La Cangreja), S Fila Costeña (Fila Cruces)."),
    ("Epidendrum ramosum",         "Bosque muy húmedo, pluvial y nuboso, 450-1600 m",           "vert. Carib. y cerca de la División Continental, Cords. de Tilarán y Central, vert. Pac. Cords. de Guanacaste y de Talamanca."),
    ("Jacquiniella teretifolia",   "Bosque húmedo, muy húmedo, pluvial y nuboso, (300-)1100-1850 m", "vert. Carib. y cerca de la División Continental, Cords. de Guanacaste, de Tilarán y Central, ambas verts. Cord. de Talamanca, centro vert. Pac., Cerros de Escazú, Cerro Turrubares."),
    ("Lepanthes ciliisepala",      "Bosque nuboso y de roble, 1400-2050 m",                     "ambas verts. Cord. Central."),
    ("Lepanthes lindleyana",       "Bosque nuboso y de roble, 1350-2200(-3000) m",              "ambas verts. Cords. Central y de Talamanca, Cerros de La Carpintera, vert. Pac. Cord. de Tilarán."),
    ("Maxillaria adolphi",         "Bosque pluvial y de roble, 2200-3300 m",                    "ambas verts. Cords. Central y de Talamanca, centro vert. Pac., Cerros de Escazú."),
    ("Phragmipedium longifolium",  "Bosque muy húmedo y pluvial, 500-1500 m",                   "vert. Carib. y cerca de la División Continental, Cords. de Tilarán y Central, N Cord. de Talamanca."),
    ("Pleurothallis cardiothallis","Bosque muy húmedo, pluvial y nuboso, 750-2000 m",            "vert. Carib. y cerca de la División Continental, todas las cords. principales."),
    ("Prosthechea livida",         "Bosque húmedo y muy húmedo, 900-1200 m",                    "vert. Pac. Cords. de Tilarán y Central, Valle Central."),
    ("Telipogon costaricensis",    "Bosque muy húmedo, pluvial y de roble, (600-)2700-3200 m",  "vert. Pac. y cerca de la División Continental, Cord. de Talamanca, Valle de General."),
    ("Andropogon glomeratus",      "Bosque húmedo, muy húmedo y pluvial, (0-)600-1750 m",       "vert. Carib. Cords. Central y de Talamanca, vecindad de Puerto Limón, ambas verts. Cord. de Tilarán, Valle Central, vert. Pac., Cerros de Escazú, S Fila Costeña (Fila Cruces)."),
    ("Aristida jorullensis",       "Bosque seco y húmedo, 0-500 m",                             "vert. Pac., llanuras de Guanacaste, O Valle Central, región de Orotina."),
    ("Arthrostylidium merostachyoides", "Bosque muy húmedo, pluvial y nuboso, (750-)1100-1700 m", "vert. Carib. y cerca de la División Continental, Cords. de Guanacaste, de Tilarán y de Talamanca."),
    ("Chusquea subtessellata",     "Bosque de roble y páramo, (2700-)3000-3800 m",              "ambas verts. Cord. de Talamanca (Cerros de La Muerte, Las Vueltas, Cuericí, Chirripó, Kámuk, Echandi, etc.)."),
    ("Dactyloctenium aegyptium",   "Bosque seco, húmedo y muy húmedo, 0-400 m",                 "toda la vert. Pac."),
    ("Dichanthelium viscidellum",  "Bosque muy húmedo, pluvial y nuboso, 700-2000 m",           "vert. Carib. Cord. de Tilarán, ambas verts. Cords. Central y de Talamanca, vert. Pac. Cord. de Guanacaste, Cerro Espíritu Santo, Tablazo, Cerros de Escazú, S Fila Costeña (Fila Cruces)."),
    ("Clusia croatii",             "Bosque húmedo, muy húmedo, pluvial y nuboso, 0-1800 m",     "vert. Carib., Llanuras de San Carlos y de Tortuguero, ambas verts. todas las cords. principales, Cerros de La Carpintera, vert. Pac., Montes del Aguacate, Tablazo, Cerros de Escazú, Cerros Turrubares y Caraigres, Fila Costeña, Valle Central, cuenca del Río Grande de Candelaria, N Valle de General."),
    ("Ipomoea pes-caprae",         "Bosque seco, húmedo y muy húmedo, 0-50(-100) m",            "vert. Carib., Llanura de Tortuguero, vecindad de Puerto Limón, Baja Talamanca, vert. Pac., N llanuras de Guanacaste (P.N. Santa Rosa), Pens. de Santa Elena y de Nicoya, Isla San Lucas, vecindad de Puntarenas a vecindad de Parrita, Uvita, región de Golfo Dulce, Pen. de Burica, Isla del Coco."),
    ("Cornus disciflora",          "Bosque muy húmedo, pluvial, nuboso y de roble, 1200-2650 m","vert. Pac. y cerca de la División Continental, Cords. de Tilarán, Central y de Talamanca, Tablazo, Cerros de Escazú, Cerro Caraigres."),
    ("Echeveria australis",        "Bosque húmedo, muy húmedo y pluvial, 1100-2200 m",          "ambas verts. Cord. Central, Cerros de La Carpintera, Valle Central, vert. Pac. Cord. de Tilarán, N Cord. de Talamanca, Tablazo, Cerros de Escazú, Cerro Caraigres."),
    ("Cucumis melo",               "Bosque seco, húmedo y muy húmedo, 0-300(-1200) m",          "vert. Carib. Cord. de Talamanca, cuenca del Río Sapoá (vecindad de La Cruz), Llanuras de San Carlos y de Tortuguero, Baja Talamanca, vert. Pac., cuenca del Río Tempisque al S hasta Río Grande de Tárcoles, Pen. de Nicoya, Valle Central, P.N. Carara, región de Puriscal, vecindad de Parrita, región de Golfo Dulce."),
    ("Gurania makoyana",           "Bosque húmedo, muy húmedo y pluvial, 0-1550 m",             "vert. Carib. Cord. Central, Llanuras de Los Guatusos, de Tortuguero y de Santa Clara, Baja Talamanca, ambas verts. Cords. de Guanacaste, de Tilarán y de Talamanca, vert. Pac., Cerro Turrubares, Fila Costeña, P.N. Carara, región de Puriscal (P.N. La Cangreja), cuenca del Río Grande de Candelaria, región de Golfo Dulce."),
    ("Melothria pendula",          "Bosque seco, húmedo, muy húmedo, pluvial y nuboso, 0-1600 m","vert. Carib. Cord. Central, Llanuras de Los Guatusos, de Tortuguero y de Santa Clara, vecindad de Puerto Limón, Baja Talamanca, ambas verts. Cords. de Guanacaste, de Tilarán y de Talamanca, vert. Pac., S Fila Costeña (J.B. Wilson), llanuras de Guanacaste, Pen. de Nicoya, Isla San Lucas, Valle Central, P.N. Carara, vecindad de Puerto Quepos, Valle de General, Dominical, Valle de Coto Brus, región de Golfo Dulce."),
    ("Psiguria warscewiczii",      "Bosque húmedo, muy húmedo y pluvial, 0-1250 m",             "vert. Carib. Cords. Central y de Talamanca, Llanuras de San Carlos, de Tortuguero (P.N. Tortuguero) y de Santa Clara, vecindad de Puerto Limón, ambas verts. Cords. de Guanacaste y de Tilarán, vert. Pac., N llanuras de Guanacaste, Pen. de Nicoya, vecindad de Esparza, Valle Central, P.N. Carara, región de Puriscal (P.N. La Cangreja), vecindad de Puerto Quepos, Pen. de Osa."),
    ("Gonocalyx pterocarpus",      "Bosque muy húmedo, pluvial, nuboso, de roble y enano, 700-2250 m", "ambas verts. Cords. de Guanacaste, de Tilarán y Central."),
    ("Sphyrospermum buxifolium",   "Bosque muy húmedo, pluvial y nuboso, 0-1900(-2350) m",     "vert. Carib., Llanuras de San Carlos y de Tortuguero, ambas verts. todas las cords. principales, Cerros de La Carpintera, vert. Pac., Cerros de Escazú."),
    ("Cnidoscolus urens",          "Bosque seco y húmedo, 0-200 m",                             "vert. Pac., llanuras de Guanacaste al S hasta vecindad de Orotina, Pen. de Santa Elena, N Pen. de Nicoya, Islas Chira y San Lucas, Islas Zopilote."),
    ("Euphorbia hirta",            "Bosque seco, húmedo, muy húmedo y pluvial, 0-1200 m",       "vert. Carib. Cords. de Guanacaste, Central (E.B. La Selva) y de Talamanca, Llanura de San Carlos, vecindad de Puerto Limón, Baja Talamanca, vert. Pac. Cord. de Tilarán, N Cord. de Talamanca, llanuras de Guanacaste al S hasta vecindad de Puntarenas, Pen. de Santa Elena, Islas Murciélago, Pen. de Nicoya, Islas Chira, San Lucas y Venado, Valle Central, P.N. Carara, N Valle de General, Uvita, región de Golfo Dulce, Isla del Coco."),
    ("Jatropha gossypiifolia",     "Bosque seco, húmedo y muy húmedo, 0-400 m",                 "vert. Carib. Cord. Central (E.B. La Selva), Llanuras de Tortuguero y de Santa Clara, vecindad de Puerto Limón, vert. Pac., N Fila Costeña, llanuras de Guanacaste al S hasta vecindad de Orotina, Pens. de Santa Elena y de Nicoya, Islas Chira, San Lucas y Venado, vecindades de Parrita y de Uvita, Pen. de Osa."),
    ("Phyllanthus niruri",         "Bosque muy húmedo, pluvial, nuboso y de roble, 500-2500 m", "ambas verts. todas las cords. principales, vert. Pac., Tablazo, Cerros de Escazú, Cerros Turrubares y Caraigres, Valle Central, cuenca del Río Grande de Candelaria, N Valle de General."),
    ("Inga sapindoides",           "Bosque húmedo, muy húmedo y pluvial, 0-1400(-1700) m",      "vert. Carib. Cords. Central y de Talamanca, Llanuras de San Carlos y de Tortuguero, Baja Talamanca, ambas verts. Cords. de Guanacaste y de Tilarán, vert. Pac. E Cord. de Talamanca, Fila Costeña, N llanuras de Guanacaste (P.N. Santa Rosa), Pen. de Nicoya, Valle Central, región de Turrubares, P.N. Carara, región de Puriscal, Valle de General, Pens. de Osa y de Burica."),
    ("Antidaphne viscoidea",       "Bosque muy húmedo, pluvial, nuboso y de roble, (700-)1100-2100(-2600) m", "ambas verts. todas las cords. principales, Valle Central, vert. Pac., región de Puriscal (P.N. La Cangreja), Cerros de Escazú."),
    ("Boehmeria aspera",           "Bosque muy húmedo, pluvial, nuboso y de roble, (100-)300-1800(-2200) m", "vert. Carib. Cords. de Guanacaste, de Tilarán y Central, E Cord. de Talamanca, ambas verts. N Cord. de Talamanca."),
    ("Stachytarpheta cayennensis", "Bosque húmedo y muy húmedo, 0-1250 m",                      "vert. Carib. Cords. Central y de Talamanca, cuenca del Río Sapoá, Llanuras de San Carlos, de Tortuguero y de Santa Clara, vecindad de Puerto Limón, Baja Talamanca, ambas verts. Cord. de Guanacaste, vert. Pac. E Cord. de Talamanca, N Fila Costeña (vecindad de Boruca), región de Mora, Pen. de Osa."),
    ("Hybanthus galeottii",        "Bosque muy húmedo y pluvial, 1100-1900 m",                  "vert. Carib. N Cord. de Talamanca, ambas verts., Cerros de La Carpintera, vert. Pac. Cord. de Talamanca, Cerros de Escazú, Cerros Turrubares y Caraigres, S Fila Costeña, Valle Central."),
    ("Allophylus psilospermus",    "Bosque húmedo, muy húmedo y pluvial, 0-1800(-2300) m",      "vert. Carib. Cord. Central, N Cord. de Talamanca, Llanuras de San Carlos y de Tortuguero, ambas verts. Cords. de Guanacaste y de Tilarán, vert. Pac. Cord. de Talamanca, Fila Costeña, P.N. Carara, región de Puriscal (P.N. La Cangreja), cuenca del Río Grande de Candelaria, vecindades de Parrita y de Puerto Quepos, N Valle de General, región de Golfo Dulce."),
    ("Paullinia cururu",           "Bosque seco, húmedo y muy húmedo, 0-1000(-1350) m",         "vert. Pac. Cords. de Guanacaste y de Tilarán, Cerro Caraigres, N Fila Costeña, llanuras de Guanacaste, Isla Bolaños, Pens. de Santa Elena y de Nicoya, Valle Central, vecindades de Tivives y de Tárcoles, P.N. Carara, región de Puriscal, cuenca del Río Grande de Candelaria, región de Golfo Dulce."),
    ("Micropholis melinoniana",    "Bosque muy húmedo y pluvial, 0-1200(-1700) m",              "vert. Carib. Cord. Central, N Cord. de Talamanca, Llanura de San Carlos (Boca Tapada), Baja Talamanca (R.N.V.S. Gandoca-Manzanillo), vert. Pac., P.N. Carara, región de Golfo Dulce."),
    ("Castilleja talamancensis",   "Bosque pluvial, de roble y páramo, 2400-3450 m",            "vert. Carib. Cord. Central (Volcán Turrialba), ambas verts. Cord. de Talamanca."),
    ("Siparuna grandiflora",       "Bosque muy húmedo, pluvial, nuboso y de roble, 0-1650(-2500) m", "vert. Carib., Llanuras de San Carlos, de Tortuguero y de Santa Clara, ambas verts. todas las cords. principales, vert. Pac., N Fila Costeña, Pen. de Osa."),
    ("Cestrum schlechtendalii",    "Bosque húmedo, muy húmedo, pluvial y nuboso, 0-1900(-2400) m", "vert. Carib., Llanuras de San Carlos, de Tortuguero y de Santa Clara, vecindad de Puerto Limón, Baja Talamanca (R.N.V.S. Gandoca-Manzanillo), ambas verts. todas las cords. principales, vert. Pac., Cerros de Escazú, Cerros Turrubares y Caraigres, Valle Central, región de Puriscal (P.N. La Cangreja), S Valle de General, región de Golfo Dulce."),
    ("Lycianthes furcatistellata", "Bosque pluvial, nuboso y de roble, 1200-2500(-2800) m",     "vert. Carib. y cerca de la División Continental, Cord. de Guanacaste, ambas verts. Cords. de Tilarán, Central y de Talamanca, vert. Pac., Montes del Aguacate."),
    ("Solanum americanum",         "Bosque seco, húmedo, muy húmedo, pluvial y nuboso, 0-1850 m","vert. Carib., Llanuras de San Carlos, de Tortuguero y de Santa Clara, ambas verts. Cords. de Tilarán, Central y de Talamanca, Valle Central, vert. Pac. Cord. de Guanacaste, Cerros Turrubares y Caraigres, llanuras de Guanacaste, cuenca baja del Río Grande de Tárcoles, cuenca del Río Grande de Candelaria, N Valle de General, vecindad de Puerto Cortés, cañón del Río Grande de Térraba, Valle de Coto Colorado, Isla del Coco."),
    ("Symplocos panamensis",       "Bosque muy húmedo y pluvial, 0-1550 m",                     "vert. Carib. Cords. de Guanacaste, Central y de Talamanca, ambas verts. Cord. de Tilarán, vert. Pac. N Cord. de Talamanca, S Fila Costeña, Pen. de Osa."),
    ("Daphnopsis americana",       "Bosque muy húmedo, pluvial, nuboso y de roble, (50-)200-2100 m", "vert. Carib. y cerca de la División Continental, N Cord. de Talamanca, Llanura de San Carlos, ambas verts. Cords. de Guanacaste, de Tilarán y Central, vert. Pac. E Cord. de Talamanca, Tablazo, Cerros de Escazú, Pen. de Osa."),
]


def main():
    extractor = GBIFExtractor()
    map_loader = ExpertMapLoader()
    cr_bounds = map_loader.load_country_boundary(config.DEFAULT_COUNTRY)
    meso_bounds = map_loader.load_mesoamerica_boundary()
    if meso_bounds is None or meso_bounds.empty:
        meso_bounds = cr_bounds

    OUT_DIR = "outputs/Maps_demo_V2_new"
    errors = []

    for sp, hab, geo in NEW_SPECIES:
        sp_slug = sp.replace(" ", "_").replace("-", "_")
        out_dir = os.path.join(OUT_DIR, sp_slug)
        os.makedirs(out_dir, exist_ok=True)

        print(f"=== {sp} ===")
        ficha = build_ficha(habitat_raw=hab, geographic_notes=geo, species=sp)
        print(f"  regions: {[r.canonical_name for r in ficha.regions]}")
        print(f"  elev:    {ficha.elevation}")
        ficha.save(os.path.join(out_dir, "ficha.json"))

        try:
            meso = extractor.fetch_occurrences_mesoamerica(sp)
            meso = extractor.clean_spatial_outliers(meso, meso_bounds)
            pres = extractor.clean_spatial_outliers(meso, cr_bounds) if meso is not None and not meso.empty else None
            if pres is None or pres.empty:
                pres = meso
            print(f"  GBIF:    {len(pres) if pres is not None else 0} pts")
        except Exception as e:
            print(f"  GBIF err: {e}")
            pres = None

        try:
            generate_distribution_map(ficha, os.path.join(out_dir, "map.png"), presencias_gdf=pres)
            print(f"  [OK]")
        except Exception as e:
            print(f"  [ERR] {e}")
            errors.append((sp, str(e)))

    print()
    print("-" * 60)
    print(f"Done. {len(NEW_SPECIES) - len(errors)}/{len(NEW_SPECIES)} OK")
    for sp, e in errors:
        print(f"  FAIL: {sp}: {e}")


if __name__ == "__main__":
    main()
