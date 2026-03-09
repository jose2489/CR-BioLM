cat("INICIANDO EXTRACCIÓN DEL CATÁLOGO...\n")

# Configuramos tu librería personal (la que creamos en el paso anterior)
mi_libreria <- Sys.getenv("R_LIBS_USER")
.libPaths(mi_libreria)
library(BIEN, lib.loc = mi_libreria)

cat("Consultando a los servidores de BIEN...\n")
cat("Descargando la lista global de especies con mapas...\n")

# Extraemos el catálogo de especies con mapas
catalogo_mapas <- BIEN_ranges_list()

# Lo guardamos como un archivo CSV en tu computadora
archivo_csv <- "catalogo_mapas_bien.csv"
write.csv(catalogo_mapas, archivo_csv, row.names = FALSE)

cat("¡ÉXITO! Catálogo descargado.\n")
cat("Se encontraron", nrow(catalogo_mapas), "especies con mapas expertos.\n")
cat("El archivo se guardó como:", archivo_csv, "\n")