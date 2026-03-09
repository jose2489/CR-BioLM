# Leer los argumentos enviados desde Python
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Error: Se requieren dos argumentos (Nombre de la Especie y Directorio de Salida).")
}

# El nombre de la especie vendra con guiones bajos desde Python, los cambiamos a espacios para BIEN
especie_input <- args[1]
especie_bien <- gsub("_", " ", especie_input)
directorio_salida <- args[2]

# Cargar librerias silenciosamente para no ensuciar la consola de Python
suppressMessages(library(BIEN))
suppressMessages(library(sf))

cat(paste("[R-BIEN] Consultando servidores para:", especie_bien, "...\n"))

# Descargar el poligono experto
rango <- BIEN_ranges_load_species(especie_bien)

# Verificar si la especie existe en la base de datos
if (is.null(rango) || nrow(rango) == 0) {
  cat("[R-BIEN] Advertencia: No se encontro mapa experto para esta especie.\n")
  quit(status = 1) # Salir con codigo de error 1 para que Python sepa que fallo
}

# Construir la ruta final y guardar el shapefile
nombre_archivo <- paste0(especie_input, ".shp")
ruta_salida <- file.path(directorio_salida, nombre_archivo)

# Guardar usando sf
st_write(rango, ruta_salida, append = FALSE, quiet = TRUE)
cat(paste("[R-BIEN] Descarga exitosa. Shapefile guardado en:", ruta_salida, "\n"))