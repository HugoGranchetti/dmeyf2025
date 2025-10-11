# ============================================================
# SCRIPT: Generar archivos Kaggle SIN simulación de ganancia
# Future (202106) NO tiene clase_ternaria
# ============================================================

require("data.table")
require("yaml")

# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_BASE <- "C:/Users/hgran/Desktop/DMEyF RStudio"
EXPERIMENTO <- 4941

# ============================================================
# CARGAR PREDICCIONES
# ============================================================

cat("\n=== GENERANDO ARCHIVOS KAGGLE ===\n")
cat("Experimento:", EXPERIMENTO, "\n\n")

# Ir a carpeta del experimento
setwd(file.path(RUTA_BASE, "exp", paste0("exp", EXPERIMENTO)))

# Cargar parámetros
PARAM <- read_yaml("PARAM.yml")

# Cargar predicciones
if (!file.exists("prediccion.txt")) {
  stop("ERROR: No se encuentra prediccion.txt")
}

tb_prediccion <- fread("prediccion.txt")
cat("Predicciones cargadas:", nrow(tb_prediccion), "registros\n")

# Ver distribución de probabilidades
cat("\n=== DISTRIBUCIÓN DE PROBABILIDADES ===\n")
print(summary(tb_prediccion$prob))

cat("\nTop 20 clientes con mayor probabilidad:\n")
print(tb_prediccion[order(-prob)][1:20])

# ============================================================
# GENERAR ARCHIVOS KAGGLE
# ============================================================

cat("\n=== GENERANDO ARCHIVOS CSV ===\n\n")

# Crear/limpiar carpeta kaggle
if (dir.exists("kaggle")) {
  unlink("kaggle", recursive = TRUE)
}
dir.create("kaggle")

# Ordenar por probabilidad descendente
setorder(tb_prediccion, -prob)

# Tabla resumen (sin ganancia)
resumen <- data.table(
  envios = integer(),
  prob_min = numeric(),
  prob_max = numeric(),
  prob_media = numeric()
)

# Generar archivos
for (envios in PARAM$cortes) {
  tb_prediccion[, Predicted := 0L]
  tb_prediccion[1:envios, Predicted := 1L]
  
  # Guardar archivo
  archivo_kaggle <- paste0("./kaggle/KA", PARAM$experimento, "_", envios, ".csv")
  fwrite(tb_prediccion[, list(numero_de_cliente, Predicted)], 
         file=archivo_kaggle, sep=",")
  
  # Estadísticas de este corte
  stats <- tb_prediccion[Predicted==1L, 
                         list(prob_min=min(prob), 
                              prob_max=max(prob), 
                              prob_media=mean(prob))]
  
  resumen <- rbind(resumen, data.table(
    envios = envios,
    prob_min = stats$prob_min,
    prob_max = stats$prob_max,
    prob_media = stats$prob_media
  ))
  
  # Mostrar
  cat("Envios=", sprintf("%5d", envios), "  ",
      "Prob: [", sprintf("%.4f", stats$prob_min), " - ", 
      sprintf("%.4f", stats$prob_max), "]  ",
      "Media: ", sprintf("%.4f", stats$prob_media), "\n", sep="")
}

# ============================================================
# ANÁLISIS Y RECOMENDACIONES
# ============================================================

cat("\n========================================\n")
cat("ANÁLISIS DE PROBABILIDADES\n")
cat("========================================\n\n")

# Identificar cortes con prob_media más alta
setorder(resumen, -prob_media)

cat("🎯 TOP 5 CORTES (por probabilidad media más alta):\n\n")
print(resumen[1:5])

cat("\n📊 ESTADÍSTICAS GENERALES:\n")
cat("Total de predicciones:", nrow(tb_prediccion), "\n")
cat("Clientes con prob > 0.50:", tb_prediccion[prob > 0.50, .N], "\n")
cat("Clientes con prob > 0.30:", tb_prediccion[prob > 0.30, .N], "\n")
cat("Clientes con prob > 0.10:", tb_prediccion[prob > 0.10, .N], "\n")
cat("Clientes con prob > 0.05:", tb_prediccion[prob > 0.05, .N], "\n")
cat("Clientes con prob > 0.01:", tb_prediccion[prob > 0.01, .N], "\n")

# Sugerir cortes basados en umbrales de probabilidad
cat("\n💡 SUGERENCIAS BASADAS EN PROBABILIDAD:\n\n")

umbrales <- c(0.50, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01)
for (umbral in umbrales) {
  n_clientes <- tb_prediccion[prob >= umbral, .N]
  if (n_clientes > 0) {
    prob_promedio <- tb_prediccion[prob >= umbral, mean(prob)]
    cat("Umbral prob >=", sprintf("%.2f", umbral), 
        "→", sprintf("%5d", n_clientes), "clientes",
        "(prob media:", sprintf("%.4f", prob_promedio), ")\n")
  }
}

cat("\n🎲 ESTRATEGIA CONSERVADORA:\n")
conservador <- tb_prediccion[prob > 0.10, .N]
cat("Enviar solo a clientes con prob > 0.10:", conservador, "envíos\n")

cat("\n⚖️ ESTRATEGIA BALANCEADA:\n")
balanceado <- tb_prediccion[prob > 0.05, .N]
cat("Enviar a clientes con prob > 0.05:", balanceado, "envíos\n")

cat("\n🚀 ESTRATEGIA AGRESIVA:\n")
agresivo <- tb_prediccion[prob > 0.01, .N]
cat("Enviar a clientes con prob > 0.01:", agresivo, "envíos\n")

# Guardar resumen
fwrite(resumen, file="resumen_probabilidades.csv")
cat("\n💾 Resumen guardado en: resumen_probabilidades.csv\n")

cat("\n========================================\n")
cat("RECOMENDACIÓN FINAL\n")
cat("========================================\n")

# Buscar el corte más cercano a la estrategia balanceada
target <- tb_prediccion[prob > 0.05, .N]
diff <- abs(PARAM$cortes - target)
mejor_corte <- PARAM$cortes[which.min(diff)]

cat("\nBasado en tu modelo (AUC = 0.93):\n")
cat("1️⃣ Primera submission: KA", PARAM$experimento, "_", mejor_corte, ".csv\n", sep="")
cat("   (Aproximadamente ", mejor_corte, " envíos, prob > 0.05)\n\n", sep="")

# Sugerir 2 adicionales alrededor
idx <- which.min(diff)
if (idx > 1) {
  cat("2️⃣ Segunda submission: KA", PARAM$experimento, "_", 
      PARAM$cortes[idx-1], ".csv (explorar menos envíos)\n", sep="")
}
if (idx < length(PARAM$cortes)) {
  cat("3️⃣ Tercera submission: KA", PARAM$experimento, "_", 
      PARAM$cortes[idx+1], ".csv (explorar más envíos)\n", sep="")
}

cat("\n📂 Archivos generados en:\n")
cat("   ", file.path(getwd(), "kaggle"), "\n")
cat("   Total:", length(PARAM$cortes), "archivos CSV\n")

cat("\n========================================\n")

format(Sys.time(), "%a %b %d %X %Y")
