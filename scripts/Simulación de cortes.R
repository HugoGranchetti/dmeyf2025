# ============================================================
# FASE 4: SIMULACIÓN (elegir mejor corte)
# ============================================================

cat("\n########################################\n")
cat("FASE 4: SIMULACIÓN\n")
cat("########################################\n\n")

cat("Mes de simulación:", paste(PARAM$simulacion, collapse = ", "), "\n\n")

# ────────────────────────────────────────────────────────────
# 1. PREPARAR DATOS DE SIMULACIÓN
# ────────────────────────────────────────────────────────────

dataset_sim <- dataset[foto_mes %in% PARAM$simulacion]

cat("Registros en simulación:", nrow(dataset_sim), "\n")

# Distribución de clases
cat("\nDistribución de clases:\n")
tabla_dist <- dataset_sim[, .N, by = clase_ternaria]
tabla_dist[, porcentaje := round(N / sum(N) * 100, 2)]
print(tabla_dist)
cat("\n")

# ────────────────────────────────────────────────────────────
# 2. PREDECIR EN SIMULACIÓN
# ────────────────────────────────────────────────────────────

cat("Generando predicciones...\n")

# Asegurarse de tener las mismas features
campos_buenos_sim <- intersect(campos_buenos, names(dataset_sim))

if (length(campos_buenos_sim) != length(campos_buenos)) {
  cat("⚠️  ADVERTENCIA: Faltan", 
      length(campos_buenos) - length(campos_buenos_sim), 
      "features en simulación\n")
}

# Predecir
predicciones_raw <- predict(
  modelo_final, 
  data.matrix(dataset_sim[, campos_buenos_sim, with = FALSE]),
  type = "response"
)

# Crear data.table con predicciones
predicciones <- data.table(
  numero_de_cliente = dataset_sim$numero_de_cliente,
  foto_mes = dataset_sim$foto_mes,
  clase_ternaria = dataset_sim$clase_ternaria,
  Predicted = predicciones_raw
)

cat("✅ Predicciones generadas:", nrow(predicciones), "\n\n")

# Estadísticas de predicciones
cat("Estadísticas de probabilidades predichas:\n")
print(summary(predicciones$Predicted))
cat("\n")

# ────────────────────────────────────────────────────────────
# 3. SIMULAR GANANCIAS PARA DIFERENTES CORTES
# ────────────────────────────────────────────────────────────

cat("Simulando ganancias para diferentes cortes...\n")
cat("Cortes a evaluar:", min(PARAM$cortes), "a", max(PARAM$cortes), 
    "cada", unique(diff(PARAM$cortes)), "\n\n")

# Ordenar predicciones de mayor a menor
predicciones <- predicciones[order(-Predicted)]

# Tabla de resultados
resultados_sim <- data.table()

for (corte in PARAM$cortes) {
  
  # Top N clientes
  pred_corte <- predicciones[1:corte]
  
  # Contar clases
  bajas <- pred_corte[clase_ternaria %in% c("BAJA+1", "BAJA+2"), .N]
  continua <- pred_corte[clase_ternaria == "CONTINUA", .N]
  
  # Calcular ganancia
  # BAJA+2: +$273,000
  # CONTINUA: -$7,000
  ganancia <- bajas * 273000 - continua * 7000
  
  # Métricas
  precision <- bajas / corte
  ganancia_por_envio <- ganancia / corte
  
  # Guardar
  resultados_sim <- rbind(resultados_sim, data.table(
    envios = corte,
    bajas_capturadas = bajas,
    continua_enviados = continua,
    precision = round(precision, 4),
    ganancia = ganancia,
    ganancia_por_envio = round(ganancia_por_envio, 2)
  ))
}

# ────────────────────────────────────────────────────────────
# 4. MOSTRAR RESULTADOS
# ────────────────────────────────────────────────────────────

cat("=== RESULTADOS DE SIMULACIÓN ===\n\n")
print(resultados_sim)
cat("\n")

# Identificar mejor corte
mejor_corte <- resultados_sim[which.max(ganancia), envios]
mejor_ganancia <- resultados_sim[which.max(ganancia), ganancia]

cat("✅ MEJOR CORTE:", mejor_corte, "envíos\n")
cat("   Ganancia estimada: $", format(mejor_ganancia, big.mark = ","), "\n")
cat("   Bajas capturadas:", resultados_sim[envios == mejor_corte, bajas_capturadas], "\n")
cat("   Precisión:", 
    round(resultados_sim[envios == mejor_corte, precision] * 100, 2), "%\n\n")

# ────────────────────────────────────────────────────────────
# 5. GRÁFICOS
# ────────────────────────────────────────────────────────────

# Gráfico 1: Ganancia vs Envíos
png("simulacion_ganancia.png", width = 800, height = 600)
plot(resultados_sim$envios, resultados_sim$ganancia / 1000, 
     type = "b", col = "blue", lwd = 2, pch = 19,
     xlab = "Número de envíos", 
     ylab = "Ganancia estimada (miles de $)",
     main = "Simulación: Ganancia vs Número de Envíos",
     cex.main = 1.3, cex.lab = 1.1)
grid()
abline(v = mejor_corte, col = "red", lty = 2, lwd = 2)
text(mejor_corte, max(resultados_sim$ganancia / 1000) * 0.9, 
     paste("Mejor:", mejor_corte), pos = 4, col = "red", cex = 1.2)
dev.off()

cat("✅ Gráfico guardado: simulacion_ganancia.png\n")

# Gráfico 2: Precisión vs Envíos
png("simulacion_precision.png", width = 800, height = 600)
plot(resultados_sim$envios, resultados_sim$precision * 100, 
     type = "b", col = "darkgreen", lwd = 2, pch = 19,
     xlab = "Número de envíos", 
     ylab = "Precisión (%)",
     main = "Simulación: Precisión vs Número de Envíos",
     cex.main = 1.3, cex.lab = 1.1)
grid()
abline(h = 2.5, col = "red", lty = 2, lwd = 1.5)  # Break-even: 2.5%
text(max(resultados_sim$envios) * 0.7, 2.8, 
     "Break-even: 2.5%", col = "red", cex = 1)
dev.off()

cat("✅ Gráfico guardado: simulacion_precision.png\n\n")

# ────────────────────────────────────────────────────────────
# 6. DIAGNÓSTICO: ¿Por qué baja la ganancia?
# ────────────────────────────────────────────────────────────

cat("=== DIAGNÓSTICO ===\n\n")

# Verificar si ganancia es decreciente desde el inicio
ganancia_inicial <- resultados_sim[1, ganancia]
ganancia_final <- resultados_sim[.N, ganancia]

if (ganancia_final < ganancia_inicial * 0.7) {
  cat("⚠️  ALERTA: Ganancia CAE más del 30% desde inicio\n")
  cat("   Ganancia en", resultados_sim[1, envios], "envíos:", 
      format(ganancia_inicial, big.mark = ","), "\n")
  cat("   Ganancia en", resultados_sim[.N, envios], "envíos:", 
      format(ganancia_final, big.mark = ","), "\n\n")
  
  cat("Posibles causas:\n")
  cat("  1. OVERFITTING: Modelo memoriza training, no generaliza\n")
  cat("  2. DRIFT TEMPORAL: 202104 muy diferente a 202102-202103\n")
  cat("  3. FEATURES CON LEAKAGE: Rankings calculados incorrectamente\n\n")
  
  cat("Recomendaciones:\n")
  cat("  - Aumentar regularización (lambda_l1, lambda_l2)\n")
  cat("  - Reducir complejidad (menos num_leaves)\n")
  cat("  - Usar validación temporal en BO\n")
  cat("  - Verificar cálculo de rankings\n\n")
}

# Verificar precisión
precision_inicial <- resultados_sim[1, precision]
precision_final <- resultados_sim[.N, precision]

if (precision_inicial > 0.10 && precision_final < 0.03) {
  cat("⚠️  ALERTA: Precisión cae drásticamente\n")
  cat("   De", round(precision_inicial * 100, 1), "% a", 
      round(precision_final * 100, 1), "%\n")
  cat("   Esto confirma OVERFITTING\n\n")
}

# Punto donde ganancia por envío se vuelve negativa
punto_negativo <- resultados_sim[ganancia_por_envio < 0]
if (nrow(punto_negativo) > 0) {
  cat("⚠️  ALERTA: Ganancia POR ENVÍO negativa desde", 
      punto_negativo[1, envios], "envíos\n")
  cat("   NO enviar más de", punto_negativo[1, envios] - 500, "clientes\n\n")
}

# ────────────────────────────────────────────────────────────
# 7. GUARDAR RESULTADOS
# ────────────────────────────────────────────────────────────

# Guardar tabla de resultados
fwrite(resultados_sim, "simulacion_resultados.csv")
cat("✅ Resultados guardados: simulacion_resultados.csv\n")

# Guardar mejor corte en PARAM
PARAM$mejor_corte_simulacion <- mejor_corte
saveRDS(PARAM, "PARAM_con_mejor_corte.rds")
cat("✅ Mejor corte guardado en PARAM\n\n")

cat("========================================\n")
cat("FIN DE SIMULACIÓN\n")
cat("========================================\n\n")
