# ============================================================
# EXPERIMENTOS sin BO
# Reutiliza hiperparámetros de experimento anterior
# ============================================================

require("data.table")
require("lightgbm")
require("yaml")

# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_BASE <- "C:/Users/hgran/Desktop/DMEyF RStudio"

# ============================================================
# PARÁMETROS DEL EXPERIMENTO
# ============================================================

PARAM <- list()
PARAM$experimento <- 4944
PARAM$semilla_primigenia <- 907871

# Training y future
PARAM$train_final <- c(202101, 202102, 202103, 202104)
PARAM$future <- c(202106)
PARAM$semilla_kaggle <- 314159
PARAM$cortes <- seq(1000, 19000, by= 500)

# SIMULACIÓN
PARAM$simulacion <- c(202104)

# Undersampling
PARAM$trainingstrategy$undersampling <- 1.0

# Parámetros fijos LightGBM
PARAM$lgbm$param_fijos <- list(
  boosting= "gbdt",
  objective= "binary",
  metric= "auc",
  first_metric_only= FALSE,
  boost_from_average= TRUE,
  feature_pre_filter= FALSE,
  force_row_wise= TRUE,
  verbosity= -100,
  seed= PARAM$semilla_primigenia,
  max_depth= -1L,
  min_gain_to_split= 0,
  min_sum_hessian_in_leaf= 0.001,
  lambda_l1= 0.0,
  lambda_l2= 0.0,
  max_bin= 31L,
  bagging_fraction= 1.0,
  pos_bagging_fraction= 1.0,
  neg_bagging_fraction= 1.0,
  is_unbalance= FALSE,
  scale_pos_weight= 1.0,
  drop_rate= 0.1,
  max_drop= 50,
  skip_drop= 0.5,
  extra_trees= FALSE
)

# ============================================================
# CARGAR HIPERPARÁMETROS DEL EXPERIMENTO 4943
# ============================================================

cat("\n=== CARGANDO HIPERPARÁMETROS DEL EXPERIMENTO 4943 ===\n")

EXPERIMENTO_ANTERIOR <- 4943
ruta_param_anterior <- file.path(RUTA_BASE, "exp", 
                                 paste0("HT", EXPERIMENTO_ANTERIOR), 
                                 "PARAM.yml")

if (file.exists(ruta_param_anterior)) {
  PARAM_anterior <- read_yaml(ruta_param_anterior)
  PARAM$out$lgbm$mejores_hiperparametros <- PARAM_anterior$out$lgbm$mejores_hiperparametros
  PARAM$out$lgbm$y <- PARAM_anterior$out$lgbm$y
  
  cat("✅ Hiperparámetros cargados desde:", ruta_param_anterior, "\n")
  cat("AUC del experimento anterior:", PARAM$out$lgbm$y, "\n\n")
  cat("Hiperparámetros a usar:\n")
  print(PARAM$out$lgbm$mejores_hiperparametros)
  cat("\n")
  
} else {
  # Si no existe el archivo, usar valores por defecto
  cat("⚠️ Usando hiperparámetros manualmente\n\n")
  
  PARAM$out$lgbm$mejores_hiperparametros <- list(
    num_iterations = 1450L,
    learning_rate = 0.035,
    feature_fraction = 0.68,
    num_leaves = 185L,
    min_data_in_leaf = 1250L
  )
  PARAM$out$lgbm$y <- 0.9302545
  
  cat("Hiperparámetros:\n")
  print(PARAM$out$lgbm$mejores_hiperparametros)
  cat("\n")
}

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

particionar <- function(data, division, agrupa= "", campo= "fold", start= 1, seed= NA) {
  if (!is.na(seed)) set.seed(seed, "L'Ecuyer-CMRG")
  bloque <- unlist(mapply(
    function(x, y) {rep(y, x)},
    division, 
    seq(from= start, length.out= length(division))
  ))
  data[, (campo) := sample(rep(bloque,ceiling(.N / length(bloque))))[1:.N], by= agrupa]
}

realidad_inicializar <- function(pfuture, pparam) {
  drealidad <- pfuture[, list(numero_de_cliente, foto_mes, clase_ternaria)]
  particionar(drealidad, division= c(3, 7), agrupa= "clase_ternaria", 
              seed= pparam$semilla_kaggle)
  return(drealidad)
}

realidad_evaluar <- function(prealidad, pprediccion) {
  prealidad[pprediccion, on= c("numero_de_cliente", "foto_mes"), predicted:= i.Predicted]
  tbl <- prealidad[, list("qty"=.N), list(fold, predicted, clase_ternaria)]
  res <- list()
  res$public <- tbl[fold==1 & predicted==1L, 
                    sum(qty*ifelse(clase_ternaria=="BAJA+2", 780000, -20000))]/0.3
  res$private <- tbl[fold==2 & predicted==1L, 
                     sum(qty*ifelse(clase_ternaria=="BAJA+2", 780000, -20000))]/0.7
  res$total <- tbl[predicted==1L, 
                   sum(qty*ifelse(clase_ternaria=="BAJA+2", 780000, -20000))]
  prealidad[, predicted:=NULL]
  return(res)
}

# ============================================================
# LECTURA DEL DATASET
# ============================================================

cat("=== LEYENDO DATASET ===\n")
ruta_dataset <- file.path(RUTA_BASE, "datasets", "competencia_01.csv.gz")

if (!file.exists(ruta_dataset)) {
  stop("ERROR: No se encuentra el archivo ", ruta_dataset)
}

dataset <- fread(ruta_dataset, stringsAsFactors= TRUE)
cat("Dataset cargado:", nrow(dataset), "registros,", ncol(dataset), "columnas\n\n")

# ============================================================
# FEATURE ENGINEERING
# ============================================================

cat("========================================\n")
cat("FEATURE ENGINEERING\n")
cat("========================================\n\n")

# Ordenar por cliente y fecha
cat("[1/3] Ordenando dataset...\n")
setorder(dataset, numero_de_cliente, foto_mes)

# Identificar columnas numéricas
columnas_excluir <- c("numero_de_cliente", "foto_mes", "clase_ternaria", "clase01")
columnas_numericas <- setdiff(
  names(dataset)[sapply(dataset, is.numeric)], 
  columnas_excluir
)

cat("Variables numéricas:", length(columnas_numericas), "\n\n")

# ──────────────────────────────────────────────────────────
# LAGS GENERALES
# ──────────────────────────────────────────────────────────

cat("[2/3] Generando LAGS generales...\n")
tiempo_inicio <- Sys.time()

# LAG-1
dataset[, paste0(columnas_numericas, "_lag1") := 
          lapply(.SD, function(x) shift(x, n=1, type="lag")), 
        by=numero_de_cliente, 
        .SDcols=columnas_numericas]

# LAG-2
dataset[, paste0(columnas_numericas, "_lag2") := 
          lapply(.SD, function(x) shift(x, n=2, type="lag")), 
        by=numero_de_cliente, 
        .SDcols=columnas_numericas]

# DELTAs
for (col in columnas_numericas) {
  col_lag1 <- paste0(col, "_lag1")
  nueva_col <- paste0(col, "_delta1")
  dataset[, (nueva_col) := get(col) - get(col_lag1)]
}

tiempo_lags <- difftime(Sys.time(), tiempo_inicio, units="mins")
cat("   Tiempo:", round(tiempo_lags, 1), "minutos\n")
cat("   Features generadas:", length(columnas_numericas) * 3, "\n\n")

# ──────────────────────────────────────────────────────────
# FEATURES ESPECÍFICAS DE LÍMITES
# ──────────────────────────────────────────────────────────

cat("[3/3] Generando features ESPECÍFICAS de Límites...\n")

# 1. Cambios porcentuales (MUY IMPORTANTES)
dataset[, Master_mlimitecompra_pct1 := 
          (Master_mlimitecompra - Master_mlimitecompra_lag1) / 
          (Master_mlimitecompra_lag1 + 1)]

dataset[, Visa_mlimitecompra_pct1 := 
          (Visa_mlimitecompra - Visa_mlimitecompra_lag1) / 
          (Visa_mlimitecompra_lag1 + 1)]

# 2. Límite total (suma de ambas tarjetas)
dataset[, limite_total := Master_mlimitecompra + Visa_mlimitecompra]
dataset[, limite_total_lag1 := shift(limite_total, 1), by=numero_de_cliente]
dataset[, limite_total_delta1 := limite_total - limite_total_lag1]

# 3. Flags de eventos críticos
dataset[, Master_reduccion_fuerte := ifelse(Master_mlimitecompra_pct1 < -0.20, 1L, 0L)]
dataset[, Visa_reduccion_fuerte := ifelse(Visa_mlimitecompra_pct1 < -0.20, 1L, 0L)]

dataset[, Master_cancelada := ifelse(
  Master_mlimitecompra == 0 & Master_mlimitecompra_lag1 > 0, 1L, 0L
)]

dataset[, Visa_cancelada := ifelse(
  Visa_mlimitecompra == 0 & Visa_mlimitecompra_lag1 > 0, 1L, 0L
)]

cat("   Features de límites agregadas: 9\n\n")

# ============================================================
# ROBUSTEZ AL AGUINALDO
# ============================================================

cat("\n=== AJUSTANDO POR AGUINALDO DE JUNIO ===\n")

# Mes
dataset[, mes := foto_mes %% 100]
dataset[, es_junio := ifelse(mes == 6, 1L, 0L)]

# ────────────────────────────────────────────────────────────
# CALCULAR AGUINALDO ESTIMADO
# ────────────────────────────────────────────────────────────

# Mejor payroll últimos 6 meses (para aguinaldo)
dataset[, mpayroll_max_6m := frollapply(mpayroll, 6, max, 
                                        align="right", fill=NA, na.rm=TRUE), 
        by=numero_de_cliente]

dataset[, aguinaldo_estimado := mpayroll_max_6m * 0.5]

# ────────────────────────────────────────────────────────────
# VARIABLES AJUSTADAS (sin aguinaldo)
# ────────────────────────────────────────────────────────────

# Payroll sin aguinaldo
dataset[, mpayroll_base := ifelse(
  es_junio == 1L & !is.na(aguinaldo_estimado),
  pmax(mpayroll - aguinaldo_estimado, 0),
  mpayroll
)]

# Caja ahorro sin aguinaldo (asumir 80% del aguinaldo va a caja)
dataset[, mcaja_ahorro_base := ifelse(
  es_junio == 1L & !is.na(aguinaldo_estimado),
  pmax(mcaja_ahorro - (aguinaldo_estimado * 0.8), 0),
  mcaja_ahorro
)]

# Cuentas saldo sin aguinaldo
dataset[, mcuentas_saldo_base := ifelse(
  es_junio == 1L & !is.na(aguinaldo_estimado),
  pmax(mcuentas_saldo - aguinaldo_estimado, 0),
  mcuentas_saldo
)]

# ────────────────────────────────────────────────────────────
# RATIOS (robustos a aguinaldo)
# ────────────────────────────────────────────────────────────

vars_sensibles <- c("mcaja_ahorro", "mcuentas_saldo", "mpayroll", "mactivos_margen")

for (var in vars_sensibles) {
  var_lag1 <- paste0(var, "_lag1")
  
  if (var_lag1 %in% names(dataset)) {
    # Ratio vs mes anterior
    var_ratio <- paste0(var, "_ratio")
    dataset[, (var_ratio) := get(var) / (get(var_lag1) + 1)]
    
    # Cambio porcentual
    var_pct <- paste0(var, "_pct")
    dataset[, (var_pct) := (get(var) - get(var_lag1)) / (get(var_lag1) + 1)]
  }
}

# ────────────────────────────────────────────────────────────
# USAR MAYO COMO REFERENCIA EN JUNIO
# ────────────────────────────────────────────────────────────

# Variables "normales" = lag1 si es junio, actual si no
for (var in c("mcaja_ahorro", "mcuentas_saldo", "mpayroll")) {
  var_lag1 <- paste0(var, "_lag1")
  var_normal <- paste0(var, "_normal")
  
  if (var_lag1 %in% names(dataset)) {
    dataset[, (var_normal) := ifelse(
      es_junio == 1L,
      get(var_lag1),
      get(var)
    )]
  }
}

# ────────────────────────────────────────────────────────────
# FLAGS Y FEATURES ADICIONALES
# ────────────────────────────────────────────────────────────

# Spike detection
dataset[, tiene_spike_saldo := ifelse(
  mcaja_ahorro_ratio > 1.3 | mcuentas_saldo_ratio > 1.3, 1L, 0L
)]

dataset[, spike_en_junio := ifelse(
  es_junio == 1L & tiene_spike_saldo == 1L, 1L, 0L
)]

# Mes como categórica
dataset[, mes_cat := as.integer(mes)]

# ────────────────────────────────────────────────────────────
# RESUMEN
# ────────────────────────────────────────────────────────────

features_aguinaldo <- sum(grepl("_base$|_ratio$|_pct$|_normal$|aguinaldo|spike|mes_cat|es_junio", 
                                names(dataset)))

cat("Features para aguinaldo generadas:", features_aguinaldo, "\n")
cat("  - Variables base (sin aguinaldo): mpayroll_base, mcaja_ahorro_base, etc.\n")
cat("  - Ratios y porcentuales: *_ratio, *_pct\n")
cat("  - Baseline de mayo: *_normal\n")
cat("  - Flags: es_junio, spike_en_junio\n")
cat("  - Estimación: aguinaldo_estimado\n\n")



# Limpiar memoria
gc()

# ============================================================
# TRAINING FINAL
# ============================================================

cat("=== TRAINING FINAL ===\n")
setwd(file.path(RUTA_BASE, "exp"))
experimento <- paste0("exp", PARAM$experimento)
dir.create(experimento, showWarnings= FALSE)
setwd(file.path(RUTA_BASE, "exp", experimento))

cat("Carpeta de trabajo:", getwd(), "\n\n")

# Preparar datos finales
dataset[, clase01 := ifelse(clase_ternaria %in% c("BAJA+1", "BAJA+2"), 1L, 0L)]
dataset_train <- dataset[foto_mes %in% PARAM$train_final]

# Verificar distribución
cat("Distribución de clases en training:\n")
print(dataset_train[, .N, by=clase_ternaria])
cat("\n")

# Campos a utilizar
campos_buenos <- setdiff(
  colnames(dataset_train), 
  c("clase_ternaria", "clase01", "azar", "training")
)

cat("Features a utilizar:", length(campos_buenos), "\n")
cat("Registros training:", nrow(dataset_train), "\n\n")

dtrain_final <- lgb.Dataset(
  data= data.matrix(dataset_train[, campos_buenos, with= FALSE]),
  label= dataset_train[, clase01]
)

# Combinar parámetros
param_final <- modifyList(PARAM$lgbm$param_fijos, 
                          PARAM$out$lgbm$mejores_hiperparametros)

# Ajustar por undersampling
param_normalizado <- copy(param_final)
param_normalizado$min_data_in_leaf <- round(
  param_final$min_data_in_leaf / PARAM$trainingstrategy$undersampling
)

cat("Parámetros finales:\n")
cat("  num_iterations:", param_normalizado$num_iterations, "\n")
cat("  learning_rate:", param_normalizado$learning_rate, "\n")
cat("  feature_fraction:", param_normalizado$feature_fraction, "\n")
cat("  num_leaves:", param_normalizado$num_leaves, "\n")
cat("  min_data_in_leaf:", param_normalizado$min_data_in_leaf, 
    "(ajustado por undersampling)\n\n")

# Entrenar modelo final
cat("Entrenando modelo final...\n")
tiempo_inicio <- Sys.time()

modelo_final <- lgb.train(data= dtrain_final, param= param_normalizado)

tiempo_training <- difftime(Sys.time(), tiempo_inicio, units="mins")
cat("Modelo entrenado en:", round(tiempo_training, 1), "minutos\n\n")

# Importancia de variables
tb_importancia <- as.data.table(lgb.importance(modelo_final))
fwrite(tb_importancia, file= "impo.txt", sep= "\t")
lgb.save(modelo_final, "modelo.txt")

cat("Top 20 variables más importantes:\n")
print(tb_importancia[1:20])
cat("\n")

# Verificar si las nuevas features aparecen
nuevas_features_importantes <- tb_importancia[
  Feature %in% c("Master_mlimitecompra_pct1", "Visa_mlimitecompra_pct1",
                 "limite_total", "limite_total_delta1",
                 "Master_reduccion_fuerte", "Visa_reduccion_fuerte",
                 "Master_cancelada", "Visa_cancelada")
]

if (nrow(nuevas_features_importantes) > 0) {
  cat("🎯 Features nuevas en el top:\n")
  print(nuevas_features_importantes)
  cat("\n")
}

# ============================================================
# SIMULACIÓN EN 202104
# ============================================================

cat("########################################\n")
cat("SIMULACIÓN (202104)\n")
cat("########################################\n\n")

dsimulacion <- dataset[foto_mes %in% PARAM$simulacion]
cat("Registros:", nrow(dsimulacion), "\n\n")

pred_simulacion <- predict(modelo_final, 
                           data.matrix(dsimulacion[, campos_buenos, with= FALSE]))

tb_sim <- dsimulacion[, list(numero_de_cliente, foto_mes, clase_ternaria)]
tb_sim[, prob := pred_simulacion]

drealidad_sim <- realidad_inicializar(dsimulacion, PARAM)

resumen_sim <- data.table(
  envios = integer(),
  total = numeric(),
  public = numeric(),
  private = numeric()
)

setorder(tb_sim, -prob)

for (envios in PARAM$cortes) {
  tb_sim[, Predicted := 0L]
  tb_sim[1:envios, Predicted := 1L]
  
  res <- realidad_evaluar(drealidad_sim, tb_sim)
  
  resumen_sim <- rbind(resumen_sim, data.table(
    envios = envios,
    total = res$total,
    public = res$public,
    private = res$private
  ))
  
  cat("Envios=", sprintf("%5d", envios), "  ",
      "TOTAL=", sprintf("%10.0f", res$total), "  ",
      "Public=", sprintf("%10.0f", res$public), "  ",
      "Private=", sprintf("%10.0f", res$private), "\n", sep="")
}

fwrite(resumen_sim, file="resumen_simulacion_202104.csv")

setorder(resumen_sim, -total)
mejor_sim <- resumen_sim[1]

cat("\n========================================\n")
cat("MEJOR ENVÍO (simulación)\n")
cat("========================================\n")
cat("Envíos:", mejor_sim$envios, "\n")
cat("Total: ", format(mejor_sim$total, big.mark=","), "\n")
cat("Public:", format(mejor_sim$public, big.mark=","), "\n")
cat("Private:", format(mejor_sim$private, big.mark=","), "\n")
cat("========================================\n\n")

rm(dsimulacion, pred_simulacion, tb_sim, drealidad_sim)
gc()

# ============================================================
# PREDICCIÓN
# ============================================================

cat("=== GENERANDO PREDICCIONES ===\n")
dfuture <- dataset[foto_mes %in% PARAM$future]
cat("Registros en future:", nrow(dfuture), "\n")

prediccion <- predict(modelo_final, 
                      data.matrix(dfuture[, campos_buenos, with= FALSE]))

tb_prediccion <- dfuture[, list(numero_de_cliente, foto_mes)]
tb_prediccion[, prob := prediccion]

fwrite(tb_prediccion, file= "prediccion.txt", sep= "\t")

cat("\nDistribución de probabilidades:\n")
print(summary(tb_prediccion$prob))

cat("\nTop 10 clientes:\n")
print(tb_prediccion[order(-prob)][1:10])
cat("\n")

# ============================================================
# GENERAR ARCHIVOS KAGGLE
# ============================================================

cat("=== GENERANDO ARCHIVOS PARA KAGGLE ===\n\n")
setorder(tb_prediccion, -prob)
dir.create("kaggle", showWarnings = FALSE)

for (envios in PARAM$cortes) {
  tb_prediccion[, Predicted := 0L]
  tb_prediccion[1:envios, Predicted := 1L]
  
  archivo_kaggle <- paste0("./kaggle/KA", PARAM$experimento, "_", envios, ".csv")
  fwrite(tb_prediccion[, list(numero_de_cliente, Predicted)], 
         file= archivo_kaggle, sep= ",")
  
  cat("✅ Generado: KA", PARAM$experimento, "_", envios, ".csv\n", sep="")
}

# ============================================================
# RECOMENDACIONES
# ============================================================

cat("\n========================================\n")
cat("RECOMENDACIONES\n")
cat("========================================\n\n")

# Sugerencia basada en probabilidades
target_conservador <- tb_prediccion[prob > 0.10, .N]
target_balanceado <- tb_prediccion[prob > 0.05, .N]
target_agresivo <- tb_prediccion[prob > 0.025, .N]

cat("Basado en umbrales de probabilidad:\n")
cat("  Conservador (prob > 0.10):", target_conservador, "envíos\n")
cat("  Balanceado (prob > 0.05):", target_balanceado, "envíos\n")
cat("  Agresivo (prob > 0.025):", target_agresivo, "envíos\n\n")

# Buscar cortes más cercanos
diff_bal <- abs(PARAM$cortes - target_balanceado)
mejor_corte <- PARAM$cortes[which.min(diff_bal)]

cat("📤 SUGERENCIA DE SUBMISSION:\n")
cat("   1️⃣ Primera: KA", PARAM$experimento, "_", mejor_corte, ".csv\n", sep="")

idx <- which.min(diff_bal)
if (idx > 1) {
  cat("   2️⃣ Segunda: KA", PARAM$experimento, "_", 
      PARAM$cortes[idx-1], ".csv\n", sep="")
}
if (idx < length(PARAM$cortes)) {
  cat("   3️⃣ Tercera: KA", PARAM$experimento, "_", 
      PARAM$cortes[idx+1], ".csv\n", sep="")
}

cat("\n📂 Archivos en:", file.path(getwd(), "kaggle"), "\n")

write_yaml(PARAM, file="PARAM.yml")

cat("\n========================================\n")
cat("EXPERIMENTO", PARAM$experimento, "COMPLETADO\n")
cat("========================================\n")
cat("Comparar con experimento 4941 en Kaggle\n")
cat("Si mejora: agregar más features de límites\n")
cat("Si no mejora: probar otras variables\n")
cat("========================================\n\n")

format(Sys.time(), "%a %b %d %X %Y")