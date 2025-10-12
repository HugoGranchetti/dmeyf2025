# ============================================================
# INICIO
# ============================================================

format(Sys.time(), "%a %b %d %X %Y")
rm(list=ls(all.names=TRUE))
gc(full=TRUE, verbose=FALSE)

RUTA_BASE <- "C:/Users/hgran/Desktop/DMEyF RStudio"

# ============================================================
# CARGA DE LIBRERÍAS
# ============================================================
cat("Cargando librerías...\n")
require("data.table")
require("parallel")
require("R.utils")
require("primes")
require("utils")
require("rlist")
require("yaml")
require("lightgbm")
require("ParBayesianOptimization")

# ============================================================
# DEFINICIÓN DE PARÁMETROS
# ============================================================
PARAM <- list()
PARAM$experimento <- 4946
PARAM$semilla_primigenia <- 907871

# BAYESIAN OPTIMIZATION
PARAM$train <- c(202102, 202103)  # 2 meses para BO

# EVALUACIÓN HOLDOUT
PARAM$test_holdout <- c(202104)

# TRAINING FINAL
PARAM$train_final <- c(202101, 202102, 202103, 202104)

# PREDICCIÓN
PARAM$future <- c(202106)

# SIMULACIÓN
PARAM$simulacion <- c(202104)

PARAM$semilla_kaggle <- 314159
PARAM$cortes <- seq(2000, 19000, by= 500)

# Undersampling
PARAM$trainingstrategy$undersampling <- 0.3

# Hyperparameter tuning
PARAM$hyperparametertuning$xval_folds <- 5

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
  pos_bagging_fraction= 1.0,
  neg_bagging_fraction= 1.0,
  is_unbalance= FALSE,
  scale_pos_weight= 1.0,
  drop_rate= 0.1,
  max_drop= 50,
  skip_drop= 0.5,
  extra_trees= FALSE
)

# Espacio de búsqueda (ParBayesianOptimization usa listas con bounds)
PARAM$hyperparametertuning$bounds <- list(
  num_iterations = c(500L, 3000L),
  learning_rate = c(0.005, 0.1),
  feature_fraction = c(0.1, 1.0),
  bagging_fraction = c(0.5, 1.0),
  num_leaves = c(10L, 200L),
  min_data_in_leaf = c(500L, 4000L)
)

PARAM$hyperparametertuning$iteraciones <- 70

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

EstimarGanancia_AUC_lightgbm <- function(num_iterations, learning_rate, feature_fraction, 
                                         bagging_fraction, num_leaves, min_data_in_leaf) {
  
  # Convertir a enteros donde corresponde
  num_iterations <- as.integer(round(num_iterations))
  num_leaves <- as.integer(round(num_leaves))
  min_data_in_leaf <- as.integer(round(min_data_in_leaf))
  
  # Construir parámetros
  param_completo <- modifyList(PARAM$lgbm$param_fijos, list(
    num_iterations = num_iterations,
    learning_rate = learning_rate,
    feature_fraction = feature_fraction,
    bagging_fraction = bagging_fraction,
    num_leaves = num_leaves,
    min_data_in_leaf = min_data_in_leaf
  ))
  
  # CV
  modelocv <- lgb.cv(
    data= dtrain,
    nfold= PARAM$hyperparametertuning$xval_folds,
    stratified= TRUE,
    param= param_completo,
    early_stopping_rounds = 100,
    verbose = -1
  )
  
  AUC <- modelocv$best_score
  rm(modelocv)
  gc(full= TRUE, verbose= FALSE)
  
  message(format(Sys.time(), "%a %b %d %X %Y"), " AUC ", round(AUC, 5))
  
  # ParBayesianOptimization requiere retornar lista con Score
  return(list(Score = AUC))
}

# ============================================================
# LECTURA DEL DATASET
# ============================================================

cat("\n=== LEYENDO DATASET ===\n")
ruta_dataset <- file.path(RUTA_BASE, "datasets", "competencia_01.csv.gz")

if (!file.exists(ruta_dataset)) {
  stop("ERROR: No se encuentra el archivo ", ruta_dataset)
}

dataset <- fread(ruta_dataset, stringsAsFactors= TRUE)
cat("Dataset cargado:", nrow(dataset), "registros,", ncol(dataset), "columnas\n")

# ============================================================
# FEATURE ENGINEERING
# ============================================================

# ────────────────────────────────────────────────────────────
# RANKINGS
# ────────────────────────────────────────────────────────────

cat("\n=== GENERANDO RANKINGS ===\n")

# Solo variables económicas (las que empiezan con 'm')
vars_economicas <- grep("^m", names(dataset), value = TRUE)
vars_economicas <- setdiff(vars_economicas, c("mes", "mes_cat"))

cat("Variables económicas a rankear:", length(vars_economicas), "\n")

# Ranking por mes, preservando solo los ceros
for (v in vars_economicas) {
  rank_col <- paste0(v, "_rank")
  
  # Rankear positivos Y negativos, preservando solo ceros
  dataset[, (rank_col) := ifelse(
    get(v) == 0 | is.na(get(v)),
    0,
    frank(get(v), na.last = "keep", ties.method = "random")
  ), by = foto_mes]
}

cat("✅ Rankings creados:", length(vars_economicas), "features\n")
cat("   (preservando solo 0s como 0, rankeando negativos)\n")

# ────────────────────────────────────────────────────────────
# LAGS Y DELTAS
# ────────────────────────────────────────────────────────────

cat("\n=== FEATURE ENGINEERING ===\n")
setorder(dataset, numero_de_cliente, foto_mes)

columnas_excluir <- c("numero_de_cliente", "foto_mes", "clase_ternaria", "clase01")
columnas_numericas <- setdiff(
  names(dataset)[sapply(dataset, is.numeric)],
  columnas_excluir
)

cat("Variables numéricas:", length(columnas_numericas), "\n")

# LAGS GENERALES
cat("Generando LAG-1...\n")
for (col in columnas_numericas) {
  nueva_col <- paste0(col, "_lag1")
  dataset[, (nueva_col) := shift(.SD, n=1, type="lag"),
          by=numero_de_cliente, .SDcols=col]
}

cat("Generando LAG-2...\n")
for (col in columnas_numericas) {
  nueva_col <- paste0(col, "_lag2")
  dataset[, (nueva_col) := shift(.SD, n=2, type="lag"),
          by=numero_de_cliente, .SDcols=col]
}

cat("Generando DELTAS...\n")
for (col in columnas_numericas) {
  col_lag1 <- paste0(col, "_lag1")
  nueva_col <- paste0(col, "_delta1")
  dataset[, (nueva_col) := get(col) - get(col_lag1)]
}

# FEATURES ESPECÍFICAS DE LÍMITES
cat("Generando features específicas de límites...\n")

dataset[, Master_mlimitecompra_pct1 := 
          (Master_mlimitecompra - Master_mlimitecompra_lag1) / 
          (Master_mlimitecompra_lag1 + 1)]

dataset[, Visa_mlimitecompra_pct1 := 
          (Visa_mlimitecompra - Visa_mlimitecompra_lag1) / 
          (Visa_mlimitecompra_lag1 + 1)]

dataset[, limite_total := Master_mlimitecompra + Visa_mlimitecompra]
dataset[, limite_total_lag1 := shift(limite_total, 1), by=numero_de_cliente]
dataset[, limite_total_delta1 := limite_total - limite_total_lag1]

dataset[, Master_reduccion_fuerte := ifelse(Master_mlimitecompra_pct1 < -0.20, 1L, 0L)]
dataset[, Visa_reduccion_fuerte := ifelse(Visa_mlimitecompra_pct1 < -0.20, 1L, 0L)]

dataset[, Master_cancelada := ifelse(
  Master_mlimitecompra == 0 & Master_mlimitecompra_lag1 > 0, 1L, 0L)]
dataset[, Visa_cancelada := ifelse(
  Visa_mlimitecompra == 0 & Visa_mlimitecompra_lag1 > 0, 1L, 0L)]

cat("Total features:", ncol(dataset), "\n\n")

# ============================================================
# ROBUSTEZ AL AGUINALDO
# ============================================================

cat("\n=== AJUSTANDO POR AGUINALDO DE JUNIO ===\n")

# Mes
dataset[, mes := foto_mes %% 100]
dataset[, es_junio := ifelse(mes == 6, 1L, 0L)]

# CALCULAR AGUINALDO ESTIMADO
dataset[, mpayroll_max_6m := frollapply(mpayroll, 6, max, 
                                        align="right", fill=NA, na.rm=TRUE), 
        by=numero_de_cliente]

dataset[, aguinaldo_estimado := mpayroll_max_6m * 0.5]

# VARIABLES AJUSTADAS (sin aguinaldo)
dataset[, mpayroll_base := ifelse(
  es_junio == 1L & !is.na(aguinaldo_estimado),
  pmax(mpayroll - aguinaldo_estimado, 0),
  mpayroll
)]

dataset[, mcaja_ahorro_base := ifelse(
  es_junio == 1L & !is.na(aguinaldo_estimado),
  pmax(mcaja_ahorro - (aguinaldo_estimado * 0.8), 0),
  mcaja_ahorro
)]

dataset[, mcuentas_saldo_base := ifelse(
  es_junio == 1L & !is.na(aguinaldo_estimado),
  pmax(mcuentas_saldo - aguinaldo_estimado, 0),
  mcuentas_saldo
)]

# RATIOS (robustos a aguinaldo)
vars_sensibles <- c("mcaja_ahorro", "mcuentas_saldo", "mpayroll", "mactivos_margen")

for (var in vars_sensibles) {
  var_lag1 <- paste0(var, "_lag1")
  
  if (var_lag1 %in% names(dataset)) {
    var_ratio <- paste0(var, "_ratio")
    dataset[, (var_ratio) := get(var) / (get(var_lag1) + 1)]
    
    var_pct <- paste0(var, "_pct")
    dataset[, (var_pct) := (get(var) - get(var_lag1)) / (get(var_lag1) + 1)]
  }
}

# FLAGS Y FEATURES ADICIONALES
dataset[, tiene_spike_saldo := ifelse(
  mcaja_ahorro_ratio > 1.3 | mcuentas_saldo_ratio > 1.3, 1L, 0L
)]

dataset[, spike_en_junio := ifelse(
  es_junio == 1L & tiene_spike_saldo == 1L, 1L, 0L
)]

dataset[, mes_cat := as.integer(mes)]

features_aguinaldo <- sum(grepl("_base$|_ratio$|_pct$|_normal$|aguinaldo|spike|mes_cat|es_junio", 
                                names(dataset)))

cat("Features para aguinaldo generadas:", features_aguinaldo, "\n\n")

gc()

# ============================================================
# FASE 1: BAYESIAN OPTIMIZATION
# ============================================================

cat("########################################\n")
cat("FASE 1: BAYESIAN OPTIMIZATION\n")
cat("########################################\n\n")

cat("Meses para BO:", paste(PARAM$train, collapse=", "), "\n")
cat("CV interno: 5-fold\n\n")

setwd(file.path(RUTA_BASE, "exp"))
experimento_folder <- paste0("HT", PARAM$experimento)
dir.create(experimento_folder, showWarnings=FALSE)
setwd(file.path(RUTA_BASE, "exp", experimento_folder))

# Crear clase01
dataset[, clase01 := ifelse(clase_ternaria %in% c("BAJA+1", "BAJA+2"), 1L, 0L)]

# Preparar datos para BO
dataset_train <- dataset[foto_mes %in% PARAM$train]

cat("Registros para BO:", nrow(dataset_train), "\n\n")

# Undersampling
set.seed(PARAM$semilla_primigenia, kind = "L'Ecuyer-CMRG")
dataset_train[, azar := runif(nrow(dataset_train))]
dataset_train[, training := 0L]
dataset_train[foto_mes %in% PARAM$train & 
                (azar <= PARAM$trainingstrategy$undersampling | 
                   clase_ternaria %in% c("BAJA+1", "BAJA+2")), 
              training := 1L]

cat("Registros con undersampling:", dataset_train[training==1L, .N], "\n\n")

# Campos buenos
campos_buenos <- setdiff(
  colnames(dataset_train), 
  c("clase_ternaria", "clase01", "azar", "training")
)

cat("Features:", length(campos_buenos), "\n\n")

# Crear dataset LightGBM (GLOBAL para que lo use la función de optimización)
dtrain <<- lgb.Dataset(
  data= data.matrix(dataset_train[training == 1L, campos_buenos, with= FALSE]),
  label= dataset_train[training == 1L, clase01],
  free_raw_data= FALSE
)

# Ejecutar BO con ParBayesianOptimization
cat("Iniciando Bayesian Optimization con ParBayesianOptimization...\n")
cat("Iteraciones:", PARAM$hyperparametertuning$iteraciones, "\n")
cat("Núcleos disponibles:", detectCores(), "\n")
cat("ADVERTENCIA: Esto toma 2-3 horas (más rápido que mlrMBO)\n\n")

set.seed(PARAM$semilla_primigenia, kind = "L'Ecuyer-CMRG")

bayesiana_salida <- bayesOpt(
  FUN = EstimarGanancia_AUC_lightgbm,
  bounds = PARAM$hyperparametertuning$bounds,
  initPoints = 10,  # Puntos aleatorios iniciales
  iters.n = PARAM$hyperparametertuning$iteraciones,
  iters.k = 1,
  otherHalting = list(timeLimit = Inf),
  acq = "ucb",  # Upper Confidence Bound
  kappa = 2.576,
  eps = 0.0,
  verbose = 1
)

# Extraer mejores hiperparámetros
tb_bayesiana <- as.data.table(bayesiana_salida$scoreSummary)
setorder(tb_bayesiana, -Score)
fwrite(tb_bayesiana, file= "BO_log.txt", sep= "\t")

# Encontrar el mejor resultado
mejor_idx <- which.max(bayesiana_salida$scoreSummary$Score)

PARAM$out$lgbm$mejores_hiperparametros <- list(
  num_iterations = as.integer(round(bayesiana_salida$scoreSummary$num_iterations[mejor_idx])),
  learning_rate = bayesiana_salida$scoreSummary$learning_rate[mejor_idx],
  feature_fraction = bayesiana_salida$scoreSummary$feature_fraction[mejor_idx],
  bagging_fraction = bayesiana_salida$scoreSummary$bagging_fraction[mejor_idx],
  num_leaves = as.integer(round(bayesiana_salida$scoreSummary$num_leaves[mejor_idx])),
  min_data_in_leaf = as.integer(round(bayesiana_salida$scoreSummary$min_data_in_leaf[mejor_idx]))
)

PARAM$out$lgbm$y <- bayesiana_salida$scoreSummary$Score[mejor_idx]

write_yaml(PARAM, file="PARAM.yml")

cat("\n========================================\n")
cat("BAYESIAN OPTIMIZATION COMPLETADA\n")
cat("========================================\n")
cat("Mejor AUC (5-fold CV):", PARAM$out$lgbm$y, "\n")
cat("Hiperparámetros óptimos:\n")
print(PARAM$out$lgbm$mejores_hiperparametros)
cat("========================================\n\n")

rm(dataset_train, dtrain)
gc()

# ============================================================
# FASE 2: EVALUACIÓN EN HOLDOUT (202104)
# ============================================================

cat("########################################\n")
cat("FASE 2: EVALUACIÓN EN HOLDOUT\n")
cat("########################################\n\n")

cat("Evaluando hiperparámetros óptimos en 202104...\n\n")

dataset_train_val <- dataset[foto_mes %in% PARAM$train]
dataset_test_holdout <- dataset[foto_mes %in% PARAM$test_holdout]

cat("Registros train (202102-202103):", nrow(dataset_train_val), "\n")
cat("Registros holdout (202104):", nrow(dataset_test_holdout), "\n\n")

# Undersampling en train
set.seed(PARAM$semilla_primigenia, kind = "L'Ecuyer-CMRG")
dataset_train_val[, azar := runif(nrow(dataset_train_val))]
dataset_train_val[, training := 0L]
dataset_train_val[(azar <= PARAM$trainingstrategy$undersampling | 
                     clase_ternaria %in% c("BAJA+1", "BAJA+2")), 
                  training := 1L]

dtrain_val <- lgb.Dataset(
  data= data.matrix(dataset_train_val[training == 1L, campos_buenos, with= FALSE]),
  label= dataset_train_val[training == 1L, clase01]
)

dtest_holdout <- lgb.Dataset(
  data= data.matrix(dataset_test_holdout[, campos_buenos, with= FALSE]),
  label= dataset_test_holdout[, clase01],
  reference= dtrain_val
)

param_holdout <- modifyList(PARAM$lgbm$param_fijos, 
                            PARAM$out$lgbm$mejores_hiperparametros)
param_holdout$min_data_in_leaf <- round(
  param_holdout$min_data_in_leaf / PARAM$trainingstrategy$undersampling
)

cat("Entrenando con hiperparámetros óptimos...\n")
modelo_holdout <- lgb.train(
  data= dtrain_val,
  param= param_holdout,
  valids= list(holdout = dtest_holdout)
)

pred_holdout <- predict(modelo_holdout, 
                        data.matrix(dataset_test_holdout[, campos_buenos, with= FALSE]))

if(require("pROC", quietly=TRUE)) {
  auc_holdout <- auc(dataset_test_holdout$clase01, pred_holdout)
  PARAM$out$auc_holdout <- as.numeric(auc_holdout)
} else {
  auc_holdout <- NA
}

cat("\n========================================\n")
cat("RESULTADO EVALUACIÓN HOLDOUT\n")
cat("========================================\n")
cat("AUC 5-fold CV (BO):", round(PARAM$out$lgbm$y, 4), "\n")
if(!is.na(auc_holdout)) {
  cat("AUC holdout (202104):", round(auc_holdout, 4), "\n\n")
  
  diferencia <- PARAM$out$lgbm$y - auc_holdout
  cat("Diferencia:", round(diferencia, 4), "\n\n")
  
  if (abs(diferencia) < 0.01) {
    cat("Interpretación: Modelo generaliza bien\n")
  } else if (diferencia > 0.02) {
    cat("Interpretación: Posible overfitting leve\n")
  } else {
    cat("Interpretación: Modelo parece robusto\n")
  }
}
cat("========================================\n\n")

rm(modelo_holdout, dtrain_val, dtest_holdout, dataset_train_val, dataset_test_holdout)
gc()

# ============================================================
# VALIDACIÓN DE ROBUSTEZ: MÚLTIPLES SEMILLAS EN HOLDOUT
# ============================================================

cat("########################################\n")
cat("VALIDACIÓN MULTI-SEMILLA (ROBUSTEZ)\n")
cat("########################################\n\n")

SEMILLAS_TEST <- c(907871, 908549, 875969, 966913, 925921)

dataset_train_val <- dataset[foto_mes %in% PARAM$train]
dataset_test_holdout <- dataset[foto_mes %in% PARAM$test_holdout]

resultados_robustez <- data.table(
  semilla = integer(),
  auc_holdout = numeric(),
  ganancia_sim = numeric()
)

cat("Evaluando robustez con", length(SEMILLAS_TEST), "semillas...\n\n")

for (i in 1:length(SEMILLAS_TEST)) {
  cat("Semilla", i, "/", length(SEMILLAS_TEST), ":", SEMILLAS_TEST[i], "\n")
  
  set.seed(SEMILLAS_TEST[i], kind = "L'Ecuyer-CMRG")
  dataset_train_val[, azar := runif(nrow(dataset_train_val))]
  dataset_train_val[, training := 0L]
  dataset_train_val[(azar <= PARAM$trainingstrategy$undersampling | 
                       clase_ternaria %in% c("BAJA+1", "BAJA+2")), 
                    training := 1L]
  
  dtrain_temp <- lgb.Dataset(
    data= data.matrix(dataset_train_val[training == 1L, campos_buenos, with= FALSE]),
    label= dataset_train_val[training == 1L, clase01]
  )
  
  param_temp <- modifyList(PARAM$lgbm$param_fijos, 
                           PARAM$out$lgbm$mejores_hiperparametros)
  param_temp$seed <- SEMILLAS_TEST[i]
  param_temp$min_data_in_leaf <- round(
    param_temp$min_data_in_leaf / PARAM$trainingstrategy$undersampling
  )
  
  modelo_temp <- lgb.train(data= dtrain_temp, param= param_temp, verbose= -1)
  
  pred_temp <- predict(modelo_temp, 
                       data.matrix(dataset_test_holdout[, campos_buenos, with= FALSE]))
  
  if(require("pROC", quietly=TRUE)) {
    auc_temp <- as.numeric(auc(dataset_test_holdout$clase01, pred_temp))
  } else {
    auc_temp <- NA
  }
  
  tb_sim_temp <- dataset_test_holdout[, list(numero_de_cliente, foto_mes, clase_ternaria)]
  tb_sim_temp[, prob := pred_temp]
  
  drealidad_temp <- realidad_inicializar(dataset_test_holdout, PARAM)
  
  setorder(tb_sim_temp, -prob)
  mejor_ganancia <- 0
  
  for (envios in seq(8000, 15000, by=1000)) {
    tb_sim_temp[, Predicted := 0L]
    tb_sim_temp[1:envios, Predicted := 1L]
    res_temp <- realidad_evaluar(drealidad_temp, tb_sim_temp)
    if (res_temp$total > mejor_ganancia) {
      mejor_ganancia <- res_temp$total
    }
  }
  
  resultados_robustez <- rbind(resultados_robustez, data.table(
    semilla = SEMILLAS_TEST[i],
    auc_holdout = auc_temp,
    ganancia_sim = mejor_ganancia
  ))
  
  cat("  AUC:", round(auc_temp, 4), 
      " | Ganancia:", format(mejor_ganancia, big.mark=","), "\n")
  
  rm(modelo_temp, dtrain_temp, tb_sim_temp, drealidad_temp)
  gc(verbose=FALSE)
}

cat("\n========================================\n")
cat("ANÁLISIS DE ROBUSTEZ\n")
cat("========================================\n\n")

cat("AUC en Holdout:\n")
cat("  Media:", round(mean(resultados_robustez$auc_holdout, na.rm=TRUE), 4), "\n")
cat("  Desv. Est.:", round(sd(resultados_robustez$auc_holdout, na.rm=TRUE), 4), "\n")
cat("  Mínimo:", round(min(resultados_robustez$auc_holdout, na.rm=TRUE), 4), "\n")
cat("  Máximo:", round(max(resultados_robustez$auc_holdout, na.rm=TRUE), 4), "\n")
cat("  Rango:", round(max(resultados_robustez$auc_holdout, na.rm=TRUE) - 
                        min(resultados_robustez$auc_holdout, na.rm=TRUE), 4), "\n\n")

cat("Ganancia Simulada:\n")
cat("  Media:", format(mean(resultados_robustez$ganancia_sim), big.mark=","), "\n")
cat("  Desv. Est.:", format(sd(resultados_robustez$ganancia_sim), big.mark=","), "\n")
cat("  Mínimo:", format(min(resultados_robustez$ganancia_sim), big.mark=","), "\n")
cat("  Máximo:", format(max(resultados_robustez$ganancia_sim), big.mark=","), "\n")
cat("  Rango:", format(max(resultados_robustez$ganancia_sim) - 
                         min(resultados_robustez$ganancia_sim), big.mark=","), "\n\n")

cv_auc <- sd(resultados_robustez$auc_holdout, na.rm=TRUE) / 
  mean(resultados_robustez$auc_holdout, na.rm=TRUE) * 100
cv_ganancia <- sd(resultados_robustez$ganancia_sim) / 
  mean(resultados_robustez$ganancia_sim) * 100

cat("Coeficiente de Variación:\n")
cat("  AUC:", round(cv_auc, 2), "%\n")
cat("  Ganancia:", round(cv_ganancia, 2), "%\n\n")

cat("INTERPRETACIÓN:\n")
if (cv_auc < 0.5) {
  cat("  ✅ Modelo MUY ROBUSTO (CV AUC < 0.5%)\n")
} else if (cv_auc < 1.0) {
  cat("  ✅ Modelo ROBUSTO (CV AUC < 1%)\n")
} else if (cv_auc < 2.0) {
  cat("  ⚠️ Modelo MODERADAMENTE ROBUSTO (CV AUC < 2%)\n")
} else {
  cat("  ❌ Modelo INESTABLE (CV AUC >= 2%)\n")
  cat("     Revisar: overfitting, features ruidosas, o datos insuficientes\n")
}

cat("\n========================================\n\n")

fwrite(resultados_robustez, file="robustez_multisemilla.csv")

rm(dataset_train_val, dataset_test_holdout)
gc()

# ============================================================
# FASE 3: TRAINING FINAL
# ============================================================

cat("########################################\n")
cat("FASE 3: TRAINING FINAL\n")
cat("########################################\n\n")

setwd(file.path(RUTA_BASE, "exp"))
experimento <- paste0("exp", PARAM$experimento)
dir.create(experimento, showWarnings= FALSE)
setwd(file.path(RUTA_BASE, "exp", experimento))

dataset_train_final <- dataset[foto_mes %in% PARAM$train_final]

cat("Meses training final:", paste(PARAM$train_final, collapse=", "), "\n")
cat("Registros:", nrow(dataset_train_final), "\n\n")

dtrain_final <- lgb.Dataset(
  data= data.matrix(dataset_train_final[, campos_buenos, with= FALSE]),
  label= dataset_train_final[, clase01]
)

param_final <- modifyList(PARAM$lgbm$param_fijos, 
                          PARAM$out$lgbm$mejores_hiperparametros)
param_normalizado <- copy(param_final)

# Ajuste 1: min_data_in_leaf (porque NO hay undersampling)
param_normalizado$min_data_in_leaf <- round(
  param_final$min_data_in_leaf / PARAM$trainingstrategy$undersampling
)

# Ajuste 2: scale_pos_weight para compensar desbalance
n_pos <- dataset_train_final[clase01 == 1L, .N]
n_neg <- dataset_train_final[clase01 == 0L, .N]
scale_pos <- n_neg / n_pos

param_normalizado$scale_pos_weight <- scale_pos

cat("Ajustes para training final (sin undersampling):\n")
cat("  min_data_in_leaf:", param_normalizado$min_data_in_leaf, "\n")
cat("  scale_pos_weight:", round(scale_pos, 2), 
    "(", n_neg, "neg /", n_pos, "pos)\n\n")

cat("Entrenando modelo final...\n")
modelo_final <- lgb.train(data= dtrain_final, param= param_normalizado)

tb_importancia <- as.data.table(lgb.importance(modelo_final))
fwrite(tb_importancia, file= "impo.txt", sep= "\t")
lgb.save(modelo_final, "modelo.txt")

cat("\nTop 20 variables:\n")
print(tb_importancia[1:20])

cat("\nFeatures nuevas de límites en top 50:\n")
nuevas_top <- tb_importancia[1:50][Feature %like% "limite|Master.*pct|Visa.*pct|cancelada|reduccion"]
if(nrow(nuevas_top) > 0) {
  print(nuevas_top)
} else {
  cat("Ninguna feature nueva en top 50\n")
}
cat("\n")

# ============================================================
# FASE 4A: SIMULACIÓN EN 202104
# ============================================================

cat("########################################\n")
cat("FASE 4A: SIMULACIÓN (202104)\n")
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
# FASE 4B: PREDICCIÓN EN 202106
# ============================================================

cat("########################################\n")
cat("FASE 4B: PREDICCIÓN (202106)\n")
cat("########################################\n\n")

# Comparar 202104 vs 202106
cat("\n=== COMPARACIÓN 202104 (abr) vs 202106 (jun) ===\n")

variables_clave <- c("mcaja_ahorro", "mpayroll", "mcuentas_saldo", 
                     "mcaja_ahorro_base", "spike_en_junio")

for (v in variables_clave) {
  if (v %in% names(dataset)) {
    cat("\n", v, ":\n")
    cat("  202104: ", summary(dataset[foto_mes == 202104, get(v)]), "\n")
    cat("  202106: ", summary(dataset[foto_mes == 202106, get(v)]), "\n")
  }
}

dfuture <- dataset[foto_mes %in% PARAM$future]
cat("Registros:", nrow(dfuture), "\n\n")

prediccion <- predict(modelo_final, 
                      data.matrix(dfuture[, campos_buenos, with= FALSE]))

tb_prediccion <- dfuture[, list(numero_de_cliente, foto_mes)]
tb_prediccion[, prob := prediccion]
fwrite(tb_prediccion, file= "prediccion_202106.txt", sep= "\t")

cat("Distribución probabilidades:\n")
print(summary(tb_prediccion$prob))
cat("\n")

umbral_conservador <- 0.10
umbral_balanceado <- 0.05
umbral_agresivo <- 0.01

n_conservador <- tb_prediccion[prob > umbral_conservador, .N]
n_balanceado <- tb_prediccion[prob > umbral_balanceado, .N]
n_agresivo <- tb_prediccion[prob > umbral_agresivo, .N]

cat("Clientes con prob >", umbral_conservador, ":", n_conservador, "\n")
cat("Clientes con prob >", umbral_balanceado, ":", n_balanceado, "\n")
cat("Clientes con prob >", umbral_agresivo, ":", n_agresivo, "\n\n")

diff_bal <- abs(PARAM$cortes - n_balanceado)
mejor_prob <- PARAM$cortes[which.min(diff_bal)]

setorder(tb_prediccion, -prob)
dir.create("kaggle", showWarnings = FALSE)

for (envios in PARAM$cortes) {
  tb_prediccion[, Predicted := 0L]
  tb_prediccion[1:envios, Predicted := 1L]
  
  archivo_kaggle <- paste0("./kaggle/KA", PARAM$experimento, "_", envios, ".csv")
  fwrite(tb_prediccion[, list(numero_de_cliente, Predicted)], 
         file= archivo_kaggle, sep= ",")
}

write_yaml(PARAM, file="PARAM.yml")

# ============================================================
# RECOMENDACIÓN FINAL
# ============================================================

cat("########################################\n")
cat("RECOMENDACIÓN DE SUBMISSION\n")
cat("########################################\n\n")

cat("OPCIÓN 1 - Simulación (202104):\n")
cat("  Subir: KA", PARAM$experimento, "_", mejor_sim$envios, ".csv\n", sep="")
cat("  Ganancia estimada:", format(mejor_sim$total, big.mark=","), "\n\n")

cat("OPCIÓN 2 - Probabilidades (202106):\n")
cat("  Subir: KA", PARAM$experimento, "_", mejor_prob, ".csv\n", sep="")
cat("  Umbral: prob >", umbral_balanceado, "\n\n")

cat("TOP 3 ALTERNATIVAS:\n")
for (i in 1:min(3, nrow(resumen_sim))) {
  cat("  ", i, ". KA", PARAM$experimento, "_", resumen_sim[i, envios], 
      ".csv (", format(resumen_sim[i, total], big.mark=","), ")\n", sep="")
}

if (mejor_sim$envios == mejor_prob) {
  cat("\n*** Ambos métodos coinciden en ", mejor_sim$envios, " envíos ***\n", sep="")
}

cat("\n########################################\n")
cat("EXPERIMENTO", PARAM$experimento, "COMPLETADO\n")
cat("########################################\n\n")

format(Sys.time(), "%a %b %d %X %Y")
