## DR-Learner: sequential tuning with separate pi, mu0, mu1, and cate params

# Fit DR-Learner with separate hyperparameters
DR_RF_learner_fit <- function(X, y, t,
                              params_pi, params_mu0, params_mu1, params_cat,
                              seed=42) {
  set.seed(seed)
  idx   <- createDataPartition(y, p=0.5, list=FALSE)
  X_tr  <- X[idx,]; y_tr<-y[idx]; t_tr<-t[idx]
  X_md  <- X[-idx,]; y_md<-y[-idx]; t_md<-t[-idx]

  # propensity
  samp_pi <- max(2, floor(params_pi$sample.fraction * nrow(X_tr)))
  pi_mod <- randomForest(as.factor(t_tr)~., data=X_tr,
                         ntree=params_pi$ntree,
                         mtry=params_pi$mtry,
                         sampsize=samp_pi,
                         nodesize=params_pi$nodesizeSpl)
  pihat <- pmin(pmax(predict(pi_mod, X_md, type="prob")[,2],1e-3),1-1e-3)

  # mu0/mu1
  samp0 <- max(2, floor(params_mu0$sample.fraction * sum(t_tr==0)))
  mu0_mod <- randomForest(y_tr[t_tr==0]~., data=X_tr[t_tr==0,],
                          ntree=params_mu0$ntree,
                          mtry=params_mu0$mtry,
                          sampsize=samp0,
                          nodesize=params_mu0$nodesizeSpl)
  samp1 <- max(2, floor(params_mu1$sample.fraction * sum(t_tr==1)))
  mu1_mod <- randomForest(y_tr[t_tr==1]~., data=X_tr[t_tr==1,],
                          ntree=params_mu1$ntree,
                          mtry=params_mu1$mtry,
                          sampsize=samp1,
                          nodesize=params_mu1$nodesizeSpl)
  mu0hat <- predict(mu0_mod, X_md); mu1hat <- predict(mu1_mod, X_md)

  # DR pseudo
  Z<-t_md; Y<-y_md
  dr_ps <- (Z/pihat - (1-Z)/(1-pihat))*Y +
           ((1-Z/pihat)*mu1hat - (1-(1-Z)/(1-pihat))*mu0hat)

  # cate
  sampc <- max(2, floor(params_cat$sample.fraction * nrow(X_md)))
  cate_mod <- randomForest(dr_ps~., data=X_md,
                            ntree=params_cat$ntree,
                            mtry=params_cat$mtry,
                            sampsize=sampc,
                            nodesize=params_cat$nodesizeAvg)

  list(pi_model=pi_mod,
       mu0_model=mu0_mod,
       mu1_model=mu1_mod,
       cate_model=cate_mod)
}

# Predict DR
DR_RF_learner_predict <- function(fit, newdata) predict(fit$cate_model, newdata)

# Sequential tuning for DR-Learner
DR_RF_learner_optimize <- function(
  train_X, train_y, train_t,
  val_X, val_CATT,
  param_ranges = list(
    mtry            = c(1,2,3,4),
    sample.fraction = c(0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6),
    nodesizeSpl     = c(1,3,5,10,15,20,25,30),
    nodesizeAvg     = c(1,3,5,10,15,20,25,30),
    ntree           = c(300,500,1000,1500,2000,3000,5000,10000)
  ),
  seed=42, verbose=TRUE
) {
  set.seed(seed)
  default <- list(ntree=1000,
                  mtry=round(ncol(train_X)/2),
                  sample.fraction=0.5,
                  nodesizeSpl=5,
                  nodesizeAvg=5)

  best_pi   <- default
  best_mu0  <- default
  best_mu1  <- default
  best_cat  <- default
  best_pehe <- Inf

  # tune pi
  if(verbose) message("Tuning pi-model...")
  for(param in names(param_ranges)) for(val in param_ranges[[param]]) {
    trial <- best_pi; trial[[param]] <- val
    fit <- tryCatch(DR_RF_learner_fit(train_X, train_y, train_t,
                                       params_pi=trial,
                                       params_mu0=best_mu0,
                                       params_mu1=best_mu1,
                                       params_cat=best_cat, seed=seed), error=function(e) NULL)
    if(is.null(fit)) next
    pehe <- PEHE(val_CATT, DR_RF_learner_predict(fit, val_X))
    if(verbose) message(sprintf("  pi %s=%s -> PEHE=%.4f", param, val, pehe))
    if(pehe<best_pehe){ best_pehe<-pehe; best_pi<-trial; if(verbose) message("   -> new best pi, PEHE=",round(pehe,4)) }
  }

  # tune mu0
  if(verbose) message("Tuning mu0-model...")
  for(param in names(param_ranges)) for(val in param_ranges[[param]]) {
    trial <- best_mu0; trial[[param]] <- val
    fit <- tryCatch(DR_RF_learner_fit(train_X, train_y, train_t,
                                       params_pi=best_pi,
                                       params_mu0=trial,
                                       params_mu1=best_mu1,
                                       params_cat=best_cat, seed=seed), error=function(e) NULL)
    if(is.null(fit)) next
    pehe <- PEHE(val_CATT, DR_RF_learner_predict(fit, val_X))
    if(verbose) message(sprintf("  mu0 %s=%s -> PEHE=%.4f", param, val, pehe))
    if(pehe<best_pehe){ best_pehe<-pehe; best_mu0<-trial; if(verbose) message("   -> new best mu0, PEHE=",round(pehe,4)) }
  }

  # tune mu1
  if(verbose) message("Tuning mu1-model...")
  for(param in names(param_ranges)) for(val in param_ranges[[param]]) {
    trial <- best_mu1; trial[[param]] <- val
    fit <- tryCatch(DR_RF_learner_fit(train_X, train_y, train_t,
                                       params_pi=best_pi,
                                       params_mu0=best_mu0,
                                       params_mu1=trial,
                                       params_cat=best_cat, seed=seed), error=function(e) NULL)
    if(is.null(fit)) next
    pehe <- PEHE(val_CATT, DR_RF_learner_predict(fit, val_X))
    if(verbose) message(sprintf("  mu1 %s=%s -> PEHE=%.4f", param, val, pehe))
    if(pehe<best_pehe){ best_pehe<-pehe; best_mu1<-trial; if(verbose) message("   -> new best mu1, PEHE=",round(pehe,4)) }
  }

  # tune cate
  if(verbose) message("Tuning cate-model...")
  for(param in names(param_ranges)) for(val in param_ranges[[param]]) {
    trial <- best_cat; trial[[param]] <- val
    fit <- tryCatch(DR_RF_learner_fit(train_X, train_y, train_t,
                                       params_pi=best_pi,
                                       params_mu0=best_mu0,
                                       params_mu1=best_mu1,
                                       params_cat=trial, seed=seed), error=function(e) NULL)
    if(is.null(fit)) next
    pehe <- PEHE(val_CATT, DR_RF_learner_predict(fit, val_X))
    if(verbose) message(sprintf("  cate %s=%s -> PEHE=%.4f", param, val, pehe))
    if(pehe<best_pehe){ best_pehe<-pehe; best_cat<-trial; if(verbose) message("   -> new best cate, PEHE=",round(pehe,4)) }
  }

  list(
    best_pi_params   = best_pi,
    best_mu0_params  = best_mu0,
    best_mu1_params  = best_mu1,
    best_cat_params  = best_cat,
    best_pehe        = best_pehe
  )
}
# ─── 1) RA‐Learner ──────────────────────────────────────────────────────────

## RA-Learner: sequential tuning with separate mu0, mu1, and cate params

# Fit RA-Learner with separate hyperparameters
RA_RF_learner_fit <- function(X, y, w,
                              params_mu0, params_mu1, params_cat,
                              seed = 42) {
  set.seed(seed)
  idx  <- createDataPartition(y, p = 0.5, list = FALSE)
  X_tr <- X[idx,]; y_tr <- y[idx]; w_tr <- w[idx]
  X_md <- X[-idx,]; y_md <- y[-idx]; w_md <- w[-idx]

  # mu0 model
  samp0 <- max(2, floor(params_mu0$sample.fraction * sum(w_tr==0)))
  mu0_mod <- randomForest(y_tr[w_tr==0] ~ ., data = X_tr[w_tr==0,],
                          ntree    = params_mu0$ntree,
                          mtry     = params_mu0$mtry,
                          sampsize = samp0,
                          nodesize = params_mu0$nodesizeSpl)
  # mu1 model
  samp1 <- max(2, floor(params_mu1$sample.fraction * sum(w_tr==1)))
  mu1_mod <- randomForest(y_tr[w_tr==1] ~ ., data = X_tr[w_tr==1,],
                          ntree    = params_mu1$ntree,
                          mtry     = params_mu1$mtry,
                          sampsize = samp1,
                          nodesize = params_mu1$nodesizeSpl)

  mu0hat <- predict(mu0_mod, X_md)
  mu1hat <- predict(mu1_mod, X_md)

  # pseudo-outcome
  pseu <- w_md * (y_md - mu0hat) + (1 - w_md) * (mu1hat - y_md)

  # cate model
  sampc <- max(2, floor(params_cat$sample.fraction * nrow(X_md)))
  cate_mod <- randomForest(pseu ~ ., data = X_md,
                            ntree    = params_cat$ntree,
                            mtry     = params_cat$mtry,
                            sampsize = sampc,
                            nodesize = params_cat$nodesizeAvg)

  list(mu0_model = mu0_mod,
       mu1_model = mu1_mod,
       cate_model = cate_mod)
}

# Predict
RA_RF_learner_predict <- function(fit, newdata) predict(fit$cate_model, newdata)

# Sequential tuning for RA-Learner
RA_RF_learner_optimize <- function(
  train_X, train_y, train_w,
  val_X, val_CATT,
  param_ranges = list(
    mtry            = c(1,2,3,4),
    sample.fraction = c(0.05,0.1,0.2),
    nodesizeSpl     = c(1,5,10),
    nodesizeAvg     = c(1,5,10),
    ntree           = c(500,1000,2000)
  ),
  seed = 42, verbose = TRUE
) {
  set.seed(seed)

  # defaults
  default <- list(ntree=1000,
                  mtry=round(ncol(train_X)/2),
                  sample.fraction=0.5,
                  nodesizeSpl=5,
                  nodesizeAvg=5)

  best_mu0  <- default
  best_mu1  <- default
  best_cat  <- default
  best_pehe <- Inf

  # tune mu0
  if (verbose) message("Tuning mu0-model...")
  for (param in names(param_ranges)) {
    for (val in param_ranges[[param]]) {
      trial <- best_mu0; trial[[param]] <- val
      fit   <- tryCatch(
        RA_RF_learner_fit(train_X, train_y, train_w,
                           params_mu0 = trial,
                           params_mu1 = best_mu1,
                           params_cat = best_cat, seed=seed),
        error=function(e) NULL
      )
      if (is.null(fit)) next
      preds <- RA_RF_learner_predict(fit, val_X)
      pehe  <- PEHE(val_CATT, preds)
      if (verbose) message(sprintf("  mu0 %s=%s -> PEHE=%.4f", param, val, pehe))
      if (pehe < best_pehe) {
        best_pehe <- pehe; best_mu0 <- trial
        if (verbose) message("   -> new best mu0, PEHE=", round(pehe,4))
      }
    }
  }

  # tune mu1
  if (verbose) message("Tuning mu1-model...")
  for (param in names(param_ranges)) {
    for (val in param_ranges[[param]]) {
      trial <- best_mu1; trial[[param]] <- val
      fit   <- tryCatch(
        RA_RF_learner_fit(train_X, train_y, train_w,
                           params_mu0 = best_mu0,
                           params_mu1 = trial,
                           params_cat = best_cat, seed=seed),
        error=function(e) NULL
      )
      if (is.null(fit)) next
      preds <- RA_RF_learner_predict(fit, val_X)
      pehe  <- PEHE(val_CATT, preds)
      if (verbose) message(sprintf("  mu1 %s=%s -> PEHE=%.4f", param, val, pehe))
      if (pehe < best_pehe) {
        best_pehe <- pehe; best_mu1 <- trial
        if (verbose) message("   -> new best mu1, PEHE=", round(pehe,4))
      }
    }
  }

  # tune cate
  if (verbose) message("Tuning cate-model...")
  for (param in names(param_ranges)) {
    for (val in param_ranges[[param]]) {
      trial <- best_cat; trial[[param]] <- val
      fit   <- tryCatch(
        RA_RF_learner_fit(train_X, train_y, train_w,
                           params_mu0 = best_mu0,
                           params_mu1 = best_mu1,
                           params_cat = trial, seed=seed),
        error=function(e) NULL
      )
      if (is.null(fit)) next
      preds <- RA_RF_learner_predict(fit, val_X)
      pehe  <- PEHE(val_CATT, preds)
      if (verbose) message(sprintf("  cate %s=%s -> PEHE=%.4f", param, val, pehe))
      if (pehe < best_pehe) {
        best_pehe <- pehe; best_cat <- trial
        if (verbose) message("   -> new best cate, PEHE=", round(pehe,4))
      }
    }
  }

  # return all best sets
  list(
    best_mu0_params = best_mu0,
    best_mu1_params = best_mu1,
    best_cat_params = best_cat,
    best_pehe       = best_pehe
  )
}


# ─── 2) PW‐Learner ──────────────────────────────────────────────────────────

# Fit PW‐Learner
PW_RF_learner_fit <- function(X, y, w, params, seed = 42) {
  set.seed(seed)
  idx  <- createDataPartition(y, p = 0.5, list = FALSE)
  X_tr <- X[idx, ];     y_tr <- y[idx];     w_tr <- w[idx]
  X_md <- X[-idx, ];    y_md <- y[-idx];    w_md <- w[-idx]

  # unpack
  ntree <- params$ntree
  mtry  <- params$mtry
  samp_tr <- floor(params$sample.fraction * nrow(X_tr))
  samp_md <- floor(params$sample.fraction * nrow(X_md))
  nspl  <- params$nodesizeSpl
  navg  <- params$nodesizeAvg

  # 1) propensity
  pi_mod <- randomForest(as.factor(w_tr) ~ ., data = X_tr,
                         ntree = ntree, mtry = mtry,
                         sampsize = samp_tr, nodesize = nspl)
  pihat  <- pmin(pmax(predict(pi_mod, X_md, type="prob")[,2],1e-3),1-1e-3)

  # 2) PW pseudo-outcome
  W <- w_md; Y <- y_md
  PW_RF_pseudo <- (W/pihat - (1 - W)/(1 - pihat)) * Y

  # 3) final regressor
  cate_mod <- randomForest(PW_RF_pseudo ~ ., data = X_md,
                           ntree = ntree, mtry = mtry,
                           sampsize = samp_md, nodesize = navg)

  list(pi_model = pi_mod, cate_model = cate_mod)
}

# Predict with PW‐Learner
PW_RF_learner_predict <- function(fit, newdata) {
  predict(fit$cate_model, newdata)
}

# Optimize PW‐Learner by PEHE
PW_RF_learner_optimize <- function(train_X, train_y, train_w,
                                val_X, val_CATT,
                                param_ranges = list(
                                  mtry            = c(1,2,3,4),
                                  sample.fraction = c(0.05,0.1,0.2),
                                  nodesizeSpl     = c(1,5,10),
                                  nodesizeAvg     = c(1,5,10),
                                  ntree           = c(500,1000,2000)
                                ),
                                seed = 42, verbose = TRUE) {
  set.seed(seed)
  best_params <- list(
    ntree = 1000,
    mtry  = round(ncol(train_X)/2),
    sample.fraction = 0.5,
    nodesizeSpl     = 5,
    nodesizeAvg     = 5
  )
  best_pehe <- Inf

  for (param in names(param_ranges)) {
    if (verbose) message("Tuning PW: ", param)
    for (value in param_ranges[[param]]) {
      trial <- best_params
      trial[[param]] <- value

      fit <- tryCatch(
        PW_RF_learner_fit(train_X, train_y, train_w, params = trial, seed = seed),
        error = function(e) { if(verbose) message("  fit failed: ",e$message); NULL }
      )
      if (is.null(fit)) next

      preds <- tryCatch(
        PW_RF_learner_predict(fit, newdata = val_X),
        error = function(e) { if(verbose) message("  pred failed: ",e$message); NULL }
      )
      if (is.null(preds)) next

      pehe <- PEHE(val_CATT, preds)
      if (verbose) message("  ",param,"=",value," → PEHE=",round(pehe,4))

      if (!is.na(pehe) && pehe < best_pehe) {
        best_pehe   <- pehe
        best_params <- trial
        if (verbose) message("   ✅ new best PEHE=",round(pehe,4))
      }
    }
  }

  best_params
}

## R-Learner with RF: sequential hyperparameter tuning

# Fit R-Learner
r_RF_learner_fit <- function(X, y, w,
                              params_m, params_pi, params_cat,
                              seed = 42) {
  set.seed(seed)
  idx   <- createDataPartition(y, p = 0.5, list = FALSE)
  X_tr  <- X[idx,];   y_tr <- y[idx];   w_tr <- w[idx]
  X_md  <- X[-idx,];  y_md <- y[-idx];  w_md <- w[-idx]

  # outcome m(x)
  samp_m <- max(2, floor(params_m$sample.fraction * nrow(X_tr)))
  m_mod <- randomForest(y_tr ~ ., data = X_tr,
                        ntree    = params_m$ntree,
                        mtry     = params_m$mtry,
                        sampsize = samp_m,
                        nodesize = params_m$nodesize)
  m_hat <- predict(m_mod, X_md)

  # propensity pi(x)
  samp_pi <- max(2, floor(params_pi$sample.fraction * nrow(X_tr)))
  pi_mod <- randomForest(as.factor(w_tr) ~ ., data = X_tr,
                         ntree    = params_pi$ntree,
                         mtry     = params_pi$mtry,
                         sampsize = samp_pi,
                         nodesize = params_pi$nodesize)
  pi_hat <- pmin(pmax(predict(pi_mod, X_md, type="prob")[,2],1e-3),1-1e-3)

  # pseudo-outcome
  resid     <- y_md - m_hat
  treat_dev <- w_md - pi_hat
  pseudo    <- resid / treat_dev
  weights   <- treat_dev^2

  # final cate-model
  samp_cat <- max(2, floor(params_cat$sample.fraction * nrow(X_md)))
  cate_mod <- randomForest(pseudo ~ ., data = X_md,
                            ntree    = params_cat$ntree,
                            mtry     = params_cat$mtry,
                            sampsize = samp_cat,
                            nodesize = params_cat$nodesize,
                            weights  = weights)

  list(m_model = m_mod, pi_model = pi_mod, cate_model = cate_mod)
}

# Predict CATE
r_RF_learner_predict <- function(fit, newdata) {
  predict(fit$cate_model, newdata)
}

# Sequential tuning: first m, then pi, then cate
r_RF_learner_optimize <- function(
  train_X, train_y, train_w,
  val_X, val_CATT,
  param_ranges = list(
    mtry            = c(1,2,3,4),
    sample.fraction = c(0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6),
    nodesize        = c(1,3,5,10,15,20,25,30),
    ntree           = c(300,500,1000,1500,2000,3000,5000,10000)
  ),
  seed = 42, verbose = TRUE
) {
  set.seed(seed)

  # defaults
  best_m  <- list(ntree=1000, mtry=round(ncol(train_X)/2), nodesize=5, sample.fraction=0.5)
  best_pi <- best_m
  best_c  <- best_m
  best_pehe <- Inf

  # tune m-model
  if (verbose) message("Tuning m-model...")
  for (param in names(param_ranges)) {
    if (verbose) message(" optimizing ", param)
    for (val in param_ranges[[param]]) {
      trial <- best_m; trial[[param]] <- val
      fit  <- tryCatch(
        r_RF_learner_fit(train_X, train_y, train_w, trial, best_pi, best_c, seed=seed),
        error = function(e) NULL
      )
      if (is.null(fit)) next
      preds <- r_RF_learner_predict(fit, val_X)
      pehe  <- PEHE(val_CATT, preds)
      if (verbose) message(sprintf("  %s=%s -> PEHE=%.4f", param, val, pehe))
      if (pehe < best_pehe) {
        best_pehe <- pehe; best_m <- trial
        if (verbose) message("   -> new best m-model, PEHE=", round(pehe,4))
      }
    }
  }

  # tune pi-model
  if (verbose) message("Tuning pi-model...")
  for (param in names(param_ranges)) {
    if (verbose) message(" optimizing ", param)
    for (val in param_ranges[[param]]) {
      trial <- best_pi; trial[[param]] <- val
      fit  <- tryCatch(
        r_RF_learner_fit(train_X, train_y, train_w, best_m, trial, best_c, seed=seed),
        error = function(e) NULL
      )
      if (is.null(fit)) next
      preds <- r_RF_learner_predict(fit, val_X)
      pehe  <- PEHE(val_CATT, preds)
      if (verbose) message(sprintf("  %s=%s -> PEHE=%.4f", param, val, pehe))
      if (pehe < best_pehe) {
        best_pehe <- pehe; best_pi <- trial
        if (verbose) message("   -> new best pi-model, PEHE=", round(pehe,4))
      }
    }
  }

  # tune cate-model
  if (verbose) message("Tuning cate-model...")
  for (param in names(param_ranges)) {
    if (verbose) message(" optimizing ", param)
    for (val in param_ranges[[param]]) {
      trial <- best_c; trial[[param]] <- val
      fit  <- tryCatch(
        r_RF_learner_fit(train_X, train_y, train_w, best_m, best_pi, trial, seed=seed),
        error = function(e) NULL
      )
      if (is.null(fit)) next
      preds <- r_RF_learner_predict(fit, val_X)
      pehe  <- PEHE(val_CATT, preds)
      if (verbose) message(sprintf("  %s=%s -> PEHE=%.4f", param, val, pehe))
      if (pehe < best_pehe) {
        best_pehe <- pehe; best_c <- trial
        if (verbose) message("   -> new best cate-model, PEHE=", round(pehe,4))
      }
    }
  }

  list(
    best_m_params  = best_m,
    best_pi_params = best_pi,
    best_cat_params= best_c,
    best_pehe      = best_pehe
  )
}