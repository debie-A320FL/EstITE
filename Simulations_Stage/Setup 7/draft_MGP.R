# Required packages
if(!require(MASS))     install.packages("MASS")
if(!require(mvtnorm))  install.packages("mvtnorm")
if(!require(cluster))  install.packages("cluster")
if(!require(scatterplot3d))  install.packages("scatterplot3d")

library(MASS)
library(mvtnorm)
library(scatterplot3d)

set.seed(123)

# PARAMETERS
d <- 5      # dimension (e.g. 3, 10, 50)
K <- 3       # number of groups
N <- 50000   # number of samples for Monte-Carlo

# 1) Generate means (mu) on the unit hypersphere
raw <- matrix(rnorm(d*K), nrow=K, ncol=d)
mu <- t(apply(raw, 1, function(x) x / sqrt(sum(x^2))))

# 2) Define covariance matrices (Σ_i)
make_ar1_cov <- function(rho, d) {
  mat <- outer(1:d, 1:d, function(i, j) rho^abs(i - j))
  return(mat)
}

# Each group gets its own AR(1)-type covariance with a random rho in (-0.9, 0.9)

rho = 0.9 # the closer to 1, the less spread, so the easier it is to distinguish the groups
rhos <- rep(rho, K)
sigmas <- lapply(rhos, function(rho) make_ar1_cov(rho, d))

# Precompute Cholesky and det
chol_list <- lapply(sigmas, chol)
det_list  <- sapply(sigmas, det)

# 3) Closed-form distances
pairs <- combn(K, 2)
DM  <- matrix(0, K, K)
DB  <- matrix(0, K, K)
rho <- matrix(0, K, K)

for(idx in 1:ncol(pairs)) {
  i <- pairs[1, idx]; j <- pairs[2, idx]
  # mixture covariance
  Sij <- (sigmas[[i]] + sigmas[[j]]) / 2
  invSij <- solve(Sij)
  
  diff <- mu[i,] - mu[j,]
  DM[i,j] <- sqrt(t(diff) %*% invSij %*% diff)
  DM[j,i] <- DM[i,j]
  
  term1 <- 0.125 * t(diff) %*% invSij %*% diff
  term2 <- 0.5 * log(det(Sij) / sqrt(det_list[i]*det_list[j]))
  DB[i,j] <- term1 + term2
  DB[j,i] <- DB[i,j]
  
  rho[i,j] <- exp(-DB[i,j])
  rho[j,i] <- rho[i,j]
}

lower_triangle_named <- function(mat) {
  idx <- which(lower.tri(mat), arr.ind = TRUE)
  data.frame(
    Group1 = idx[,1],
    Group2 = idx[,2],
    Value  = mat[idx]
  )
}

scalar_products <- mu %*% t(mu)

cat("Scalar product matrix (cosine between group centers):\n")
print(round(scalar_products, 3))

cat("Lower triangle (scalar products between centers):\n")
print(lower_triangle_named(scalar_products))

cat("Mahalanobis distances (lower triangle):\n")
print(lower_triangle_named(DM))

cat("Bhattacharyya distances (lower triangle):\n")
print(lower_triangle_named(DB))

cat("Overlap coefficients (rho = exp(-DB)) (lower triangle):\n")
print(lower_triangle_named(rho))

# 4) Monte-Carlo sampling & classification
# draw true labels
zs <- sample(1:K, size=N, replace=TRUE, prob=rep(1/K, K))
X  <- matrix(0, nrow=N, ncol=d)
for(i in 1:K) {
  ni <- sum(zs == i)
  if(ni > 0) {
    X[zs==i, ] <- mvrnorm(ni, mu=mu[i,], Sigma=sigmas[[i]])
  }
}

# compute log-densities and assign
logdens <- sapply(1:K, function(i) {
  dmvnorm(X, mean=mu[i,], sigma=sigmas[[i]], log=TRUE) + log(1/K)
})
pred <- max.col(logdens)

# overall misclassification rate
err_rate <- mean(pred != zs)
cat(sprintf("Overall error rate: %.3f\n", err_rate))

# confusion matrix
conf <- table(True=zs, Pred=pred)
print(conf)

# silhouette score (on full data may be slow for N=50k—sample 5k for silhouette)
ssample <- sample(1:N, min(5000, N))
sil <- silhouette(zs[ssample], dist(X[ssample, ]))
cat(sprintf("Average silhouette width: %.3f\n", mean(sil[, 3])))

# 5) Random-projection visualization (2D & 3D)
# 2D projection
P2 <- qr.Q(qr(matrix(rnorm(d*2), ncol=2)))
Y2 <- X %*% P2
plot(Y2, col=zs, pch=20, main="2D Random Projection")

# 3D projection (if d > 2)
if(require(scatterplot3d)) {
  P3 <- qr.Q(qr(matrix(rnorm(d*3), ncol=3)))
  Y3 <- X %*% P3
  scatterplot3d::scatterplot3d(
    Y3, color=zs, pch=20, main="3D Random Projection"
  )
} else {
  message("Install 'scatterplot3d' for 3D plotting.")
}
