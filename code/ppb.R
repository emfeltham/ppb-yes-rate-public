# ppb.R
# R port of the PPB point estimate and confidence intervals in inference.jl.
# Base R only. Usage:
#   source("code/ppb.R")
#   ppb(hits = 42, ns = 50, fa = 17, nn = 50)
#   ppb_interval(42, 50, 17, 50)              # Clopper-Pearson, coverage >= 95%
#   adjusted_ppb_interval(42, 50, 17, 50)     # add-one adjusted Wald
#   ppb_difference_interval(42, 50, 17, 50, 35, 50, 20, 50)
# Arguments are counts: hits of ns signal trials, false alarms of nn noise
# trials. These are intervals for fixed binomial probabilities, not for
# heterogeneous, paired or clustered participant data.

.check_alpha <- function(alpha) {
  if (!(length(alpha) == 1 && is.finite(alpha) && alpha > 0 && alpha < 1))
    stop("alpha must be finite and strictly between zero and one")
}

.check_counts <- function(x, n) {
  if (!(n > 0)) stop("trial denominator must be positive")
  if (!(x >= 0 && x <= n)) stop("count must lie between zero and its denominator")
}

ppb <- function(hits, ns, fa, nn) {
  .check_counts(hits, ns); .check_counts(fa, nn)
  hits / ns + fa / nn
}

clopper_pearson_interval <- function(x, n, alpha = 0.05) {
  .check_counts(x, n); .check_alpha(alpha)
  c(lower = if (x == 0) 0 else qbeta(alpha / 2, x, n - x + 1),
    upper = if (x == n) 1 else qbeta(1 - alpha / 2, x + 1, n - x))
}

ppb_interval <- function(hits, ns, fa, nn, alpha = 0.05) {
  .check_alpha(alpha)
  h <- clopper_pearson_interval(hits, ns, alpha / 2)
  f <- clopper_pearson_interval(fa, nn, alpha / 2)
  c(lower = unname(h["lower"] + f["lower"]), upper = unname(h["upper"] + f["upper"]))
}

wald_interval <- function(hits, ns, fa, nn, alpha = 0.05) {
  .check_counts(hits, ns); .check_counts(fa, nn); .check_alpha(alpha)
  h <- hits / ns; f <- fa / nn
  hw <- qnorm(1 - alpha / 2) * sqrt(h * (1 - h) / ns + f * (1 - f) / nn)
  c(lower = h + f - hw, upper = h + f + hw)
}

adjusted_ppb_interval <- function(hits, ns, fa, nn, alpha = 0.05) {
  .check_counts(hits, ns); .check_counts(fa, nn); .check_alpha(alpha)
  h <- (hits + 1) / (ns + 2); f <- (fa + 1) / (nn + 2)
  hw <- qnorm(1 - alpha / 2) * sqrt(h * (1 - h) / (ns + 2) + f * (1 - f) / (nn + 2))
  c(lower = max(0, h + f - hw), upper = min(2, h + f + hw))
}

ppb_difference_interval <- function(hits1, ns1, fa1, nn1,
                                    hits2, ns2, fa2, nn2, alpha = 0.05) {
  .check_alpha(alpha)
  h1 <- clopper_pearson_interval(hits1, ns1, alpha / 4)
  f1 <- clopper_pearson_interval(fa1, nn1, alpha / 4)
  h2 <- clopper_pearson_interval(hits2, ns2, alpha / 4)
  f2 <- clopper_pearson_interval(fa2, nn2, alpha / 4)
  c(lower = unname(h1["lower"] + f1["lower"] - h2["upper"] - f2["upper"]),
    upper = unname(h1["upper"] + f1["upper"] - h2["lower"] - f2["lower"]))
}
