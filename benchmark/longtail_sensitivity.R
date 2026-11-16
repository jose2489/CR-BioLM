# Sensitivity of the long-tail model (data/confirmatory/longtail_model.R) to the mix of
# question types. The number of records L changes which questions a species gets (park
# questions need a park record) and how their keys are computed, so the slope of
# correctness on log(1 + L) could come from that mix rather than from the models.
# Three variants of the pre-registered model, each reporting the per-condition slope:
#   base          the pre-registered model
#   type_slope    plus a random slope of log(1 + L) by question type
#   type_fixed    plus a fixed question type x log(1 + L) interaction
#   no_park       the base model without the three park question types
# Run: Rscript benchmark/longtail_sensitivity.R benchmark/data/confirmatory/answers_long.csv
lib <- Sys.getenv("R_LIBS_USER"); .libPaths(c(lib, .libPaths()))
suppressMessages(library(lme4))
d0 <- read.csv(commandArgs(TRUE)[1], encoding = "UTF-8")
d0 <- d0[d0$arm %in% c("D", "F", "A1F", "CF"), ]
d0$arm <- factor(d0$arm, levels = c("A1F", "CF", "D", "F"))
d0$lrec <- log1p(d0$n_records)
d0$lglob <- log10(d0$n_global)
ctl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

slopes <- function(m) {
  co <- fixef(m); v <- as.matrix(vcov(m))
  sapply(c("A1F", "CF", "D", "F"), function(a) {
    if (a == "A1F") return(c(est = co[["lrec"]], se = sqrt(v["lrec", "lrec"])))
    k <- paste0("arm", a, ":lrec")
    c(est = co[["lrec"]] + co[[k]],
      se = sqrt(v["lrec", "lrec"] + v[k, k] + 2 * v["lrec", k]))
  }, simplify = FALSE)
}
fit <- function(f, d) {
  full <- glmer(f, data = d, family = binomial, control = ctl)
  null <- update(full, . ~ . - arm:lrec)
  lrt <- anova(null, full)
  list(n = nrow(d), lrt_p = lrt$`Pr(>Chisq)`[2], singular = isSingular(full),
       slopes = slopes(full))
}
base <- correct ~ arm * lrec + lglob + (1 | species) + (1 | qtype)
out <- list(
  base = fit(base, d0),
  type_slope = fit(correct ~ arm * lrec + lglob + (1 | species) + (1 + lrec | qtype), d0),
  type_fixed = fit(correct ~ arm * lrec + qtype:lrec + lglob + (1 | species) + (1 | qtype), d0),
  no_park = fit(base, d0[!grepl("^park_", d0$qtype), ]))
cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 5, pretty = TRUE))
