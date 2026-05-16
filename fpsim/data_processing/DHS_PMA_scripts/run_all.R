# Master script to control which data processing scripts to run

# Set to TRUE to run, FALSE to skip
run_age_at_first_birth <- TRUE
run_age_partnership <- TRUE
run_ageparity <- TRUE
run_birth_spacing <- TRUE
run_breastfeeding_stats <- TRUE
run_contra_coeffs <- TRUE
run_education <- TRUE
run_extract_urban <- TRUE
run_make_matrices <- TRUE
run_method_mix <- TRUE
run_postpartum_recode <- TRUE
run_sexual_activity <- TRUE
run_sexual_debut_age_probs <- TRUE
run_subnational <- TRUE
run_time_on_method <- TRUE
run_wealth <- TRUE

# Run scripts conditionally
if (run_age_at_first_birth) source("age_at_first_birth.R")
if (run_age_partnership) source("age_partnership.R")
if (run_ageparity) source("ageparity.R")
if (run_birth_spacing) source("birth_spacing.R")
if (run_breastfeeding_stats) source("breastfeeding_stats.R")
if (run_contra_coeffs) source("contra_coeffs.R")
if (run_education) source("education_dhs_pma.R")
if (run_extract_urban) source("extract_urban.R")
if (run_make_matrices) source("make_matrices.R")
if (run_method_mix) source("method_mix.R")
if (run_postpartum_recode) source("postpartum_recode.R")
if (run_sexual_activity) source("sexual_activity.R")
if (run_sexual_debut_age_probs) source("sexual_debut_age_probs.R")
if (run_subnational) source("subnational.R")
if (run_time_on_method) source("time_on_method.R")
if (run_wealth) source("wealth.R")
