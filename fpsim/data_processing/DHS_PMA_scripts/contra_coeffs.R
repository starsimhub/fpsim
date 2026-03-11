###############################################################################
# Generate Contraceptive Use Coefficients from DHS Data
# Using DHS individual recode (IR) data
# -----------------------------------------------------------------------------
# Outputs:
# For Simple Choice Model:
# - contra_coef_simple.csv
# - contra_coef_simple_pp1.csv
# - contra_coef_simple_pp6.csv
#
# For Standard Choice Model:
# - contra_coef_mid.csv
# - contra_coef_mid_pp1.csv
# - contra_coef_mid_pp6.csv
# - splines_25_40.csv
#
###############################################################################

# -------------------------------
# 1. Setup
# -------------------------------

# Clear environment (preserve run control variables)
run_vars <- ls(pattern = "^run_")
rm(list = setdiff(ls(), run_vars))

# Load user configuration
source("./config.R")

# Install packages
required_packages <- c(
  "tidyverse", "haven", "survey", "splines", "glmnet", "boot",
  "lavaan", "tidySEM", "ggpubr", "extrafont", "ggalluvial"
)
installed <- rownames(installed.packages())
for (pkg in required_packages) {
  if (!pkg %in% installed) install.packages(pkg, repos = "https://cloud.r-project.org/")
  library(pkg, character.only = TRUE)
}
options(survey.lonely.psu = "adjust")

# -------------------------------
# 2. Load and Clean Data
# -------------------------------
# Filter if region and region_code are defined
if (exists("region_variable") && exists("region") && exists("region_code")) {
  dhs_data <- read_dta(dhs_path) %>% 
    filter(.data[[region_variable]] == region_code)
} else {
  dhs_data <- read_dta(dhs_path) 
}


All_data <- dhs_data %>%
  mutate(
    age = as.numeric(v012),
    age_grp = cut(age, c(0, 18, 20, 25, 35, 50)),
    live_births = as.numeric(v219),
    urban = ifelse(v025 == 1, 1, 0),
    pp.time = v222,
    pregnant = v213,
    fp_ever_user = ifelse(v302a %in% c(1, 2), 1, 0),
    yrs.edu = ifelse(v133 == 98, NA, v133),
    edu.level = factor(case_when(
      v106 == 0 ~ "None",
      v106 == 1 ~ "Primary",
      v106 %in% c(2, 3) ~ "Secondary"
    ), levels = c("None", "Primary", "Secondary")),
    wealthquintile = v190,
    current_contra = ifelse(v312 == 0, 0, 1),
    cal = str_squish(substr(vcal_1, 1, 36)),
    first_char = substr(cal, 1, 1),
    chars = strsplit(cal, ""),
    repeat_end = map2_int(chars, first_char, ~ match(TRUE, .x != .y, nomatch = length(.x) + 1)),
    remainder = ifelse(repeat_end > str_length(cal), "", map2_chr(chars, repeat_end, ~ paste0(.x[.y:length(.x)], collapse = ""))),
    prior_user = ifelse(vcal_1 == "", NA,
                        ifelse(repeat_end > str_length(cal) & first_char != "0", TRUE,
                               map_lgl(remainder, ~ str_detect(.x, "[^0LPTB]"))))
  )

# -------------------------------
# 3. Create Survey Design Objects
# -------------------------------
filter.data.notpp <- All_data %>% filter((pp.time > 6 | is.na(pp.time)) & pregnant == 0)
filter.data.pp1 <- All_data %>% filter(pp.time < 2)
filter.data.pp6 <- All_data %>% filter(pp.time < 6)

# create survey objects
options(survey.lonely.psu="adjust")
svydes = svydesign(id = ~v001, strata = ~v023, weights = filter.data.notpp$v005/1000000, data=filter.data.notpp, nest = T)
svydes.pp1 = svydesign(id = ~v001, strata = ~v023, weights = filter.data.pp1$v005/1000000, data=filter.data.pp1, nest = T)
svydes.pp6 = svydesign(id = ~v001, strata = ~v023, weights = filter.data.pp6$v005/1000000, data=filter.data.pp6, nest = T)

# -------------------------------
# 4. Define Model Functions
# -------------------------------
simple.function <- function(svychoice) {
  model <- svyglm(current_contra ~ age_grp * prior_user, family = quasibinomial(), design = svychoice)
  summary(model)$coefficients %>%
    as.data.frame() %>%
    rownames_to_column("rhs")
}

standard.function <- function(svychoice) {
  model <- svyglm(current_contra ~ ns(age, knots = c(25, 40)) * prior_user +
                    edu.level + live_births + urban + wealthquintile,
                  family = quasibinomial(), design = svychoice)
  summary(model)$coefficients %>%
    as.data.frame() %>%
    rownames_to_column("rhs") %>%
    mutate(rhs = rhs %>%
             str_replace_all("live_births", "parity") %>%
             str_replace_all("yrs.edu", "edu_attainment") %>%
             str_replace_all("current_contra", "contraception"))
}

# -------------------------------
# 5. Validate Model Results Functions
# -------------------------------

validate_simple_coef_structure <- function(coef_df, model_name) {
  # Extract age group coefficients (not interactions)
  age_grp_coefs <- coef_df %>%
    filter(str_detect(rhs, "^age_grp") & !str_detect(rhs, ":"))
  
  # Extract age group x prior_user interaction coefficients
  interaction_coefs <- coef_df %>%
    filter(str_detect(rhs, "age_grp.*:.*prior_userTRUE"))
  
  n_age_grps <- nrow(age_grp_coefs)
  n_interactions <- nrow(interaction_coefs)
  
  if (n_age_grps != n_interactions) {
    cat(sprintf("\n=== WARNING: %s COEFFICIENT VALIDATION ===\n", toupper(model_name)))
    cat("WARNING: Inconsistent age group coefficient structure!\n")
    cat(sprintf("Age group coefficients: %d\n", n_age_grps))
    cat(sprintf("Age group x prior_user interactions: %d\n", n_interactions))
    cat("\nAge group coefficients found:\n")
    cat(paste("- ", age_grp_coefs$rhs, collapse = "\n"), "\n")
    cat("\nInteraction coefficients found:\n")
    cat(paste("- ", interaction_coefs$rhs, collapse = "\n"), "\n")
    cat("\nThis could indicate:\n")
    cat("- Missing age groups in the data\n")
    cat("- Model convergence issues for specific age x prior_user combinations\n")
    cat("- Insufficient data for some age group x prior_user interactions\n")
    cat("- Data filtering removed observations from certain age groups\n")
    cat("=== END WARNING ===\n\n")
  } else {
    cat(sprintf("✓ %s coefficient structure valid: %d age groups, %d interactions\n", 
                model_name, n_age_grps, n_interactions))
  }
}

validate_standard_coef_structure <- function(coef_df, model_name) {
  # Extract spline basis coefficients (not interactions)
  spline_coefs <- coef_df %>%
    filter(str_detect(rhs, "^ns\\(age") & !str_detect(rhs, ":"))
  
  # Extract spline x prior_user interaction coefficients
  interaction_coefs <- coef_df %>%
    filter(str_detect(rhs, "ns\\(age.*:.*prior_userTRUE"))
  
  n_splines <- nrow(spline_coefs)
  n_interactions <- nrow(interaction_coefs)
  
  if (n_splines != n_interactions) {
    cat(sprintf("\n=== WARNING: %s COEFFICIENT VALIDATION ===\n", toupper(model_name)))
    cat("WARNING: Inconsistent spline coefficient structure!\n")
    cat(sprintf("Spline basis coefficients: %d\n", n_splines))
    cat(sprintf("Spline x prior_user interactions: %d\n", n_interactions))
    cat("\nSpline coefficients found:\n")
    cat(paste("- ", spline_coefs$rhs, collapse = "\n"), "\n")
    cat("\nInteraction coefficients found:\n")
    cat(paste("- ", interaction_coefs$rhs, collapse = "\n"), "\n")
    cat("\nThis could indicate:\n")
    cat("- Model convergence issues for spline x prior_user interactions\n")
    cat("- Insufficient data for some spline basis x prior_user combinations\n")
    cat("- Numerical instability in spline fitting\n")
    cat("- Prior user variable has insufficient variation across age range\n")
    cat("=== END WARNING ===\n\n")
  } else {
    cat(sprintf("✓ %s coefficient structure valid: %d spline basis terms, %d interactions\n", 
                model_name, n_splines, n_interactions))
  }
}

# -------------------------------
# 6. Run and Save Models
# -------------------------------
# Create country-based output directory if it doesn't exist
if (exists("region") && exists("region_code")) {
  output_dir <- file.path(output_dir, paste0(country, "_", region), 'data')
} else {
  output_dir <- file.path(output_dir, country, 'data')
}

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

if (model_type %in% c("simple", "both")) {
  simple_coef <- simple.function(svydes)
  simple_coef_pp1 <- simple.function(svydes.pp1)
  simple_coef_pp6 <- simple.function(svydes.pp6)
  
  # Validate coefficient structures
  validate_simple_coef_structure(simple_coef, "contra_coef_simple")
  validate_simple_coef_structure(simple_coef_pp1, "contra_coef_simple_pp1")
  validate_simple_coef_structure(simple_coef_pp6, "contra_coef_simple_pp6")
  
  # Save results
  write_csv(simple_coef, file.path(output_dir, "contra_coef_simple.csv"))
  write_csv(simple_coef_pp1, file.path(output_dir, "contra_coef_simple_pp1.csv"))
  write_csv(simple_coef_pp6, file.path(output_dir, "contra_coef_simple_pp6.csv"))
}

if (model_type %in% c("standard", "both")) {
  standard_coef <- standard.function(svydes)
  standard_coef_pp1 <- standard.function(svydes.pp1)
  standard_coef_pp6 <- standard.function(svydes.pp6)
  
  # Validate coefficient structures
  validate_standard_coef_structure(standard_coef, "contra_coef_mid")
  validate_standard_coef_structure(standard_coef_pp1, "contra_coef_mid_pp1")
  validate_standard_coef_structure(standard_coef_pp6, "contra_coef_mid_pp6")
  
  # Save results
  write_csv(standard_coef, file.path(output_dir, "contra_coef_mid.csv"))
  write_csv(standard_coef_pp1, file.path(output_dir, "contra_coef_mid_pp1.csv"))
  write_csv(standard_coef_pp6, file.path(output_dir, "contra_coef_mid_pp6.csv"))

  # Create and save natural spline basis for ages 15 to 49 with internal knots at 25 and 40
  age_range <- 15:49
  splines_df <- as.data.frame(ns(age_range, knots = c(25, 40)))

  # Rename the spline basis columns for clarity
  names(splines_df) <- c("knot_1", "knot_2", "knot_3")

  # Add a column for the corresponding ages
  splines_df$age <- age_range

  # Write to CSV
  write_csv(splines_df, file.path(output_dir, "splines_25_40.csv"))
}
