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

rm(list = ls())  # Clear environment

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
# 5. Run and Save Models
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
  write_csv(simple.function(svydes), file.path(output_dir, "contra_coef_simple.csv"))
  write_csv(simple.function(svydes.pp1), file.path(output_dir, "contra_coef_simple_pp1.csv"))
  write_csv(simple.function(svydes.pp6), file.path(output_dir, "contra_coef_simple_pp6.csv"))
}

if (model_type %in% c("standard", "both")) {
  write_csv(standard.function(svydes), file.path(output_dir, "contra_coef_mid.csv"))
  write_csv(standard.function(svydes.pp1), file.path(output_dir, "contra_coef_mid_pp1.csv"))
  write_csv(standard.function(svydes.pp6), file.path(output_dir, "contra_coef_mid_pp6.csv"))

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
