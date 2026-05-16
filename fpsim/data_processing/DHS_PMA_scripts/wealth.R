###############################################################################
# Extract Empowerment Indicators from DHS Data
# Using DHS individual recode (IR) data
# -----------------------------------------------------------------------------
# Creates: wealth.csv: Percent in each wealth quintile (v190)
#
###############################################################################

# -------------------------------
# 1. Setup
# -------------------------------

rm(list = ls())

# Load user configuration
source("./config.R")

required_packages <- c("tidyverse", "haven", "survey")
installed <- rownames(installed.packages())
for (pkg in required_packages) {
  if (!pkg %in% installed) install.packages(pkg)
  library(pkg, character.only = TRUE)
}

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

data <- dhs_data %>%
  mutate(
    age = v012,
    parity = v220,
    edu = v133,
    paid_employment = case_when(
      v731 %in% c(1, 2, 3) & v741 %in% c(1, 2, 3) ~ 1,
      v731 == 0 | v741 == 0 ~ 0
    ),
    decision.wages = case_when(
      v739 %in% c(1, 2, 3) ~ 1,
      v739 %in% c(4, 5) ~ 0
    ),
    decision_wages = case_when(
      paid_employment == 1 & decision.wages == 1 ~ 1,
      paid_employment == 0 | decision.wages == 0 ~ 0
    ),
    sexual_autonomy = case_when(
      v850a == 1 ~ 1,
      v850a == 8 ~ 0.5,
      v850a == 0 ~ 0
    ),
    decision_health = case_when(
      v743a %in% c(1, 2, 3) ~ 1,
      v743a %in% c(4, 5) ~ 0
    ),
    decision_purchase = case_when(
      v743b %in% c(1, 2, 3) ~ 1,
      v743b %in% c(4, 5) ~ 0
    ),
    urban = ifelse(v025 == 1, 1, 0),
    age_partner = v511
  )

# -------------------------------
# 3. Create Survey Design Object
# -------------------------------
svydes = svydesign(id = data$v001, strata=data$v023, weights = data$v005/1000000, data=data)

# -------------------------------
# 4. Calculate Wealth Quintile Proportions
# -------------------------------
table_wealth <- as.data.frame(svytable(~v190, svydes)) %>%
  mutate(percent = Freq / sum(Freq)) %>%
  rename(quintile = v190) %>%
  dplyr::select(quintile, percent)

# -------------------------------
# 5. Save Output
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

write_csv(table_wealth, file.path(output_dir, "wealth.csv"))
