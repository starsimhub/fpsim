###############################################################################
# Calculate Sexual Activity by Age Group
# DHS IR Data: Percent of women active in past 4 weeks among those ever active
#
# Creates: sexually_active.csv
###############################################################################

# -------------------------------
# 1. Setup
# -------------------------------

rm(list = ls())

# Load user configuration
source("./config.R")

# Install and load required packages
required_packages <- c("tidyverse", "haven", "survey", "dplyr")
installed_packages <- rownames(installed.packages())
for (pkg in required_packages) {
  if (!pkg %in% installed_packages) install.packages(pkg)
  library(pkg, character.only = TRUE)
}

# -------------------------------
# 2. Load and Clean Data
# -------------------------------
# Filter if region and region_code are defined
if (exists("region_variable") && exists("region") && exists("region_code")) {
  dhs_data <- read_dta(dhs_path,
                       col_select = c("v005", "v021", "v023", "v012", "v536", region_variable)) 
  dhs_data <- dhs_data %>% 
    filter(.data[[region_variable]] == region_code)
} else {
  dhs_data <- read_dta(dhs_path,
                       col_select = c("v005", "v021", "v023", "v012", "v536")) 
}

dhs_data <- dhs_data %>%
  mutate(
    active = case_when(
      is.na(v536) | v536 == 0 ~ NA_real_,  # never had sex or missing
      v536 == 1 ~ 1,                      # active in last 4 weeks
      TRUE ~ 0                            # not active in last 4 weeks
    ),
    age = cut(v012, breaks = seq(0, 55, by = 5), right = FALSE, labels = seq(0, 50, by = 5)),
    wt = v005 / 1e6
  )

# -------------------------------
# 3. Create Survey Design and Calculate Activity Rates
# -------------------------------
survey_design <- svydesign(
  id = ~v021,
  strata = ~v023,
  weights = ~wt,
  data = dhs_data
)

# Calculate percent active in past 4 weeks among ever active, by age group
activity_summary <- as.data.frame(svytable(~active + age, survey_design)) %>%
  group_by(age) %>%
  mutate(
    probs = Freq / sum(Freq) * 100,
    probs = replace_na(probs, 0)
  ) %>%
  filter(active == 1) %>%
  dplyr::select(age, probs)

# Convert age to numeric and remove duplicate 50s
activity_summary <- activity_summary %>%
  mutate(age = as.integer(as.character(age))) %>%
  distinct(age, .keep_all = TRUE)

# Add a row for age 50 only if it doesn't already exist
if (!50 %in% activity_summary$age) {
  age_50_row <- activity_summary %>% filter(age == max(age)) %>% mutate(age = 50)
  activity_summary <- bind_rows(activity_summary, age_50_row)
}

# -------------------------------
# 4. Save Output to Country Directory
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

write.csv(activity_summary, file.path(output_dir, "sexually_active.csv"), row.names = FALSE)
