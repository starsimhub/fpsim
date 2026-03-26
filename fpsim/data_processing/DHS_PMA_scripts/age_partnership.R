###############################################################################
# Calculate the proportion of women entering partnership by age
# Using DHS individual recode (IR) data
#
# Creates: age_partnership.csv
###############################################################################

# -------------------------------
# 1. Setup
# -------------------------------

# Clear environment (preserve run control variables)
run_vars <- ls(pattern = "^run_")
rm(list = setdiff(ls(), run_vars))

# Load user configuration
source("./config.R")

# Install and load required packages
required_packages <- c("tidyverse", "withr", "haven", "survey")
installed_packages <- rownames(installed.packages())

for (pkg in required_packages) {
  if (!pkg %in% installed_packages) {
    install.packages(pkg, repos = "https://cloud.r-project.org/")
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
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
  mutate(age_partner = v511)  # v511: age at first cohabitation

# -------------------------------
# 3. Create Survey Design Object
# -------------------------------
svydesign_obj = svydesign(id = data$v001, strata=data$v023, weights = data$v005/1000000, data=data)  # Ensure means and SEM are properly adjusted for the survey design

# -------------------------------
# 4. Calculate Proportion by Age at Partnership
# -------------------------------
table_partnership <- as.data.frame(svytable(~age_partner, svydesign_obj)) %>%
  mutate(percent = Freq/sum(Freq)) %>% dplyr::select(-Freq)

# -------------------------------
# 5. Save Output to Country Directory
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

write.csv(table_partnership, file.path(output_dir, "age_partnership.csv"), row.names = FALSE)
