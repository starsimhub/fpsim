###############################################################################
# Calculate the proportion of women living in urban areas
# Using DHS individual recode (IR) data
#
# Creates: urban.csv
###############################################################################

# -------------------------------
# 1. Setup
# -------------------------------

# Clear environment
rm(list = ls())

# Load user configuration
source("./config.R")

# Install and load required packages
required_packages <- c("tidyverse", "haven", "survey")
installed_packages <- rownames(installed.packages())

for (pkg in required_packages) {
  if (!pkg %in% installed_packages) {
    install.packages(pkg)
  }
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
  mutate(urban = ifelse(v025 == 1, 1, 0))  # v025: 1 = urban, 2 = rural

# -------------------------------
# 3. Create Survey Design Object
# -------------------------------
svydesign_obj = svydesign(id = data$v001, strata=data$v023, weights = data$v005/1000000, data=data)

# -------------------------------
# 4. Calculate Proportion Urban
# -------------------------------
urban_table <- as.data.frame(svymean(~urban, svydesign_obj)) %>% rename(urban.se = urban)  # Rename standard error column for clarity

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

write.csv(urban_table, file.path(output_dir, "urban.csv"), row.names = FALSE)
