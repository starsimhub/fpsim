###############################################################################
# Calculate Birth Spacing Intervals for Calibration
# Using Most Recent DHS IR Recode Data
#
# Creates: birth_spacing_dhs.csv
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
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}

# -------------------------------
# 2. Load and Prepare DHS Data
# -------------------------------

# multiple births = starts_with("b0")
# preceding birth intervals (in months) = starts_with("b11") 
#dhs_data <- read_dta(dhs_path,
#  col_select = c("v005", "v021", "v023", "v102", "caseid", starts_with("b0"),  
#    starts_with("b11")))

# Filter if region and region_code are defined
if (exists("region_variable") && exists("region") && exists("region_code")) {
  dhs_data <- read_dta(dhs_path,
                       col_select = c("v005", "v021", "v023", "v102", "caseid", starts_with("b0"),  
                                      starts_with("b11"), region_variable)) 
  dhs_data <- dhs_data %>% 
    filter(.data[[region_variable]] == region_code)
} else {
  dhs_data <- read_dta(dhs_path,
                       col_select = c("v005", "v021", "v023", "v102", "caseid", starts_with("b0"),  
                                    starts_with("b11"))) 
}

# Apply gather and mutate after optional filtering
dhs_data <- dhs_data %>%
  gather(var, val, -v005, -v021, -v023, -v102, -caseid) %>%
  separate(var, into = c("var", "num"), sep = "_") %>%
  spread(var, val) %>%
  filter(b0 < 2) %>%  # remove multiples
  mutate(
    space_mo = b11,                         # birth interval in months
    wt = v005 / 1e6,                        # DHS sample weight
    urban_rural = factor(v102, levels = c(1, 2), labels = c("urban", "rural"))
  )

# -------------------------------
# 3. Calculate Weighted Birth Spacing Table
# -------------------------------
design <- svydesign(
  id = ~v021,
  strata = ~v023,
  weights = ~wt,
  data = dhs_data
)

spacing <- svytable(~space_mo + urban_rural, design) %>%
  as.data.frame()

# Ensure weights column is always a float
spacing$Freq <- as.numeric(spacing$Freq)

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

write.csv(spacing, file.path(output_dir, "birth_spacing_dhs.csv"), row.names = FALSE)
