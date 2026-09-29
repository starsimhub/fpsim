###############################################################################
# Estimate Breastfeeding Duration from DHS Data (IR Recode Data)
# Using survey-weighted analysis and truncated normal distribution
#
# Creates: bf_stats.csv
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
required_packages <- c("tidyverse", "withr", "survey", "fitdistrplus", "truncdist", "readstata13", "haven")
installed_packages <- rownames(installed.packages())

for (pkg in required_packages) {
  if (!pkg %in% installed_packages) {
    install.packages(pkg, repos = "https://cloud.r-project.org/")
  }
  library(pkg, character.only = TRUE)
}

# -------------------------------
# 2. Load and Clean DHS Data
# -------------------------------

# Read DHS individual recode file (must include m4_* and b19_* for breastfeeding analysis)
# Filter if region and region_code are defined
if (exists("region_variable") && exists("region") && exists("region_code")) {
  dhs_data <- read_dta(dhs_path) %>% 
    filter(.data[[region_variable]] == region_code)
} else {
  dhs_data <- read_dta(dhs_path) 
}


# Select and process relevant variables
dat_nonmiss <- dhs_data %>%
  # select categories of breastfeeding, months breastfed, age of baby in months, and survey variables
  dplyr::select(starts_with("m4_"), starts_with("b19_"), v005, v021, v023, caseid) %>%
  # make long format with a row for each birth
  gather(var, val, -v005, -v021, -v023, -caseid) %>% separate(var, into = c("var", "birth")) %>% mutate(birth = as.numeric(birth)) %>% spread(var, val) %>%
  mutate(age_months = as.numeric(b19), # ensure age is numeric
         wt = v005/1e6, # calucalte survey weight
         still_bf = ifelse(m4 == 95 | m4 == "still breastfeeding", 1, 0), # create indicator for still breastfeeding, 'still breastfeeding' is numeric code = 95
         age_grp = floor(age_months / 2) * 2) %>% # 2 month age bins (it's too noisy with individual age bins)
  filter(age_months < 36 & !is.na(m4))  # include babies less than 3 years old and non-missing data
#print(dat_nonmiss)

# -------------------------------
# 3. Calculate Survey-Weighted BF Proportions
# -------------------------------

# survey weighted table of proportion still breastfeeding of that age group
summary_table <- svyby(~still_bf, ~age_grp,
                       svydesign(id=~v021, weights=~wt, strata=~v023, data = dat_nonmiss),
                       FUN = svymean) %>%
  arrange(age_grp) %>% mutate(p_stop = lag(still_bf) - still_bf,  # estimate of stopping probability, basically the PDF
                              counts = ifelse(p_stop<0, 0, round(p_stop*10000))) %>% # create pseudo-counts to simulate a sample size, change and negatives to 0
  filter(!is.na(p_stop)) # remove month 0
#print(summary_table)

# -------------------------------
# 4. Fit Truncated Normal Distribution
# -------------------------------

# Create a vector of repeated ages
ages_rep <- rep(summary_table$age_grp, summary_table$counts)
#print(ages_rep)
# Fit truncated normal distribution
fit_tnorm <- fitdist(ages_rep, "norm", start = list(mean = mean(ages_rep), sd = sd(ages_rep)))
summary(fit_tnorm)

# -------------------------------
# 5. Validate Fit Results
# -------------------------------

# Check if fitting was successful and produced valid results
if (is.null(fit_tnorm) || is.null(fit_tnorm$estimate) || any(is.na(fit_tnorm$estimate)) || any(is.infinite(fit_tnorm$estimate))) {
  cat("\n=== WARNING: BREASTFEEDING DATA VALIDATION ===\n")
  cat("ERROR: Distribution fitting failed or produced invalid results!\n")
  cat("Potential causes:\n")
  cat("- Insufficient breastfeeding data in survey (fewer than ~10 observations)\n")
  cat("- Missing m4_* variables (breastfeeding status) in DHS data\n") 
  cat("- Missing b19_* variables (child age in months) in DHS data\n")
  cat("- All children in survey are still breastfeeding (no variation in cessation)\n")
  cat("- Survey weights resulted in zero or negative stopping probabilities\n")
  cat("- Data quality issues (e.g., unrealistic ages or breastfeeding durations)\n")
  cat(sprintf("Data summary: %d total records, %d after filtering, %d age groups with data\n", 
              nrow(dhs_data), nrow(dat_nonmiss), nrow(summary_table)))
  if (exists("ages_rep")) {
    cat(sprintf("Distribution fitting attempted on %d data points\n", length(ages_rep)))
  }
  cat("No bf_stats.csv file will be created.\n")
  cat("=== END WARNING ===\n\n")
  
  # Exit script early
  stop("Script terminated due to distribution fitting failure")
} else {
  cat("✓ Breastfeeding distribution fitting successful\n")
}


# -------------------------------
# 6. Save Output to Country Directory
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
estimates <- data.frame(
  parameter = names(fit_tnorm$estimate),
  value = as.numeric(fit_tnorm$estimate)
)

write.csv(estimates, file.path(output_dir, "bf_stats.csv"), row.names = FALSE)
