###############################################################################
# Calculate Method Mix from DHS Data
# Using DHS individual recode (IR) data
#
# Creates: mix.csv, use.csv
###############################################################################

# -------------------------------
# 1. Setup
# -------------------------------

rm(list = ls())

# Load user configuration
source("./config.R")

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
data <- read_dta(dhs_path) %>%
  mutate(
    method = case_when(
      v312 == 1 ~ "Pill",
      v312 == 2 ~ "IUDs",
      v312 == 3 ~ "Injectables",
      v312 %in% c(4, 13, 15, 16, 17, 18) ~ "Other modern",
      v312 %in% c(5, 14) ~ "Condoms",
      v312 %in% c(6, 7) ~ "BTL",
      v312 %in% c(8, 10, 12) ~ "Other traditional",
      v312 == 9 ~ "Withdrawal",
      v312 == 11 ~ "Implants"
    ),
    use = ifelse(is.na(v312), NA, ifelse(v312 == 0, 0, 1)),
    wt = v005 / 1e6
  )

# -------------------------------
# 3. Create Survey Design Object
# -------------------------------
svydesign_obj <- svydesign(id = ~v021, strata = ~v023, weights = ~wt, data = data)

# -------------------------------
# 4. Calculate Method Mix
# -------------------------------
method_mix <- svytable(~method, svydesign_obj) %>%
  as.data.frame() %>%
  mutate(perc = Freq / sum(Freq) * 100)

# -------------------------------
# 5. Calculate current use
# -------------------------------
use <- svytable(~use, svydesign_obj) %>%
  as.data.frame() %>%
  mutate(perc = Freq / sum(Freq) * 100)

# -------------------------------
# 6. Save Output to Country Directory
# -------------------------------
output_dir <- file.path(output_dir, country)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

write.csv(method_mix, file.path(output_dir, "mix.csv"), row.names = FALSE)
write.csv(use, file.path(output_dir, "use.csv"), row.names = FALSE)
