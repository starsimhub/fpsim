###############################################################################
# Postpartum Analysis: Sexual Activity & LAM by Months Postpartum
# Using DHS individual recode (IR) data
#
# Creates: sexually_active_pp.csv, lam.csv
# -----------------------------------------------------------------------------
# Author: Marita Zimmermann
###############################################################################

# -------------------------------
# 1. Setup
# -------------------------------

rm(list = ls())

# Load user configuration
source("./config.R")

required_packages <- c("tidyverse", "haven", "scales")
installed_packages <- rownames(installed.packages())

for (pkg in required_packages) {
  if (!pkg %in% installed_packages) install.packages(pkg)
  library(pkg, character.only = TRUE)
}

# -------------------------------
# 2. Load and Clean DHS Data
# -------------------------------
if (exists("region_variable") && exists("region") && exists("region_code")) {
  dhs_data <- read_dta(dhs_path) %>% 
    filter(.data[[region_variable]] == region_code)
} else {
  dhs_data <- read_dta(dhs_path) 
}
# Calculate postpartum months and filter for recent births
dhs.pp <- dhs_data %>%
  mutate(b19_01 = v008 - b3_01) %>% # have to calculate months pp
  filter(b19_01 < 24 & b9_01 == 0) %>% # keep if most recent birth is under 24 months, living with mother
  mutate(breastmilk = ifelse(m4_1 == 95, T, F), # breastfeeding https://dhsprogram.com/data/Guide-to-DHS-Statistics/Breastfeeding_and_Complementary_Feeding.htm#Calculation
         water = ifelse(v409 == 1, T, F),
         non.milk.liquids = ifelse(v409a == 1 | v410 == 1 | v410a == 1 | v412c == 1 | v413 == 1 | v413a == 1 | v413b == 1 | v413c == 1 | v413d == 1, T, F),
         non.milk.liquids = ifelse(is.na(non.milk.liquids), F, non.milk.liquids),
         other.milk = ifelse(v411 == 1 | v411a == 1, T, F),
         solids = ifelse(v412a == 1 | v412b == 1 | v414a == 1 | v414b == 1 | v414c == 1 | v414d == 1 | v414e == 1 | v414f == 1 | v414g == 1 | v414h == 1 | v414i == 1 | v414j == 1 | v414k == 1 | v414l == 1 | v414m == 1 | v414n == 1 | v414o == 1 | v414p == 1 | v414q == 1 | v414r == 1 | v414s == 1 | v414t == 1 | v414u == 1 | v414v == 1 | v414w == 1 | m39a_1 == 1, T, F),
         solids = ifelse(is.na(solids), F, solids),
         exclusive.bf = ifelse(breastmilk == T & non.milk.liquids == F & other.milk == F & solids == F, T, F), # exclusively breastfeeding (only breastmilk and plain water)
         amenorrhea = ifelse((m6_1 >= b19_01 & m6_1 < 96) | m6_1 == 96, T, F),
         abstinent = ifelse(m8_1 >= b19_01 & m8_1 <= 96, T, F), # if duration of abstinence is greater than months pp and
         exclusive.bf_amerorrhea = ifelse(exclusive.bf == T | amenorrhea == T, T, F),
         exclusive.bf_and_amerorrhea = ifelse(exclusive.bf == T & amenorrhea == T, T, F),
         exclusive.bf_amerorrhea_abstinent = ifelse(exclusive.bf_amerorrhea == T | abstinent == T, T, F),
         s.active_2 = case_when(v536 == 1 ~ T, v536 == 2 | v536 == 3 ~ F),
         wt = v005/1000000)

# -------------------------------
# 3. Summarize Results by Month Postpartum
# -------------------------------
dhs.pp.results <- dhs.pp %>%
  rename(months_postpartum = b19_01) %>%
  group_by(months_postpartum) %>%
  mutate_at(c("exclusive.bf", "amenorrhea", "exclusive.bf_and_amerorrhea"), ~ifelse(breastmilk, ., NA)) %>% # take out values for those who are not breastfeeding for these becasue we want percentages among breastfeeding women for the model
  summarise(abstinent = weighted.mean(abstinent, wt, na.rm = T), s.active_1 = 1-abstinent,
            s.active_2 = weighted.mean(s.active_2, wt, na.rm = T),
            exclusive.bf = weighted.mean(exclusive.bf, wt, na.rm = T),
            amenorrhea = weighted.mean(amenorrhea, wt, na.rm = T),
            exclusive.bf_and_amerorrhea = weighted.mean(exclusive.bf_and_amerorrhea, wt, na.rm = T),
            n = sum(!is.na(abstinent)))

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

# Sexual Activity Output: Percentage by month postpartum
  # method 1 is using duration of postpartum abstinence with months pp, and method 2 is using sexually active in last 4 weeks with months pp
  # Note that method two overlooks if someone was sexually active pp, then stopped having sex in the last month
  # We are currently using method 2 in the model
sexually.active.results <- dhs.pp.results %>% dplyr::select(months_postpartum, n, s.active_1, s.active_2)
sexually.active.model <- sexually.active.results %>%
  dplyr::select(month = months_postpartum, probs = s.active_2)
write.csv(sexually.active.model, file.path(output_dir, "sexually_active_pp.csv"), row.names = F)

# LAM output
  # percentage of women on LAM by month pp (breastfeeding, amenorrhea, and both)
LAM.results <- dhs.pp.results %>% dplyr::select(months_postpartum, n, exclusive.bf, amenorrhea, exclusive.bf_and_amerorrhea)
LAM.model <- LAM.results %>% filter(months_postpartum<12) %>% dplyr::select(month = months_postpartum, rate = exclusive.bf_and_amerorrhea)

write.csv(LAM.model, file.path(output_dir, "lam.csv"), row.names = FALSE)
