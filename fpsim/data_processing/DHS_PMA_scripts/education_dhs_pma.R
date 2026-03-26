################################################################################
# -- Extract education data from DHS and PMA data datasets
# From DHS (https://dhsprogram.com/) we use the IR (Individual Recode) to extract:
# - education objective (expressed in percentage of women that will aim to
#                        complete X years of education, sratified by type of
#                        residential setting).
# - education initialization (expressed in (edu) number of years a woman aged (age)
#                             will have likely completed).
#
# From PMA (https://datalab.pmadata.org/) we use all phases of the most recent year of data to extract:
# - education stopping criteria (expressed in percentage of women aged (age)
#                                years old and a given parity that would dropout
#                                from their education if they got pregnant.
#
# Creates: edu_initialization.csv, edu_objective.csv, edu_stop.csv
#
# ---------------------------------------------------------------------------
# Based on original analysis by Marita Zimmermann, August 2023
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
required_packages <- c("tidyverse", "haven", "survey", "withr")
installed_packages <- rownames(installed.packages())

for (pkg in required_packages) {
  if (!pkg %in% installed_packages) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}

# -------------------------------
# 2. Load and Process DHS Data
# -------------------------------

# -- Load all the data -- #
dhs_household_data <- read_dta(dhs_household_path)
# Filter if region and region_code are defined
if (exists("region_variable") && exists("region") && exists("region_code")) {
  dhs_data <- read_dta(dhs_path) %>% 
    filter(.data[[region_variable]] == region_code)
} else {
  dhs_data <- read_dta(dhs_path) 
}

# -- Make new columns with names that are more readable -- #
data <- dhs_data %>%
  mutate(
    age = v012,
    parity = v220,
    edu = v133,
    urban = ifelse(v025 == 1, 1, 0)
  )

svydesign_obj_1 = svydesign(
  id = data$v001,
  strata = data$v023,
  weights = data$v005 / 1000000,
  data = data
)

# -- preprocess household data
data.h <- dhs_household_data %>%
  select(starts_with("hv103"), starts_with("hv105"), starts_with("hv104"), starts_with("hv109"), 
         starts_with("hhid"), starts_with("hv001"), starts_with("hv023"), starts_with("hv005")) %>%
  gather(var, val, -c(hhid, hv001, hv023, hv005)) %>% mutate(num = substr(var,7,9), var = substr(var,1,5)) %>% spread(var, val) %>%
  filter(hv103 == 1 & hv105 %in% c(6:14) & hv104 == 2) %>% # de facto hh member, age 6-15, female
  rename(edu = hv109, age = hv105) %>%
  mutate(parity = 0) # assume girls under age 15 are parity 0
svydesign_obj_hh = svydesign(id = data.h$hv001, strata=data.h$hv023, weights = data.h$hv005/1000000, data=data.h)


# -- Preprocess education data -- #
table.edu.mean <-
  as.data.frame(svyby(~ edu, ~ age, svydesign_obj_1, svymean))

# Define some age-related constants
dhs_min_age <- 15
dhs_max_age <- 49
fpsim_max_age_pregnant <- 50  # Maximum age to become pregnant
fpsim_max_age <- 99

# Create projections for older and younger women by slope
age_lower_bound <- 20
age_upper_bound <- 43

table.edu.lag <- table.edu.mean %>%
  mutate(
    slope = (edu - lag(edu, 1)) / (age - lag(age, 1)),
    group = case_when(age < age_lower_bound ~ 1, age > age_upper_bound ~
                        2)
  ) %>%
  group_by(group) %>% mutate(avg.slope = ifelse(group == 1, mean(slope, na.rm = T), NA),
                             avg.edu = ifelse(group == 2, mean(edu), NA))
# Visualize education data
table.edu.mean %>%
  ggplot() +
  geom_line(aes(y = edu, x = age)) +
  geom_ribbon(aes(
    ymin = edu - se,
    ymax = edu + se,
    x = age
  ), alpha = 0.5) +
  geom_segment(aes(
    x = dhs_max_age + 1,
    y = min(table.edu.lag$avg.edu, na.rm = T),
    xend = 70,
    yend = min(table.edu.lag$avg.edu, na.rm = T)
  )) +
  geom_segment(aes(
    y = 0,
    x = dhs_min_age - min(table.edu.lag$edu, na.rm = T) / min(table.edu.lag$avg.slope, na.rm = T),
    xend = dhs_min_age,
    yend = min(table.edu.lag$edu, na.rm = T)
  )) +
  ylab("Mean years of education") + xlab("Age") +
  theme_bw(base_size = 13)


# Generate education table for initialization of education parameters
table.edu.inital <-
  data.frame(age = c(1:dhs_min_age - 1, dhs_max_age+1:fpsim_max_age)) %>%
  mutate(edu = ifelse(
    age < dhs_min_age,
    pmax(
      min(table.edu.lag$edu, na.rm = T) - min(table.edu.lag$avg.slope, na.rm = T) *
        (dhs_min_age - age),
      0
    ),
    min(table.edu.lag$avg.edu, na.rm = T)
  )) %>%
  bind_rows(table.edu.mean)

# For initialising education objective, we look at the distribution of
# years of education for all women over age 20 and living in urban settings
# (assumed that they have finished edu)
min_age <- 20
data.20 <- data %>% filter(age > min_age)
svydesign_obj_2 = svydesign(
  id = data.20$v001,
  strata = data.20$v023,
  weights = data.20$v005 / 1000000,
  data = data.20
)

table.edu.20 <-
  as.data.frame(svytable(~ edu + urban, svydesign_obj_2)) %>% group_by(urban) %>% mutate(percent = Freq /
                                                                                           sum(Freq)) %>% dplyr::select(-Freq)
current_levels_num <- sort(as.numeric(levels(table.edu.20$edu)))

# -- Fill out missing edu level values, edu=22 and edu=23 are not found
# in the data but fpsim needs a continuous range of edu values
all_levels <-
  expand.grid(edu = as.character(min(current_levels_num):max(current_levels_num)),
              urban = unique(table.edu.20$urban))

table.edu.20 <-
  right_join(table.edu.20, all_levels, by = c("edu", "urban"))
table.edu.20$percent[is.na(table.edu.20$percent)] <- 0

table.edu.20$edu <- as.numeric(as.character(table.edu.20$edu))
table.edu.20$urban <- as.numeric(as.character(table.edu.20$urban))

table.edu.20 <- table.edu.20 %>%
  arrange(urban, edu)

table.edu.20$urban <- as.factor(table.edu.20$urban)
table.edu.20$edu <- as.factor(table.edu.20$edu)

table.edu.20 %>%
  ggplot() +
  geom_line(aes(
    y = percent,
    x = edu,
    group = urban,
    color = urban
  )) +
  ylab("Percent of women") +
  theme_bw(base_size = 13)

# -- Create a table of the distribution of women by age and parity
# For age 15+
table.edu.ind <- as.data.frame(svytable(~age+edu+parity, svydesign_obj_1))

# For women who have a birth before age 15, create matrix for parity 1+ and age<15 using birth calendar
data.edu <- data %>%
  select(caseid, v001, v023, v005, v212, v011, age = v012, edu = v133, starts_with("b3"), starts_with("bord")) %>%
  filter(v212 < 15) %>% # age at first birth under 15
  gather(var, val, -caseid, -v212, -v011, -age, -edu, -v001, -v023, -v005) %>% separate(var, c("var", "order")) %>% spread(var, val) %>% # one row per pregnancy
  mutate(age_at_birth = floor((b3 - v011)/12)) %>%
  filter(age_at_birth < 15) %>% # look at birth under age 15
  mutate(edu_at_birth = pmin(age_at_birth - 6, edu)) # assume school starts at age 6, women continue education from then until age at birth or until their actual education level, whichever is less
svydesign_obj_3 = svydesign(id = data.edu$v001, strata=data.edu$v023, weights = data.edu$v005/1000000, data=data.edu)
table.edu.young <- as.data.frame(svytable(~age_at_birth+edu_at_birth+bord, svydesign_obj_3)) %>%
  rename(age = age_at_birth, edu = edu_at_birth, parity = bord)

# For 0 parity under age 15, create data frame for education for girls age 6-15 from household survey
table.edu.hh <- as.data.frame(svytable(~age+edu+parity, svydesign_obj_hh)) 

# combine age group tables
table.edu <- table.edu.ind %>% 
  bind_rows(table.edu.young) %>%
  bind_rows(table.edu.hh) %>%
  mutate(edu = as.numeric(as.character(edu)),
         age = as.numeric(as.character(age))) %>%
  group_by(age, parity) %>% arrange(-edu) %>%
  mutate(total = sum(Freq), sum = cumsum(Freq), cum.percent = sum/total) %>% # percentage with each year of education in the age/parity group
  select(-total, -sum, -Freq)


# -------------------------------
# 3. Load and Process PMA Data
# -------------------------------

# -- PMA DATA for education interruption following pregnancy
# We use PMA data because it has age at stopping education (DHS doesn't)
# Access PMA data here: https://www.pmadata.org/data/available-datasets
# Analysis uses the most recent three phases of the Household & Female survey
# Data available for Burkina Faso, Cote d'Ivoire, DRC, Ghana, India, Indonesia, Kenya, Niger, Nigeria, and Uganda

# Load multiple datasets and add a column wave to each of them
data1 <- read_dta(pma1_path)
data1 <- data1 %>% mutate(wave = 1, RE_ID = as.character(RE_ID), country = as.character(country), strata = as.character(strata))

data2 <- read_dta(pma2_path)
data2 <- data2 %>% mutate(wave = 2, RE_ID = as.character(RE_ID), country = as.character(country), strata = as.character(strata))

# Added flexibility for if 2, 3, or 4 PMA stata files are available
if (exists("pma3_path") && exists("pma4_path")){
  data3 <- read_dta(pma3_path)
  data3 <- data3 %>% mutate(wave = 3, RE_ID = as.character(doi_corrected), country = as.character(country), strata = as.character(strata))
  data4 <- read_dta(pma4_path)
  data4 <- data4 %>% mutate(wave = 4, RE_ID = as.character(doi_corrected), country = as.character(country), strata = as.character(strata))
  df_list <- list(data1, data2, data3, data4)
} else if (exists("pma3_path")){
  data3 <- read_dta(pma3_path)
  data3 <- data3 %>% mutate(wave = 3, RE_ID = as.character(doi_corrected), country = as.character(country), strata = as.character(strata))
  df_list <- list(data1, data2, data3)
} else {
  df_list <- list(data1, data2)
}

# Found that columns with "doi_corrected" and "consent_obtained"
# often were different data types, fixed with code below
df_list <- lapply(df_list, function(x) {
  cols_to_fix <- grep("doi_corrected", names(x), value = TRUE)
  for (col in cols_to_fix) {
    x[[col]] <- as.character(x[[col]])
  }
  return(x)
})
df_list <- lapply(df_list, function(x) {
  cols_to_fix <- grep("consent_obtained", names(x), value = TRUE)
  for (col in cols_to_fix) {
    x[[col]] <- as.character(x[[col]])
  }
  return(x)
})

data.raw.pma <- bind_rows(df_list)

# recode data for school and birth timing
weeks_in_a_year <- 52
data.pma <- data.raw.pma %>% filter(!is.na(FQweight)) %>%
  mutate(
    birthage1 = floor(as.numeric(
      difftime(
        parse_date_time(first_birth, "Y-m-d"),
        parse_date_time(birthdateSIF, "Y-m-d"),
        units = "weeks"
      ) / weeks_in_a_year
    )),
    # age at first birth
    school_birth1 = ifelse(between(
      school_left_age, birthage1 - 1, birthage1 + 1
    ), 1, 0),
    # stop school within one year of first birth
    school_birth1 = ifelse(is.na(school_birth1), 0, school_birth1),
    # replace NA with no
    birthage2 = floor(as.numeric(
      difftime(
        parse_date_time(recent_birthSIF, "Y-m-d"),
        parse_date_time(birthdateSIF, "Y-m-d"),
        units = "weeks"
      ) / weeks_in_a_year
    )),
    # age at most recent (but not first) birth
    school_birth2 = ifelse(between(
      school_left_age, birthage2 - 1, birthage2 + 1
    ), 1, 0),
    # stopped school within one year of recent birth
    school_birth2 = ifelse(is.na(school_birth2), 0, school_birth2),
    # replace NA with no
    recentbyoung = ifelse(recent_birthSIF != first_birthSIF &
                            birthage2 <= 22, 1, 0)
  ) %>% # was recent birth at age 22 or younger
  filter(birth_events > 0) # only use data for women who have had a birth


# EA_ID is not present in Nigeria PMA files
# Try using 'EA_ID' as the id, 
# if errors out because not in the PMA file, then try 'Cluster_ID'
if ("EA_ID" %in% names(data.pma)) {
  svydesign_obj_3 <- svydesign(
    id = ~ EA_ID,
    strata = ~ strata,
    weights = ~ FQweight,
    data = data.pma,
    nest = TRUE
  )
} else if ("Cluster_ID" %in% names(data.pma)) {
  svydesign_obj_3 <- svydesign(
    id = ~ Cluster_ID,
    strata = ~ strata,
    weights = ~ FQweight,
    data = data.pma,
    nest = TRUE
  )
} else {
  stop("Neither EA_ID nor Cluster_ID found in data.pma")
}

# table of probability of stopping school by age and parity
stop.school <-
  as.data.frame(prop.table(svytable( ~ school_birth1 + birthage1, svydesign_obj_3), margin = 2)) %>%
  filter(school_birth1 == 1) %>% rename(`1` = Freq, age = birthage1) %>% dplyr::select(-school_birth1) %>%
  left_join(
    stop.school2 <-
      as.data.frame(prop.table(
        svytable( ~ school_birth2 + birthage2, svydesign_obj_3), margin = 2
      )) %>% filter(school_birth2 == 1) %>% rename(`2+` = Freq, age = birthage2) %>% dplyr::select(-school_birth2)
  ) %>%
  gather(parity, percent,-age)
stop.school %>%
  ggplot() +
  geom_point(aes(y = percent, x = age, color = parity))


# -- Fill out missing percent values, age=40 and age=45 are not found, and we also have
# age 59 and 60, though their percent is NA. fpsim's max age to become pregnant is 50.
# Here we preprocess data to cover  the range between the minimum age present in the data and
# fpsim_max_age_pregnant
age_num <- sort(as.numeric(levels(stop.school$age)))

all_levels <-
  expand.grid(age = as.character(min(age_num):max(age_num)),
              parity = unique(stop.school$parity))

stop.school <-
  right_join(stop.school, all_levels, by = c("age", "parity"))
stop.school$percent[is.na(stop.school$percent)] <- 0

stop.school$age <- as.numeric(as.character(stop.school$age))
stop.school <- stop.school %>% filter(age <= fpsim_max_age_pregnant)

stop.school <- stop.school %>%
  arrange(parity, age)

stop.school$age <- as.factor(stop.school$age)
stop.school$parity <- as.factor(stop.school$parity)

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

write.csv(table.edu.inital, file.path(output_dir, "edu_initialization.csv"), row.names = FALSE)
write.csv(table.edu.20, file.path(output_dir, "edu_objective.csv"), row.names = FALSE)
write.csv(stop.school, file.path(output_dir, "edu_stop.csv"), row.names = FALSE)
write.csv(table.edu, file.path(output_dir, "education.csv"), row.names = F)

