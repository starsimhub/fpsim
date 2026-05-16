###############################################################################
# Create Switching Matrix for ABM
# Using DHS individual recode (IR) calendar data
#
# Creates: method_mix_matrix_switch.csv
# ----------------------------------------------------------------
# Author: Marita Zimmermann
###############################################################################

# -------------------------------
# 1. Setup
# -------------------------------

rm(list = ls())

# Load user configuration
source("./config.R")

required_packages <- c("tidyverse", "haven", "withr", "labelled",
                       "lubridate", "expm", "data.table")
installed_packages <- rownames(installed.packages())

for (pkg in required_packages) {
  if (!pkg %in% installed_packages) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}

# -------------------------------
# 2. Load DHS Calendar Data
# -------------------------------
# Filter if region and region_code are defined
if (exists("region_variable") && exists("region") && exists("region_code")) {
  dhs_data <- read_dta(dhs_path) %>% 
    filter(.data[[region_variable]] == region_code)
} else {
  dhs_data <- read_dta(dhs_path) 
}

# -------------------------------
# 3. Data Manipulation
#      Keep only calendar data, age, and parity
# -------------------------------

data <- dhs_data %>%
  mutate(wt = v005/1000000,
         age = as.numeric(v012),                                                                            # convert age to numeric
         sex.active = v527,                                                                                 # Time since the last sexual relations as reported by the respondent.  The first digit gives the units in which the respondent gave her answer:  1 - Days ago, 2 - Weeks ago, 3 - Months ago, 4 - Years ago, with 9 meaning a special answer was given.  The last two digits give the time in the units given.
         parity1=as.numeric(v220)) %>%                                                                      # convert parity to numeric, living children categorical 0-5, 6+
  dplyr::select(wt, age, parity1, vcal_1, sex.active) %>%
  mutate(cal = str_squish(vcal_1),                                                                          # remove whitespace
         cal = gsub("8","W",cal),                                                                           # recode contraceptive methods, Periodic abstinence/rhythm to other traditional
         cal = gsub("S","M",cal), cal = gsub("4", "M", cal), cal = gsub("C", "M", cal),                     # recode contraceptive methods to other modern, std days, diaphragm, female condom
         cal = gsub("E", "M", cal), cal = gsub("F", "M", cal), cal = gsub("7", "M", cal),                   # recode contraceptive methods to other modern, emergency contraception, foam and jelly, male sterilization
         cal_clean = sapply(strsplit(cal, split = ""),  function(str) {paste(rev(str), collapse = "")}),    # reverse string so oldest is first
         pattern_split = strsplit(cal_clean,""),                                                            # split into individual characters
         month.a = lapply(pattern_split, function(m) m[1:79]),                                              # codes for months 1-79
         month.b = lapply(pattern_split, function(m) m[2:80]),                                              # codes for months 2-80
         obs = 1:nrow(.)) %>%                                                                               # create obs variable to identify the individual
  unnest(c(month.a, month.b)) %>%                                                                           # create long format
  group_by(obs) %>% mutate(month = row_number(),                                                            # create variable for month from beginning of calendar
                           age = age - (79-month)/12,                                                       # adjust for age in calendar
                           age_grp = case_when(age <= 18 ~ "<18",                                           # define age groups
                                               age > 18 & age <= 20 ~ "18-20",
                                               age > 20 & age <= 25 ~ "20-25",
                                               age > 25 & age <= 35 ~ "25-35",
                                               age > 35 ~ ">35"),
                           month.6 = lead(month.a,6),                                                       # code for what happened 6 months after current month (for use in pp calculation)
                           age_grp.6 = lead(age_grp, 5),                                                    # code for age 6 months after current month (for use in pp calculation)
                           pp.6 = str_detect(substr(cal_clean, month-6, month), "B"),                       # postpartum in past 6 months
                           pp.12 = str_detect(substr(cal_clean, month-12, month), "B"),                     # postpartum in past 12 months
                           b.month = ifelse(month.a == "B", month, NA)) %>% fill(b.month) %>% mutate(       # month of most recent birth
                           month.pp = ifelse(pp.12, month-b.month, NA),                                     # number of months pp
                           termination = str_detect(substr(cal_clean, month-6, month), "T")) %>%            # termination within past 6 months
  filter(!is.na(month.a) & !is.na(month.b) & month.a != "P" & month.b != "P" &  month.a != "T" & month.b != "T" &  month.a != "L" & month.b != "L") %>% # remove missing values, remove pregnancy, remove lac.am, remove termination
  mutate_at(c("month.a", "month.b", "month.6"), ~factor(.,
                                         levels = c("0",    "1",    "2",   "3",          "5",      "6",               "9",          "N",       "W",          "M",         "B",     "A"),
                                         labels = c("None", "Pill", "IUD", "Injectable", "Condom", "F.sterilization", "Withdrawal", "Implant", "Other.trad", "Other.mod", "Birth", "Abstinence")))

# -------------------------------
# 4. Generate Switching Matrices
# -------------------------------

matrices <- data %>%
  mutate(postpartum = case_when(month.a == "Birth" ~ 1L, pp.6 == F & termination == F ~ 0L), From = month.a, To = month.b) %>%
  filter(!is.na(postpartum)) %>%                                                                            # Not using months 1-5 postpartum
  bind_rows(data %>% filter(month.a == "Birth" & !is.na(month.6) & month.6 != "Lac.Am" & month.6 != "Termination" & month.6 != "Pregnant") %>%  # Duplicate pp rows so we can have 6 month pp timepoint
              mutate(postpartum = 6L, age_grp = age_grp.6, From = month.b, To = month.6)) %>%              # For the 6 month pp matrices, replace age group with age group at the 6 month point
  filter(!(postpartum == 0L & sex.active > 311)) %>%                                                      # Filter out women in the non-postartum matrix who have not has sex in the past year+
  group_by(postpartum, From, age_grp) %>% mutate(n = sum(wt)) %>% ungroup %>%                               # sum total in 'from' each method
  group_by(postpartum, From, To, n, age_grp) %>% summarise(Freq = sum(wt)) %>% mutate(Freq = Freq/n) %>% ungroup %>%
  full_join(expand.grid(age_grp = unique(data$age_grp), From = unique(levels(data$month.a))[1:11], To = unique(levels(data$month.a))[1:11], postpartum = c(6L, 0L)) %>% # Add in any missing rows to make sure matrices will be square
              filter(From != 'Birth' & From != "None" & To != "Birth")) %>%                                 # Don't need from none or birth or to birth in the switching matrices so remove from new rows
  mutate(Freq = ifelse(is.na(Freq), ifelse(From == To, 1, 0), Freq), n = ifelse(is.na(n), 0, n)) %>%  ungroup %>% # for added rows, set p(x to x) to 1 and all others to 0, and set n to 0
  spread(To, Freq, fill = 0) %>% filter(rowSums(.[5:14], na.rm = T) > 0) %>%                                # spread to wide format by each method, for missing fill with 0, for some reason adding columns with 0 for everything so filtered those out
  filter(!(From == "F.sterilization" & F.sterilization == 0)) %>%                                           # one woman in <18 is going from sterilization to none and there's no other women in that category so filter her out)
  mutate(group = paste0("grp", postpartum, age_grp)) %>% as.data.table()                                    # create a group variable and make into a data table for matrix manipulation

# create matrix for switching methods at the end of time on method
matrices_switch <- data %>%
  mutate(postpartum = case_when(month.a == "Birth" ~ 1L, pp.6 == F & termination == F ~ 0L), From = month.a, To = month.b) %>%
  filter(!is.na(postpartum)) %>%                                                                            # Not using months 1-5 postpartum
  bind_rows(data %>% filter(month.a == "Birth" & !is.na(month.6) & month.6 != "Lac.Am" & month.6 != "Termination" & month.6 != "Pregnant") %>%  # Duplicate pp rows so we can have 6 month pp timepoint
              mutate(postpartum = 6L, age_grp = age_grp.6, From = month.b, To = month.6)) %>%              # For the 6 month pp matrices, replace age group with age group at the 6 month point
  filter(!(postpartum == 0L & sex.active > 311)) %>%                                                      # Filter out women in the non-postartum matrix who have not has sex in the past year+
  filter(!(postpartum == 6L & From != "None")) %>%                                                           # Only want from none for 6 mo pp
  filter(From != To) %>%                                                                                    # Take out same to same
  filter(To != "None") %>%                                                                                  # Don't want discontinuation in here
  group_by(postpartum, From, age_grp) %>% mutate(n = sum(wt)) %>% ungroup %>%                               # sum total in 'from' each method
  group_by(postpartum, From, To, n, age_grp) %>% summarise(Freq = sum(wt)) %>% mutate(Freq = Freq/n) %>% ungroup %>%
  spread(To, Freq, fill = 0)  %>%                                                                           # spread to wide format by each method, for missing fill with 0
  mutate(group = paste0("grp", postpartum, age_grp)) %>% as.data.table()                                    # create a group variable and make into a data table for matrix manipulation


# -------------------------------
# 5. Prepare and Save Output
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

# Save CSV
write.csv(matrices_switch, file.path(output_dir, "method_mix_matrix_switch.csv"), row.names = F)
