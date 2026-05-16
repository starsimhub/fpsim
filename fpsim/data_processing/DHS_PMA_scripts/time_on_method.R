##############################################################################
# -- Analysis of time on contraceptive method using DHS calendar data
# Using DHS individual recode (IR) data
#
# Creates: method_time_coefficients.csv
##############################################################################

# -------------------------------
# 1. Setup
# -------------------------------

rm(list = ls())

# Load user configuration
source("./config.R")

required_packages <- c("survival", "ggplot2", "survey", "ciTools", "tidyverse",
                       "survminer", "flexsurv", "withr", "haven", "data.table",
                       "mvtnorm", "extrafont", "dplyr")

installed_packages <- rownames(installed.packages())

for (pkg in required_packages) {
  if (!pkg %in% installed_packages) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}

# -------------------------------
# 2. Data Cleaning
# -------------------------------

# List of variables we want to use
varlist <- c("v000", #country-survey code
             "v001", #cluster
             "v005", #ind weights
             "v008", #date of interview (CMC)
             "v011", #respondent's age (CMC)
             "v012", #respondent age
             "v021", #PSU
             "v022", #strata
             "v101", #region
             "v130",
             "v133", #education in years
             "v212", #age at first birth
             "v218", #living children
             "v190", #wealth index
             "vcal_1", #contraceptive calendar
             "v025") #urban/rural

# Read and process IR data
# Filter if region and region_code are defined
if (exists("region_variable") && exists("region") && exists("region_code")) {
  dhs_data <- read_dta(dhs_path,
                       col_select = any_of(c(varlist, region_variable))) 
  dhs_data <- dhs_data %>% 
    filter(.data[[region_variable]] == region_code)
} else {
  dhs_data <- read_dta(dhs_path,
                       col_select = any_of(varlist)) 
}

# -- Survival analysis data organization -- #
data.surv <- dhs_data %>%
  mutate(cc = str_sub(v000,1,2),
         wt = v005/1000000,
         urban = ifelse(v025 == 1, 1, 0),
         live_births = v218,
         wealth_index = v190,
         edu_years = ifelse(v133 == 99, NA, v133),
         age = as.numeric(v012)) %>%                                                                        # convert age to numeric
  filter(age < 50 & !is.na(vcal_1) & vcal_1 != "") %>%                                                      # Keep only women of reproductive age with calendar data
  dplyr::select(wt, age, live_births, edu_years, wealth_index, urban, vcal_1, v000, v021, v022) %>%
  mutate(cal = str_squish(vcal_1),                                                                          # remove whitespace
         cal = gsub("8","W",cal),                                                                           # recode contraceptive methods, Periodic abstinence/rhythm to other traditional
         cal = gsub("S","M",cal), cal = gsub("4", "M", cal), cal = gsub("D", "M", cal), cal = gsub("C", "M", cal),  # recode contraceptive methods to other modern, std days, diaphragm x2, female condom
         cal = gsub("E", "M", cal), cal = gsub("F", "M", cal), cal = gsub("7", "M", cal),                   # recode NUR-ISTERATE from SA to injectable, per https://userforum.dhsprogram.com/index.php?t=msg&goto=25256&&srch=vcal#msg_25256
         cal = gsub("I","3",cal),                                                                           # recode contraceptive methods, Periodic abstinence/rhythm to other traditionalhttp://127.0.0.1:42617/graphics/plot_zoom_png?width=2560&height=1378
         cal_clean = sapply(strsplit(cal, split = ""),  function(str) {paste(rev(str), collapse = "")}),    # reverse string so oldest is first
         pattern_split = strsplit(cal_clean,""),                                                            # split into individual characters
         method = lapply(pattern_split, function(m) m[1:80]),                                               # codes for months
         obs = 1:nrow(.)) %>%                                                                               # create obs variable to identify the individual
  unnest(c(method)) %>%                                                                                     # create long format
  group_by(obs) %>% mutate(month = row_number(),                                                            # create variable for month from beginning of calendar
                           age = age - (79-month)/12,                                                       # adjust for age in calendar
                           age_grp = cut(age, c(0,18,20,25,35,50)),                                            # define age groups
                           age_grp_fact = factor(age_grp,levels=c('(20,25]','(0,18]','(18,20]','(25,35]','(35,50]')), # factor age group
                           new.method = ifelse(method != lag(method,1) | month == 1,1,0),                   # Indicator for a month where a new method is started (the first month of a switch/discontinue), with methods including pregnant, none, etc
                           method.month = rowid(method, cumsum(new.method == 1)),                           # Months on individual method
                           on_method = ifelse(method %in% c("1", "2", "3", "5", "9", "N", "W", "M"), 1, 0), # Identify relevant methods
                           cumulative_method_months = cumsum(on_method),                                    # Cumulative months for relevant methods
                           methods_so_far = purrr::accumulate(method, ~ unique(c(.x, .y))),                 # List of methods up to current month
                           methods_so_far = purrr::map(methods_so_far, ~ .x[.x %in% c("1", "2", "3", "5", "9", "N", "W", "M")]),
                           unique_methods_count = sapply(methods_so_far, length),                           # count unique methods
                           births = cumsum(method == "B"),                                                  # count births
                           parity = ifelse(live_births - births<0, 0, live_births - births),
                           expected_edu = round(age,0) - 5,
                           edu = ifelse(edu_years<expected_edu, edu_years, expected_edu)) %>%
  filter(lead(method.month, 1) == 1) %>%                                                                    # Keep only last row of each method with number of months on that method
  mutate(discontinued = ifelse(month == max(month), 0, 1)) %>%                                              # Indicator for observed switch (1) or censored (0)
  filter(month != min(month)) %>%                                                                           # Take out first method because we don't know when started
  filter(method != "P" & method != "B" & method != "T" & method != "L" & method != "L") %>%                 # Don't need to look at these methods
  mutate(method = factor(method, levels = c("0",    "1",    "2",   "3",          "5",      "9",          "N",       "W",          "M"),
                         labels = c("None", "Pill", "IUD", "Injectable", "Condom", "Withdrawal", "Implant", "Other.trad", "Other.mod"))) %>%
  dplyr::select(obs, v000, v021, v022, wt, age, age_grp, age_grp_fact, parity, edu, wealth_index, urban, cumulative_method_months, unique_methods_count, method, month, method.month, discontinued)

# Dataset to use for modeling
df_country <- data.surv

# -------------------------------
# 3. Accelerated Failure Time Model
# -------------------------------

results <- list()

# Loop through each method
for (m in c("None", "Pill", "IUD", "Injectable", "Condom", "Withdrawal",
            "Implant", "Other.trad", "Other.mod")) {

  df_method <- df_country[!is.na(df_country$method) & df_country$method == m, ]
  if (nrow(df_method) == 0) next

  lowest_aic <- Inf
  best_model <- NULL
  best_dist <- NULL

  # Loop through distributions
  for (d in c("exponential", "weibull", "llogis", "lnorm", "gamma")) {
    df_method <- df_method[order(df_method$obs), ]
    df.first.switch <- df_method[!duplicated(df_method$obs), ]

    aft <- tryCatch(
      flexsurvreg(Surv(method.month, discontinued) ~ age_grp_fact,
                  data = df.first.switch, dist = d),
      error = function(e) NULL
    )

    if (!is.null(aft) && aft$AIC < lowest_aic) {
      lowest_aic <- aft$AIC
      best_model <- aft
      best_dist <- d
    }
  }

  if (!is.null(best_model)) {
    results[[m]] <- list(
      best_distribution = best_dist,
      lowest_aic = lowest_aic,
      model = best_model
    )
  }
}

# -------------------------------
# 4. Extract and Save Coefficients
# -------------------------------

coef_list <- list()

# Iterate through the results list
for (m in names(results)) {
    model_info <- results[[m]]

    if (!is.null(model_info$model)) {
      model_summary <- model_info$model
      res <- as.data.frame(model_summary$res.t)

      output <- res %>%
        mutate(coef = rownames(res),
               method = m,
               functionform = model_info$best_distribution)

      coef_list[[length(coef_list) + 1]] <- output
    }
  }

# Combine into one data frame
coef_res <- bind_rows(coef_list)


coef_fpsim <- coef_res %>%
  mutate(ifelse(functionform == "lnorm", "lognormal", functionform)) %>%
  dplyr::select(estimate = est, coef, se, method, functionform)

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

# Save CSV to ./<country>/data/method_time_coefficients.csv
write.csv(coef_fpsim, file.path(output_dir, "method_time_coefficients.csv"), row.names = F)
