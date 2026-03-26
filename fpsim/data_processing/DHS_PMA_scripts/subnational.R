################################################################
# Calculate sub-national parameters for calibration            #  
# Using most recent DHS and PMA data                           #
# Ethiopia                                                     #
# March 2024                                                 #
################################################################


# Clear environment (preserve run control variables)
run_vars <- ls(pattern = "^run_")
rm(list = setdiff(ls(), run_vars))


# -- Libraries -- #
library(tidyverse)      
library(withr)  
library(haven)   
library(survey)
library(readstata13) 
library(zoo)


# -- Set Working Directory -- #
dir <- "~/Data" ## directory with data 
setwd(dir) 
getwd()

# ---- Load DHS Data ---- 
ETKR71FL <- read.dta13("DHS/ETKR71FL.DTA") ## (Child Recode - KR)
ETIR71FL <- read.dta13("DHS/ETIR71FL.DTA") ## (Individual Recode - IR)
# Source https://dhsprogram.com/data/dataset_admin/index.cfm
# recode https://www.dhsprogram.com/publications/publication-dhsg4-dhs-questionnaires-and-manuals.cfm

# Individual Recode Data
data.ir <- ETIR71FL %>% # clean IR data
  mutate(age = v012, activity = v536, parity = v220, edu = v133, method = v312, urban = v025,
         age_group = recode(factor(v013), 
                            "1" = "15-19", 
                            "2" = "20-24", 
                            "3" = "25-29", 
                            "4" = "30-34", 
                            "5"  = "35-39", 
                            "6"  = "40-44", 
                            "7"  = "45-49"), 
         region = recode(factor(v024), #rename the regions - keep spelling consistent
                           "tigray" = "Tigray", 
                           "afar" = "Afar", 
                           "amhara" = "Amhara", 
                           "oromia" = "Oromia", 
                           "somali" = "Somali", 
                           "benishangul" = "Benishangul-Gumuz", 
                           "snnpr" = "SNNPR", 
                           "gambela" = "Gambela", 
                           "harari" = "Harari", 
                           "addis adaba" = "Addis Ababa", 
                           "dire dawa" = "Dire Dawa"),
         use = ifelse(v312 == "not using", 0, 1), 
         wgt = v005/1000000)

svydes1 = svydesign( #create design object for weighted analysis of individual recode data
  id = data.ir$v001, 
  strata=data.ir$v023,
  weights = data.ir$v005/1000000,
  data=data.ir) 

regions <- unique(data.ir$region)



## ---- Birth Spacing ----
# Define a function to process data for each region
spacing_region <- function(region_name, data) {
  data <- data %>%
    filter(region == region_name) %>%
    dplyr::select(v005, caseid, v102,               # weight, individual id, urban/rural
                  starts_with("b0"),                # multiples
                  starts_with("b11")) %>%           # preceding birth interval
    gather(var, val, -v005, -caseid, -v102) %>%
    separate(var, c("var", "num"), sep = "_") %>%
    spread(var, val) %>%
    filter(b0 != "2nd of multiple" & b0 != "3rd of multiple") %>% # remove multiples
    mutate(space_mo = as.numeric(b11), wt = v005/1000000, urban_rural = v102) 
  
  # Calculate weighted data table of number of births (Freq) in each individual birth spacing (by month) category
  spacing <- as.data.frame(svytable(~ space_mo + v102, svydesign(id=~caseid, weights=~wt, data = data)))
  
  return(spacing)
}

# Apply the function to each region and store results in a list
birth_spacing <- map(setNames(regions, regions), ~spacing_region(.x, data.ir))
birth_spacing_region <- bind_rows(birth_spacing, .id = "region") %>%
  arrange(region, v102)

write.csv(birth_spacing_region, file = "~/Data/Output/Subnational/birth_spacing_dhs_region.csv", row.names = FALSE)



## ---- Breastfeeding duration ----
require(fitdistrplus) 

#gumbel distribution definitions  
dgumbel <- function(x, a, b) 1/b*exp((a-x)/b)*exp(-exp((a-x)/b)) 
pgumbel <- function(q, a, b) exp(-exp((a-q)/b)) 
qgumbel <- function(p, a, b) a-b*log(-log(p)) 

# Create an empty dataframe to store the combined results
bf_stats_region <- data.frame()

# Loop through each unique region (v024)
for (region in unique(ETKR71FL$v024)) { ## Use Child Recode Data ##
  cat("Processing region:", region, "\n")
  
  # Subset data for the current region
  region_data <- subset(ETKR71FL, v024 == region)
  
  # Filter data for months of breastfeeding < 93
  dat_nonmiss <- subset(region_data, m5 < 93)
  dat_nonmiss$m5_weighted <- (dat_nonmiss$v005 / 1000000) * dat_nonmiss$m5
  
  # Fit gumbel distribution
  fitgumbel <- fitdist(dat_nonmiss$m5, "gumbel", start = list(a = 1, b = 1))
  
  # Extract parameters
  parameters <- as.data.frame(fitgumbel$estimate)
  parameters$region <- region
  
  # Convert row names to new column - parameters
  parameters$parameter <- rownames(parameters)
  parameters$parameter <- gsub("\\d+", "", parameters$parameter)  #reecode all parameters to 'a' or 'b'
  
  # Clean up date frame
  parameters <- parameters[, c("region", "parameter", "fitgumbel$estimate")] #reorder columns
  parameters$region <- str_to_title(parameters$region) # capitalize regions
  parameters$region[parameters$region == "Snnpr"] <- "SNNPR"   # Adjust SNNPR label
  parameters$region[parameters$region == "Benishagul"] <- "Benishagul-Gumuz"   #Adjust Benishangul-Gumuz label

  # Bind the current region's results to the bf_stats_region dataframe
  bf_stats_region <- rbind(bf_stats_region, parameters) %>%
    arrange(region, parameter)
}

write.csv(bf_stats_region, file = "~/Data/Output/Subnational/bf_stats_region.csv", row.names = FALSE)



## ---- Postpartum Results ----   

### ---- Sexual Activity PP ----   

# Create an empty dataframe to store the results
postpartum_region <- data.frame()

# Loop through each unique region
for (region in regions) { 
  # Subset data for the current region 
  region_data <- data.ir[data.ir$region == region, ] 
  
  # Your data manipulation code here...
  dhs.pp <- region_data %>% 
    mutate(b19_01 = v008 - b3_01) %>% 
    filter(b19_01 < 24 & b9_01 == "respondent") %>% 
    mutate(breastmilk = ifelse(m4_1 == "still breastfeeding", TRUE, FALSE),
           water = v409 == "yes",
           non.milk.liquids = ifelse(v409a == "yes" | v410 == "yes" | v410a == "yes" | v412c == "yes" | v413 == "yes" | v413a == "yes" | v413b == "yes" | v413c == "yes" | v413d == "yes", TRUE, FALSE),
           non.milk.liquids = ifelse(is.na(non.milk.liquids), FALSE, non.milk.liquids),
           other.milk = v411 == "yes" | v411a == "yes",
           solids = ifelse(v412a == "yes" | v412b == "yes" | v414a == "yes" | v414b == "yes" | v414c == "yes" | v414d == "yes" | v414e == "yes" | v414f == "yes" | v414g == "yes" | v414h == "yes" | v414i == "yes" | v414j == "yes" | v414k == "yes" | v414l == "yes" | v414m == "yes" | v414n == "yes" | v414o == "yes" | v414p == "yes" | v414q == "yes" | v414r == "yes" | v414s == "yes" | v414t == "yes" | v414u == "yes" | v414v == "yes" | v414w == "yes" | m39a_1 == "yes", TRUE, FALSE),
           solids = ifelse(is.na(solids), FALSE, solids),
           exclusive.bf = breastmilk & !non.milk.liquids & !other.milk & !solids,
           amenorrhea = (m6_1 >= b19_01 & m6_1 < 96) | m6_1 == 96,
           abstinent = m8_1 >= b19_01 & m8_1 <= 96,
           exclusive.bf_amerorrhea = exclusive.bf | amenorrhea,
           exclusive.bf_and_amerorrhea = exclusive.bf & amenorrhea,
           exclusive.bf_amerorrhea_abstinent = exclusive.bf_amerorrhea | abstinent,
           s.active_2 = v536 == "active in last 4 weeks",
           wt = v005/1000000)
  
  dhs.pp.results <- dhs.pp %>% 
    rename(months_postpartum = b19_01) %>% 
    group_by(months_postpartum) %>%
    mutate_at(c("exclusive.bf", "amenorrhea", "exclusive.bf_and_amerorrhea"), ~ifelse(breastmilk, ., NA)) %>%
    summarise(abstinent = weighted.mean(abstinent, wt, na.rm = TRUE),
              s.active_1 = 1 - abstinent,
              s.active_2 = weighted.mean(s.active_2, wt, na.rm = TRUE),
              exclusive.bf = weighted.mean(exclusive.bf, wt, na.rm = TRUE),
              amenorrhea = weighted.mean(amenorrhea, wt, na.rm = TRUE),
              exclusive.bf_and_amerorrhea = weighted.mean(exclusive.bf_and_amerorrhea, wt, na.rm = TRUE),
              n = sum(!is.na(abstinent)))
  
  # Add region information
  dhs.pp.results$region <- region
  
  # Combine results with postpartum_region dataframe
  postpartum_region <- bind_rows(postpartum_region, dhs.pp.results)
}

sexual_activity_pp_region <- postpartum_region %>% 
  dplyr::select(region, months_postpartum, s.active_2) %>%
  arrange(region, months_postpartum)


# Save results for sexual activity parameter by region
write.csv(sexual_activity_pp_region, "~/Data/Output/Subnational/sexual_activity_pp_region.csv", row.names = FALSE)


### ---- Lactational Amenorrhea ----   


# percentage of women on LAM by month pp (breastfeeding, amenorrhea, and both) 
lam_results_region <- postpartum_region %>% 
  dplyr::select(region, months_postpartum, exclusive.bf_and_amerorrhea) %>%
  arrange(region, months_postpartum)

#save results for LAM parameter
write.csv(lam_results_region, "~/Data/Output/Subnational/lam_region.csv", row.names = F) 



## ---- Regional Proportion ----  

table.region <- as.data.frame(svymean(~region, svydes1)) %>% #percentage living in each region 
  arrange(row.names(.)) 

row.names(table.region) <- gsub("^region", "", row.names(table.region)) #clean rows 
rownames(table.region) <- c("Addis Ababa", "Afar", "Amhara", "Benishangul-Gumuz", "Dire Dawa",
                            "Gambela", "Harari", "Oromia", "SNNPR", "Somali", "Tigray") #rename row

table.region.urban <- as.data.frame(svyby(~urban, ~region, svydes1, svymean)) %>% 
  arrange(row.names(.)) 
table.region.urban <- cbind(table.region[1], table.region.urban[2]) 
colnames(table.region.urban) <- c("mean", "urban") 
write.csv(table.region.urban, file = "~/Data/Output/Subnational/region_urban.csv", row.names = TRUE) 



## ---- Sexual Activity ----  

activity <- as.data.frame(svyby(~activity, ~age_group + ~region, svydes1, svymean)) %>% 
  arrange(row.names(.)) 

sexual_activity <- activity[c(1,2,4)] %>% 
  mutate(perc = `activityactive in last 4 weeks`*100, 
         age_group = as.numeric(str_extract(age_group, "\\d+"))) %>%
  dplyr::select(region, age_group, perc)
rownames(sexual_activity) <- NULL

#adding age_group 0, 5, 10
extra_rows <- activity %>%
  group_by(region) %>%
  summarise(age_group = c(0, 5, 10), perc = c(0, 0, 0))

#adding age_group 50, which duplicates age_group 45 values
extra_row_50 <- sexual_activity %>%
  group_by(region) %>%
  summarise(perc_45 = perc[age_group == 45]) %>%
  mutate(age_group =  50) %>%
  rename(perc = perc_45) %>%
  dplyr::select(region, age_group, perc)


sexual_activity_region <- rbind(sexual_activity, extra_rows, extra_row_50) %>%
  arrange(region, age_group)

write.csv(sexual_activity_region, file = '~/Data/Output/Subnational/sexual_activity_region.csv', row.names = FALSE) 



## ---- Sexual Debut ----  

dhs_debut <- data.ir[data.ir$v531 != 0 & data.ir$v531 < 50 & data.ir$v531 >= 10, ]  

# Initialize an empty dataframe to store the results 
debut_age_region <- data.frame() 

for (region in regions) { 
  # Subset data for the current region 
  region_data <- dhs_debut[dhs_debut$region == region, ] 
  
  # Calculate weighted probabilities for the current region 
  age_probs <- aggregate(wgt ~ v531, data = region_data, sum)  
  age_probs <- data.frame(age_probs)   
  names(age_probs) <- c('v531', 'wgt')   
  
  age_probs$prob_weights <- age_probs$wgt / sum(age_probs$wgt)   
  
  # Remove the period and number after the region 
  region_name <- gsub("\\..*", "", region) 
  
  if(tolower(region_name) == "snnpr") {
    region_name <- toupper(region_name)
  } else {
    region_name <- tools::toTitleCase(region_name)
  }
  # Create a new dataframe without the wgt column and with renamed columns 
  result <- data.frame(region = region_name, age = age_probs$v531, prob = age_probs$prob_weights) 
  
  # Bind the current region's results to the final dataframe 
  debut_age_region <- rbind(debut_age_region, result) 
} 

# Reset row names 
debut_age_region <- debut_age_region[order(debut_age_region$region),]
row.names(debut_age_region) <- NULL 

write.csv(debut_age_region, file = '~/Data/Output/Subnational/sexual_debut_region.csv', row.names = FALSE) 




# ---- Load PMA Data ---- 

data.pma <- read_dta("PMA/PMAET_HQFQ_2019_CrossSection_v2.0_30Apr2021.dta") %>% 
  filter(!is.na(FQweight)) %>% 
  mutate(region2 = recode(factor(region), #rename the regions 
                          "1" = "Tigray", 
                          "2" = "Afar", 
                          "3" = "Amhara", 
                          "4" = "Oromia", 
                          "5" = "Somali", 
                          "6" = "Benishangul-Gumuz", 
                          "7" = "SNNPR", 
                          "8" = "Gambela", 
                          "9" = "Harari", 
                          "10" = "Addis Ababa", 
                          "11" = "Dire Dawa")) 

svydes2 <- svydesign(id = ~EA_ID, strata = ~strata, weights =  ~FQweight, data = data.pma, nest = T) 



## ---- Barriers ---- 

#Calculate barriers to use at the national level 
reasons <- unlist(strsplit(svydes2$variables$why_not_using, " ")) 

# Create a table of counts for each response 
reasons_counts <- as.data.frame(table(reasons)) 
reasons_counts <- reasons_counts %>% 
  filter(reasons != "dnk", reasons != "nr", reasons != "other") 

reasons_mapping <- list( 
  "No need" = c("breastfeed", "infreq_sex", "menopousal", "not_marr", "not_mens", "subfecund", "part_away"), 
  "Opposition" = c("part_oppo", "religious", "respo_oppo", "upto_god"), 
  "Knowledge" = c("dnk_method", "dnk_where"), 
  "Access" = c("cost_much", "incoven", "not_avai", "pre_mth_una", "too_far"), 
  "Health" = c("fear_sideef", "health_conc", "inter_body")
  ) 

# Create a new variable 'group' based on the mapping 
reasons_counts$group <- sapply(reasons_counts$reasons, function(reason) { 
  for (group in names(reasons_mapping)) { 
    if (reason %in% reasons_mapping[[group]]) { 
      return(group) 
    } 
  } 
}) 

# Aggregate frequencies by group 
reasons_national <- reasons_counts %>% 
  group_by(group) %>% 
  summarise(freq = sum(Freq)) %>% 
  mutate(perc = (freq / sum(freq)) * 100) 

# Create a data frame with reasons and region 
reasons_data <- data.frame(why_not_using = svydes2$variables$why_not_using, svydes2$variables$region) %>% 
  filter(why_not_using != "") 

# Split the multiple-choice variable into separate responses 
reasons_data <- reasons_data %>% 
  separate_rows(why_not_using, sep = " ") 

# Create a table of counts for each response 
reasons_data <- reasons_data %>% 
  filter(reasons_data$why_not_using != "dnk", 
         reasons_data$why_not_using != "nr", 
         reasons_data$why_not_using != "other") 

reasons_counts <- as.data.frame(table(reasons_data$why_not_using, reasons_data$svydes2.variables.region)) 
reasons_counts$Var2 <- as.character(reasons_counts$Var2) 

# Create a dataframe for region mapping 
region_mapping <- data.frame( 
  region = c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"), 
  Region = c("Tigray", "Afar", "Amhara","Oromia", "Somali", "Benishangul-Gumuz", 
             "SNNPR", "Gambela", "Harari", "Addis Ababa", "Dire Dawa") 
) 

reasons_counts <- reasons_counts %>% 
  left_join(region_mapping, by = c("Var2" = "region")) 

# Create a new variable 'group' based on the mapping 
reasons_counts$group <- sapply(reasons_counts$Var1, function(reason) { 
  for (group in names(reasons_mapping)) { 
    if (reason %in% reasons_mapping[[group]]) { 
      return(group) 
    } 
  } 
}) 

# Aggregate frequencies by group, region, and calculate the percentage 
barriers_region <- reasons_counts %>% 
  group_by(group, Region) %>% 
  summarise(perc = sum(Freq) / sum(reasons_counts$Freq) * 100) %>%
  group_by(Region) %>%
  mutate(perc = perc / sum(perc)*100) %>%
  dplyr::select(Region, group, perc)


colnames(barriers_region) <- c("region", "barrier", "per")

write.csv(barriers_region, file = "~/Data/Output/Subnational/barriers_region.csv", row.names = FALSE)





