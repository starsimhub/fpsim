# Example FPsim usage from R.
#
# Run with e.g.:
#
#    Rscript example_sim.R

# Imports
library(ggplot2)
library(reticulate)
fp <- import('fpsim')
sp <- import('fp_analyses.senegal_parameters')

# Create
pars <- sp$make_pars()
pars$n_agents <- 500 # Set population size
sim <- fp$Sim(pars=pars)

# Run
sim$run()

# Plot
df = data.frame(t=sim$results$t, pop_size=sim$results$pop_size_months)
ggplot(df, aes(x=t, y=pop_size)) + geom_point()
