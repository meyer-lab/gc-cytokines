"""
The file that runs the fitting process.
"""
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
from .fit import build_model, IL2_activity_input


def generate_plot(sampling_data, rate):
#    plt.hist(sampling_data[rate], bins=np.logspace(-3.3, 2.7, 8))
    plt.hist(np.log(sampling_data[rate]), bins=20)
#    plt.gca().set_xscale("log")
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='minor',      # only minor ticks are affected, major ticks left alone
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='on') # labels along the bottom edge are on
    plt.title(rate)
    plt.xlabel(rate + ' value (log distribution)')
    plt.ylabel('relative probability')
    plt.show()

M = build_model()
M.build()

# name of the file we're loading is dependent on the output of run_fit.py
sampling_data = pm.backends.text.load("IL2_model_results", model=M.M) # loads the results from running fit.py... we are not unpickling here

#generate_plot(sampling_data, 'k4fwd')
#generate_plot(sampling_data, 'k5rev')
#generate_plot(sampling_data, 'k6rev')

pm.plots.traceplot(sampling_data) # this generates the traceplots so that we can track Y and other variables during the fitting process
plt.show()

# plot a scatter plot of k5rev against k6rev
def scatter_plot(sampling_data, rate1, rate2):
    plt.scatter(np.log(sampling_data[rate1]), np.log(sampling_data[rate2]))
    plt.xlabel(rate1 + ' (log distribution)')
    plt.ylabel(rate2 + ' (log distribution)')
    plt.title(rate2 + ' vs. ' + rate1)
    plt.show()

#scatter_plot(sampling_data, 'k5rev', 'k6rev')
#scatter_plot(sampling_data, 'k4fwd', 'k6rev')

#can call this function to get a graph similar to that which was published
def plot_IL2_percent_activity(y0, IL2, rxnRates, trafRates):
    new_table = IL2_activity_input(y0, IL2, rxnRates, trafRates)

    x = np.log10(new_table[:, 0]) # changing the x values to the log10(nM) values that were in the published graph

    plt.rcParams.update({'font.size': 8})
    plt.xlabel("IL2 concentration (log(nm))")
    plt.ylabel("percent activation of pSTAT")
    plt.scatter(x[:], new_table[:,1])
    plt.show()
