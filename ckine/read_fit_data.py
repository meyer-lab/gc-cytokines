import matplotlib.pyplot as plt
from fit import build_model
import pymc3 as pm
import numpy as np

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

sampling_data = pm.backends.text.load("IL2_model_results", model=M.M)

#generate_plot(sampling_data, 'k4fwd')
#generate_plot(sampling_data, 'k5rev')
#generate_plot(sampling_data, 'k6rev')

pm.plots.traceplot(sampling_data)
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