import matplotlib.pyplot as plt
from fit import build_model
import pymc3 as pm
import numpy as np

def generate_plot(sampling_data, rate):
#    plt.hist(sampling_data[rate], bins=np.logspace(-3.3, 2.7, 8))
    plt.hist(sampling_data[rate], bins=np.logspace(-3.6, 3.0, 9))
    plt.gca().set_xscale("log")
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='minor',      # only minor ticks are affected, major ticks left alone
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='on') # labels along the bottom edge are on
    plt.title(rate)
    plt.xlabel('IL2 conc. [nM]')
    plt.show()

M = build_model()
M.build()

sampling_data = pm.backends.text.load("IL2_model_results", model=M.M)

generate_plot(sampling_data, 'k4fwd')
generate_plot(sampling_data, 'k5rev')
generate_plot(sampling_data, 'k6rev')
