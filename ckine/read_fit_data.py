import matplotlib.pyplot as plt
from fit import build_model
import pymc3 as pm

def generate_plot(sampling_data, rate):
    plt.hist(sampling_data[rate], 100)
    plt.show()

M = build_model()
M.build()

sampling_data = pm.backends.text.load("IL2_model_results", model=M.M)

generate_plot(sampling_data, 'k4fwd')
#generate_plot(sampling_data, 'k5rev')
#generate_plot(sampling_data, 'k6rev')
