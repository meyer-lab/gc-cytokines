import pickle as pk
import bz2
import matplotlib.pyplot as plt
from fit import build_model, IL2_sum_squared_dist, IL2_activity_values

def read_data(name):
    filename = './' + name + '.pkl'
    data = pk.load(bz2.BZ2File(filename, 'rb'))
    return data

def generate_plot(sampling_data, rate):
    plt.hist(sampling_data.trace[rate], 100)
    plt.show()

sampling_data = read_data("IL2_model_results")

generate_plot(sampling_data, 'k4fwd')
#generate_plot(sampling_data, 'k5rev')
#generate_plot(sampling_data, 'k6rev')
