import pickle as pk
import bz2
import matplotlib.pyplot as plt

def read_data(name):
    filename = './type-I-ckine_model/ckine/' + name + '.pkl'
    data = pk.load(bz2.BZ2File(filename, 'rb'))
    return data

def generate_plot(rate):
    sampling_data = read_data("IL2_model_results")
    plot = plt.hist(sampling_data.M.trace[rate],100)
    return plot

generate_plot('k4fwd')
generate_plot('k5rev')
generate_plot('k6rev')
        