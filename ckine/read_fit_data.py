import pickle as pk
import bz2

def read_data(name):
    filename = './type-I-ckine_model/ckine/' + name + '.pkl'
    data = pk.load(bz2.BZ2File(filename, 'rb'))
    return data

read_data("build_results")
read_data("sampling_results")