"""
This file contains functions that are used in multiple figures.
"""
import os
from os.path import join
import string
import pickle
import itertools
import pymc3 as pm, os
import seaborn as sns
import numpy as np
import pandas as pds
import matplotlib.cm as cm
from matplotlib import gridspec, pyplot as plt
from ..model import nParams
from ..fit import build_model as build_model_2_15
from ..fit_others import build_model as build_model_4_7

def getSetup(figsize, gridd, mults=None, multz=None, empts=[]):
    """ Establish figure set-up with subplots. """
    sns.set(style="whitegrid",
            font_scale=0.7,
            color_codes=True,
            palette="colorblind",
            rc={'grid.linestyle':'dotted',
                'axes.linewidth':0.6})

    # Setup plotting space
    f = plt.figure(figsize=figsize)

    # Make grid
    gs1 = gridspec.GridSpec(*gridd)

    # Get list of axis objects
    if mults is None:
        ax = [f.add_subplot(gs1[x]) for x in range(gridd[0] * gridd[1])]
    else:
        ax = [f.add_subplot(gs1[x]) if x not in mults else f.add_subplot(gs1[x:x + multz[x]]) for x in range(
            gridd[0] * gridd[1]) if not any([x - j in mults for j in range(1, max(multz.values()))]) and x not in empts]

    return (ax, f)

def subplotLabel(ax, letter, hstretch=1):
    """ Label each subplot """
    ax.text(-0.2 / hstretch, 1.2, letter, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top')

def traf_names():
    """ Returns a list of the trafficking parameters in order they appear within unkVec. """
    return ['endo', 'activeEndo', 'sortF', 'kRec', 'kDeg']

def plot_conf_int(ax, x_axis, y_axis, color, label):
    """ Calculates the 95% confidence interval for y-axis data and then plots said interval. The percentiles are found along axis=1. """
    y_axis_top = np.percentile(y_axis, 97.5, axis=1)
    y_axis_bot = np.percentile(y_axis, 2.5, axis=1)
    ax.fill_between(x_axis, y_axis_top, y_axis_bot, color=color, alpha=0.5, label=label)

def plot_values(ax1, factors, component_x, component_y, ax_pos):
    """Plot the values decomposition factors matrix."""
    #Generate a plot for component x vs component y of the factors[3] above representing our values
    # The markers are for the following elements in order: 'IL2 & IL15 Combined', 'IL7', 'IL9', 'IL4','IL21','IL2Ra', 'IL2Rb', 'gc', 'IL15Ra', 'IL7Ra', 'IL9R', 'IL4Ra','IL21Ra','IL2Ra', 'IL2Rb', 'gc', 'IL15Ra', 'IL7Ra', 'IL9R', 'IL4Ra','IL21Ra.'
    #Set Active to color red. Set Surface to color blue. Set Total to color black
    markersLigand = itertools.cycle(('^', 'D', 's', 'X', 'o'))

    labelLigand = itertools.cycle(('Combined IL2-15 Activity', 'IL7 Activity', 'IL9 Activity', 'IL4 Activity', 'IL21 Activity'))

    for q,p in zip(factors[0:5, component_x - 1], factors[0:5, component_y - 1]):
            ax1.plot(q, p, linestyle = '', c = 'm', marker = next(markersLigand), label = next(labelLigand))

    if ax_pos == 3:
        ax1.legend(loc='upper left', bbox_to_anchor=(1.2, 1.025))


def plot_timepoint(ax, factors, component_x, component_y):
    """Plot the timepoint decomposition in the first column of figS2."""
    print(factors.shape)
    ax.plot(factors[:, component_x - 1], factors[:, component_y - 1], color = 'k')
    ax.scatter(factors[-1, component_x - 1], factors[-1, component_y - 1], s = 12, color = 'b')

def plot_cells(ax, factors, component_x, component_y, cell_names, ax_pos):
    """This function plots the combination decomposition based on cell type."""
    colors = cm.rainbow(np.linspace(0, 1, len(cell_names)))
    markersCells = ['^', '*', 'D', 's', 'X', 'o', '^', '4', 'P', '*', 'D', 's', 'X' ,'o', 'd', '1', '2', '3', '4', 'h', 'H', 'X', 'v', '*', '+', '8', 'P', 'p', 'D', '_','D', 's', 'X', 'o']

    for ii in range(len(factors[:, component_x - 1])):
        ax.scatter(factors[ii, component_x - 1], factors[ii, component_y - 1], c = colors[ii], marker = markersCells[ii], label = cell_names[ii])
    if ax_pos == 5 and factors.shape[1] <= 10:
        ax.legend(loc='upper left', bbox_to_anchor=(3.6, 1.7))

    elif ax_pos == 5:
        ax.legend(loc='upper left', bbox_to_anchor=(3.6, 0.5))

def plot_ligands(ax, factors, component_x, component_y):
    "This function is to plot the ligand combination dimension of the values tensor."
    ax.scatter(factors[:,component_x - 1], factors[:,component_y - 1])

def overlayCartoon(figFile, cartoonFile, x, y, scalee=1):
    """ Add cartoon to a figure file. """
    import svgutils.transform as st

    # Overlay Figure 4 cartoon
    template = st.fromfile(figFile)
    cartoon = st.fromfile(cartoonFile).getroot()

    cartoon.moveto(x, y, scale=scalee)

    template.append(cartoon)
    template.save(figFile)

def plot_timepoints(ax, factors):
    """Function to put all timepoint plots in one figure."""
    ts = np.logspace(-3., np.log10(4 * 60.), 100)
    ts = np.insert(ts, 0, 0.0)
    colors = ['b', 'k', 'r', 'y', 'm', 'g']
    for ii in range(factors.shape[1]):
        ax.plot(ts, factors[:,ii], c = colors[ii], label = 'Component ' + str(ii+1))
        ax.scatter(ts[-1], factors[-1, ii], s = 12, color = 'k')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Component')
    ax.legend()

def import_samples_2_15():
    """ This function imports the csv results of IL2-15 fitting into a numpy array called unkVec. """
    bmodel = build_model_2_15()
    n_params = nParams()

    path = os.path.dirname(os.path.abspath(__file__))
    trace = pm.backends.text.load(join(path, '../../IL2_model_results'), bmodel.M)
    kfwd = trace.get_values('kfwd', chains=[0])
    rxn = trace.get_values('rxn', chains=[0])
    endo_activeEndo = trace.get_values('endo', chains=[0])
    sortF = trace.get_values('sortF', chains=[0])
    kRec_kDeg = trace.get_values('kRec_kDeg', chains=[0])
    exprRates = trace.get_values('IL2Raexpr', chains=[0])

    unkVec = np.zeros((n_params, 500))
    for ii in range (0, 500):
        unkVec[:, ii] = np.array([0., 0., 0., 0., 0., 0., kfwd[ii], rxn[ii, 0], rxn[ii, 1], rxn[ii, 2], rxn[ii, 3], rxn[ii, 4], rxn[ii, 5], 1., 1., 1., 1., endo_activeEndo[ii, 0], endo_activeEndo[ii, 1], sortF[ii], kRec_kDeg[ii, 0], kRec_kDeg[ii, 1], exprRates[ii, 0], exprRates[ii, 1], exprRates[ii, 2], exprRates[ii, 3], 0., 0., 0., 0.])

    return unkVec

def import_samples_4_7():
    ''' This function imports the csv results of IL4-7 fitting into a numpy array called unkVec. '''
    bmodel = build_model_4_7()
    n_params = nParams()

    path = os.path.dirname(os.path.abspath(__file__))
    trace = pm.backends.text.load(join(path, '../../IL4-7_model_results'), bmodel.M)
    kfwd = trace.get_values('kfwd', chains=[0])
    k27rev = trace.get_values('k27rev', chains=[0])
    k33rev = trace.get_values('k33rev', chains=[0])
    endo_activeEndo = trace.get_values('endo', chains=[0])
    sortF = trace.get_values('sortF', chains=[0])
    kRec_kDeg = trace.get_values('kRec_kDeg', chains=[0])
    scales = trace.get_values('scales', chains=[0])
    GCexpr = (328. * endo_activeEndo[:, 0]) / (1. + ((kRec_kDeg[:, 0]*(1.-sortF[:, 0])) / (kRec_kDeg[:, 1]*sortF[:, 0]))) # constant according to measured number per cell
    IL7Raexpr = (2591. * endo_activeEndo[:, 0]) / (1. + ((kRec_kDeg[:, 0]*(1.-sortF[:, 0])) / (kRec_kDeg[:, 1]*sortF[:, 0]))) # constant according to measured number per cell
    IL4Raexpr = (254. * endo_activeEndo[:, 0]) / (1. + ((kRec_kDeg[:, 0]*(1.-sortF[:, 0])) / (kRec_kDeg[:, 1]*sortF[:, 0]))) # constant according to measured number per cell

    unkVec = np.zeros((n_params, 500))
    for ii in range (0, 500):
        unkVec[:, ii] = np.array([0., 0., 0., 0., 0., 0., kfwd[ii], 1., 1., 1., 1., 1., 1., k27rev[ii], 1., k33rev[ii], 1., endo_activeEndo[ii, 0], endo_activeEndo[ii, 1], sortF[ii], kRec_kDeg[ii, 0], kRec_kDeg[ii, 1], 0., 0., np.squeeze(GCexpr[ii]), 0., np.squeeze(IL7Raexpr[ii]), 0., np.squeeze(IL4Raexpr[ii]), 0.])

    return unkVec, scales

def load_cells():
    """ Loads CSV file that gives Rexpr levels for different cell types. """
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    expr_filename = os.path.join(fileDir, './ckine/data/expr_table.csv')
    data = pds.read_csv(expr_filename) # Every column in the data represents a specific cell
    cell_names = data.columns.values.tolist()[1::] #returns the cell names from the pandas dataframe (which came from csv)
    return data, cell_names
