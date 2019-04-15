"""
This file contains functions that are used in multiple figures.
"""
import os
from os.path import join
import tensorly as tl
import pymc3 as pm
import seaborn as sns
import numpy as np
import pandas as pds
import matplotlib.cm as cm
from matplotlib import gridspec, pyplot as plt
from matplotlib.lines import Line2D
from ..model import nParams
from ..fit import build_model as build_model_2_15
from ..fit_others import build_model as build_model_4_7
from ..tensor_generation import prepare_tensor

n_ligands = 4
values, _, mat, _, _ = prepare_tensor(n_ligands)
values = tl.tensor(values)


def getSetup(figsize, gridd, mults=None, multz=None, empts=None):
    """ Establish figure set-up with subplots. """
    sns.set(style="whitegrid",
            font_scale=0.7,
            color_codes=True,
            palette="colorblind",
            rc={'grid.linestyle': 'dotted',
                'axes.linewidth': 0.6})

    # create empty list if empts isn't specified
    if empts is None:
        empts = []

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


def set_bounds(ax, compNum):
    """Add labels and bounds"""
    ax.set_xlabel('Component ' + str(compNum))
    ax.set_ylabel('Component ' + str(compNum + 1))

    x_max = np.max(np.absolute(np.asarray(ax.get_xlim()))) * 1.1
    y_max = np.max(np.absolute(np.asarray(ax.get_ylim()))) * 1.1

    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-y_max, y_max)


def plot_ligands(ax, factors, component_x, component_y, ax_pos, fig3=True):
    "This function is to plot the ligand combination dimension of the values tensor."
    markers = ['^', '*', 'x']
    cmap = sns.color_palette("hls", n_ligands)

    legend_shape = [Line2D([0], [0], color='k', marker=markers[0], label='IL-2', linestyle=''),
                    Line2D([0], [0], color='k', label='IL-15', marker=markers[1], linestyle=''),
                    Line2D([0], [0], color='k', label='IL-2 mut', marker=markers[2], linestyle='')]

    for ii in range(int(factors.shape[0] / n_ligands)):
        idx = range(ii * n_ligands, (ii + 1) * n_ligands)
        if ii == 0 and ax_pos == 5 and fig3:
            legend = "full"
        elif ii == 0 and ax_pos == 2 and fig3 is False:
            legend = "full"
        else:
            legend = False
        sns.scatterplot(x=factors[idx, component_x - 1], y=factors[idx, component_y - 1], marker=markers[ii], hue=np.log10(np.sum(mat[idx, :], axis=1)), ax=ax, palette=cmap, s=100, legend=legend)
        h, l = ax.get_legend_handles_labels()
        if ax_pos == 5 and fig3:
            legend1 = ax.legend(handles=h, loc=2)
            ax.add_artist(legend1)
            legend2 = ax.legend(handles=legend_shape, loc=3)
            ax.add_artist(legend2)
        elif ax_pos == 2 and not fig3:
            legend1 = ax.legend(handles=h, loc=2)
            ax.add_artist(legend1)
            legend2 = ax.legend(handles=legend_shape, loc=3)
            ax.add_artist(legend2)

    ax.set_title('Ligands')
    set_bounds(ax, component_x)


def subplotLabel(ax, letter, hstretch=1):
    """ Label each subplot """
    ax.text(-0.2 / hstretch, 1.2, letter, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top')


def traf_names():
    """ Returns a list of the trafficking parameters in order they appear within unkVec. """
    return [r'$k_{endo}$', r'$k_{endo,a}$', r'$f_{sort}$', r'$k_{rec}$', r'$k_{deg}$']


def plot_conf_int(ax, x_axis, y_axis, color, label=None):
    """ Calculates the 95% confidence interval for y-axis data and then plots said interval. The percentiles are found along axis=1. """
    y_axis_top = np.percentile(y_axis, 97.5, axis=1)
    y_axis_bot = np.percentile(y_axis, 2.5, axis=1)
    ax.fill_between(x_axis, y_axis_top, y_axis_bot, color=color, alpha=0.5, label=label)


def plot_cells(ax, factors, component_x, component_y, cell_names, ax_pos, fig3=True):
    """This function plots the combination decomposition based on cell type."""
    colors = cm.rainbow(np.linspace(0, 1, len(cell_names)))
    markersCells = ['^', '*', 'D', 's', 'X', 'o', '4', 'H', 'P', '*', 'D', 's', 'X'] #'o', 'd', '1', '2', '3', '4', 'h', 'H', 'X', 'v', '*', '+', '8', 'P', 'p', 'D', '_','D', 's', 'X', 'o'

    for ii in range(len(factors[:, component_x - 1])):
        ax.scatter(factors[ii, component_x - 1], factors[ii, component_y - 1], c = [colors[ii]], marker = markersCells[ii], label = cell_names[ii])

    if ax_pos == 1:
        ax.legend()

    elif ax_pos == 4 and fig3:
        ax.legend()
    ax.set_title('Cells')

    set_bounds(ax, component_x)


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
        ax.plot(ts, factors[:, ii], c=colors[ii], label='Component ' + str(ii + 1))
        ax.scatter(ts[-1], factors[-1, ii], s=12, color='k')

    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Component')
    ax.set_title('Time')
    ax.legend()


def import_samples_2_15(Traf=True, ret_trace=False):
    """ This function imports the csv results of IL2-15 fitting into a numpy array called unkVec. """
    bmodel = build_model_2_15(traf=Traf)
    n_params = nParams()

    path = os.path.dirname(os.path.abspath(__file__))

    if Traf:
        trace = pm.backends.text.load(join(path, '../../IL2_model_results'), bmodel.M)
    else:
        trace = pm.backends.text.load(join(path, '../../IL2_15_no_traf'), bmodel.M)

    # option to return trace instead of numpy array
    if ret_trace:
        return trace

    scales = trace.get_values('scales')
    num = scales.size
    kfwd = trace.get_values('kfwd')
    rxn = trace.get_values('rxn')
    exprRates = trace.get_values('IL2Raexpr')

    if Traf:
        endo = trace.get_values('endo')
        activeEndo = trace.get_values('activeEndo')
        sortF = trace.get_values('sortF')
        kRec = trace.get_values('kRec')
        kDeg = trace.get_values('kDeg')
    else:
        endo = np.zeros((num))
        activeEndo = np.zeros((num))
        sortF = np.zeros((num))
        kRec = np.zeros((num))
        kDeg = np.zeros((num))

    unkVec = np.zeros((n_params, num))
    for ii in range(num):
        unkVec[:, ii] = np.array([0., 0., 0., 0., 0., 0., kfwd[ii], rxn[ii, 0], rxn[ii, 1], rxn[ii, 2], rxn[ii, 3], rxn[ii, 4], rxn[ii, 5], 1., 1., 1., 1., endo[ii],
                                  activeEndo[ii], sortF[ii], kRec[ii], kDeg[ii], exprRates[ii, 0], exprRates[ii, 1], exprRates[ii, 2], exprRates[ii, 3], 0., 0., 0., 0.])

    return unkVec, scales


def import_samples_4_7():
    ''' This function imports the csv results of IL4-7 fitting into a numpy array called unkVec. '''
    bmodel = build_model_4_7()
    n_params = nParams()

    path = os.path.dirname(os.path.abspath(__file__))
    trace = pm.backends.text.load(join(path, '../../IL4-7_model_results'), bmodel.M)
    kfwd = trace.get_values('kfwd')
    k27rev = trace.get_values('k27rev')
    k33rev = trace.get_values('k33rev')
    endo = trace.get_values('endo')
    activeEndo = trace.get_values('activeEndo')
    sortF = trace.get_values('sortF')
    kRec = trace.get_values('kRec')
    kDeg = trace.get_values('kDeg')
    scales = trace.get_values('scales')
    GCexpr = (328. * endo) / (1. + ((kRec * (1. - sortF)) / (kDeg * sortF)))  # constant according to measured number per cell
    IL7Raexpr = (2591. * endo[0]) / (1. + ((kRec * (1. - sortF)) / (kDeg * sortF)))  # constant according to measured number per cell
    IL4Raexpr = (254. * endo) / (1. + ((kRec * (1. - sortF)) / (kDeg * sortF)))  # constant according to measured number per cell
    num = scales.shape[0]

    unkVec = np.zeros((n_params, num))
    for ii in range(num):
        unkVec[:, ii] = np.array([0., 0., 0., 0., 0., 0., kfwd[ii], 1., 1., 1., 1., 1., 1., k27rev[ii], 1., k33rev[ii], 1., endo[ii],
                                  activeEndo[ii], sortF[ii], kRec[ii], kDeg[ii], 0., 0., GCexpr[ii], 0., IL7Raexpr[ii], 0., IL4Raexpr[ii], 0.])

    return unkVec, scales


def load_cells():
    """ Loads CSV file that gives Rexpr levels for different cell types. """
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    expr_filename = os.path.join(fileDir, './ckine/data/expr_table.csv')
    data = pds.read_csv(expr_filename)  # Every column in the data represents a specific cell
    cell_names = data.columns.values.tolist()[1::]  # returns the cell names from the pandas dataframe (which came from csv)
    return data, cell_names


def kfwd_info(unkVec):
    """ Gives the mean and standard deviation of a kfwd distribution. We need this since we are not using violin plots for this rate. """
    mean = np.mean(unkVec[6])
    std = np.std(unkVec[6])
    return mean, std

def import_Rexpr():
    """ Loads CSV file containing Rexpr levels from preliminary Visterra data. """
    path = os.path.dirname(os.path.dirname(__file__))
    data = pds.read_csv(join(path, 'data/Receptor_levels_4_8_19.csv')) # Every row in the data represents a specific cell
    numpy_data = data.values[:, 1:] # returns data values in a numpy array
    cell_names = list(data.values[:, 0])
    return numpy_data, cell_names
