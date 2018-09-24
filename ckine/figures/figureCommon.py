"""
This file contains functions that are used in multiple figures.
"""
import string
import os
import pickle
import itertools
import seaborn as sns
import numpy as np
import pandas as pds
import matplotlib
import matplotlib.cm as cm
from matplotlib import gridspec, pyplot as plt


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
