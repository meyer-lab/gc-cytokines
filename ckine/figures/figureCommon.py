"""
This file contains functions that are used in multiple figures.
"""
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.cm as cm
from matplotlib import gridspec, pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from ..tensor import find_R2X
from ..imports import import_pstat


matplotlib.rcParams['legend.labelspacing'] = 0.2
matplotlib.rcParams['legend.fontsize'] = 8
matplotlib.rcParams['xtick.major.pad'] = 2
matplotlib.rcParams['ytick.major.pad'] = 2
matplotlib.rcParams['xtick.minor.pad'] = 1.9
matplotlib.rcParams['ytick.minor.pad'] = 1.9
matplotlib.rcParams['legend.handletextpad'] = 0.5
matplotlib.rcParams['legend.handlelength'] = 0.5
matplotlib.rcParams['legend.framealpha'] = 0.5
matplotlib.rcParams['legend.markerscale'] = 0.7
matplotlib.rcParams['legend.borderpad'] = 0.35


def getSetup(figsize, gridd, multz=None, empts=None):
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

    if multz is None:
        multz = dict()

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=True)
    gs1 = gridspec.GridSpec(*gridd, figure=f)

    # Get list of axis objects
    x = 0
    ax = list()
    while x < gridd[0] * gridd[1]:
        if x not in empts and x not in multz.keys():  # If this is just a normal subplot
            ax.append(f.add_subplot(gs1[x]))
        elif x in multz.keys():  # If this is a subplot that spans grid elements
            ax.append(f.add_subplot(gs1[x:x + multz[x] + 1]))
            x += multz[x]
        x += 1

    return (ax, f)


def set_bounds(ax, compNum):
    """ Set bounds of component plots. """
    x_max = np.max(np.absolute(np.asarray(ax.get_xlim()))) * 1.1
    y_max = np.max(np.absolute(np.asarray(ax.get_ylim()))) * 1.1

    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-y_max, y_max)


def plot_R2X(ax, tensor, factors_list):
    """Function to plot R2X bar graph."""
    R2X_array = list()
    for _, factors in enumerate(factors_list):
        R2X_array.append(find_R2X(tensor, factors))

    ax.plot(range(1, len(factors_list) + 1), R2X_array, 'ko', label='Overall R2X')
    ax.set_ylabel('R2X')
    ax.set_xlabel('Number of Components')
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(1, len(factors_list) + 1))
    ax.set_xticklabels(np.arange(1, len(factors_list) + 1))


def subplotLabel(ax, letter, hstretch=1):
    """ Label each subplot """
    ax.text(-0.2 / hstretch, 1.2, letter, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top')


def traf_names():
    """ Returns a list of the trafficking parameters in order they appear within unkVec. """
    return [r'$\mathrm{k_{endo}}$', r'$\mathrm{k_{endo,a}}$', r'$\mathrm{k_{rec}}$', r'$\mathrm{k_{deg}}$']


def plot_conf_int(ax, x_axis, y_axis, color, label=None):
    """ Shades the 25-75 percentiles dark and the 10-90 percentiles light. The percentiles are found along axis=1. """
    y_axis_top = np.percentile(y_axis, 90., axis=1)
    y_axis_bot = np.percentile(y_axis, 10., axis=1)
    ax.fill_between(x_axis, y_axis_top, y_axis_bot, color=color, alpha=0.4)

    y_axis_top = np.percentile(y_axis, 75., axis=1)
    y_axis_bot = np.percentile(y_axis, 25., axis=1)
    ax.fill_between(x_axis, y_axis_top, y_axis_bot, color=color, alpha=0.7, label=label)
    if label is not None:
        ax.legend()


def plot_cells(ax, factors, component_x, component_y, cell_names):
    """This function plots the combination decomposition based on cell type."""
    colors = cm.rainbow(np.linspace(0, 1, len(cell_names)))
    markersCells = ['^', '*', 'D', 's', 'X', 'o', '4', 'H', 'P', '*', 'D', 's', 'X']

    for ii, _ in enumerate(factors[:, component_x - 1]):
        ax.scatter(factors[ii, component_x - 1], factors[ii, component_y - 1], c=[colors[ii]], marker=markersCells[ii], label=cell_names[ii])

    ax.set_title('Cells')
    ax.set_xlabel('Component ' + str(component_x))
    ax.set_ylabel('Component ' + str(component_y))
    set_bounds(ax, component_x)
    ax.legend()


def overlayCartoon(figFile, cartoonFile, x, y, scalee=1, scale_x=1, scale_y=1):
    """ Add cartoon to a figure file. """
    import svgutils.transform as st

    # Overlay Figure cartoons
    template = st.fromfile(figFile)
    cartoon = st.fromfile(cartoonFile).getroot()

    cartoon.moveto(x, y, scale=scalee)
    cartoon.scale_xy(scale_x, scale_y)

    template.append(cartoon)
    template.save(figFile)


def plot_ligands(ax, factors, ligand_names, cutoff=0.0):
    """Function to put all ligand decomposition plots in one figure."""
    ILs, _, _, _ = import_pstat()  # Cytokine stimulation concentrations in nM
    n_ligands = len(ligand_names)
    ILs = np.flip(ILs)
    colors = sns.color_palette()
    legend_shape = []
    markers = ['^', '.', 'd']

    for ii, name in enumerate(ligand_names):
        legend_shape.append(Line2D([0], [0], color='k', marker=markers[ii], label=name, linestyle='')) # Make ligand legend elements

    for ii in range(factors.shape[1]):
        componentLabel = True
        for jj in range(n_ligands):
            idx = range(jj * len(ILs), (jj + 1) * len(ILs))

            # If the component value never gets over cutoff, then don't plot the line
            if np.max(factors[idx, ii]) > cutoff:
                if componentLabel:
                    ax.plot(ILs, factors[idx, ii], color=colors[ii], label='Component ' + str(ii + 1))
                    componentLabel = False
                else:
                    ax.plot(ILs, factors[idx, ii], color=colors[ii])
                ax.scatter(ILs, factors[idx, ii], color=colors[ii], marker=markers[jj])

    ax.add_artist(ax.legend(handles=legend_shape, loc=4))

    ax.set_xlabel('Ligand Concentration (nM)')
    ax.set_ylabel('Component')
    ax.set_xscale('log')
    ax.set_title('Ligands')

    # Put a legend to the right of the current axis
    ax.legend(loc=3)


def plot_timepoints(ax, ts, factors):
    """Function to put all timepoint plots in one figure."""
    colors = sns.color_palette()
    for ii in range(factors.shape[1]):
        ax.plot(ts, factors[:, ii], c=colors[ii], label='Component ' + str(ii + 1))

    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Component')
    ax.set_title('Time')
    ax.legend()


def kfwd_info(unkVec):
    """ Gives the mean and standard deviation of a kfwd distribution. We need this since we are not using violin plots for this rate. """
    mean = np.mean(unkVec[6])
    std = np.std(unkVec[6])
    return mean, std


def legend_2_15(ax, location="center right"):
    """ Plots a legend for all the IL-2 and IL-15 related plots in its own subpanel. """
    legend_elements = [Patch(facecolor='darkorchid', label='IL-2'),
                       Patch(facecolor='goldenrod', label='IL-15'),
                       Line2D([0], [0], marker='o', color='w', label='IL-2Rα+',
                              markerfacecolor='k', markersize=8),
                       Line2D([0], [0], marker='^', color='w', label='IL-2Rα-',
                              markerfacecolor='k', markersize=8)]
    ax.legend(handles=legend_elements, loc=location)
    ax.axis('off')  # remove the grid


def plot_scaled_pstat(ax, cytokC, pstat):
    """ Plots pSTAT5 data scaled by the average activity measurement. """
    # plot pstat5 data for each time point
    ax.scatter(cytokC, pstat[3, :], c="indigo", s=2)  # 0.5 hr
    ax.scatter(cytokC, pstat[2, :], c="teal", s=2)  # 1 hr
    ax.scatter(cytokC, pstat[1, :], c="forestgreen", s=2)  # 2 hr
    ax.scatter(cytokC, pstat[0, :], c="darkred", s=2)  # 4 hr
