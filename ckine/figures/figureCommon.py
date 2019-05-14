"""
This file contains functions that are used in multiple figures.
"""
import seaborn as sns
import numpy as np
import matplotlib.cm as cm
from matplotlib import gridspec, pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch



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

    # ax = []

    # Get list of axis objects
    if mults is None:
        ax = [f.add_subplot(gs1[x]) for x in range(gridd[0] * gridd[1]) if x not in empts]
    else:
        ax = [f.add_subplot(gs1[x]) if x not in mults else f.add_subplot(gs1[x:x + multz[x]]) for x in range(
            gridd[0] * gridd[1]) if not any([x - j in mults for j in range(1, max(multz.values()))]) and x not in empts]

    # shrink the padding between ticks and axes
    for a in ax:
        a.tick_params(axis='both', pad=-2)

    return (ax, f)


def set_bounds(ax, compNum):
    """Add labels and bounds"""
    ax.set_xlabel('Component ' + str(compNum))
    ax.set_ylabel('Component ' + str(compNum + 1))

    x_max = np.max(np.absolute(np.asarray(ax.get_xlim()))) * 1.1
    y_max = np.max(np.absolute(np.asarray(ax.get_ylim()))) * 1.1

    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-y_max, y_max)


def plot_ligands(ax, factors, component_x, component_y, ax_pos, n_ligands, mesh, fig3=True):
    "This function is to plot the ligand combination dimension of the values tensor."
    markers = ['^', '*', 'x']
    cmap = sns.color_palette("hls", n_ligands)

    legend_shape = [Line2D([0], [0], color='k', marker=markers[0], label='IL-2', linestyle=''),
                    Line2D([0], [0], color='k', label='IL-15', marker=markers[1], linestyle=''),
                    Line2D([0], [0], color='k', label='IL-2 mut', marker=markers[2], linestyle='')]

    for ii in range(int(factors.shape[0] / n_ligands)):
        idx = range(ii * n_ligands, (ii + 1) * n_ligands)
        if ii == 0 and ax_pos == 4 and fig3:
            legend = "full"
        elif ii == 0 and ax_pos == 2 and fig3 is False:
            legend = "full"
        else:
            legend = False
        sns.scatterplot(x=factors[idx, component_x - 1], y=factors[idx, component_y - 1], marker=markers[ii], hue=np.log10(np.sum(mesh[idx, :], axis=1)), ax=ax, palette=cmap, s=100, legend=legend)
        h, _ = ax.get_legend_handles_labels()
        if ax_pos == 4 and fig3:
            ax.add_artist(ax.legend(handles=h, loc=2))
            ax.add_artist(ax.legend(handles=legend_shape, loc=3))
        elif ax_pos == 2 and not fig3:
            ax.add_artist(ax.legend(handles=h, loc=2))
            ax.add_artist(ax.legend(handles=legend_shape, loc=3))

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
    """ Shades the 25-75 percentiles dark and the 10-90 percentiles light. The percentiles are found along axis=1. """
    y_axis_top = np.percentile(y_axis, 90., axis=1)
    y_axis_bot = np.percentile(y_axis, 10., axis=1)
    ax.fill_between(x_axis, y_axis_top, y_axis_bot, color=color, alpha=0.4, label=label)

    y_axis_top = np.percentile(y_axis, 75., axis=1)
    y_axis_bot = np.percentile(y_axis, 25., axis=1)
    ax.fill_between(x_axis, y_axis_top, y_axis_bot, color=color, alpha=0.7, label=label)


def plot_cells(ax, factors, component_x, component_y, cell_names, ax_pos, fig3=True):
    """This function plots the combination decomposition based on cell type."""
    colors = cm.rainbow(np.linspace(0, 1, len(cell_names)))
    markersCells = ['^', '*', 'D', 's', 'X', 'o', '4', 'H', 'P', '*', 'D', 's', 'X']  # 'o', 'd', '1', '2', '3', '4', 'h', 'H', 'X', 'v', '*', '+', '8', 'P', 'p', 'D', '_','D', 's', 'X', 'o'

    for ii in range(len(factors[:, component_x - 1])):
        ax.scatter(factors[ii, component_x - 1], factors[ii, component_y - 1], c=[colors[ii]], marker=markersCells[ii], label=cell_names[ii])

    if ax_pos in (1, 2):
        ax.legend()

    elif ax_pos == 3 and fig3:
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


def kfwd_info(unkVec):
    """ Gives the mean and standard deviation of a kfwd distribution. We need this since we are not using violin plots for this rate. """
    mean = np.mean(unkVec[6])
    std = np.std(unkVec[6])
    return mean, std


def legend_2_15(ax):
    """ Plots a legend for all the IL-2 and IL-15 related plots in its own subpanel. """
    legend_elements = [Patch(facecolor='darkorchid', label='IL-2'),
                       Patch(facecolor='goldenrod', label='IL-15'),
                       Line2D([0], [0], marker='o', color='w', label='IL-2Rα+ cells',
                              markerfacecolor='k', markersize=8),
                       Line2D([0], [0], marker='^', color='w', label='IL-2Rα- cells',
                              markerfacecolor='k', markersize=8)]
    ax.legend(handles=legend_elements, loc='center', mode="expand", fontsize="large")
    ax.axis('off')  # remove the grid


def plot_scaled_pstat(ax, cytokC, pstat):
    """ Plots pSTAT5 data scaled by the average activity measurement. """
    tps = np.array([0.5, 1., 2., 4.])
    avg_activity = np.sum(pstat) / tps.size
    # plot pstat5 data for each time point
    ax.scatter(cytokC, pstat[3, :] / avg_activity, c="indigo", s=2)  # 0.5 hr
    ax.scatter(cytokC, pstat[2, :] / avg_activity, c="teal", s=2)  # 1 hr
    ax.scatter(cytokC, pstat[1, :] / avg_activity, c="forestgreen", s=2)  # 2 hr
    ax.scatter(cytokC, pstat[0, :] / avg_activity, c="darkred", s=2)  # 4 hr
