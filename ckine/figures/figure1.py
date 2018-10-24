"""
This creates Figure 1.
"""
from os.path import join
import os
import string
import numpy as np
import seaborn as sns
import pandas as pd
from .figureCommon import subplotLabel, getSetup, traf_names, plot_conf_int, import_samples_2_15
from ..plot_model_prediction import surf_IL2Rb, pstat, surf_gc

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 7), (3, 4), mults=[0, 10], multz={0: 2, 10: 2}, empts=[3])

    # blank out first two axes for cartoon
    ax[0].axis('off')

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    unkVec = import_samples_2_15()
    pstat_act(ax[1], unkVec)
    surf_perc(ax[2:4], 'IL2Rb', unkVec)
    violinPlots(ax[6:8], unkVec)
    rateComp(ax[8], unkVec)


    f.tight_layout()

    return f


def surf_perc(ax, species, unkVec):
    """ Calculates the percent of IL2Rb or gc on the cell surface over the course of 90 mins. Cell environments match those of surface IL2Rb data collected by Ring et al. """
    if species == 'IL2Rb':
        surf = surf_IL2Rb() # load proper class
        # overlay experimental data
        path = os.path.dirname(os.path.abspath(__file__))
        data_minus = pd.read_csv(join(path, "../data/IL2Ra-_surface_IL2RB_datasets.csv")).values # imports file into pandas array
        data_plus = pd.read_csv(join(path, "../data/IL2Ra+_surface_IL2RB_datasets.csv")).values # imports file into pandas array
        ax[0].scatter(data_minus[:,0], data_minus[:,1] * 10., color='darkorchid', marker='^', edgecolors='k', zorder=100) # 1nM of IL2 in 2Ra-
        ax[0].scatter(data_minus[:,0], data_minus[:,2] * 10., color='goldenrod', marker='^', edgecolors='k', zorder=101) # 1nM of IL15 in 2Ra-
        ax[1].scatter(data_minus[:,0], data_minus[:,5] * 10., color='darkorchid', marker='^', edgecolors='k', zorder=100) # 500nM of IL2 in 2Ra-
        ax[1].scatter(data_minus[:,0], data_minus[:,6] * 10., color='goldenrod', marker='^', edgecolors='k', zorder=101) # 500nM of IL15 in 2Ra-
        ax[0].scatter(data_plus[:,0], data_plus[:,1] * 10., color='darkorchid', marker='o', edgecolors='k', zorder=100) # 1nM of IL2 in 2Ra+
        ax[0].scatter(data_plus[:,0], data_plus[:,2] * 10., color='goldenrod', marker='o', edgecolors='k', zorder=101) # 1nM of IL15 in 2Ra+
        ax[1].scatter(data_plus[:,0], data_plus[:,5] * 10., color='darkorchid', marker='o', edgecolors='k', zorder=100) # 500nM of IL2 in 2Ra+
        ax[1].scatter(data_plus[:,0], data_plus[:,6] * 10., color='goldenrod', marker='o', edgecolors='k', zorder=101) # 500nM of IL15 in 2Ra+

    if species == 'gc':
        surf = surf_gc()    # load proper class

    y_max = 100.
    ts = np.array([0., 2., 5., 15., 30., 60., 90.])
    size = len(ts)
    results = np.zeros((size, 500, 4, 2)) # 3rd dim is cell condition (IL2Ra+/- and cytokC), 4th dim is cytok species

    for ii in range(0,500):
        output = surf.calc(unkVec[:, ii], ts) * y_max
        results[:, ii, 2, 0] = output[0:(size)]
        results[:, ii, 3, 0] = output[(size):(size*2)]
        results[:, ii, 0, 0] = output[(size*2):(size*3)]
        results[:, ii, 1, 0] = output[(size*3):(size*4)]
        results[:, ii, 2, 1] = output[(size*4):(size*5)]
        results[:, ii, 3, 1] = output[(size*5):(size*6)]
        results[:, ii, 0, 1] = output[(size*6):(size*7)]
        results[:, ii, 1, 1] = output[(size*7):(size*8)]

    for n in range(4):
        # plot results within confidence intervals
        plot_conf_int(ax[n % 2], ts, results[:, :, n, 0], "darkorchid", "IL2")
        plot_conf_int(ax[n % 2], ts, results[:, :, n, 1], "goldenrod", "IL15")

    # label axes and titles
    ax[1].set(xlabel="Time (min)", ylabel=("Surface " + str(species) + " (%)"), title="YT-1 Cells and 500 nM")
    ax[1].set_ylim(0,115)
    ax[0].set(xlabel="Time (min)", ylabel=("Surface " + str(species) + " (%)"), title="YT-1 Cells and 1 nM")
    ax[0].set_ylim(0,115)


def pstat_act(ax, unkVec):
    """ This function generates the pSTAT activation levels for each combination of parameters in unkVec. The results are plotted and then overlayed with the values measured by Ring et al. """
    pstat5 = pstat()
    PTS = 30
    cytokC = np.logspace(-3.3, 2.7, PTS)
    y_max = 100.
    IL2_plus = np.zeros((PTS, 500))
    IL15_minus = IL2_plus.copy()
    IL15_plus = IL2_plus.copy()
    IL2_minus = IL2_plus.copy()

    # calculate activity for each unkVec for all conc.
    for ii in range(0,500):
        output = pstat5.calc(unkVec[:, ii], cytokC) * y_max
        IL2_plus[:, ii] = output[0:PTS]
        IL2_minus[:, ii] = output[PTS:(PTS*2)]
        IL15_plus[:, ii] = output[(PTS*2):(PTS*3)]
        IL15_minus[:, ii] = output[(PTS*3):(PTS*4)]

    # plot confidence intervals based on model predictions
    plot_conf_int(ax, np.log10(cytokC), IL2_minus, "darkorchid", "IL2")
    plot_conf_int(ax, np.log10(cytokC), IL15_minus, "goldenrod", "IL15")
    plot_conf_int(ax, np.log10(cytokC), IL2_plus, "darkorchid", "IL2")
    plot_conf_int(ax, np.log10(cytokC), IL15_plus, "goldenrod", "IL15")

    # plot experimental data
    path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(join(path, "../data/IL2_IL15_extracted_data.csv")).values # imports file into pandas array
    ax.scatter(data[:,0], data[:,2], color='darkorchid', marker='^', edgecolors='k', zorder=100, label="IL2, 2Ra-") # IL2 in 2Ra-
    ax.scatter(data[:,0], data[:,3], color='goldenrod', marker='^', edgecolors='k', zorder=101, label="IL15, 2Ra-") # IL15 in 2Ra-
    ax.scatter(data[:,0], data[:,6], color='darkorchid', marker='o', edgecolors='k', zorder=102, label="IL2, 2Ra+") # IL2 in 2Ra+
    ax.scatter(data[:,0], data[:,7], color='goldenrod', marker='o', edgecolors='k', zorder=103, label="IL15, 2Ra+") # IL15 in 2Ra+
    ax.set(ylabel='Percent of maximal p-STAT5 (%)', xlabel='log10 of cytokine concentration (nM)', title='YT-1 Cell Activity')
    ax.legend(loc='upper left', bbox_to_anchor=(1.5, 1))

def violinPlots(ax, unkVec):
    """ Create violin plots of model posterior. """
    unkVec = unkVec.transpose()

    traf = pd.DataFrame(unkVec[:, 17:22])
    Rexpr = pd.DataFrame(unkVec[:, 22:26])

    traf.columns = traf_names()
    b = sns.violinplot(data=np.log10(traf), ax=ax[0], linewidth=0, bw=10)
    b.set_xticklabels(b.get_xticklabels(), rotation=40, rotation_mode="anchor", ha="right", fontsize=8, position=(0, 0.075))
    b.set(title="Trafficking parameters", ylabel="log10 of 1/min")

    Rexpr.columns = ['IL2Ra', 'IL2Rb', 'gc', 'IL15Ra']
    c = sns.violinplot(data=np.log10(Rexpr), ax=ax[1], linewidth=0, bw=10)
    c.set_xticklabels(c.get_xticklabels(), rotation=40, rotation_mode="anchor", ha="right", fontsize=8, position=(0, 0.075))
    c.set(title="Receptor expression rates", ylabel="log10 of #/cell/min")


def rateComp(ax, unkVec):
    """ This function compares the analogous reverse rxn distributions from IL2 and IL15 in a violin plot. """

    # assign values from unkVec
    k4rev, k5rev, k16rev, k17rev, k22rev, k23rev = unkVec[7, :], unkVec[8, :], unkVec[9, :], unkVec[10, :], unkVec[11, :], unkVec[12, :]

    # plug in values from measured constants into arrays of size 500
    kfbnd = 0.60 # Assuming on rate of 10^7 M-1 sec-1
    k1rev = np.full((500), (kfbnd * 10))    # doi:10.1016/j.jmb.2004.04.038, 10 nM
    k2rev = np.full((500), (kfbnd * 144))   # doi:10.1016/j.jmb.2004.04.038, 144 nM
    k13rev = np.full((500), (kfbnd * 0.065))    # based on the multiple papers suggesting 30-100 pM
    k14rev = np.full((500), (kfbnd * 438))  # doi:10.1038/ni.2449, 438 nM

    # proportions known through measurements
    k10rev = 12.0 * k5rev / 1.5 # doi:10.1016/j.jmb.2004.04.038
    k11rev = 63.0 * k5rev / 1.5 # doi:10.1016/j.jmb.2004.04.038

    # detailed balance
    k12rev = k1rev * k11rev / k2rev # loop for IL2_IL2Ra_IL2Rb
    k9rev = k10rev * k11rev / k4rev
    k8rev = k10rev * k12rev / k5rev
    k24rev = k13rev * k23rev / k14rev
    k21rev = k22rev * k23rev / k16rev
    k20rev = k22rev * k24rev / k17rev

    # add each rate duo as separate column in dataframe
    df = pd.DataFrame({'k1_k13': np.append(k1rev, k13rev), 'k2_k14': np.append(k2rev, k14rev), 'k4_k16': np.append(k4rev, k16rev), 'k5_k17': np.append(k5rev, k17rev), 'k8_k20': np.append(k8rev, k20rev), 'k9_k21': np.append(k9rev, k21rev), 'k10_k22': np.append(k10rev, k22rev), 'k11_k23': np.append(k11rev, k23rev), 'k12_k24': np.append(k12rev, k24rev)})

    # add labels for IL2 and IL15
    df['cytokine'] = 'IL2'
    df.loc[500:1000, 'cytokine'] = 'IL15'

    # melt into long form and take log value
    melted = pd.melt(df, id_vars='cytokine', var_name='rate', value_name='log10 of 1/nM/min')
    melted.loc[:, 'log10 of 1/nM/min'] = np.log10(melted.loc[:, 'log10 of 1/nM/min'])

    col_list = ["violet", "goldenrod"]
    col_list_palette = sns.xkcd_palette(col_list)
    cmap = sns.set_palette(col_list_palette)

    # plot with hue being cytokine species
    a = sns.violinplot(x='rate', y='log10 of 1/nM/min', data=melted, hue='cytokine', ax=ax, cmap=cmap, linewidth=0, bw=15, scale='width')
    a.scatter(2.75, np.log10(kfbnd * 10), color="darkviolet")   # overlay point for k1rev
    a.scatter(3.20, np.log10(kfbnd * 0.065), color='goldenrod') # overlay point for k13rev
    a.scatter(3.7, np.log10(kfbnd * 144), color="darkviolet")   # overlay point for k2rev
    a.scatter(4.15, np.log10(kfbnd * 468), color='goldenrod') # overlay point for k14rev
    a.set_title("Analogous reverse reaction rates")
