"""
This creates Figure 1.
"""
from os.path import join
import os
import string
import numpy as np
import seaborn as sns
import pandas as pd
from .figureCommon import subplotLabel, getSetup, traf_names, plot_conf_int, import_samples_2_15, kfwd_info
from ..plot_model_prediction import surf_IL2Rb, pstat, surf_gc

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 7), (3, 4), mults=[0, 6], multz={0: 2, 6: 2}, empts=[3])

    # blank out first two axes for cartoon
    ax[0].axis('off')

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    unkVec, scales = import_samples_2_15()
    kfwd_avg, kfwd_std = kfwd_info(unkVec)
    print("kfwd = " + str(kfwd_avg) + " +/- " + str(kfwd_std))
    pstat_act(ax[1], unkVec, scales)
    surf_perc(ax[2:4], 'IL-2Rβ', unkVec)
    violinPlots(ax[5:8], unkVec, scales)
    rateComp(ax[4], unkVec)

    f.tight_layout(w_pad=1.3)

    return f


def surf_perc(ax, species, unkVec):
    """ Calculates the percent of IL2Rb or gc on the cell surface over the course of 90 mins. Cell environments match those of surface IL2Rb data collected by Ring et al. """
    if species == 'IL-2Rβ':
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
    results = np.zeros((size, unkVec.shape[1], 4, 2)) # 3rd dim is cell condition (IL2Ra+/- and cytokC), 4th dim is cytok species

    output = surf.calc(unkVec, ts) * y_max # run the simulation
    # split according to experimental conditions
    results[:, :, 2, 0] = output[:, 0:(size)].T
    results[:, :, 3, 0] = output[:, (size):(size*2)].T
    results[:, :, 0, 0] = output[:, (size*2):(size*3)].T
    results[:, :, 1, 0] = output[:, (size*3):(size*4)].T
    results[:, :, 2, 1] = output[:, (size*4):(size*5)].T
    results[:, :, 3, 1] = output[:, (size*5):(size*6)].T
    results[:, :, 0, 1] = output[:, (size*6):(size*7)].T
    results[:, :, 1, 1] = output[:, (size*7):(size*8)].T

    for n in range(4):
        # plot results within confidence intervals
        plot_conf_int(ax[n % 2], ts, results[:, :, n, 0], "darkorchid", "IL-2")
        plot_conf_int(ax[n % 2], ts, results[:, :, n, 1], "goldenrod", "IL-15")

    # label axes and titles
    ax[1].set(xlabel="Time (min)", ylabel=("Surface " + str(species) + " (%)"), title="YT-1 cells and 500 nM")
    ax[1].set_ylim(0,115)
    ax[0].set(xlabel="Time (min)", ylabel=("Surface " + str(species) + " (%)"), title="YT-1 cells and 1 nM")
    ax[0].set_ylim(0,115)


def pstat_act(ax, unkVec, scales):
    """ This function generates the pSTAT activation levels for each combination of parameters in unkVec. The results are plotted and then overlayed with the values measured by Ring et al. """
    pstat5 = pstat()
    PTS = 30
    cytokC = np.logspace(-3.3, 2.7, PTS)
    y_max = 100.
    IL2_plus = np.zeros((unkVec.shape[1], PTS))
    IL15_minus = IL2_plus.copy()
    IL15_plus = IL2_plus.copy()
    IL2_minus = IL2_plus.copy()

    output = pstat5.calc(unkVec, scales, cytokC) * y_max # calculate activity for all unkVecs and concs
    # split according to experimental condition
    IL2_plus = output[:, 0:PTS].T
    IL2_minus = output[:, PTS:(PTS*2)].T
    IL15_plus = output[:, (PTS*2):(PTS*3)].T
    IL15_minus = output[:, (PTS*3):(PTS*4)].T

    # plot confidence intervals based on model predictions
    plot_conf_int(ax, np.log10(cytokC), IL2_minus, "darkorchid", "IL-2")
    plot_conf_int(ax, np.log10(cytokC), IL15_minus, "goldenrod", "IL-15")
    plot_conf_int(ax, np.log10(cytokC), IL2_plus, "darkorchid")
    plot_conf_int(ax, np.log10(cytokC), IL15_plus, "goldenrod")

    # plot experimental data
    path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(join(path, "../data/IL2_IL15_extracted_data.csv")).values # imports file into pandas array
    ax.scatter(data[:,0], data[:,2], color='darkorchid', marker='^', edgecolors='k', zorder=100, label="IL-2, 2Rα-") # IL2 in 2Ra-
    ax.scatter(data[:,0], data[:,3], color='goldenrod', marker='^', edgecolors='k', zorder=101, label="IL-15, 2Rα-") # IL15 in 2Ra-
    ax.scatter(data[:,0], data[:,6], color='darkorchid', marker='o', edgecolors='k', zorder=102, label="IL-2, 2Rα+") # IL2 in 2Ra+
    ax.scatter(data[:,0], data[:,7], color='goldenrod', marker='o', edgecolors='k', zorder=103, label="IL-15, 2Rα+") # IL15 in 2Ra+
    ax.set(ylabel='pSTAT5 (% of max)', xlabel=r'Cytokine concentration (log$_{10}$[nM])', title='YT-1 cell activity')

    ax.legend()

def violinPlots(ax, unkVec, scales, Fig1=True):
    """ Create violin plots of model posterior. """
    unkVec = unkVec.transpose()
    traf = pd.DataFrame(unkVec[:, 17:22])
    Rexpr = pd.DataFrame(unkVec[:, 22:26])
    scales = pd.DataFrame(scales)

    if Fig1:
        traf.columns = traf_names()
        b = sns.violinplot(data=np.log10(traf), ax=ax[0], linewidth=0.5)
        b.set_xticklabels(b.get_xticklabels(), rotation=40, rotation_mode="anchor", ha="right", fontsize=8, position=(0, 0.075))
        b.set(title="Trafficking parameters", ylabel=r"$\mathrm{log_{10}(\frac{1}{min})}$")

    Rexpr.columns = ['IL-2Rα', 'IL-2Rβ', r'$\gamma_{c}$', 'IL-15Rα']
    if Fig1:
        c = sns.violinplot(data=np.log10(Rexpr), ax=ax[1], linewidth=0.5)
    else:
        c = sns.violinplot(data=np.log10(Rexpr), ax=ax[0], linewidth=0.5)
    c.set_xticklabels(c.get_xticklabels(), rotation=40, rotation_mode="anchor", ha="right", fontsize=8, position=(0, 0.075))
    c.set(title="Receptor expression rates", ylabel=r"$\mathrm{log_{10}(\frac{num}{cell * min})}$")

    sc_ax = 1 # subplot number for the scaling constant
    if Fig1:
        sc_ax = 2
    scales.columns = [r'$C_{5}$']
    d = sns.violinplot(data=scales, ax=ax[sc_ax], linewidth=0.5)
    d.set_title("value")
    d.set_title("pSTAT5 scaling constant")

def rateComp(ax, unkVec):
    """ This function compares the analogous reverse rxn distributions from IL2 and IL15 in a violin plot. """
    # assign values from unkVec
    k4rev, k5rev, k16rev, k17rev, k22rev, k23rev = unkVec[7, :], unkVec[8, :], unkVec[9, :], unkVec[10, :], unkVec[11, :], unkVec[12, :]
    split = unkVec.shape[1]
    # plug in values from measured constants into arrays of size 500
    kfbnd = 0.60 # Assuming on rate of 10^7 M-1 sec-1
    k1rev = np.full((split), (kfbnd * 10))    # doi:10.1016/j.jmb.2004.04.038, 10 nM
    k2rev = np.full((split), (kfbnd * 144))   # doi:10.1016/j.jmb.2004.04.038, 144 nM
    k13rev = np.full((split), (kfbnd * 0.065))    # based on the multiple papers suggesting 30-100 pM
    k14rev = np.full((split), (kfbnd * 438))  # doi:10.1038/ni.2449, 438 nM

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
    df = pd.DataFrame({r'$k_{1}, k_{13}$': np.append(k1rev, k13rev), r'$k_{2}, k_{14}$': np.append(k2rev, k14rev), r'$k_{4}, k_{16}$': np.append(k4rev, k16rev), r'$k_{5}, k_{17}$': np.append(k5rev, k17rev), r'$k_{8}, k_{20}$': np.append(k8rev, k20rev), r'$k_{9}, k_{21}$': np.append(k9rev, k21rev), r'$k_{10}, k_{22}$': np.append(k10rev, k22rev), r'$k_{11}, k_{23}$': np.append(k11rev, k23rev), r'$k_{12}, k_{24}$': np.append(k12rev, k24rev)})

    # add labels for IL2 and IL15
    df['cytokine'] = 'IL-2'
    df.loc[split:(split*2), 'cytokine'] = 'IL-15'

    # melt into long form and take log value
    melted = pd.melt(df, id_vars='cytokine', var_name='rate', value_name=r"$\mathrm{log_{10}(\frac{1}{min})}$")
    melted.loc[:, r"$\mathrm{log_{10}(\frac{1}{min})}$"] = np.log10(melted.loc[:, r"$\mathrm{log_{10}(\frac{1}{min})}$"])

    col_list = ["violet", "goldenrod"]
    col_list_palette = sns.xkcd_palette(col_list)
    sns.set_palette(col_list_palette)

    # plot with hue being cytokine species
    a = sns.violinplot(x='rate', y=r"$\mathrm{log_{10}(\frac{1}{min})}$", data=melted, hue='cytokine', ax=ax, linewidth=0, scale='width')
    a.scatter(-0.3, np.log10(kfbnd * 10), color="darkviolet")   # overlay point for k1rev
    a.scatter(0.1, np.log10(kfbnd * 0.065), color='goldenrod') # overlay point for k13rev
    a.scatter(0.7, np.log10(kfbnd * 144), color="darkviolet")   # overlay point for k2rev
    a.scatter(1.1, np.log10(kfbnd * 468), color='goldenrod') # overlay point for k14rev
    a.set_title("Analogous reverse reaction rates")
