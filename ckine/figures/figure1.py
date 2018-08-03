"""
This creates Figure 1.
"""
from os.path import join
import pymc3 as pm, os
import numpy as np
import seaborn as sns
import pandas as pd
import string
from ..fit import build_model
from .figureCommon import subplotLabel, getSetup, traf_names, plot_conf_int
from ..plot_model_prediction import surf_IL2Rb, pstat, surf_gc
from ..model import nParams


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 7), (3, 4), mults=[0, 8], multz={0: 2, 8: 2})

    # blank out first two axes for cartoon
    ax[0].axis('off')

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    unkVec = import_samples()
    pstat_act(ax[1:3], unkVec)
    surf_perc(ax[3:7], 'IL2Rb', unkVec)
    rateComp(ax[7], unkVec)
    violinPlots(ax[8:10], unkVec)

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
        ax[2].scatter(data_plus[:,0], data_plus[:,1] * 10., color='darkorchid', marker='^', edgecolors='k', zorder=100) # 1nM of IL2 in 2Ra+
        ax[2].scatter(data_plus[:,0], data_plus[:,2] * 10., color='goldenrod', marker='^', edgecolors='k', zorder=101) # 1nM of IL15 in 2Ra+
        ax[3].scatter(data_plus[:,0], data_plus[:,5] * 10., color='darkorchid', marker='^', edgecolors='k', zorder=100) # 500nM of IL2 in 2Ra+
        ax[3].scatter(data_plus[:,0], data_plus[:,6] * 10., color='goldenrod', marker='^', edgecolors='k', zorder=101) # 500nM of IL15 in 2Ra+

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
        plot_conf_int(ax[n], ts, results[:, :, n, 0], "darkorchid", "IL2")
        plot_conf_int(ax[n], ts, results[:, :, n, 1], "goldenrod", "IL15")
        # label axes and show legends
        ax[n].set(xlabel="time (min)", ylabel=("surface " + str(species) + " (%)"))
        ax[n].legend()

    # set titles
    ax[0].set_title("1 nM and IL2Ra-")
    ax[1].set_title("500 nM and IL2Ra-")
    ax[2].set_title("1 nM and IL2Ra+")
    ax[3].set_title("500 nM and IL2Ra+")


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
    plot_conf_int(ax[0], np.log10(cytokC), IL2_minus, "darkorchid", "IL2")
    plot_conf_int(ax[0], np.log10(cytokC), IL15_minus, "goldenrod", "IL15")
    plot_conf_int(ax[1], np.log10(cytokC), IL2_plus, "darkorchid", "IL2")
    plot_conf_int(ax[1], np.log10(cytokC), IL15_plus, "goldenrod", "IL15")

    # plot experimental data
    path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(join(path, "../data/IL2_IL15_extracted_data.csv")).values # imports file into pandas array
    ax[0].scatter(data[:,0], data[:,2], color='darkorchid', marker='^', edgecolors='k', zorder=100) # IL2 in 2Ra-
    ax[0].scatter(data[:,0], data[:,3], color='goldenrod', marker='^', edgecolors='k', zorder=101) # IL15 in 2Ra-
    ax[1].scatter(data[:,0], data[:,6], color='darkorchid', marker='^', edgecolors='k', zorder=100) # IL2 in 2Ra+
    ax[1].scatter(data[:,0], data[:,7], color='goldenrod', marker='^', edgecolors='k', zorder=101) # IL15 in 2Ra+
    ax[0].set(ylabel='Maximal p-STAT5 (% x 100)', xlabel='log10 of cytokine concentration (nM)', title='IL2Ra- YT-1 cells')
    ax[1].set(ylabel='Maximal p-STAT5 (% x 100)', xlabel='log10 of cytokine concentration (nM)', title='IL2Ra+ YT-1 cells')
    ax[0].legend()
    ax[1].legend()


def import_samples():
    """ This function imports the csv results into a numpy array called unkVec. """
    bmodel = build_model()
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

def violinPlots(ax, unkVec):
    """ Create violin plots of model posterior. """
    unkVec = unkVec.transpose()

    traf = pd.DataFrame(unkVec[:, 17:22])
    Rexpr = pd.DataFrame(unkVec[:, 22:26])

    traf.columns = traf_names()
    b = sns.violinplot(data=np.log10(traf), ax=ax[0], linewidth=0, bw=10)
    b.set_xticklabels(b.get_xticklabels(), rotation=40, rotation_mode="anchor", ha="right", fontsize=8, position=(0, 0.075))
    b.set(title="Trafficking parameters", ylabel="log10 of value")

    Rexpr.columns = ['IL2Ra', 'IL2Rb', 'gc', 'IL15Ra']
    c = sns.violinplot(data=np.log10(Rexpr), ax=ax[1], linewidth=0, bw=10)
    c.set_xticklabels(c.get_xticklabels(), rotation=40, rotation_mode="anchor", ha="right", fontsize=8, position=(0, 0.075))
    c.set(title="Receptor expression rates", ylabel="log10 of value")


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

    # detailed balance
    k10rev = 12.0 * k5rev / 1.5 # doi:10.1016/j.jmb.2004.04.038
    k11rev = 63.0 * k5rev / 1.5 # doi:10.1016/j.jmb.2004.04.038
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
    melted = pd.melt(df, id_vars='cytokine', var_name='rate', value_name='log10 of value')
    melted.loc[:, 'log10 of value'] = np.log10(melted.loc[:, 'log10 of value'])

    col_list = ["violet", "goldenrod"]
    col_list_palette = sns.xkcd_palette(col_list)
    cmap = sns.set_palette(col_list_palette)

    # plot with hue being cytokine species
    a = sns.violinplot(x='rate', y='log10 of value', data=melted, hue='cytokine', ax=ax, cmap=cmap, linewidth=0, bw=15, scale='width')
    a.set_title("Analogous reverse reaction rates")
