"""
This creates Figure 1.
"""
from .figureCommon import subplotLabel, getSetup, rev_rxn_names, traf_names, Rexpr_names
from ..plot_model_prediction import surf_IL2Rb, pstat, surf_gc
from ..model import nParams
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm, os
from os.path import join
from ..fit import build_model



def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 6), (4, 4))

    subplotLabel(ax[0], 'A')
    
    #surf_perc(ax[0:4], 'IL2Rb')
    #pstat_act(ax[4:6])
    #surf_perc(ax[8:12], 'gc')
    #violinPlots(ax[12:15])

    f.tight_layout()

    return f

def plot_structure(IL2vec, IL15vec, title, ax, x_axis, data_type, species='NONE'):
    ax.set_title(title)
    ax.set_ylim(0,120)
    if (data_type=='surf'):
        ax.scatter(x_axis, IL2vec, color='r', label='IL2', alpha=0.7)
        ax.scatter(x_axis, IL15vec, color='g', label='IL15', alpha=0.7)
        ax.set_ylabel("Surface " + str(species) + " (% x 100)")
        ax.set_xlabel("Time (min)")
    elif (data_type=='act'):
        ax.scatter(np.log10(x_axis), IL2vec, color='r', alpha=0.5, label="IL2")
        ax.scatter(np.log10(x_axis), IL15vec, color='g', alpha=0.5, label='IL15')
        ax.set_ylabel('Maximal p-STAT5 (% x 100)')
        ax.set_xlabel('log10 of cytokine concentration (nM)')
    else:
        print('invalid data_type')   
    # ax.legend()

def surf_perc(ax, species):
    if (species == 'IL2Rb'):
        surf = surf_IL2Rb()
    elif (species == 'gc'):
        surf = surf_gc()
    else:
        print('not a valid species')
        return -1

    y_max = 100.
    ts = np.array([0., 2., 5., 15., 30., 60., 90.])
    size = len(ts)
    unkVec = import_samples()

    for ii in range(0,500):
        output = surf.calc(unkVec[:, ii], ts) * y_max
        IL2_1_plus = output[0:(size)]
        IL2_500_plus = output[(size):(size*2)]
        IL2_1_minus = output[(size*2):(size*3)]
        IL2_500_minus = output[(size*3):(size*4)]
        IL15_1_plus = output[(size*4):(size*5)]
        IL15_500_plus = output[(size*5):(size*6)]
        IL15_1_minus = output[(size*6):(size*7)]
        IL15_500_minus = output[(size*7):(size*8)]

        plot_structure(IL2_1_minus, IL15_1_minus, '1 nM and IL2Ra-', ax[0], ts, 'surf', species)
        plot_structure(IL2_500_minus, IL15_500_minus, "500 nM and IL2Ra-", ax[1], ts, 'surf', species)
        plot_structure(IL2_1_plus, IL15_1_plus, "1 nM and IL2Ra+", ax[2], ts, 'surf', species)
        plot_structure(IL2_500_plus, IL15_500_plus, "500 nM and IL2Ra+", ax[3], ts, 'surf', species)

    
def pstat_act(ax):
    pstat5 = pstat()
    PTS = 30
    cytokC = np.logspace(-3.3, 2.7, PTS)
    y_max = 100.
    unkVec = import_samples()

    for ii in range(0,500):
        output = pstat5.calc(unkVec[:, ii], cytokC) * y_max
        IL2_plus = output[0:PTS]
        IL2_minus = output[PTS:(PTS*2)]
        IL15_plus = output[(PTS*2):(PTS*3)]
        IL15_minus = output[(PTS*3):(PTS*4)]

        plot_structure(IL2_minus, IL15_minus, "IL2Ra- YT-1 cells", ax[0], cytokC, 'act')
        plot_structure(IL2_plus, IL15_plus, "IL2Ra+ YT-1 cells", ax[1], cytokC, 'act')

    
def import_samples():
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
        unkVec[:, ii] = np.array([0., 0., 0., 0., 0., 0., kfwd[ii], rxn[ii, 0], rxn[ii, 1], rxn[ii, 2], rxn[ii, 3], rxn[ii, 4], rxn[ii, 5], rxn[ii, 6], rxn[ii, 7], rxn[ii, 8], rxn[ii, 9], endo_activeEndo[ii, 0], endo_activeEndo[ii, 1], sortF[ii], kRec_kDeg[ii, 0], kRec_kDeg[ii, 1], exprRates[ii, 0], exprRates[ii, 1], exprRates[ii, 2], exprRates[ii, 3], 0., 0., 0., 0.])
    
    return unkVec

def violinPlots(ax):
    """ Create violin plots of model posterior. """
    unkVec = import_samples()
    unkVec = unkVec.transpose()
    
    rev_rxn = pd.DataFrame(unkVec[:, 7:17])
    traf = pd.DataFrame(unkVec[:, 17:22])
    Rexpr = pd.DataFrame(unkVec[:, 22:])
    
    rev_rxn.columns = rev_rxn_names()
    sns.violinplot(data=rev_rxn, ax=ax[0])  # creates names based on dataframe columns
    
    traf.columns = traf_names()
    sns.violinplot(data=traf, ax=ax[1])
    
    Rexpr.columns = Rexpr_names()
    sns.violinplot(data=Rexpr, ax=ax[2])

