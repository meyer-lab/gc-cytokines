"""
This creates Figure 2.
"""
from .figureCommon import subplotLabel, getSetup, traf_names, Rexpr_names
from ..plot_model_prediction import pstat
from ..model import nParams, getTotalActiveSpecies, runCkineU, getSurfaceGCSpecies, runCkineY0, getTotalActiveCytokine
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm, os
from os.path import join
from ..fit_others import build_model



def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 4))

    # Blank out for the cartoon
    ax[0].axis('off')

    subplotLabel(ax[0], 'A')
    pstat_plot(ax[1])
    violinPlots(ax[2:6])
    surf_gc(ax[6], 100.)
    surf_gc(ax[7], 1000.)
    plot_pretreat(ax[8])
    
    

    f.tight_layout()

    return f

def import_samples():
    ''' Imports the csv files into a proper unkVec'''
    bmodel = build_model()
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
        unkVec[:, ii] = np.array([0., 0., 0., 0., 0., 0., kfwd[ii], 1., 1., 1., 1., 1., 1., k27rev[ii], 1., k33rev[ii], 1., 
            endo_activeEndo[ii, 0], endo_activeEndo[ii, 1], sortF[ii], kRec_kDeg[ii, 0], kRec_kDeg[ii, 1], 0., 0.,
            np.squeeze(GCexpr[ii]), 0., np.squeeze(IL7Raexpr[ii]), 0., np.squeeze(IL4Raexpr[ii]), 0.])
    
    return unkVec, scales

def pstat_calc(unkVec, scales, cytokC):
    ''' This function performs the calculations necessary to produce the Gonnord Figures S3B and S3C. '''
    # import function returns from model.py
    activity = getTotalActiveSpecies().astype(np.float64)
    ts = np.array([10.]) # was 10. in literature
    
    def singleCalc(unkVec, cytokine, conc):
        ''' This function generates the active vector for a given unkVec, cytokine, and concentration. '''
        unkVec = unkVec.copy()
        unkVec[cytokine] = conc

        returnn, retVal = runCkineU(ts, unkVec)

        assert retVal >= 0

        return np.dot(returnn, activity)
    
    assert unkVec.size == nParams()
    actVec_IL7 = np.fromiter((singleCalc(unkVec, 2, x) for x in cytokC), np.float64)
    actVec_IL4 = np.fromiter((singleCalc(unkVec, 4, x) for x in cytokC), np.float64)
    
    actVec = np.concatenate((actVec_IL4 * scales[0], actVec_IL7 * scales[1]))
    return actVec
    
def pstat_plot(ax):
    ''' This function calls the pstat_calc function to re-generate Gonnord figures S3B and S3C with our own fitting data. '''
    PTS = 30
    cytokC_4 = np.array([5., 50., 500., 5000., 50000., 250000.]) / 14900. # 14.9 kDa according to sigma aldrich
    cytokC_7 = np.array([1., 10., 100., 1000., 10000., 100000.]) / 17400. # 17.4 kDa according to prospec bio
    cytokC_common = np.logspace(-3.8, 1.5, num=PTS)
    unkVec, scales = import_samples()
    path = os.path.dirname(os.path.abspath(__file__))
    dataIL4 = pd.read_csv(join(path, "../data/Gonnord_S3B.csv")).values # imports IL4 file into pandas array
    dataIL7 = pd.read_csv(join(path, "../data/Gonnord_S3C.csv")).values # imports IL7 file into pandas array
    
    def plot_structure(IL4vec, IL7vec, title, ax):
        ax.set_title(title)
        ax.plot(np.log10(cytokC_common), IL4vec, color='powderblue', alpha=0.5, label="IL4")
        ax.plot(np.log10(cytokC_common), IL7vec, color='b', alpha=0.5, label='IL7') 
        ax.set_ylabel('pSTAT activation' )
        ax.set_xlabel('cytokine concentration (nM)')
        # ax.legend()

    for ii in range(0,500):
        output = pstat_calc(unkVec[:,ii], scales[ii,:], cytokC_common)
        IL4_output = output[0:PTS]
        IL7_output = output[PTS:(PTS*2)]

        plot_structure(IL4_output, IL7_output, "PBMCs stimulated for 10 min.", ax)
        
    ax.scatter(np.log10(cytokC_4), dataIL4[:,1], color='powderblue', marker='^', edgecolors='k', zorder=100)
    ax.scatter(np.log10(cytokC_4), dataIL4[:,2], color='powderblue', marker='^', edgecolors='k', zorder=200)
    ax.scatter(np.log10(cytokC_7), dataIL7[:,1], color='b', marker='^', edgecolors='k', zorder=300)
    ax.scatter(np.log10(cytokC_7), dataIL7[:,2], color='b', marker='^', edgecolors='k', zorder=400)
        
def violinPlots(ax):
    """ Create violin plots of model posterior. """
    unkVec, scales = import_samples()
    unkVec = unkVec.transpose()
    
    rxn = np.array([unkVec[:, 6], unkVec[:, 13], unkVec[:, 15]])
    rxn = rxn.transpose()
    rxn = pd.DataFrame(rxn)
    traf = pd.DataFrame(unkVec[:, 17:22])
    Rexpr = np.array([unkVec[:, 24], unkVec[:, 26], unkVec[:, 28]])
    Rexpr = Rexpr.transpose()
    Rexpr = pd.DataFrame(Rexpr)
    scales = pd.DataFrame(scales)
    
    rxn.columns = ['kfwd', 'k27rev', 'k33rev']
    a = sns.violinplot(data=np.log10(rxn), ax=ax[0])  # creates names based on dataframe columns
    a.set_xticklabels(a.get_xticklabels(),
                       rotation=40,
                       rotation_mode="anchor",
                       ha="right",
                       fontsize=8,
                       position=(0, 0.075))
    
    
    traf.columns = traf_names()
    b = sns.violinplot(data=traf, ax=ax[1])
    b.set_xticklabels(b.get_xticklabels(),
                       rotation=40,
                       rotation_mode="anchor",
                       ha="right",
                       fontsize=8,
                       position=(0, 0.075))
    
    Rexpr.columns = ['GCexpr', 'IL7Raexpr', 'IL4Raexpr']
    c = sns.violinplot(data=Rexpr, ax=ax[2])
    c.set_xticklabels(c.get_xticklabels(),
                       rotation=40,
                       rotation_mode="anchor",
                       ha="right",
                       fontsize=8,
                       position=(0, 0.075))
    
    scales.columns = ['IL4 scale', 'IL7 scale']
    sns.violinplot(data=scales, ax=ax[3])
    

def pretreat_calc(unkVec, pre_IL4, pre_IL7):
    ''' This function performs the calculations necessary to produce the Gonnord Figures S3B and S3C. '''
    # import function returns from model.py
    activity = getTotalActiveSpecies().astype(np.float64)
    ts = np.array([10.]) # was 10. in literature
    IL4_stim_conc = 100. / 14900. # concentration used for IL4 stimulation
    IL7_stim_conc = 50. / 17400. # concentration used for IL7 stimulation
    
    def singleCalc_4stim(unkVec, pre_cytokine, conc):
        ''' This function generates the IL4 active vector for a given unkVec, cytokine used for inhibition and concentration of pretreatment cytokine. '''
        unkVec2 = unkVec.copy()
        unkVec2[pre_cytokine] = conc

        y0, retVal = runCkineU(ts, unkVec2)

        assert retVal >= 0

        unkVec2[4] = IL4_stim_conc # add in IL4 while leaving IL7 in system
        returnn, retVal = runCkineY0(y0, ts, unkVec2)
        
        return getTotalActiveCytokine(4, np.squeeze(returnn)) # only look at active species associated with IL4
    
    def singleCalc_7stim(unkVec, pre_cytokine, conc):
        unkVec2 = unkVec.copy()
        unkVec2[pre_cytokine] = conc

        y0, retVal = runCkineU(ts, unkVec2)

        assert retVal >= 0

        unkVec2[2] = IL7_stim_conc # add in IL7 while leaving IL4 in the system
        returnn, retVal = runCkineY0(y0, ts, unkVec2)
        
        return getTotalActiveCytokine(2, np.squeeze(returnn)) # only look at active species associated with IL7
    
    assert unkVec.size == nParams()
    actVec_IL4stim = np.fromiter((singleCalc_4stim(unkVec, 2, x) for x in pre_IL7), np.float64)
    actVec_IL7stim = np.fromiter((singleCalc_7stim(unkVec, 4, x) for x in pre_IL4), np.float64)
    
    actVec = np.concatenate((actVec_IL4stim, actVec_IL7stim))
    
    def singleCalc_no_pre(unkVec, cytokine, conc):
        ''' This function generates the active vector for a given unkVec, cytokine, and concentration. '''
        unkVec = unkVec.copy()
        unkVec[cytokine] = conc
        returnn, retVal = runCkineU(ts, unkVec)
        assert retVal >= 0
        return np.dot(returnn, activity)
    
    IL4stim_no_pre = singleCalc_no_pre(unkVec, 4, IL4_stim_conc)
    IL7stim_no_pre = singleCalc_no_pre(unkVec, 2, IL7_stim_conc)
    
    result = np.concatenate(((1-(actVec_IL4stim/IL4stim_no_pre)), (1-(actVec_IL7stim/IL7stim_no_pre))))
    return result * 100.


def plot_pretreat(ax):
    unkVec, scales = import_samples()
    path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(join(path, "../data/Gonnord_S3D.csv")).values 
    IL7_pretreat_conc = data[:, 0] / 17400. # concentrations used for IL7 pretreatment followed by IL4 stimulation
    IL4_pretreat_conc = data[:, 5] / 14900. # concentrations used for IL4 pretreatment followed by IL7 stimulation 
    num = 20
    pre_IL4 = np.logspace(-3.8, 1.0, num=num)
    pre_IL7 = np.logspace(-3.8, 1.0, num=num)
    for ii in range(25):
        output = pretreat_calc(unkVec[:, ii], pre_IL4, pre_IL7)
        IL4_stim = output[0:num]
        IL7_stim = output[num:(num*2)]
        ax.plot(np.log10(pre_IL7), IL4_stim, color='powderblue', zorder=ii)
        ax.plot(np.log10(pre_IL4), IL7_stim, color='b', zorder=ii)
    
    ax.set_title('IL-4 and IL-7 crosstalk')
    ax.set_ylim(0,120)
    ax.set_ylabel("Percent inhibition (% x 100)")
    ax.set_xlabel("log10 of pretreatment concentration (nM)")
    
    # add experimental data to plots
    ax.scatter(np.log10(IL7_pretreat_conc), data[:, 1], color='powderblue', zorder=100, marker='^', edgecolors='k', 
              label='IL-4 stim. (IL-7 pre.)')
    ax.scatter(np.log10(IL7_pretreat_conc), data[:, 2], color='powderblue', zorder=101, marker='^', edgecolors='k')
    ax.scatter(np.log10(IL7_pretreat_conc), data[:, 3], color='powderblue', zorder=102, marker='^', edgecolors='k')
    ax.scatter(np.log10(IL4_pretreat_conc), data[:, 6], color='b', zorder=103, marker='^', edgecolors='k', 
              label='IL-7 stim. (IL-4 pre.)')
    ax.scatter(np.log10(IL4_pretreat_conc), data[:, 7], color='b', zorder=104, marker='^', edgecolors='k')
    ax.scatter(np.log10(IL4_pretreat_conc), data[:, 8], color='b', zorder=105, marker='^', edgecolors='k')


    

def surf_gc(ax, cytokC_pg):
    size = 40
    ts = np.linspace(0., 100., num=size)
    output = calc_surf_gc(ts, cytokC_pg)
    IL4vec = output[:, 0:size]
    IL7vec = output[:, size:(size*2)]
    
    for ii in range(500):
        ax.plot(ts, IL4vec[ii, :], color='powderblue', label='IL4', alpha=0.5, zorder=ii)
        ax.plot(ts, IL7vec[ii, :], color='b', label='IL7', alpha=0.5, zorder=ii)
    
    ax.set_title(str(cytokC_pg) + ' pg/mL of stimulation')
    ax.set_ylim(0,120)
    ax.set_ylabel("Surface gamma chain (% x 100)")
    ax.set_xlabel("Time (min)")
    
def calc_surf_gc(t, cytokC_pg):
    gc_species_IDX = getSurfaceGCSpecies()
    unkVec, scales = import_samples()
    
    def singleCalc(unkVec, cytokine, conc, t):
        """ Calculates the surface gc over time for one condition. """
        unkVec = unkVec.copy()
        unkVec[cytokine] = conc
        returnn, retVal = runCkineU(t, unkVec)
        assert retVal >= 0
        a = np.dot(returnn, gc_species_IDX)
        return a
    
    size = t.size
    result = np.zeros((500,size*2))
    for ii in range(500):
        # calculate IL4 stimulation
        a = singleCalc(unkVec[:, ii], 4, (cytokC_pg / 14900.), t)
        # calculate IL7 stimulation
        b = singleCalc(unkVec[:, ii], 2, (cytokC_pg / 17400.), t)
        result[ii, :] = np.concatenate((a, b))
        
    return (result / np.max(result)) * 100.


def data_path():
    path = os.path.dirname(os.path.abspath(__file__))
    dataIL4 = pd.read_csv(join(path, "../data/Gonnord_S3B.csv")).values # imports IL4 file into pandas array
    dataIL7 = pd.read_csv(join(path, "../data/Gonnord_S3C.csv")).values
    data_pretreat = pd.read_csv(join(path, "../data/Gonnord_S3D.csv")).values 
    return (dataIL4, dataIL7, data_pretreat)

