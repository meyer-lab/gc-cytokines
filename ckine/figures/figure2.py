"""
This creates Figure 2.
"""
from os.path import join
import os
import string
import numpy as np
import seaborn as sns
import pandas as pd
from .figureCommon import subplotLabel, getSetup, traf_names, plot_conf_int, import_samples_4_7
from ..model import nParams, getTotalActiveSpecies, runCkineU, getSurfaceGCSpecies, runCkinePreT, getTotalActiveCytokine

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 9), (3, 3))

    # Blank out for the cartoon
    ax[0].axis('off')

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    unkVec, scales = import_samples_4_7()
    pstat_plot(ax[1], unkVec, scales)
    plot_pretreat(ax[2], unkVec, scales, "Cross-talk pSTAT inhibition")
    surf_gc(ax[3], 100., unkVec)
    violinPlots(ax[4:8], unkVec, scales)

    unkVec_noActiveEndo = unkVec.copy()
    unkVec_noActiveEndo[18] = 0.0   # set activeEndo rate to 0
    plot_pretreat(ax[8], unkVec_noActiveEndo, scales, "Inhibition without active endocytosis")

    f.tight_layout()

    return f

def pstat_calc(unkVec, scales, cytokC):
    ''' This function performs the calculations necessary to produce the Gonnord Figures S3B and S3C. '''
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
    actVecIL7 = np.fromiter((singleCalc(unkVec, 2, x) for x in cytokC), np.float64)
    actVecIL4 = np.fromiter((singleCalc(unkVec, 4, x) for x in cytokC), np.float64)

    # incorporate IC50 scale
    actVecIL4 = actVecIL4  / (actVecIL4 + scales[0])
    actVecIL7 = actVecIL7 / (actVecIL7 + scales[1])

    # normalize each actVec by its maximum... do I need to be doing this?
    actVecIL4 = actVecIL4 / np.max(actVecIL4)
    actVecIL7 = actVecIL7 / np.max(actVecIL7)
    return np.concatenate((actVecIL4, actVecIL7))

def pstat_plot(ax, unkVec, scales):
    ''' This function calls the pstat_calc function to re-generate Gonnord figures S3B and S3C with our own fitting data. '''
    PTS = 30
    cytokC_4 = np.array([5., 50., 500., 5000., 50000., 250000.]) / 14900. # 14.9 kDa according to sigma aldrich
    cytokC_7 = np.array([1., 10., 100., 1000., 10000., 100000.]) / 17400. # 17.4 kDa according to prospec bio
    cytokC_common = np.logspace(-3.8, 1.5, num=PTS)
    path = os.path.dirname(os.path.abspath(__file__))
    dataIL4 = pd.read_csv(join(path, "../data/Gonnord_S3B.csv")).values # imports IL4 file into pandas array
    dataIL7 = pd.read_csv(join(path, "../data/Gonnord_S3C.csv")).values # imports IL7 file into pandas array
    IL4_data_max = np.amax(np.concatenate((dataIL4[:,1], dataIL4[:,2])))
    IL7_data_max = np.amax(np.concatenate((dataIL7[:,1], dataIL7[:,2])))

    IL4_output = np.zeros((PTS, 500))
    IL7_output = IL4_output.copy()
    for ii in range(0,500):
        output = pstat_calc(unkVec[:,ii], scales[ii,:], cytokC_common)
        IL4_output[:, ii] = output[0:PTS]
        IL7_output[:, ii] = output[PTS:(PTS*2)]

    # plot confidence intervals based on model predictions
    plot_conf_int(ax, np.log10(cytokC_common), IL4_output, "powderblue", "IL4")
    plot_conf_int(ax, np.log10(cytokC_common), IL7_output, "b", "IL7")

    # overlay experimental data
    ax.scatter(np.log10(cytokC_4), dataIL4[:,1] / IL4_data_max, color='powderblue', marker='^', edgecolors='k', zorder=100)
    ax.scatter(np.log10(cytokC_4), dataIL4[:,2] / IL4_data_max, color='powderblue', marker='^', edgecolors='k', zorder=200)
    ax.scatter(np.log10(cytokC_7), dataIL7[:,1] / IL7_data_max, color='b', marker='^', edgecolors='k', zorder=300)
    ax.scatter(np.log10(cytokC_7), dataIL7[:,2] / IL7_data_max, color='b', marker='^', edgecolors='k', zorder=400)
    ax.set(ylabel='Fraction of maximal p-STAT', xlabel='log10 of stimulation concentration (nM)', title="pSTAT activity")
    ax.legend()

def violinPlots(ax, unkVec, scales):
    """ Create violin plots of model posterior. """
    unkVec = unkVec.transpose()

    rxn = np.array([unkVec[:, 6], unkVec[:, 13], unkVec[:, 15]]) # kfwd, k27rev, k33rev
    rxn = rxn.transpose()
    rxn = pd.DataFrame(rxn)
    traf = pd.DataFrame(unkVec[:, 17:22])
    Rexpr = np.array([unkVec[:, 24], unkVec[:, 26], unkVec[:, 28]])
    Rexpr = Rexpr.transpose()
    Rexpr = pd.DataFrame(Rexpr)
    scales = pd.DataFrame(scales)

    rxn.columns = ['kfwd', 'k27rev', 'k33rev']
    a = sns.violinplot(data=np.log10(rxn), ax=ax[0])  # creates names based on dataframe columns
    a.set_xticklabels(a.get_xticklabels(), rotation=40, rotation_mode="anchor", ha="right", fontsize=8, position=(0,0.045))
    a.set_ylabel("log10 of rate")
    a.set_title("Reverse reaction rates")

    traf.columns = traf_names()
    b = sns.violinplot(data=traf, ax=ax[1])
    b.set_xticklabels(b.get_xticklabels(), rotation=40, rotation_mode="anchor", ha="right", fontsize=8, position=(0,0.045))
    b.set_ylabel("Rate (1/min)")
    b.set_title("Trafficking parameters")

    Rexpr.columns = ['GCexpr', 'IL7Raexpr', 'IL4Raexpr']
    c = sns.violinplot(data=Rexpr, ax=ax[2])
    c.set_xticklabels(c.get_xticklabels(), rotation=40, rotation_mode="anchor", ha="right", fontsize=8, position=(0,0.045))
    c.set_ylabel("Rate (#/cell/min)")
    c.set_title("Receptor expression rates")

    scales.columns = ['IL4 scale', 'IL7 scale']
    d = sns.violinplot(data=scales, ax=ax[3])
    d.set_xticklabels(d.get_xticklabels(), rotation=40, rotation_mode="anchor", ha="right", fontsize=8, position=(0,0.045))


def pretreat_calc(unkVec, scales, pre_conc):
    ''' This function performs the calculations necessary to produce the Gonnord Figures S3B and S3C. '''
    activity = getTotalActiveSpecies().astype(np.float64)
    ts = np.array([10.]) # was 10. in literature
    IL4_stim_conc = 100. / 14900. # concentration used for IL4 stimulation
    IL7_stim_conc = 50. / 17400. # concentration used for IL7 stimulation

    def singleCalc(unkVec, pre_cytokine, pre_conc, stim_cytokine, stim_conc):
        """ Calculate for single case. """
        unkVec2 = unkVec.copy()
        unkVec2[pre_cytokine] = pre_conc
        ligands = np.zeros((6))
        ligands[stim_cytokine] = stim_conc
        ligands[pre_cytokine] = pre_conc # pretreatment ligand stays in system
        returnn, retVal = runCkinePreT(ts, ts, unkVec2, ligands)
        assert retVal >= 0
        return getTotalActiveCytokine(stim_cytokine, np.squeeze(returnn)) # only look at active species associated with IL4

    assert unkVec.size == nParams()
    actVec_IL4stim = np.fromiter((singleCalc(unkVec, 2, x, 4, IL4_stim_conc) for x in pre_conc), np.float64)
    actVec_IL7stim = np.fromiter((singleCalc(unkVec, 4, x, 2, IL7_stim_conc) for x in pre_conc), np.float64)

    # incorporate IC50
    actVec_IL4stim = actVec_IL4stim  / (actVec_IL4stim + scales[0])
    actVec_IL7stim = actVec_IL7stim  / (actVec_IL7stim + scales[1])

    def singleCalc_no_pre(unkVec, cytokine, conc):
        ''' This function generates the active vector for a given unkVec, cytokine, and concentration. '''
        unkVec = unkVec.copy()
        unkVec[cytokine] = conc
        returnn, retVal = runCkineU(ts, unkVec)
        assert retVal >= 0
        return np.dot(returnn, activity)

    IL4stim_no_pre = singleCalc_no_pre(unkVec, 4, IL4_stim_conc)
    IL7stim_no_pre = singleCalc_no_pre(unkVec, 2, IL7_stim_conc)

    # incorporate IC50
    IL4stim_no_pre = IL4stim_no_pre  / (IL4stim_no_pre + scales[0])
    IL7stim_no_pre = IL7stim_no_pre  / (IL7stim_no_pre + scales[1])

    return np.concatenate(((1-(actVec_IL4stim/IL4stim_no_pre)), (1-(actVec_IL7stim/IL7stim_no_pre))))


def plot_pretreat(ax, unkVec, scales, title):
    """ Generates plots that mimic the percent inhibition after pretreatment in Gonnord Fig S3. """
    path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(join(path, "../data/Gonnord_S3D.csv")).values
    IL7_pretreat_conc = data[:, 0] / 17400. # concentrations used for IL7 pretreatment followed by IL4 stimulation
    IL4_pretreat_conc = data[:, 5] / 14900. # concentrations used for IL4 pretreatment followed by IL7 stimulation
    PTS = 30
    pre_conc = np.logspace(-3.8, 1.0, num=PTS)
    IL4_stim = np.zeros((PTS, 500))
    IL7_stim = IL4_stim.copy()

    for ii in range(500):
        output = pretreat_calc(unkVec[:, ii], scales[ii, :], pre_conc)
        IL4_stim[:, ii] = output[0:PTS]
        IL7_stim[:, ii] = output[PTS:(PTS*2)]

    plot_conf_int(ax, np.log10(pre_conc), IL4_stim, "powderblue", "IL-4 stim. (IL-7 pre.)")
    plot_conf_int(ax, np.log10(pre_conc), IL7_stim, "b", "IL-7 stim. (IL-4 pre.)")
    ax.set(title=title, ylabel="Fraction of inhibition", xlabel="log10 of pretreatment concentration (nM)")

    # add experimental data to plots
    ax.scatter(np.log10(IL7_pretreat_conc), data[:, 1] / 100., color='powderblue', zorder=100, marker='^', edgecolors='k')
    ax.scatter(np.log10(IL7_pretreat_conc), data[:, 2] / 100., color='powderblue', zorder=101, marker='^', edgecolors='k')
    ax.scatter(np.log10(IL7_pretreat_conc), data[:, 3] / 100., color='powderblue', zorder=102, marker='^', edgecolors='k')
    ax.scatter(np.log10(IL4_pretreat_conc), data[:, 6] / 100., color='b', zorder=103, marker='^', edgecolors='k')
    ax.scatter(np.log10(IL4_pretreat_conc), data[:, 7] / 100., color='b', zorder=104, marker='^', edgecolors='k')
    ax.scatter(np.log10(IL4_pretreat_conc), data[:, 8] / 100., color='b', zorder=105, marker='^', edgecolors='k')
    ax.legend()


def surf_gc(ax, cytokC_pg, unkVec):
    """ Generate a plot that shows the relative amount of gc on the cell surface under IL4 and IL7 stimulation. """
    PTS = 40
    ts = np.linspace(0., 100., num=PTS)
    output = calc_surf_gc(ts, cytokC_pg, unkVec)
    IL4vec = np.transpose(output[:, 0:PTS])
    IL7vec = np.transpose(output[:, PTS:(PTS*2)])
    plot_conf_int(ax, ts, IL4vec, "powderblue", "IL4")
    plot_conf_int(ax, ts, IL7vec, "b", "IL7")
    ax.set(title=("Ligand conc.: " + str(cytokC_pg) + ' pg/mL'), ylabel="surface gamma chain (% x 100)", xlabel="time (min)")
    ax.legend()

def calc_surf_gc(t, cytokC_pg, unkVec):
    """ Calculates the percent of gc on the surface over time while under IL7 and IL4 stimulation. """
    gc_species_IDX = getSurfaceGCSpecies()

    def singleCalc(unkVec, cytokine, conc, t):
        """ Calculates the surface gc over time for one condition. """
        unkVec = unkVec.copy()
        unkVec[cytokine] = conc
        returnn, retVal = runCkineU(t, unkVec)
        assert retVal >= 0
        a = np.dot(returnn, gc_species_IDX)
        return a

    result = np.zeros((500,(t.size)*2))
    for ii in range(500):
        # calculate IL4 stimulation
        a = singleCalc(unkVec[:, ii], 4, (cytokC_pg / 14900.), t)
        # calculate IL7 stimulation
        b = singleCalc(unkVec[:, ii], 2, (cytokC_pg / 17400.), t)
        result[ii, :] = np.concatenate((a, b))

    return (result / np.max(result)) * 100.

def data_path():
    """ Loads the Gonnard data from the appropriate CSV files. """
    path = os.path.dirname(os.path.abspath(__file__))
    dataIL4 = pd.read_csv(join(path, "../data/Gonnord_S3B.csv")).values # imports IL4 file into pandas array
    dataIL7 = pd.read_csv(join(path, "../data/Gonnord_S3C.csv")).values
    data_pretreat = pd.read_csv(join(path, "../data/Gonnord_S3D.csv")).values
    return (dataIL4, dataIL7, data_pretreat)
