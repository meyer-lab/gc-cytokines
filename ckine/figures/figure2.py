"""
This creates Figure 2.
"""
from os.path import join
import os
import string
import numpy as np
import seaborn as sns
import pandas as pd
from .figureCommon import subplotLabel, getSetup, traf_names, plot_conf_int, import_samples_4_7, kfwd_info
from ..model import nParams, getTotalActiveSpecies, runCkineUP, getSurfaceGCSpecies, getTotalActiveCytokine

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 9), (3, 3))

    # Blank out for the cartoon
    ax[0].axis('off')

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    unkVec, scales = import_samples_4_7()
    kfwd_avg, kfwd_std = kfwd_info(unkVec)
    print("kfwd = " + str(kfwd_avg) + " +/- " + str(kfwd_std))
    pstat_plot(ax[1], unkVec, scales)
    plot_pretreat(ax[2], unkVec, scales, "Cross-talk pSTAT inhibition")
    surf_gc(ax[7], 100., unkVec)
    violinPlots(ax[3:7], unkVec, scales)

    unkVec_noActiveEndo = unkVec.copy()
    unkVec_noActiveEndo[18] = 0.0   # set activeEndo rate to 0
    plot_pretreat(ax[8], unkVec_noActiveEndo, scales, "Inhibition without active endocytosis")

    f.tight_layout()

    return f

def pstat_calc(unkVec, scales, cytokC):
    ''' This function performs the calculations necessary to produce the Gonnord Figures S3B and S3C. '''
    activity = getTotalActiveSpecies().astype(np.float64)
    ts = np.array([10.]) # was 10. in literature
    assert unkVec.shape[0] == nParams()
    K = unkVec.shape[1] # should be 500

    def parallelCalc(unkVec, cytokine, conc):
        ''' This function generates the active vector for a given 2D unkVec, cytokine, and concentration. '''
        unkVec = unkVec.copy()
        unkVec[cytokine, :] = conc
        unkVec = np.transpose(unkVec).copy() # transpose the matrix (save view as a new copy)
        returnn, retVal = runCkineUP(ts, unkVec)
        assert retVal >= 0
        return np.dot(returnn, activity)

    # find cytokine activity under various stimulation concentrations
    actVecIL7 = np.zeros((K, len(cytokC)))
    actVecIL4 = actVecIL7.copy()
    for x, conc in enumerate(cytokC):
        actVecIL7[:, x] = parallelCalc(unkVec, 2, conc)
        actVecIL4[:, x] = parallelCalc(unkVec, 4, conc)

    for ii in range(K):
        # incorporate IC50 (sigmoidal) scale
        actVecIL4[ii] = actVecIL4[ii]  / (actVecIL4[ii] + scales[ii, 0])
        actVecIL7[ii] = actVecIL7[ii] / (actVecIL7[ii] + scales[ii, 1])
        # normalize from 0-1
        actVecIL4[ii] = actVecIL4[ii] / np.max(actVecIL4[ii])
        actVecIL7[ii] = actVecIL7[ii] / np.max(actVecIL7[ii])

    return np.concatenate((actVecIL4, actVecIL7))

def pstat_plot(ax, unkVec, scales):
    ''' This function calls the pstat_calc function to re-generate Gonnord figures S3B and S3C with our own fitting data. '''
    PTS = 30
    K = unkVec.shape[1] # should be 500
    cytokC_4 = np.array([5., 50., 500., 5000., 50000., 250000.]) / 14900. # 14.9 kDa according to sigma aldrich
    cytokC_7 = np.array([1., 10., 100., 1000., 10000., 100000.]) / 17400. # 17.4 kDa according to prospec bio
    cytokC_common = np.logspace(-3.8, 1.5, num=PTS)
    path = os.path.dirname(os.path.abspath(__file__))
    dataIL4 = pd.read_csv(join(path, "../data/Gonnord_S3B.csv")).values # imports IL4 file into pandas array
    dataIL7 = pd.read_csv(join(path, "../data/Gonnord_S3C.csv")).values # imports IL7 file into pandas array
    IL4_data_max = np.amax(np.concatenate((dataIL4[:,1], dataIL4[:,2])))
    IL7_data_max = np.amax(np.concatenate((dataIL7[:,1], dataIL7[:,2])))

    output = pstat_calc(unkVec, scales, cytokC_common) # run simulation
    # split according to cytokine and transpose for input into plot_conf_int
    IL4_output = output[0:K].T
    IL7_output = output[K:(K*2)].T

    # plot confidence intervals based on model predictions
    plot_conf_int(ax, np.log10(cytokC_common), IL4_output * 100., "powderblue", "IL-4")
    plot_conf_int(ax, np.log10(cytokC_common), IL7_output * 100., "b", "IL-7")

    # overlay experimental data
    ax.scatter(np.log10(cytokC_4), (dataIL4[:,1] / IL4_data_max) * 100., color='powderblue', marker='^', edgecolors='k', zorder=100)
    ax.scatter(np.log10(cytokC_4), (dataIL4[:,2] / IL4_data_max) * 100., color='powderblue', marker='^', edgecolors='k', zorder=200)
    ax.scatter(np.log10(cytokC_7), (dataIL7[:,1] / IL7_data_max) * 100., color='b', marker='^', edgecolors='k', zorder=300)
    ax.scatter(np.log10(cytokC_7), (dataIL7[:,2] / IL7_data_max) * 100., color='b', marker='^', edgecolors='k', zorder=400)
    ax.set(ylabel='pSTAT5/6 (% of max)', xlabel=r'Cytokine concentration (log$_{10}$[nM])', title='PBMC activity')
    ax.legend()

def violinPlots(ax, unkVec, scales):
    """ Create violin plots of model posterior. """
    unkVec = unkVec.transpose()

    rxn = np.array([unkVec[:, 13], unkVec[:, 15]]) #k27rev, k33rev
    rxn = rxn.transpose()
    rxn = pd.DataFrame(rxn)
    traf = pd.DataFrame(unkVec[:, 17:22])
    Rexpr = np.array([unkVec[:, 24], unkVec[:, 26], unkVec[:, 28]])
    Rexpr = Rexpr.transpose()
    Rexpr = pd.DataFrame(Rexpr)
    scales = pd.DataFrame(scales)

    rxn.columns = [r'$k_{27}$', r'$k_{33}$']
    a = sns.violinplot(data=np.log10(rxn), ax=ax[0], linewidth=0.5)  # creates names based on dataframe columns
    a.set_ylabel(r"$\mathrm{log_{10}(\frac{1}{min})}$")
    a.set_title("Reverse reaction rates")

    traf.columns = traf_names()
    b = sns.violinplot(data=np.log10(traf), ax=ax[1], linewidth=0.5)
    b.set_xticklabels(b.get_xticklabels(), rotation=40, rotation_mode="anchor", ha="right", fontsize=8, position=(0,0.045))
    b.set_ylabel(r"$\mathrm{log_{10}(\frac{1}{min})}$")
    b.set_title("Trafficking parameters")

    Rexpr.columns = [r'$\gamma_{c}$', 'IL-7Rα', 'IL-4Rα']
    c = sns.violinplot(data=np.log10(Rexpr), ax=ax[2], linewidth=0.5)
    c.set_xticklabels(c.get_xticklabels(), rotation=40, rotation_mode="anchor", ha="right", fontsize=8, position=(0,0.045))
    c.set_ylabel(r"$\mathrm{log_{10}(\frac{num}{cell * min})}$")
    c.set_title("Receptor expression rates")

    scales.columns = [r'$C_{6}$', r'$C_{5}$']
    d = sns.violinplot(data=scales, ax=ax[3], linewidth=0.5)
    d.set_ylabel("value")
    d.set_title("pSTAT scaling constants")

def pretreat_calc(unkVec, scales, pre_conc):
    ''' This function performs the calculations necessary to produce the Gonnord Figures S3B and S3C. '''
    activity = getTotalActiveSpecies().astype(np.float64)
    ts = np.array([10.]) # was 10. in literature
    IL4_stim_conc = 100. / 14900. # concentration used for IL4 stimulation
    IL7_stim_conc = 50. / 17400. # concentration used for IL7 stimulation
    assert unkVec.shape[0] == nParams()
    K = unkVec.shape[1] # should be 500
    N = len(pre_conc)

    def parallelCalc(unkVec, pre_cytokine, pre_conc, stim_cytokine, stim_conc):
        """ Calculate pSTAT activity for single case pretreatment case. Simulation run in parallel. """
        unkVec2 = unkVec.copy()
        unkVec2[pre_cytokine, :] = pre_conc
        unkVec2[stim_cytokine, :] = stim_conc
        ligands = np.zeros(6)
        ligands[pre_cytokine] = pre_conc # pretreatment ligand stays in system
        unkVec2 = np.transpose(unkVec2).copy() # transpose the matrix (save view as a new copy)

        returnn, retVal = runCkineUP(ts, unkVec2, preT=ts, prestim=ligands)
        assert retVal >= 0
        ret = np.zeros((returnn.shape[0]))
        for ii in range(returnn.shape[0]):
            ret[ii] = getTotalActiveCytokine(stim_cytokine, np.squeeze(returnn[ii])) # only look at active species associated with the active cytokine
        return ret

    # run two-cytokine simulation for varying pretreatment concnetrations
    actVec_IL4stim = np.zeros((K, N))
    actVec_IL7stim = actVec_IL4stim.copy()
    for x in range(N):
        actVec_IL4stim[:, x] = parallelCalc(unkVec, 2, pre_conc[x], 4, IL4_stim_conc)
        actVec_IL7stim[:, x] = parallelCalc(unkVec, 4, pre_conc[x], 2, IL7_stim_conc)

    def parallelCalc_no_pre(unkVec, cytokine, conc):
        ''' This function generates the active vector for a given 2D unkVec, cytokine, and concentration. '''
        unkVec = unkVec.copy()
        unkVec[cytokine, :] = conc
        unkVec = np.transpose(unkVec).copy() # transpose the matrix (save view as a new copy)
        returnn, retVal = runCkineUP(ts, unkVec)
        assert retVal >= 0
        return np.dot(returnn, activity)

    # run simulation with just one cytokine
    IL4stim_no_pre = parallelCalc_no_pre(unkVec, 4, IL4_stim_conc)
    IL7stim_no_pre = parallelCalc_no_pre(unkVec, 2, IL7_stim_conc)

    ret1 = np.zeros((K, N)) # arrays to hold inhibition calculation
    ret2 = ret1.copy()
    # incorporate IC50 and find inhibition
    for ii in range(K):
        actVec_IL4stim[ii] = actVec_IL4stim[ii] / (actVec_IL4stim[ii] + scales[ii, 0])
        actVec_IL7stim[ii] = actVec_IL7stim[ii] / (actVec_IL7stim[ii] + scales[ii, 1])
        IL4stim_no_pre[ii] = IL4stim_no_pre[ii] / (IL4stim_no_pre[ii] + scales[ii, 0])
        IL7stim_no_pre[ii] = IL7stim_no_pre[ii] / (IL7stim_no_pre[ii] + scales[ii, 1])
        ret1[ii] = 1 - (actVec_IL4stim[ii] / IL4stim_no_pre[ii])
        ret2[ii] = 1 - (actVec_IL7stim[ii] / IL7stim_no_pre[ii])

    return np.concatenate((ret1, ret2))

def plot_pretreat(ax, unkVec, scales, title):
    """ Generates plots that mimic the percent inhibition after pretreatment in Gonnord Fig S3. """
    path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(join(path, "../data/Gonnord_S3D.csv")).values
    IL7_pretreat_conc = data[:, 0] / 17400. # concentrations used for IL7 pretreatment followed by IL4 stimulation
    IL4_pretreat_conc = data[:, 5] / 14900. # concentrations used for IL4 pretreatment followed by IL7 stimulation
    PTS = 30
    K = unkVec.shape[1] # should be 500
    pre_conc = np.logspace(-3.8, 1.0, num=PTS)

    output = pretreat_calc(unkVec, scales, pre_conc) # run simulation
    # split according to cytokine and transpose so it works with plot_conf_int
    IL4_stim = output[0:K].T
    IL7_stim = output[K:(K*2)].T

    plot_conf_int(ax, np.log10(pre_conc), IL4_stim * 100., "powderblue", "IL-4 stim. (IL-7 pre.)")
    plot_conf_int(ax, np.log10(pre_conc), IL7_stim * 100., "b", "IL-7 stim. (IL-4 pre.)")
    ax.set(title=title, ylabel="Inhibition (% of no pretreat)", xlabel=r'Pretreatment concentration (log$_{10}$[nM])')

    # add experimental data to plots
    ax.scatter(np.log10(IL7_pretreat_conc), data[:, 1], color='powderblue', zorder=100, marker='^', edgecolors='k')
    ax.scatter(np.log10(IL7_pretreat_conc), data[:, 2], color='powderblue', zorder=101, marker='^', edgecolors='k')
    ax.scatter(np.log10(IL7_pretreat_conc), data[:, 3], color='powderblue', zorder=102, marker='^', edgecolors='k')
    ax.scatter(np.log10(IL4_pretreat_conc), data[:, 6], color='b', zorder=103, marker='^', edgecolors='k')
    ax.scatter(np.log10(IL4_pretreat_conc), data[:, 7], color='b', zorder=104, marker='^', edgecolors='k')
    ax.scatter(np.log10(IL4_pretreat_conc), data[:, 8], color='b', zorder=105, marker='^', edgecolors='k')
    ax.legend()


def surf_gc(ax, cytokC_pg, unkVec):
    """ Generate a plot that shows the relative amount of gc on the cell surface under IL4 and IL7 stimulation. """
    PTS = 40
    ts = np.linspace(0., 100., num=PTS)
    output = calc_surf_gc(ts, cytokC_pg, unkVec)
    IL4vec = np.transpose(output[:, 0:PTS])
    IL7vec = np.transpose(output[:, PTS:(PTS*2)])
    plot_conf_int(ax, ts, IL4vec, "powderblue", "IL-4")
    plot_conf_int(ax, ts, IL7vec, "b", "IL-7")
    ax.set(title=("Ligand conc: " + str(round(cytokC_pg, 0)) + ' pg/mL'), ylabel=r"Surface $\gamma_{c}$ (%)", xlabel="Time (min)")
    ax.set_ylim(0,115)
    ax.legend()

def calc_surf_gc(t, cytokC_pg, unkVec):
    """ Calculates the percent of gc on the surface over time while under IL7 and IL4 stimulation. """
    gc_species_IDX = getSurfaceGCSpecies()
    PTS = len(t)
    K = unkVec.shape[1]

    def parallelCalc(unkVec, cytokine, conc, t):
        """ Calculates the surface gc over time for one condition. """
        unkVec = unkVec.copy()
        unkVec[cytokine, :] = conc
        unkVec = np.transpose(unkVec).copy() # transpose the matrix (save view as a new copy)
        returnn, retVal = runCkineUP(t, unkVec)
        assert retVal >= 0
        return np.dot(returnn, gc_species_IDX)

    # calculate IL4 stimulation
    a = parallelCalc(unkVec, 4, (cytokC_pg / 14900.), t).reshape((K,PTS))
    # calculate IL7 stimulation
    b = parallelCalc(unkVec, 2, (cytokC_pg / 17400.), t).reshape((K,PTS))
    # concatenate results and normalize to 100%
    result = np.concatenate((a, b), axis=1)
    return (result / np.max(result)) * 100.

def data_path():
    """ Loads the Gonnard data from the appropriate CSV files. """
    path = os.path.dirname(os.path.abspath(__file__))
    dataIL4 = pd.read_csv(join(path, "../data/Gonnord_S3B.csv")).values # imports IL4 file into pandas array
    dataIL7 = pd.read_csv(join(path, "../data/Gonnord_S3C.csv")).values
    data_pretreat = pd.read_csv(join(path, "../data/Gonnord_S3D.csv")).values
    return (dataIL4, dataIL7, data_pretreat)
