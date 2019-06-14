"""
This creates Figure 2.
"""
from os.path import join
import os
import string
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
from .figureCommon import subplotLabel, getSetup, traf_names, plot_conf_int, kfwd_info
from ..model import nParams, getTotalActiveSpecies, runCkineUP, getSurfaceGCSpecies, getTotalActiveCytokine
from ..imports import import_samples_4_7, import_samples_2_15


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 7), (3, 3))

    # Blank out for the cartoon
    ax[0].axis('off')

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    full_unkVec_2_15, _ = import_samples_2_15()
    full_unkVec_4_7, full_scales_4_7 = import_samples_4_7()  # full version used for violin plots
    unkVec_4_7, scales_4_7 = import_samples_4_7(N=100)  # a subsampled version used for simulation

    kfwd_avg, kfwd_std = kfwd_info(full_unkVec_4_7)
    print("kfwd = " + str(kfwd_avg) + " +/- " + str(kfwd_std))
    pstat_plot(ax[1], unkVec_4_7, scales_4_7)
    plot_pretreat(ax[2], unkVec_4_7, scales_4_7, "Cross-talk pSTAT inhibition")
    traf_violin(ax[6], full_unkVec_4_7)
    rexpr_violin(ax[7], full_unkVec_4_7)
    misc_violin(ax[8], full_unkVec_4_7, full_scales_4_7)
    surf_gc(ax[4], 100., full_unkVec_4_7)
    unkVec_noActiveEndo = unkVec_4_7.copy()
    unkVec_noActiveEndo[18] = 0.0   # set activeEndo rate to 0
    plot_pretreat(ax[3], unkVec_noActiveEndo, scales_4_7, "Cross-talk w/o active endocytosis")

    relativeGC(ax[5], full_unkVec_2_15, full_unkVec_4_7)  # plot last to avoid coloring all other violins purple

    f.tight_layout()

    return f


def pstat_calc(unkVec, scales, cytokC):
    ''' This function performs the calculations necessary to produce the Gonnord Figures S3B and S3C. '''
    activity = getTotalActiveSpecies().astype(np.float64)
    ts = np.array([10.])  # was 10. in literature
    assert unkVec.shape[0] == nParams()
    K = unkVec.shape[1]  # should be 500

    def parallelCalc(unkVec, cytokine, conc):
        ''' This function generates the active vector for a given 2D unkVec, cytokine, and concentration. '''
        unkVec = unkVec.copy()
        unkVec[cytokine, :] = conc
        unkVec = np.transpose(unkVec).copy()  # transpose the matrix (save view as a new copy)
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
        actVecIL4[ii] = actVecIL4[ii] / (actVecIL4[ii] + scales[ii, 0])
        actVecIL7[ii] = actVecIL7[ii] / (actVecIL7[ii] + scales[ii, 1])
        # normalize from 0-1
        actVecIL4[ii] = actVecIL4[ii] / np.max(actVecIL4[ii])
        actVecIL7[ii] = actVecIL7[ii] / np.max(actVecIL7[ii])

    return np.concatenate((actVecIL4, actVecIL7))


def pstat_plot(ax, unkVec, scales):
    ''' This function calls the pstat_calc function to re-generate Gonnord figures S3B and S3C with our own fitting data. '''
    PTS = 30
    K = unkVec.shape[1]  # should be 500
    cytokC_4 = np.array([5., 50., 500., 5000., 50000., 250000.]) / 14900.  # 14.9 kDa according to sigma aldrich
    cytokC_7 = np.array([1., 10., 100., 1000., 10000., 100000.]) / 17400.  # 17.4 kDa according to prospec bio
    cytokC_common = np.logspace(-3.8, 1.5, num=PTS)
    dataIL4 = pd.read_csv(join(os.path.dirname(os.path.abspath(__file__)), "../data/Gonnord_S3B.csv")).values  # imports IL4 file into pandas array
    dataIL7 = pd.read_csv(join(os.path.dirname(os.path.abspath(__file__)), "../data/Gonnord_S3C.csv")).values  # imports IL7 file into pandas array
    IL4_data_max = np.amax(np.concatenate((dataIL4[:, 1], dataIL4[:, 2])))
    IL7_data_max = np.amax(np.concatenate((dataIL7[:, 1], dataIL7[:, 2])))

    output = pstat_calc(unkVec, scales, cytokC_common)  # run simulation
    # split according to cytokine and transpose for input into plot_conf_int
    IL4_output = output[0:K].T
    IL7_output = output[K:(K * 2)].T

    # plot confidence intervals based on model predictions
    plot_conf_int(ax, np.log10(cytokC_common), IL4_output * 100., "powderblue")
    plot_conf_int(ax, np.log10(cytokC_common), IL7_output * 100., "b")

    # overlay experimental data
    ax.scatter(np.log10(cytokC_4), (dataIL4[:, 1] / IL4_data_max) * 100., color='powderblue', marker='^', edgecolors='k', zorder=100)
    ax.scatter(np.log10(cytokC_4), (dataIL4[:, 2] / IL4_data_max) * 100., color='powderblue', marker='^', edgecolors='k', zorder=200)
    ax.scatter(np.log10(cytokC_7), (dataIL7[:, 1] / IL7_data_max) * 100., color='b', marker='^', edgecolors='k', zorder=300)
    ax.scatter(np.log10(cytokC_7), (dataIL7[:, 2] / IL7_data_max) * 100., color='b', marker='^', edgecolors='k', zorder=400)
    ax.set(ylabel='pSTAT5/6 (% of max)', xlabel=r'Cytokine concentration (log$_{10}$[nM])', title='PBMC activity')


def traf_violin(ax, unkVec):
    """ Create violin plot of trafficking parameters. """
    unkVec = unkVec.transpose()
    traf = np.concatenate((unkVec[:, 17:19], unkVec[:, 20:22]), axis=1)
    traf = pd.DataFrame(traf)

    traf.columns = traf_names()
    a = sns.violinplot(data=np.log10(traf), ax=ax, linewidth=0.5, color="grey")
    a.set_xticklabels(a.get_xticklabels(), rotation=25, rotation_mode="anchor", ha="right", fontsize=8, position=(0, 0.04))
    a.set_ylabel(r"$\mathrm{log_{10}(\frac{1}{min})}$")
    a.set_title("Trafficking parameters")


def rexpr_violin(ax, unkVec):
    """ Create violin plot of receptor expression rates. """
    unkVec = unkVec.transpose()
    Rexpr = np.array([unkVec[:, 24], unkVec[:, 26], unkVec[:, 28]])
    Rexpr = Rexpr.transpose()
    Rexpr = pd.DataFrame(Rexpr)

    Rexpr.columns = [r'$\gamma_{c}$', 'IL-7Rα', 'IL-4Rα']
    col_list = ["grey", "blue", "lightblue"]
    col_list_palette = sns.xkcd_palette(col_list)
    a = sns.violinplot(data=np.log10(Rexpr), ax=ax, linewidth=0.5, palette=col_list_palette)
    a.set_xticklabels(a.get_xticklabels(), rotation=25, rotation_mode="anchor", ha="right", fontsize=8, position=(0, 0.03))
    a.set_ylabel(r"$\mathrm{log_{10}(\frac{num}{cell * min})}$")
    a.set_title("Receptor expression rates")


def misc_violin(ax, unkVec, scales):
    """ Create violin plot of activity scaling constants, sortF, and kfwd. """
    scales6 = scales[:, 0] / np.max(scales[:, 0])
    scales5 = scales[:, 1] / np.max(scales[:, 1])
    misc = np.vstack((scales6, scales5, unkVec[19, :], unkVec[6, :] / np.max(unkVec[6, :])))
    misc = pd.DataFrame(misc.T)

    misc.columns = [r'$C_{6}$ / ' + '{:.2E}'.format(np.max(scales[:, 0])), r'$C_{5}$ / ' + '{:.2E}'.format(np.max(scales[:, 1])),
                    r'$f_{sort}$', r'$k_{fwd}$ / ' + "{:.2E}".format(np.max(unkVec[6, :]))]
    a = sns.violinplot(data=misc, ax=ax, linewidth=0.5, color="grey")
    a.set_xticklabels(a.get_xticklabels(), rotation=25, rotation_mode="anchor", ha="right", fontsize=8, position=(0, 0.02))
    a.set_ylabel("value")
    a.set_title("Miscellaneous parameters")


def pretreat_calc(unkVec, scales, pre_conc):
    ''' This function performs the calculations necessary to produce the Gonnord Figures S3B and S3C. '''
    activity = getTotalActiveSpecies().astype(np.float64)
    ts = np.array([10.])  # was 10. in literature
    IL4_stim_conc = 100. / 14900.  # concentration used for IL4 stimulation
    IL7_stim_conc = 50. / 17400.  # concentration used for IL7 stimulation
    assert unkVec.shape[0] == nParams()
    K = unkVec.shape[1]  # should be 500
    N = len(pre_conc)

    def parallelCalc(unkVec, pre_cytokine, pre_conc, stim_cytokine, stim_conc):
        """ Calculate pSTAT activity for single case pretreatment case. Simulation run in parallel. """
        unkVec2 = unkVec.copy()
        unkVec2[pre_cytokine, :] = pre_conc
        unkVec2[stim_cytokine, :] = stim_conc
        ligands = np.zeros(6)
        ligands[pre_cytokine] = pre_conc  # pretreatment ligand stays in system
        unkVec2 = np.transpose(unkVec2).copy()  # transpose the matrix (save view as a new copy)

        returnn, retVal = runCkineUP(ts, unkVec2, preT=ts, prestim=ligands)
        assert retVal >= 0
        ret = np.zeros((returnn.shape[0]))
        for ii in range(returnn.shape[0]):
            ret[ii] = getTotalActiveCytokine(stim_cytokine, np.squeeze(returnn[ii]))  # only look at active species associated with the active cytokine
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
        unkVec = np.transpose(unkVec).copy()  # transpose the matrix (save view as a new copy)
        returnn, retVal = runCkineUP(ts, unkVec)
        assert retVal >= 0
        return np.dot(returnn, activity)

    # run simulation with just one cytokine
    IL4stim_no_pre = parallelCalc_no_pre(unkVec, 4, IL4_stim_conc)
    IL7stim_no_pre = parallelCalc_no_pre(unkVec, 2, IL7_stim_conc)

    ret1 = np.zeros((K, N))  # arrays to hold inhibition calculation
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
    IL7_pretreat_conc = data[:, 0] / 17400.  # concentrations used for IL7 pretreatment followed by IL4 stimulation
    IL4_pretreat_conc = data[:, 5] / 14900.  # concentrations used for IL4 pretreatment followed by IL7 stimulation
    PTS = 30
    K = unkVec.shape[1]  # should be 500
    pre_conc = np.logspace(-3.8, 1.0, num=PTS)

    output = pretreat_calc(unkVec, scales, pre_conc)  # run simulation
    # split according to cytokine and transpose so it works with plot_conf_int
    IL4_stim = output[0:K].T
    IL7_stim = output[K:(K * 2)].T

    plot_conf_int(ax, np.log10(pre_conc), IL4_stim * 100., "powderblue", "IL-4 stim/IL-7 pretreat")
    plot_conf_int(ax, np.log10(pre_conc), IL7_stim * 100., "b", "IL-7 stim/IL-4 pretreat")
    ax.set(title=title, ylabel="Inhibition (% of no pretreat)", xlabel=r'Pretreatment concentration (log$_{10}$[nM])')

    # add experimental data to plots
    ax.scatter(np.log10(IL7_pretreat_conc), data[:, 1], color='powderblue', zorder=100, marker='^', edgecolors='k')
    ax.scatter(np.log10(IL7_pretreat_conc), data[:, 2], color='powderblue', zorder=101, marker='^', edgecolors='k')
    ax.scatter(np.log10(IL7_pretreat_conc), data[:, 3], color='powderblue', zorder=102, marker='^', edgecolors='k')
    ax.scatter(np.log10(IL4_pretreat_conc), data[:, 6], color='b', zorder=103, marker='^', edgecolors='k')
    ax.scatter(np.log10(IL4_pretreat_conc), data[:, 7], color='b', zorder=104, marker='^', edgecolors='k')
    ax.scatter(np.log10(IL4_pretreat_conc), data[:, 8], color='b', zorder=105, marker='^', edgecolors='k')


def surf_gc(ax, cytokC_pg, unkVec):
    """ Generate a plot that shows the relative amount of gc on the cell surface under IL4 and IL7 stimulation. """
    PTS = 40
    ts = np.linspace(0., 100., num=PTS)
    output = calc_surf_gc(ts, cytokC_pg, unkVec)
    IL4vec = np.transpose(output[:, 0:PTS])
    IL7vec = np.transpose(output[:, PTS:(PTS * 2)])
    plot_conf_int(ax, ts, IL4vec, "powderblue")
    plot_conf_int(ax, ts, IL7vec, "b")
    ax.set(title=(r"$\gamma_{c}$ depletion at " + str(int(cytokC_pg)) + ' pg/mL'), ylabel=r"Surface $\gamma_{c}$ (%)", xlabel="Time (min)")
    ax.set_ylim(0, 115)


def calc_surf_gc(t, cytokC_pg, unkVec):
    """ Calculates the percent of gc on the surface over time while under IL7 and IL4 stimulation. """
    gc_species_IDX = getSurfaceGCSpecies()
    PTS = len(t)
    K = unkVec.shape[1]

    def parallelCalc(unkVec, cytokine, conc, t):
        """ Calculates the surface gc over time for one condition. """
        unkVec = unkVec.copy()
        unkVec[cytokine, :] = conc
        unkVec = np.transpose(unkVec).copy()  # transpose the matrix (save view as a new copy)
        returnn, retVal = runCkineUP(t, unkVec)
        assert retVal >= 0
        return np.dot(returnn, gc_species_IDX)

    # calculate IL4 stimulation
    a = parallelCalc(unkVec, 4, (cytokC_pg / 14900.), t).reshape((K, PTS))
    # calculate IL7 stimulation
    b = parallelCalc(unkVec, 2, (cytokC_pg / 17400.), t).reshape((K, PTS))
    # concatenate results and normalize to 100%
    result = np.concatenate((a, b), axis=1)
    return (result / np.max(result)) * 100.


def data_path():
    """ Loads the Gonnard data from the appropriate CSV files. """
    path = os.path.dirname(os.path.abspath(__file__))
    dataIL4 = pd.read_csv(join(path, "../data/Gonnord_S3B.csv")).values  # imports IL4 file into pandas array
    dataIL7 = pd.read_csv(join(path, "../data/Gonnord_S3C.csv")).values
    data_pretreat = pd.read_csv(join(path, "../data/Gonnord_S3D.csv")).values
    return (dataIL4, dataIL7, data_pretreat)


def relativeGC(ax, unkVec2, unkVec4):
    """ This function compares the relative complex affinities for GC. The rates included in this violin plot will be k4rev, k10rev,
    k17rev, k22rev, k27rev, and k33rev. We're currently ignoring k31rev (IL9) and k35rev (IL21) since we don't fit to any of its data. """

    # assign values from unkVec
    kfwd_2, kfwd_4, k4rev, k5rev = unkVec2[6, :], unkVec4[6, :], unkVec2[7, :], unkVec2[8, :]
    k16rev, k17rev, k22rev, k27rev, k33rev = unkVec2[9, :], unkVec2[10, :], unkVec2[11, :], unkVec4[13, :], unkVec4[15, :]

    # back-out k10 with ratio
    k10rev = 12.0 * k5rev / 1.5  # doi:10.1016/j.jmb.2004.04.038

    # add each rate duo as separate column in dataframe
    df = pd.DataFrame({'2·2Rα': kfwd_2 / k4rev, '2·2Rβ': kfwd_2 / k5rev, '2·2Rα·2Rβ': kfwd_2 / k10rev, '15·15Rα': kfwd_2 / k16rev,
                       '15·2Rβ': kfwd_2 / k17rev, '15·15Rα·2Rβ': kfwd_2 / k22rev, '7·7Rα': kfwd_4 / k27rev, '4·4Rα': kfwd_4 / k33rev})

    sns.set_palette(sns.xkcd_palette(["violet", "violet", "violet", "goldenrod", "goldenrod", "goldenrod", "blue", "lightblue"]))

    a = sns.violinplot(data=np.log10(df), ax=ax, linewidth=0, scale='width')
    a.set_xticklabels(a.get_xticklabels(), rotation=25, rotation_mode="anchor", ha="right", fontsize=8, position=(0, 0.02))
    a.set(title=r"Relative $\gamma_{c}$ affinity", ylabel=r"$\mathrm{log_{10}(K_{a})}$")
    
# TODO: Add Ka units.
