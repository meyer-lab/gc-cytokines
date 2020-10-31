"""
This creates Figure 1.
"""
from os.path import join
import os
import numpy as np
import seaborn as sns
import pandas as pd
from .figureCommon import subplotLabel, getSetup, traf_names, plot_conf_int, global_legend
from ..plot_model_prediction import parallelCalc
from ..model import getSurfaceIL2RbSpecies, getSurfaceGCSpecies, getTotalActiveSpecies
from ..imports import import_samples_2_15


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7.5, 6), (3, 4), multz={0: 2, 10: 1}, empts=[2])

    ax[0].axis("off")  # blank out first axis for cartoon

    subplotLabel(ax, hstretch={0: 3.8, 8: 3.25})

    unkVec, scales = import_samples_2_15(N=100)  # use these for simulations
    full_unkVec, full_scales = import_samples_2_15()  # use these for violin plots
    pstat_act(ax[1], unkVec, scales)
    IL2Rb_perc(ax[2:4], unkVec)
    gc_perc(ax[4], unkVec)
    violinPlots(ax[5:8], full_unkVec, full_scales)
    rateComp(ax[8], full_unkVec)
    global_legend(ax[1], exppred=False)
    legend = ax[1].get_legend()
    labels = (x.get_text() for x in legend.get_texts())
    ax[1].legend(legend.legendHandles, labels, loc="lower right")

    return f


def IL2Rb_perc(ax, unkVec):
    """ Calculates the percent of IL2Rb on the cell surface over the course of 90 mins. Cell environments match those of surface IL2Rb data collected by Ring et al. """
    IL2Rb_species_IDX = getSurfaceIL2RbSpecies()

    # overlay experimental data
    path = os.path.dirname(os.path.abspath(__file__))
    data_minus = pd.read_csv(join(path, "../data/IL2Ra-_surface_IL2RB_datasets.csv")).values  # imports file into pandas array
    data_plus = pd.read_csv(join(path, "../data/IL2Ra+_surface_IL2RB_datasets.csv")).values  # imports file into pandas array
    ax[0].scatter(data_minus[:, 0], data_minus[:, 1] * 10.0, color="darkorchid", marker="^", edgecolors="k", zorder=100)  # 1nM of IL2 in 2Ra-
    ax[0].scatter(data_minus[:, 0], data_minus[:, 2] * 10.0, color="goldenrod", marker="^", edgecolors="k", zorder=101)  # 1nM of IL15 in 2Ra-
    ax[1].scatter(data_minus[:, 0], data_minus[:, 5] * 10.0, color="darkorchid", marker="^", edgecolors="k", zorder=100)  # 500nM of IL2 in 2Ra-
    ax[1].scatter(data_minus[:, 0], data_minus[:, 6] * 10.0, color="goldenrod", marker="^", edgecolors="k", zorder=101)  # 500nM of IL15 in 2Ra-
    ax[0].scatter(data_plus[:, 0], data_plus[:, 1] * 10.0, color="darkorchid", marker="o", edgecolors="k", zorder=100)  # 1nM of IL2 in 2Ra+
    ax[0].scatter(data_plus[:, 0], data_plus[:, 2] * 10.0, color="goldenrod", marker="o", edgecolors="k", zorder=101)  # 1nM of IL15 in 2Ra+
    ax[1].scatter(data_plus[:, 0], data_plus[:, 5] * 10.0, color="darkorchid", marker="o", edgecolors="k", zorder=100)  # 500nM of IL2 in 2Ra+
    ax[1].scatter(data_plus[:, 0], data_plus[:, 6] * 10.0, color="goldenrod", marker="o", edgecolors="k", zorder=101)  # 500nM of IL15 in 2Ra+

    y_max = 100.0
    ts = np.array([0.0, 2.0, 5.0, 15.0, 30.0, 60.0, 90.0])
    results = np.zeros((ts.size, unkVec.shape[1], 4, 2))  # 3rd dim is cell condition (IL2Ra+/- and cytokC), 4th dim is cytok species

    # set IL2Ra concentrations
    unkVecIL2RaMinus = unkVec.copy()
    unkVecIL2RaMinus[22, :] = 0.0

    # calculate IL2 stimulation
    a = parallelCalc(unkVec, 0, 1.0, ts, IL2Rb_species_IDX)
    b = parallelCalc(unkVec, 0, 500.0, ts, IL2Rb_species_IDX)
    c = parallelCalc(unkVecIL2RaMinus, 0, 1.0, ts, IL2Rb_species_IDX)
    d = parallelCalc(unkVecIL2RaMinus, 0, 500.0, ts, IL2Rb_species_IDX)

    # calculate IL15 stimulation
    e = parallelCalc(unkVec, 1, 1.0, ts, IL2Rb_species_IDX)
    f = parallelCalc(unkVec, 1, 500.0, ts, IL2Rb_species_IDX)
    g = parallelCalc(unkVecIL2RaMinus, 1, 1.0, ts, IL2Rb_species_IDX)
    h = parallelCalc(unkVecIL2RaMinus, 1, 500.0, ts, IL2Rb_species_IDX)

    output = np.concatenate((a, b, c, d, e, f, g, h), axis=1) * y_max
    output /= a[:, 0][:, np.newaxis]  # normalize by a[0] for each row

    # split according to experimental conditions
    results[:, :, 2, 0] = output[:, 0: (ts.size)].T
    results[:, :, 3, 0] = output[:, (ts.size): (ts.size * 2)].T
    results[:, :, 0, 0] = output[:, (ts.size * 2): (ts.size * 3)].T
    results[:, :, 1, 0] = output[:, (ts.size * 3): (ts.size * 4)].T
    results[:, :, 2, 1] = output[:, (ts.size * 4): (ts.size * 5)].T
    results[:, :, 3, 1] = output[:, (ts.size * 5): (ts.size * 6)].T
    results[:, :, 0, 1] = output[:, (ts.size * 6): (ts.size * 7)].T
    results[:, :, 1, 1] = output[:, (ts.size * 7): (ts.size * 8)].T

    for n in range(4):
        # plot results within confidence intervals
        plot_conf_int(ax[n % 2], ts, results[:, :, n, 0], "darkorchid")
        plot_conf_int(ax[n % 2], ts, results[:, :, n, 1], "goldenrod")

    # label axes and titles
    ax[1].set(xlabel="Time (min)", ylabel=("Surface IL-2Rβ (%)"), title="500 nM")
    ax[1].set_ylim(0, 115)
    ax[1].set_xticks(np.arange(0, 105, step=15))
    ax[0].set(xlabel="Time (min)", ylabel=("Surface IL-2Rβ (%)"), title="1 nM")
    ax[0].set_ylim(0, 115)
    ax[0].set_xticks(np.arange(0, 105, step=15))


def gc_perc(ax, unkVec):
    """ Calculates the amount of gc that stays on the cell surface and compares it to experimental values in Mitra paper. """
    gc_species_IDX = getSurfaceGCSpecies()
    # overlay experimental data
    path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(join(path, "../data/mitra_surface_gc_depletion.csv")).values  # imports file into pandas array
    ts = data[:, 0]
    ax.scatter(ts, data[:, 1], color="darkorchid", marker="^", edgecolors="k", zorder=100)  # 1000 nM of IL2 in 2Ra-

    # set IL2 concentrations
    unkVecIL2RaMinus = unkVec.copy()
    unkVecIL2RaMinus[22, :] = 0.0

    # calculate IL2 stimulation
    output = parallelCalc(unkVecIL2RaMinus, 0, 1000.0, ts, gc_species_IDX)

    output /= output[:, 0][:, np.newaxis]  # normalize by output[0] for each row
    output *= 100.0

    plot_conf_int(ax, ts, output.T, "darkorchid")

    # label axes and titles
    ax.set(xlabel="Time (min)", ylabel=r"Surface $\gamma_{c}$ (%)", title="1000 nM")
    ax.set_ylim(0, 115)
    ax.set_xticks(np.arange(0, 300, step=60))


def pstat_act(ax, unkVec, scales):
    """ This function generates the pSTAT activation levels for each combination of parameters in unkVec. The results are plotted and then overlayed with the values measured by Ring et al. """
    PTS = 30
    cytokC = np.logspace(-3.3, 2.7, PTS)
    y_max = 100.0
    IL2_plus = np.zeros((unkVec.shape[1], PTS))
    IL15_minus = IL2_plus.copy()
    IL15_plus = IL2_plus.copy()
    IL2_minus = IL2_plus.copy()

    activity = getTotalActiveSpecies().astype(np.float64)
    ts = np.array([500.0])  # was 500. in literature

    unkVec_IL2Raminus = unkVec.copy()
    unkVec_IL2Raminus[22, :] = np.zeros(unkVec.shape[1])  # set IL2Ra expression rates to 0

    actVec_IL2 = np.zeros((unkVec.shape[1], len(cytokC)))
    actVec_IL2_IL2Raminus = actVec_IL2.copy()
    actVec_IL15 = actVec_IL2.copy()
    actVec_IL15_IL2Raminus = actVec_IL2.copy()

    # Calculate activities
    for x, conc in enumerate(cytokC):
        actVec_IL2[:, x] = parallelCalc(unkVec, 0, conc, ts, activity).T
        actVec_IL2_IL2Raminus[:, x] = parallelCalc(unkVec_IL2Raminus, 0, conc, ts, activity).T
        actVec_IL15[:, x] = parallelCalc(unkVec, 1, conc, ts, activity).T
        actVec_IL15_IL2Raminus[:, x] = parallelCalc(unkVec_IL2Raminus, 1, conc, ts, activity).T

    # put together into one vector & normalize by scale
    actVec = np.concatenate((actVec_IL2, actVec_IL2_IL2Raminus, actVec_IL15, actVec_IL15_IL2Raminus), axis=1)
    actVec = actVec / (actVec + scales[:, np.newaxis])
    output = actVec / actVec.max(axis=1, keepdims=True) * y_max  # normalize by the max value of each row

    # split according to experimental condition
    IL2_plus = output[:, 0:PTS].T
    IL2_minus = output[:, PTS: (PTS * 2)].T
    IL15_plus = output[:, (PTS * 2): (PTS * 3)].T
    IL15_minus = output[:, (PTS * 3): (PTS * 4)].T

    # plot confidence intervals based on model predictions
    plot_conf_int(ax, cytokC, IL2_minus, "darkorchid")
    plot_conf_int(ax, cytokC, IL15_minus, "goldenrod")
    plot_conf_int(ax, cytokC, IL2_plus, "darkorchid")
    plot_conf_int(ax, cytokC, IL15_plus, "goldenrod")

    # plot experimental data
    cytokCplt = np.logspace(-3.3, 2.7, 8)
    path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(join(path, "../data/IL2_IL15_extracted_data.csv")).values  # imports file into pandas array
    ax.scatter(cytokCplt, data[:, 2], color="darkorchid", marker="^", edgecolors="k", zorder=100)  # IL2 in 2Ra-
    ax.scatter(cytokCplt, data[:, 3], color="goldenrod", marker="^", edgecolors="k", zorder=101)  # IL15 in 2Ra-
    ax.scatter(cytokCplt, data[:, 6], color="darkorchid", marker="o", edgecolors="k", zorder=102)  # IL2 in 2Ra+
    ax.scatter(cytokCplt, data[:, 7], color="goldenrod", marker="o", edgecolors="k", zorder=103)  # IL15 in 2Ra+
    ax.set(ylabel="pSTAT5 (% of max)", xlabel="Ligand Concentration (nM)", title="YT-1 Cell Activity")
    ax.set_xscale("log")
    ax.set_xticks([10e-5, 10e-2, 10e1])
    ax.xaxis.set_tick_params(labelsize=7)


def violinPlots(ax, unkVec, scales, Traf=True):
    """ Create violin plots of model posterior. """
    unkVec = unkVec.transpose()
    traf = np.concatenate((unkVec[:, 17:19], unkVec[:, 20:22]), axis=1)
    traf = pd.DataFrame(traf)
    Rexpr = pd.DataFrame(unkVec[:, 22:26])
    scaless = scales / np.max(scales)
    kfwd = unkVec[:, 6] / np.max(unkVec[:, 6])
    if Traf:  # include sortF
        misc = np.vstack((scaless, unkVec[:, 19], kfwd))
        misc = pd.DataFrame(misc.T)
        misc.columns = [r"$\mathrm{C_{5}}$ / " + "{:.2E}".format(np.max(scales)), "Sorting Fraction", "Cmplx form. rate / " + "{:.2E}".format(np.max(unkVec[:, 6]))]
    else:  # ignore sortF
        misc = np.vstack((scaless, kfwd))
        misc = pd.DataFrame(misc.T)
        misc.columns = [r"$\mathrm{C_{5}}$ / " + "{:.2E}".format(np.max(scales)), "Cmplx form. rate / " + "{:.2E}".format(np.max(unkVec[:, 6]))]

    Rexpr.columns = ["IL-2Rα", "IL-2Rβ", r"$\mathrm{γ_{c}}$", "IL-15Rα"]
    col_list = ["violet", "violet", "grey", "goldenrod"]
    col_list_palette = sns.xkcd_palette(col_list)
    a = sns.violinplot(data=np.log10(Rexpr), ax=ax[0], linewidth=0.5, palette=col_list_palette)
    a.set(title="Receptor Expression Rates", ylabel=r"$\mathrm{log_{10}(\frac{\#}{cell * min})}$")
    a.set_xticklabels(a.get_xticklabels(), rotation=25, rotation_mode="anchor", ha="right", fontsize=8, position=(0, 0.02))

    if Traf:
        traf.columns = traf_names()
        b = sns.violinplot(data=np.log10(traf), ax=ax[1], linewidth=0.5, color="grey")
        b.set_xticklabels(b.get_xticklabels(), rotation=35, rotation_mode="anchor", ha="right", fontsize=5, position=(0, 0.05))
        b.set(title="Trafficking Parameters", ylabel=r"$\mathrm{log_{10}(\frac{1}{min})}$")

    sc_ax = 1  # subplot number for the scaling constant
    if Traf:
        sc_ax = 2
    c = sns.violinplot(data=misc, ax=ax[sc_ax], linewidth=0.5, color="grey")
    if Traf:
        c.set_xticklabels(c.get_xticklabels(), rotation=35, rotation_mode="anchor", ha="right", fontsize=5, position=(0, 0.05))
    else:
        c.set_xticklabels(c.get_xticklabels(), rotation=35, rotation_mode="anchor", ha="right", fontsize=8, position=(0, 0.05))
    c.set(ylabel="Value", title="Misc. Parameters")


def rateComp(ax, unkVec, fsize=5):
    """ This function compares the analogous reverse rxn distributions from IL2 and IL15 in a violin plot. """
    # assign values from unkVec
    kfwd, k4rev, k5rev, k16rev, k17rev, k22rev, k23rev = unkVec[6, :], unkVec[7, :], unkVec[8, :], unkVec[9, :], unkVec[10, :], unkVec[11, :], unkVec[12, :]
    split = unkVec.shape[1]
    # plug in values from measured constants into arrays of size 500
    kfbnd = 0.60  # Assuming on rate of 10^7 M-1 sec-1
    k1rev = np.full((split), (kfbnd * 10))  # doi:10.1016/j.jmb.2004.04.038, 10 nM
    k2rev = np.full((split), (kfbnd * 144))  # doi:10.1016/j.jmb.2004.04.038, 144 nM
    k13rev = np.full((split), (kfbnd * 0.065))  # based on the multiple papers suggesting 30-100 pM
    k14rev = np.full((split), (kfbnd * 438))  # doi:10.1038/ni.2449, 438 nM
    kfbndCol = np.full((split), (kfbnd))

    # proportions known through measurements
    k10rev = 12.0 * k5rev / 1.5  # doi:10.1016/j.jmb.2004.04.038
    k11rev = 63.0 * k5rev / 1.5  # doi:10.1016/j.jmb.2004.04.038

    # detailed balance
    k12rev = k1rev * k11rev / k2rev  # loop for IL2_IL2Ra_IL2Rb
    k9rev = k10rev * k11rev / k4rev
    k8rev = k10rev * k12rev / k5rev
    k24rev = k13rev * k23rev / k14rev
    k21rev = k22rev * k23rev / k16rev
    k20rev = k22rev * k24rev / k17rev

    # add each rate duo as separate column in dataframe
    df = pd.DataFrame(
        {
            "(2)·2Rα, (15)·15Rα": np.append(kfbndCol / k1rev, kfbndCol / k13rev),
            "(2)·2Rβ, (15)·2Rβ": np.append(kfbndCol / k2rev, kfbndCol / k14rev),
            r"($\mathrm{γ_{c}}$)·2·2Rα, ($\mathrm{γ_{c}}$)·15·15Rα": np.append(kfwd / k4rev, kfwd / k16rev),
            r"($\mathrm{γ_{c}}$)·2·2Rβ, ($\mathrm{γ_{c}}$)·15·2Rβ": np.append(kfwd / k5rev, kfwd / k17rev),
            r"(2Rα)·2·2Rβ·$\mathrm{γ_{c}}$, (15Rα)·15·2Rβ·$\mathrm{γ_{c}}$": np.append(kfwd / k8rev, kfwd / k20rev),
            r"(2Rβ)·2·2Rα·$\mathrm{γ_{c}}$, (2Rβ)·15·15Rα·$\mathrm{γ_{c}}$": np.append(kfwd / k9rev, kfwd / k21rev),
            r"($\mathrm{γ_{c}}$)·2·2Rα·2Rβ, ($\mathrm{γ_{c}}$)·15·15Rα·2Rβ": np.append(kfwd / k10rev, kfwd / k22rev),
            "(2Rβ)·2·2Rα, (2Rβ)·15·15Rα": np.append(kfwd / k11rev, kfwd / k23rev),
            "(15Rα)·2·2Rβ, (15Rα)·15·2Rβ": np.append(kfwd / k12rev, kfwd / k24rev),
        }
    )

    # add labels for IL2 and IL15
    df["cytokine"] = "IL-2"
    df.loc[split: (split * 2), "cytokine"] = "IL-15"

    # melt into long form and take log value
    melted = pd.melt(df, id_vars="cytokine", var_name="Rate", value_name=r"$\mathrm{log_{10}(K_{a})}$")
    melted.loc[:, r"$\mathrm{log_{10}(K_{a})}$"] = np.log10(melted.loc[:, r"$\mathrm{log_{10}(K_{a})}$"])

    col_list = ["violet", "goldenrod"]
    col_list_palette = sns.xkcd_palette(col_list)
    sns.set_palette(col_list_palette)

    # plot with hue being cytokine species
    a = sns.violinplot(x="Rate", y=r"$\mathrm{log_{10}(K_{a})}$", data=melted, hue="cytokine", ax=ax, linewidth=0, scale="width")
    a.get_legend().remove()  # remove the legend
    a.scatter(-0.3, np.log10(1 / 10), color="darkviolet")  # overlay point for k1rev
    a.scatter(0.1, np.log10(1 / 0.065), color="goldenrod")  # overlay point for k13rev
    a.scatter(0.7, np.log10(1 / 144), color="darkviolet")  # overlay point for k2rev
    a.scatter(1.1, np.log10(1 / 468), color="goldenrod")  # overlay point for k14rev
    a.set_xticklabels(a.get_xticklabels(), rotation=45, rotation_mode="anchor", ha="right", fontsize=fsize)
    a.set(title=r"Species Association Constants", ylabel=r"$\mathrm{log_{10}(K_{a})}$", xlabel="")
