"""
This file includes various methods for flow cytometry analysis.
"""

from pathlib import Path
from pylab import *
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import scale
from FlowCytometryTools import FCMeasurement
from FlowCytometryTools.core.gates import CompositeGate
from FlowCytometryTools import QuadGate, ThresholdGate, PolyGate
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy.optimize import least_squares


def importF(pathname, WellRow):
    """
    Import FCS files. Variable input: name of path name to file. Output is a list of Data File Names in FCT Format
    Title/file names are returned in the array file --> later referenced in other functions as title/titles input argument
    """
    # Declare arrays and int
    file = []
    sample = []
    z = 0
    # Read in user input for file path and assign to array file
    pathlist = Path(r"" + str(pathname)).glob("**/*.fcs")
    for path in pathlist:
        wellID = path.name.split("_")[1]
        if wellID[0] == WellRow:
            file.append(str(path))
    file.sort()
    assert file != []
    # Go through each file and assign the file contents to entry in the array sample
    for entry in file:
        sample.append(FCMeasurement(ID="Test Sample" + str(z), datafile=entry))
        z += 1
    # Returns the array sample which contains data of each file in folder (one file per entry in array)
    return sample, file


# *********************************** Gating Fxns *******************************************

#Panel 1 gates:naïve and memory T-regulatory and T-helper cells
def cd3cd4():
    """Function for gating CD3+CD4+ cells (generates T cells)"""
    cd3cd4 = PolyGate([(5.2e03,5.4e03),(5.8e03,7.9e03),(7.2e03,7.9e03),(7.2e03,5.7e03)], ('VL6-H','VL4-H'),region='in', name = 'cd3cd4')
    cd3cd4_gate = cd3cd4
    return cd3cd4_gate

def thelper():
    """Fucntion creating and returning T helper gate on CD3+CD4+ cells"""
    thelp = PolyGate([(4.1e03, 2e03), (4.1e03, 4.6e03), (5.2e03, 5.55e03), (6.5e03,5.55e03 ), (6.5e03, 2e03)], ('BL1-H','VL1-H'),region = 'out', name = 'thelp')
    thelp_gate = thelp
    return thelp_gate

def treg():#switch to polygates
    """Function creating and returning the T reg gate on CD3+CD4+ cells"""
    treg = PolyGate([(2.5e03, 4.65e03), (2.5e03, 5.8e03), (4.4e03, 5.8e03), (5.1e03,5.56e03 ), (3.9e03, 4.65e03)], ('BL1-H', 'VL1-H'), region='in', name='treg')
    treg_gate = treg
    return treg_gate

def tregMem():#switch to polygates
    """Function for creating and returning the Treg Memory gate on Treg CD3+CD4+ cells"""
    treg1 = QuadGate((1e+03, 1e+03), ('BL1-H', 'VL1-H'), region='top right', name='treg1')
    treg2 = QuadGate((0, 0), ('BL1-H', 'VL1-H'), region='bottom left', name='treg2')
    cd45 = ThresholdGate(6300, ('BL3-H'), region="below", name='cd45')
    tregMem_gate = treg1 & treg2 & cd3cd4() & cd45
    return tregMem_gate

def tregN():#switch to polygates
    """Function for creating and returning the T reg gate on CD4+ cells"""
    treg1 = QuadGate((1e+03, 1e+03), ('BL1-H', 'VL1-H'), region='top right', name='treg1')
    treg2 = QuadGate((0, 0), ('BL1-H', 'VL1-H'), region='bottom left', name='treg2')
    cd45 = ThresholdGate(6300, ('BL3-H'), region="above", name='cd45')
    tregN_gate = treg1 & treg2 & cd3cd4() & cd45
    return tregN_gate

def THelpMem():#switch to polygates
    """Function for creating and returning the non T reg gate on CD4+ cells"""
    thelp1 = QuadGate((2e+03,2e+03), ('BL1-H','VL1-H'),region='top right', name = 'thelp1')
    thelp2 = QuadGate((0,0), ('BL1-H','VL1-H'),region='bottom left', name = 'thelp2')
    cd45 = ThresholdGate(6300, ('BL3-H'), region="below", name='cd45')
    Thelp_gate = thelper() & cd3cd4() & cd45
    return Thelp_gate

def THelpN():#switch to polygates
    """Function for creating and returning the non T reg gate on CD4+ cells"""
    thelp1 = QuadGate((5.115e+03, 3.470e+02), ('BL1-H', 'VL1-H'), region="top left", name='thelp1')
    thelp2 = QuadGate((2.586e+03, 5.245e+03), ('BL1-H', 'VL1-H'), region="bottom right", name='thelp2')
    cd45 = ThresholdGate(6300, ('BL3-H'), region="above", name='cd45')
    ThelpN_gate = thelp1 & thelp2 & cd3cd4() & cd45
    return ThelpN_gate

#Panel 2 gates: NK and CD56bright NK cells
def nk():
    """Function for creating and returning the NK gate"""
    # NK cells: Take quad gates for NK cells and combine them to create single, overall NK gate
    nk1 = QuadGate((6.468e03, 4.861e03), ("BL3-H", "VL4-H"), region="top left", name="nk1")
    nk2 = QuadGate((5.550e03, 5.813e03), ("BL3-H", "VL4-H"), region="bottom right", name="nk2")
    nk_gate = nk1 & nk2
    return nk_gate

def bnk():
    """Function for creating and returning the BNK gate"""
    # Bright NK cells: Take quad gates for bright NK cells and combine them to create single, overall bright NK gate
    bnk1 = QuadGate((7.342e03, 4.899e03), ("BL3-H", "VL4-H"), region="top left", name="bnk1")
    bnk2 = QuadGate((6.533e03, 5.751e03), ("BL3-H", "VL4-H"), region="bottom right", name="bnk2")
    bnk_gate = bnk1 & bnk2
    return bnk_gate

#Panel 3 gates: naïve and memory cytotoxic T cells
def cd3cd8():
    """Function for creating and returning the CD3+CD8+ gate"""
    cd3cd81 = QuadGate((9.016e03, 5.976e03), ("VL6-H", "VL4-H"), region="top left", name="cd3cd81")
    cd3cd82 = QuadGate((6.825e03, 7.541e03), ("VL6-H", "VL4-H"), region="bottom right", name="cd3cd82")
    cd3cd8_gate = cd3cd81 & cd3cd82
    return cd3cd8_gate

def cytoTMem():
    cd45 = ThresholdGate(6300, ('BL3-H'), region="above", name='cd45')
    cytoTMem_gate = cd3cd8() & cd45
    return cytoTMem_gate

def cytoTN():
    cd45 = ThresholdGate(6300, ('BL3-H'), region="below", name='cd45')
    cytoTN_gate = cd3cd4() & cd45
    return cytoTN_gate

#Panel 4
#TODO

#Not using below
def cellCount(sample_i, gate, Tcells=True):
    """
    Function for returning the count of cells in a single .fcs. file of a single cell file. Arguments: single sample/.fcs file and the gate of the
    desired cell output.
    """
    # Import single file and save data to a variable --> transform to logarithmic scale
    if Tcells:
        channels = ["BL1-H", "VL1-H", "VL4-H", "BL3-H"]
    else:
        channels = ["BL1-H", "RL1-H", "VL4-H"]
    smpl = sample_i.transform("hlog", channels=channels)
    # Apply T reg gate to overall data --> i.e. step that detrmines which cells are T reg
    cells = smpl.gate(gate)
    # Number of events (AKA number of cells)
    cell_count = cells.get_data().shape[0]
    # print(cell_count)
    # print('Number of Treg cells:' + str(treg_count))
    return cell_count


def rawData(sample_i, gate, Tcells=True):
    """
    Function that returns the raw data of certain cell population in a given file. Arguments: sample_i is a single entry/.fcs file and the gate
    of the desired cell population.
    """
    if Tcells:
        channels = ["BL1-H", "VL1-H", "VL4-H", "BL3-H"]
    else:
        channels = ["BL1-H", "RL1-H", "VL4-H"]
    smpl = sample_i.transform("hlog", channels=channels)
    # Apply T reg gate to overall data --> i.e. step that detrmines which cells are T reg
    cells = smpl.gate(gate)
    # Get raw data of t reg cells in file
    cell_data = cells.get_data()
    return cell_data


def tcells(sample_i, treg_gate, nonTreg_gate, title):
    """
    Function that is used to plot the Treg and NonTreg gates in CD4+EDIT cells. Treg (yellow) and Non Treg (green). sample_i is an indivual flow cytommetry file/data.
    """
    # Data to use is on CD4+ cells
    # Apply new T reg and Non treg gate
    # Assign data of current file for analysis to variable smpl and transform to log scale
    smpl = sample_i.transform('hlog', channels=["VL3-H", "VL4-H", "BL1-H", "VL1-H"])
    # Create data set to only include CD4 cells
    cd4_gate = cd4()
    cd4_cells = smpl.gate(cd4_gate)
    # CD25 v. Foxp33: VL1 v. BL1
    # Treg
    # Apply T reg gate to overall data --> step that determines which cells are Treg
    treg_cells = smpl.gate(treg_gate)
    # Non Tregs
    # Apply non T reg gate to overall data --> step that detrmines which cells are non T reg
    nonTreg_cells = smpl.gate(nonTreg_gate)

    # Declare figure and axis
    _, ax = plt.subplots()
    # Plot the treg gate
    treg_cells.plot(["BL1-H", "VL1-H"], color="teal")
    # Plot the non Treg gate
    nonTreg_cells.plot(["BL1-H", "VL1-H"], color="cyan")
    # Plot all of the cells in the file
    ax.set_title("T Reg + Non T Reg - Gating - " + str(title), fontsize=12)
    cd4_cells.plot(["BL1-H", "VL1-H"])
    plt.xlabel("Foxp3", fontsize=12)
    plt.ylabel("CD25", fontsize=12)
    # Set values for legend
    bar_T = ax.bar(np.arange(0, 10), np.arange(1, 11), color="teal")
    bar_NT = ax.bar(np.arange(0, 10), np.arange(30, 40), bottom=np.arange(1, 11), color="cyan")
    ax.legend([bar_T, bar_NT], ("T Reg", "Non T Reg"), loc="upper left")


def nk_bnk_plot(sample_i, nk_gate, bnk_gate, title):
    """
    Function that plots the graph of NK and Bright NK cells (both are determined by same x, y-axis). Arguemnt 1: current sample (a single file).
    Argument 2: the gate for NK. Argument 3: the gate for bright NK.
    """
    smpl = sample_i.transform("hlog", channels=["BL1-H", "VL4-H", "RL1-H"])

    # CD3 v. CD56: VL4 v. BL1
    # NK
    # Apply NK gate to overall data --> step that determines which cells are NK
    nk_cells = smpl.gate(nk_gate)
    # CD56 Bright NK
    # Apply Bright NK gate to overall data --> step that determines which cells are Bright NK
    bnk_cells = smpl.gate(bnk_gate)

    _, ax1 = plt.subplots()
    ax1.set_title("CD56 BrightNK + NK - Gating - " + str(title), fontsize=12)
    nk_cells.plot(["BL1-H", "VL4-H"], color="y", label="NK")
    bnk_cells.plot(["BL1-H", "VL4-H"], color="g", label="Bright NK")
    smpl.plot(["BL1-H", "VL4-H"])

    bar_NK = ax1.bar(np.arange(0, 10), np.arange(1, 11), color="y")
    bar_BNK = ax1.bar(np.arange(0, 10), np.arange(30, 40), bottom=np.arange(1, 11), color="g")
    ax1.legend([bar_NK, bar_BNK], ("NK", "BNK"), loc="upper left")


def cd_plot(sample_i, cd_gate, title):
    """
    Function that plots the graph of CD cells. Argument 1: current sample (a single file). Argument 2: the gate for CD cells. Argument 3: the value
    of the current i in a for loop --> use
    when plotting multiple files.
    """
    smpl = sample_i.transform("hlog", channels=["BL1-H", "VL4-H", "RL1-H"])
    # CD3 v. CD8: VL4 v. RL1
    # CD3+CD8+
    # Apply CD cell gate to overall data --> step that determines which cells are CD
    cd_cells = smpl.gate(cd_gate)

    _, ax2 = plt.subplots()
    ax2.set_title("CD3+CD8+ - Gating - " + str(title), fontsize=20)
    cd_cells.plot(["RL1-H", "VL4-H"], color="b")
    smpl.plot(["RL1-H", "VL4-H"])

    bar_CD = ax2.bar(np.arange(0, 10), np.arange(1, 11), color="b")
    ax2.legend([bar_CD], ("CD3+8+"), loc="upper left")


def count_data(sampleType, gate, Tcells=True):
    """
    Used to count the number of cells and store the data of all of these cells in a folder with multiple files --> automates the process sampleType
    is NK or T cell data, gate is the desired cell population.
    Sample type: is the overall importF assignment for T or NK (all the T cell files, all NK cell files)
    """
    # declare the arrays to store the data
    count_array = []
    data_array = []
    # create the for loop to file through the data and save to the arrays
    # using the functions created above for a singular file
    for _, sample in enumerate(sampleType):
        count_array.append(cellCount(sample, gate, Tcells))
        data_array.append(rawData(sample, gate, Tcells))
    # returns the array for count of cells and the array where each entry is the data for the specific cell population in that .fcs file
    return count_array, data_array


def plotAll(sampleType, check, gate1, gate2, titles):
    """
    Ask the user to input 't' for t cell, 'n' for nk cell, and 'c' for cd cell checks are used to determine if user input a T-cell, NK-cell, or
    CD-cell gate automates the process for plotting multiple files.
    """
    if check == "t":
        for i, sample in enumerate(sampleType):
            title = titles[i].split("/")
            title = title[len(title) - 1]
            tcells(sample, gate1, gate2, title)
    elif check == "n":
        for i, sample in enumerate(sampleType):
            title = titles[i].split("/")
            title = title[len(title) - 1]
            nk_bnk_plot(sample, gate1, gate2, title)
    elif check == "c":
        for i, sample in enumerate(sampleType):
            title = titles[i].split("/")
            title = title[len(title) - 1]
            cd_plot(sample, gate1, title)
            

#Importing and transforming Panel 1            
pathName = "Documents/Desktop/PBMC receptor quant/04-23/Plate 1/Plate 1 - Panel 1 IL2R"
sample, file = importF(pathName, "A")

datafile = r'Documents/Desktop/PBMC receptor quant/04-23/Plate 1/Plate 1 - Panel 1 IL2R/Well_A01_Lymphocytes.fcs'
sample = FCMeasurement(ID='Test Sample', datafile=datafile)
tsample = sample.transform('hlog', channels=['BL1-H', 'VL1-H', 'VL6-H', 'VL4-H','BL3-H'])

#Plotting panel 1 (naïve and memory T-regulatory and T-helper cells)

fig = figure(figsize=(14,9))
fig.subplots_adjust(hspace=.25)
ax1 = subplot(221)
xscale("symlog") #Remove
yscale("symlog") #Remove
sample.plot(['VL4-H', 'VL6-H'], cmap=cm.viridis, gates=cd3cd4(), gate_lw=2) #Replace sample with tsample to use hyperlogged data
title('Singlet Lymphocytes')
ylim(1, 1e7) #Remove
ax1.set_ylabel('CD4')
ax1.set_xlabel('CD3')

cd3cd4gated_sample = tsample.gate(cd3cd4())

ax2 = subplot(222)
cd3cd4gated_sample.plot(['VL1-H','BL1-H'], cmap=cm.viridis, gates=(thelper(),treg()), gate_lw=2)
title('CD3+CD4+ Cells')
ax2.set_ylabel('CD127')
ax2.set_xlabel('CD25')

ThelpGated_sample = cd3cd4gated_sample.gate(thelper())
TregGated_sample = cd3cd4gated_sample.gate(treg())

ax3 = subplot(223)
ThelpGated_sample.plot(['BL3-H'], color='blue');
title('T helper')
ax3.set_xlabel('CD45Ra');

ax4= subplot(224)
TregGated_sample.plot(['BL3-H'], color='blue');
title('T reg')
ax4.set_xlabel('CD45Ra');

#Importing and transforming Panel 2
pathName2 = "Documents/Desktop/PBMC receptor quant/04-23/Plate 1/Plate 1 - Panel 2 IL2R"
sample2, file2 = importF(pathName2, "B")

datafile2 = r'Documents/Desktop/PBMC receptor quant/04-23/Plate 1/Plate 1 - Panel 2 IL2R/Well_B01_Lymphocytes.fcs'
sample2 = FCMeasurement(ID='Test Sample', datafile=datafile2)
tsample2 = sample.transform('hlog', channels=['BL3-H', 'VL4-H'])

#Plotting panel 2 (NK and CD56Bright NK cells)
fig2= figure()
tsample2.plot(['VL4-H', 'BL3-H'], cmap=cm.viridis);