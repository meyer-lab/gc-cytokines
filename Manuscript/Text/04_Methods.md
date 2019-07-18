# Methods

All analysis was implemented in Python, and can be found at <https://github.com/meyer-lab/type-I-ckine-model>, release 1.0.

## Model

### Base model

Cytokine binding to receptors was modeled using ordinary differential equations (ODEs). The ligands of interest were six cytokines: IL-2, IL-15, IL-4, IL-7, IL-9, and IL-21. IL-2 and IL-15 each had two private receptors, one being a signaling-deficient α-chain (IL-2Rα and IL-15Rα) and the other being signaling-competent IL-2Rβ. The other four cytokines each had one signaling-competent private receptor (IL-7Rα, IL-9R, IL-4Rα, IL-21Rα). Ligands must bind first to a private receptor and then can dimerize to other private receptors or the common γ~c~ receptor thereafter. Direct binding of ligand to γ~c~ was not included because such a reaction was not observed experimentally [@Voss2428].

Our model accounted for endosomal trafficking, allowing all species to be endocytosed, recycled, and degraded. Each receptor was synthesized and placed on the cell surface at a unique rate. *All binding events occurring on the cell surface also occurred in the endosome with the same rate parameters* (Pretty sure the binding is 5x weaker in the endosome because of its acidic environment). The surface and endosomal ligand concentrations were assumed to be equal for each cytokine. The number densities of each complex and receptor were different between the cell surface and endosome. 

The rate of change for free receptors and complexes were modeled through ODEs. There were 28 ODEs that represented the rate of change for free receptors and complexes on the cell surface, 28 ODEs that represented the rate of change for free receptors and complexes in the endosome, and 6 ODEs that represented the rate of change of ligand concentration. For our model, all of the free receptors and complexes were measured in units of number per cell and all ligands were measured in units of concentration (nM). Due to these unit choices for our species, the rate constants for ligand binding to a free receptors had units of $\mathrm{\frac{1}{nM \times min}}$, rate constants for the forward dimerization of free receptor to complex had units of $\mathrm{\frac{cell}{min \times number}}$, and the dissociation of a complex into another complex and free receptor had units of $\mathrm{min^{-1}}$.

Cytokine signaling follows the JAK-STAT signaling pathway which is initiated when two JAK subunits come in contact with one another. JAK proteins are found on the intracellular regions of the γ~c~, IL-2Rβ, IL-4Rα, IL-7Rα, IL-9R, and IL-21Rα receptors; therefore all complexes which contained two signaling-competent receptors were deemed to be active species.

Initial values were calculated by assuming steady-state in the absence of ligand.

Differential equation solving was performed using the SUNDIALS solvers in C++, with a Python interface for all other code [@hindmarsh2005sundials]. Model sensitivities were calculated using the adjoint solution [@CAO2002171]. Calculating the adjoint requires the partial derivatives of the differential equations both with respect to the species and unknown parameters. Constructing these can be tedious and error-prone. Therefore, we calculated these algorithmically using forward-pass autodifferentiation implemented in Adept-2 [@hogan_robin_j_2017]. A model and sensitivities tolerance of $10^{-9}$ and $10^{-3}$, respectively, were used throughout. We used unit tests for conservation of mass, equilibrium, and detailed balance to help ensure model correctness.

### Parameters and assumptions

All ligand-receptor binding processes had a forward rate constant of k~bnd~ which was assumed to be on rate of $10^7$ $\mathrm{\frac{1}{M * sec}}$ (0.6 $\mathrm{\frac{1}{nM * min}}$). All forward dimerization reaction rates were represented by k~fwd~. All reverse reaction rates were unique. We used detailed balance to define 8 unknown reverse reaction rate constants. Many of the equilibrium dissociation constants (K~d~) were found in published surface plasmon resonance and isothermal calorimetry experiments.

+--------------------+------------------+--------------------------------------+
| K~d~               | Value            | Reference                            |
+====================+==================+======================================+
| IL-2                                                                         |
+--------------------+------------------+--------------------------------------+
| k~2~ / k~bnd~      | 144 nM           | [@Rickert_receptor_const]            |
+--------------------+------------------+--------------------------------------+
| k~10~ / k~fwd~     | 12 nM            | [@Rickert_receptor_const]            |
+--------------------+------------------+--------------------------------------+
| k~11~ / k~fwd~     | 63 nM            | [@Rickert_receptor_const]            |
+--------------------+------------------+--------------------------------------+
| k~5~ / k~fwd~      | 1.5 nM           | [@Rickert_receptor_const]            |
+--------------------+------------------+--------------------------------------+
| IL-15                                                                        |
+--------------------+------------------+--------------------------------------+
| k~14~ / k~bnd~     | 438 nM           | [@ring_mechanistic_2012]             |
+--------------------+------------------+--------------------------------------+

: Experimentally measured K~d~ values in literature.


Experimentally-derived affinities of 1.0 nM [@Gonnordeaal1253], 59 nM [@Walsh_IL7_2012], 0.1 nM [@Renauld5690], and 0.07 nM [@Gonnordeaal1253] were used for IL-4, IL-7, IL-9, and IL-21 binding to their cognate private receptors, respectively. IL-2 and -15 were assumed to have affinities of 10 nM and 0.065 nM for their alpha receptors, respectively [@Rickert_receptor_const; @Mortier20012006; @DUBOIS2002537].

The rate of endocytosis is quantified by a constant of k~endo,a~ for active complexes and k~endo~ for all other species. The fraction of all endosomal species sent to lysosomes is f~sort~. All endosomal species not sent to lysosomes are recycled back to the cell surface. The rate constants to quantify degradation and recycling are k~deg~ and k~rec~, respectively. There is no autocrine ligand produced by the cells. Receptors can be synthesized by the cells and placed on the cell surface; receptor synthesis rates are specific to each receptor. The volume of the entire endosome was 10 fl and the surface area of the endosome is half the size of the cell surface [@MEYER201525].


### Model fitting

We used Markov chain Monte Carlo to fit the unknown parameters in our model to experimental data obtained by Ring, et al. and Gonnord, et al. [@ring_mechanistic_2012; @Gonnordeaal1253]. Experimental measurements include p-STAT activity under stimulation with varying concentrations of IL-2, IL-15, IL-4, and IL-7 as well as time-course measurements of surface IL-2Rβ upon IL-2 and IL-15 stimulation. YT-1 human NK cells were used for all data-sets involving IL-2 and IL-15. Human PBMCs were used for all data-sets involving IL-4 and IL-7. All YT-1 cell experiments were repeated in the absence of IL-2Rα. All of the data used from the Ring paper can be found in Figure 5 of said paper and all of the Gonnord data can be found in Figure S3 of said paper. Measurements of receptor counts at steady state in Gonnord et al. were used to solve for IL-7Rα, IL-4Rα, and γ~c~ expression rates in human PBMCs. We created functions that calculated p-STAT activity and percent of surface IL-2Rβ under various cellular conditions in order to calculate the residuals between our model predictions and experimental data. 

When fitting our model to the data, we used PyMC model that incorporates Bayesian Statistics to store the likelihood of the model at a given point. The fitting process varied the unknown rate parameters in an attempt to minimize the residuals between our model predictions and experimental data. All unknown rate parameters were assumed to have a lognormal distribution with a standard deviation of 0.1; the only exception to these distributions was f~sort~ which was assumed to have a beta distribution with shape parameters of α=20 and β=40. Executing this fitting process yielded likelihood distributions of each unknown parameter and sum of squared error between model prediction and experimental data at each point of experimental data. The Geweke criterion metric was used to verify fitting convergance for all versions of the model ([@Fig:supp2]) [@Geweke92].

### Tensor Generation and Factorization

To perform factorization of model predictions, we generated a three-dimensional (timepoints $\times$ cell types $\times$ ligand) dataset for the ligand activities over time. We predicted simulating 10 cell types with IL-2, IL-7, IL-15, or a mutant IL-2Rα affinity-enriched IL-2. The receptor expression rates were set based on measured surface abundance ([@Fig:tfac]B). The ligand stimulation concentrations matched those from our experiments ([@Fig:supp5]D).

Before decomposition, each tensor was variance scaled across each cell population. Tensor decomposition was performed using the Python package TensorLy [@TensorlyArxiv]. The number of components was set based on the minimum required number to reconstruct >95% of the variance in the original tensor (R2X).

## Experimental Methods

### Receptor abundance quantitation

Cryopreserved PBMCs (ATCC, PCS-800-011, lot#81115172) were thawed to room temperature and slowly diluted with 9 mL pre-warmed RPMI-1640 medium (Gibco, 11875-093) supplemented with 10% fetal bovine serum (FBS, Seradigm, 1500-500, lot#322B15). Media was removed, and cells washed once more with 10 mL warm RPMI-1640 + 10% FBS. Cells were brought to $1.5\times10^{6}$ cells/mL, distributed at 250,000 cells per well in a 96-well V-bottom plate, and allowed to recover 2 hours at 37℃ in an incubator at 5% CO2. Cells were washed twice with PBS + 0.1% BSA (PBSA, Gibco, 15260-037, Lot#2000843) then suspended in 50 µL PBSA + 10% FBS for 10 min on ice to reduce background binding to IgG.

Antibodies were diluted in PBSA + 10% FBS and cells were stained for 1 hour at 4℃ in darkness with a gating panel (Panel 1, Panel 2, Panel 3, or Panel 4) and one anti-receptor antibody, or an equal concentration of matched isotype/fluorochrome control antibody. Stain for CD25 was included in Panel 1 when CD122, CD132, CD127, or CD215 was being measured (CD25 is used to separate Tregs from other CD4+ T cells).

Compensation beads (Simply Cellular Compensation Standard, Bangs Labs, 550, lot#12970) and quantitation standards (Quantum Simply Cellular anti-Mouse IgG or anti-Rat IgG, Bangs Labs, 815, Lot# 13895, 817, Lot# 13294) were prepared for compensation and standard curve. One well was prepared for each fluorophore with 2 µL antibody in 50 µL PBSA and the corresponding beads. Bead standards were incubated for 1 hr at room temperature in the dark.

Both beads and cells were washed twice with PBSA. Cells were suspended in 120 µL per well PBSA, and beads to 50 uL, and analyzed using an IntelliCyt iQue Screener PLUS with VBR configuration (Sartorius) with a sip time of 35 seconds and beads 30 seconds. Antibody number was calculated from fluorescence intensity by subtracting isotype control values from matched receptor stains and calibrated using the two lowest binding quantitation standards. Treg cells could not be gated in the absence of CD25, so CD4+ T cells were used as the isotype control to measure CD25 in Treg populations. Cells were gated as shown in [@Fig:gating]. Measurements were performed using four independent staining procedures over two days. Separately, the analysis was performed with anti-receptor antibodies at 3x normal concentration to verify that receptor binding was saturated.

### pSTAT5 Measurement of IL-2 and IL-15 Signaling in PBMCs

Human PBMCs were thawed, distributed across a 96-well plate, and allowed to recover as described above. IL-2 (R&D Systems, 202-IL-010) or IL-15 (R&D Systems, 247-ILB-025) were diluted in RPMI-1640 without FBS and added to the indicated concentrations. To measure pSTAT5, media was removed, and cells fixed in 100 µL of 10% formalin (Fisher Scientific, SF100-4) for 15 minutes at room temperature. Formalin was removed, cells were placed on ice, and cells were gently suspended in 50 µL of cold methanol (-30℃). Cells were stored overnight at -30℃. Cells were then washed twice with PBSA, split into two identical plates, and stained 1 hour at room temperature in darkness using antibody panels 4 and 5 with 50 µL per well. Cells were suspended in 100 µL PBSA per well, and beads to 50 uL, and analyzed on an IntelliCyt iQue Screener PLUS with VBR configuration (Sartorius) using a sip time of 35 seconds and beads 30 seconds. Compensation was performed as above. Populations were gated as shown in [@Fig:gating], and the median pSTAT5 level extracted for each population in each well.
