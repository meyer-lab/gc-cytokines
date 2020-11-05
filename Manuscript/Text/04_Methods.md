# Methods

All analysis was implemented in Python, and can be found at <https://github.com/meyer-lab/gc-cytokines>, release 1.0.

### Base model

Cytokine (IL-2, -4, -7, -9, -15, & -21) binding to receptors was modeled using ordinary differential equations (ODEs). IL-2 and -15 each had two private receptors, one being a signaling-deficient α-chain (IL-2Rα & -15Rα) and the other being signaling-competent IL-2Rβ. The other four cytokines each had one signaling-competent private receptor (IL-7Rα, -9R, -4Rα, & -21Rα). JAK-STAT signaling is initiated when JAK-binding motifs are brought together. JAK binding sites are found on the intracellular regions of the γ~c~, IL-2Rβ, IL-4Rα, IL-7Rα, IL-9R, and IL-21Rα receptors; therefore all complexes which contained two signaling-competent receptors were deemed to be active species. Ligands were assumed to first bind a private receptor and then can dimerize with other private receptors or γ~c~ thereafter. Direct binding of ligand to γ~c~ was not included due to its very weak or absent binding [@Voss2428].

In addition to binding interactions, our model incorporated receptor-ligand trafficking. Receptor synthesis was assumed to occur at a constant rate. The endocytosis rate was defined separately for active (k~endo,a~) and inactive (k~endo~) receptors. f~sort~ fraction of species in the endosome were ultimately trafficked to the lysosome, and active species in the endosome had a sorting fraction of 1.0. All endosomal species not sent to lysosomes were recycled back to the cell surface. The lysosomal degradation and recycling rate constants were defined as k~deg~ and k~rec~, respectively. We assumed no autocrine ligand was produced by the cells. We assumed an endosomal volume of 10 fL and endosomal surface area half that of the plasma membrane [@MEYER201525]. All binding events were assumed to occur with 5-fold greater disassociation rate in the endosome due to its acidic pH [@Fallon2000].

Free receptors and complexes were measured in units of number per cell and soluble ligands were measured in units of concentration (nM). Due to these unit choices for our species, the rate constants for ligand binding to a free receptors had units of nM^-1^ min^-1^, rate constants for the forward dimerization of free receptor to complex had units of cell min^-1^ number^-1^. Dissociation rates had units of min^-1^. All ligand-receptor binding processes had an assumed forward rate (k~bnd~) of 10^7^ M^-1^ sec^-1^. All forward dimerization reaction rates were assumed to be identical, represented by k~fwd~. Reverse reaction rates were unique. Experimentally-derived affinities of 1.0 [@Gonnordeaal1253], 59 [@Walsh_IL7_2012], 0.1 [@Renauld5690], and 0.07 nM [@Gonnordeaal1253] were used for IL-4, -7, -9, and -21 binding to their cognate private receptors, respectively. IL-2 and -15 were assumed to have affinities of 10 nM and 0.065 nM for their respective α-chains [@Rickert_receptor_const; @Mortier20012006; @DUBOIS2002537], and affinities of 144 nM and 438 nM for their respective β-chains [@Rickert_receptor_const]. Rates k~5~, k~10~, and k~11~ were set to their experimentally-determined dissassociation constants of 1.5, 12, and 63 nM [@Rickert_receptor_const].

Initial values were calculated by assuming steady-state in the absence of ligand. Differential equation solving was performed using the SUNDIALS solvers in C++, with a Python interface for all other code [@hindmarsh2005sundials]. Model sensitivities were calculated using the adjoint solution [@CAO2002171]. Calculating the adjoint requires the partial derivatives of the differential equations both with respect to the species and unknown parameters. Constructing these can be tedious and error-prone. Therefore, we calculated these algorithmically using forward-pass autodifferentiation implemented in Adept-2 [@hogan_robin_j_2017]. A model and sensitivities tolerance of 10^-9^ and 10^-3^, respectively, were used throughout. We used unit tests for conservation of mass, equilibrium, and detailed balance to ensure model correctness.

### Full Model ODEs

Below are the ODEs pertaining to IL-2 binding and unbinding events.

$$
\frac{dIL2R𝛼}{dt} = -kfbnd * IL2R𝛼 * IL2 + k1rev * IL2·IL2R𝛼 - kfwd * IL2Ra * IL2·IL2R𝛽·𝛾_c + k8rev * IL2·IL2R𝛼·IL2R𝛽·𝛾_c - kfwd * IL2R𝛼 * IL2·IL2R𝛽 + k12rev * IL2·IL2R𝛼·IL2R𝛽
$$

$$
\frac{dIL2R𝛽}{dt} = -kfbnd * IL2R𝛽 * IL2 + k2rev * IL2·IL2R𝛽 - kfwd * IL2R𝛽 * IL2·IL2R𝛼·𝛾_c + k9rev * IL2·IL2R𝛼·IL2R𝛽·𝛾_c - kfwd * IL2R𝛽 * IL2·IL2R𝛼 + k11rev * IL2·IL2R𝛼·IL2R𝛽
$$

$$
\frac{d𝛾_c}{dt} = -kfwd * IL2·IL2R𝛽 * 𝛾_c + k5rev * IL2·IL2R𝛽·𝛾_c - kfwd * IL2·IL2R𝛼 * 𝛾_c + k4rev * IL2·IL2R𝛼·𝛾_c - kfwd * IL2·IL2R𝛼·IL2R𝛽 * 𝛾_c + k10rev * IL2·IL2R𝛼·IL2R𝛽·𝛾_c
$$

$$
\frac{dIL2·IL2R𝛼}{dt} = -kfwd * IL2·IL2R𝛼 * IL2R𝛽 + k11rev * IL2·IL2R𝛼·IL2R𝛽 - kfwd * IL2·IL2R𝛼 * 𝛾_c + k4rev * IL2·IL2R𝛼·𝛾_c + kfbnd * IL2 * IL2R𝛼 - k1rev * IL2·IL2R𝛼
$$

$$
\frac{dIL2·IL2R𝛽}{dt} = -kfwd * IL2·IL2R𝛽 * IL2R𝛼 + k12rev * IL2·IL2R𝛼·IL2R𝛽 - kfwd * IL2·IL2R𝛽 * 𝛾_c + k5rev * IL2·IL2R𝛽·𝛾_c + kfbnd * IL2 * IL2R𝛽 - k2rev * IL2·IL2R𝛽
$$

$$
\frac{dIL2·IL2R𝛼·IL2R𝛽}{dt} = -kfwd * IL2·IL2R𝛼·IL2R𝛽 * 𝛾_c + k10rev * IL2·IL2R𝛼·IL2R𝛽·𝛾_c + kfwd * IL2·IL2R𝛼 * IL2R𝛽 - k11rev * IL2·IL2R𝛼·IL2R𝛽 + kfwd * IL2·IL2R𝛽 * IL2R𝛼 - k12rev * IL2·IL2R𝛼·IL2R𝛽
$$

$$
\frac{dIL2·IL2R𝛼·𝛾_c}{dt} = -kfwd * IL2·IL2R𝛼·𝛾_c * IL2R𝛽  + k9rev * IL2·IL2R𝛼·IL2R𝛽·𝛾_c + kfwd * IL2·IL2R𝛼 * 𝛾_c - k4rev * IL2·IL2R𝛼·𝛾_c 
$$

$$
\frac{dIL2·IL2R𝛽·𝛾_c}{dt} = -kfwd * IL2·IL2R𝛽·𝛾_c * IL2R𝛼  + k8rev * IL2·IL2R𝛼·IL2R𝛽·𝛾_c + kfwd * IL2·IL2R𝛽 * 𝛾_c - k5rev * IL2·IL2R𝛽·𝛾_c 
$$

$$
\frac{dIL2·IL2R𝛼·IL2R𝛽·𝛾_c}{dt} = kfwd * IL2·IL2R𝛽·𝛾_c * IL2R𝛼  - k8rev * IL2·IL2R𝛼·IL2R𝛽·𝛾_c + kfwd * IL2·IL2R𝛼·𝛾_c * IL2R𝛽 - k9rev * IL2·IL2R𝛼·IL2R𝛽·𝛾_c + kfwd * IL2·IL2R𝛼·IL2R𝛽 * 𝛾_c - k10rev * IL2·IL2R𝛼·IL2R𝛽·𝛾_c
$$

The ODEs for IL-15 binding and unbinding events are of the same form as those for IL-2, with IL-2, and IL-2R𝛼 having analogous species IL-15, and IL-15R𝛼. The analogous reverse binding rates are as follows:

$$
k1rev = k13rev, \  k2rev = k14rev, \  k4rev = k16rev, \  k5rev = k17rev, \  k8rev = k20rev, \  k9rev = k21rev, \  k10rev = k22rev, \  k11rev = k23rev, \  k12rev = k24rev
$$

Below are the ODEs pertaining to IL-4 binding and unbinding events.

$$
\frac{dIL4R𝛼}{dt} = -kfbnd * IL4 * IL4R𝛼 + k32rev*IL4·IL4R𝛼
$$

$$
\frac{dIL4·IL4R𝛼}{dt} = - kfwd * IL4·IL4R𝛼 * 𝛾_c + k33rev·IL4·IL4R𝛼·𝛾_c + kfbnd * IL4 * IL4R𝛼
$$

$$
\frac{dIL4·IL4R𝛼·𝛾_c}{dt} = kfwd * IL4·IL4R𝛼 * 𝛾_c - k33rev·IL4·IL4R𝛼·𝛾_c
$$

The ODEs for IL-7 binding and unbinding events are of the same form as those for IL-4, with IL-4, and IL-4R𝛼 having analogous species IL-7, and IL-7R𝛼. The analogous reverse binding rates are as follows:

$$
k33rev = k27rev, \  k32rev = k25rev 
$$

All of the above reactions also occur in the endosome for trafficked species, with reverse binding rates being 5 fold larger due to pH discrepancies between the endosome and extracellular space.

Trafficking is calculated for each species using the below equations.

$$
\frac{d \ Extracellular \ Active \ Species}{dt} = -Extracellular \ Active \ Species * (k_{endo} + k_{endo,a})
$$

$$
\frac{d \ Intracellular \ Active \ Species}{dt} = Extracellular \ Active \ Species * (k_{endo} + k_{endo,a}) / 0.5 - k_{deg}*Intracellular \ Active \ Species
$$

$$
\frac{d \ Extracellular \ Inactive \ Species}{dt} = -Extracellular \ Inactive \ Species * (k_{endo}) + k_{rec}*(1-f_{sort})*Intracellular \ Inactive \ Species * 0.5
$$

$$
\frac{d \ Intracellular \ Inactive \ Species}{dt} = Extracellular \ Inactive \ Species * (k_{endo}) / 0.5 - k_{rec}*(1-f_{sort})*Intracellular \ Inactive \ Species - (k_{deg}*f_{sort})*Intracellular \ Inactive \ Species
$$

### Model fitting

We used Markov chain Monte Carlo to fit the unknown parameters in our model using previously published cytokine response data [@ring_mechanistic_2012; @Gonnordeaal1253]. Experimental measurements include pSTAT activity under stimulation with varying concentrations of IL-2, -15, -4, and -7 as well as time-course measurements of surface IL-2Rβ upon IL-2 and -15 stimulation. YT-1 human NK cells were used for all data-sets involving IL-2 and IL-15. Human PBMC-derived CD4+TCR+CCR7^high^ cells were used for all IL-4 and -7 response data. All YT-1 cell experiments were performed both with the wild-type cell line, lacking IL-2Rα, and cells sorted for expression of the receptor. Data from Ring *et al* and Gonnord *et al* can be found in Figure 5 and Figure S3 of each paper, respectively [@ring_mechanistic_2012; @Gonnordeaal1253]. Measurements of receptor counts at steady state in Gonnord *et al* were used to solve for IL-7Rα, IL-4Rα, and γ~c~ expression rates in human PBMCs.

Fitting was performed with the Python package PyMC3. All unknown rate parameters were assumed to have a lognormal distribution with a standard deviation of 0.1; the only exception to these distributions was f~sort~ which was assumed to have a beta distribution with shape parameters of α=20 and β=40. Executing this fitting process yielded likelihood distributions of each unknown parameter and sum of squared error between model prediction and experimental data at each point of experimental data. The Geweke criterion metric was used to verify fitting convergence for all versions of the model ([@Fig:supp2]) [@Geweke92].

### Tensor Generation and Factorization

To perform tensor factorization we generated a three- (timepoints $\times$ cell types $\times$ ligand) or four-dimensional (timepoints $\times$ cell types $\times$ concentration $\times$ mutein) data tensor of predicted or measured ligand-induced signaling. Before decomposition, the tensor was variance scaled across each cell population. Tensor decomposition was performed using the Python package TensorLy [@TensorlyArxiv]. Except where indicated otherwise, tensor decomposition was performed using non-negative canonical polyadic decomposition. Where indicated, non-negative Tucker decomposition was used.

### Receptor abundance quantitation

Cryopreserved PBMCs (ATCC, PCS-800-011, lot#81115172) were thawed to room temperature and slowly diluted with 9 mL pre-warmed RPMI-1640 medium (Gibco, 11875-093) supplemented with 10% fetal bovine serum (FBS, Seradigm, 1500-500, lot#322B15). Media was removed, and cells washed once more with 10 mL warm RPMI-1640 + 10% FBS. Cells were brought to 1.5x10^6^ cells/mL, distributed at 250,000 cells per well in a 96-well V-bottom plate, and allowed to recover 2 hrs at 37℃ in an incubator at 5% CO2. Cells were then washed twice with PBS + 0.1% BSA (PBSA, Gibco, 15260-037, Lot#2000843) and suspended in 50 µL PBSA + 10% FBS for 10 min on ice to reduce background binding to IgG.

Antibodies were diluted in PBSA + 10% FBS and cells were stained for 1 hr at 4℃ in darkness with a gating panel (Panel 1, Panel 2, Panel 3, or Panel 4) and one anti-receptor antibody, or an equal concentration of matched isotype/fluorochrome control antibody. Stain for CD25 was included in Panel 1 when CD122, CD132, CD127, or CD215 was being measured (CD25 is used to separate T~reg~s from other CD4+ T cells).

Compensation beads (Simply Cellular Compensation Standard, Bangs Labs, 550, lot#12970) and quantitation standards (Quantum Simply Cellular anti-Mouse IgG or anti-Rat IgG, Bangs Labs, 815, Lot#13895, 817, Lot#13294) were prepared for compensation and standard curve. One well was prepared for each fluorophore with 2 µL antibody in 50 µL PBSA and the corresponding beads. Bead standards were incubated for 1 hr at room temperature in the dark.

Both beads and cells were washed twice with PBSA. Cells were suspended in 120 µL per well PBSA, and beads to 50 µL, and analyzed using an IntelliCyt iQue Screener PLUS with VBR configuration (Sartorius) with a sip time of 35 and 30 secs for cells and beads, respectively. Antibody number was calculated from fluorescence intensity by subtracting isotype control values from matched receptor stains and calibrated using the two lowest binding quantitation standards. T~reg~ cells could not be gated in the absence of CD25, so CD4+ T cells were used as the isotype control to measure CD25 in T~reg~ populations. Cells were gated ([@Fig:gating]), and then measurements were performed using four independent staining procedures over two days. Separately, the analysis was performed with anti-receptor antibodies at 3x normal concentration to verify that receptor binding was saturated. Replicates were summarized by geometric mean.

### pSTAT5 Measurement of IL-2 and -15 Signaling in PBMCs

Human PBMCs were thawed, distributed across a 96-well plate, and allowed to recover as described above. IL-2 (R&D Systems, 202-IL-010) or IL-15 (R&D Systems, 247-ILB-025) were diluted in RPMI-1640 without FBS and added to the indicated concentrations. To measure pSTAT5, media was removed, and cells fixed in 100 µL of 10% formalin (Fisher Scientific, SF100-4) for 15 mins at room temperature. Formalin was removed, cells were placed on ice, and cells were gently suspended in 50 µL of cold methanol (-30℃). Cells were stored overnight at -30℃. Cells were then washed twice with PBSA, split into two identical plates, and stained 1 hr at room temperature in darkness using antibody panels 4 and 5 with 50 µL per well. Cells were suspended in 100 µL PBSA per well, and beads to 50 µL, and analyzed on an IntelliCyt iQue Screener PLUS with VBR configuration (Sartorius) using a sip time of 35 seconds and beads 30 seconds. Compensation was performed as above. Populations were gated ([@Fig:gating]), and the median pSTAT5 level extracted for each population in each well.

### Recombinant proteins

IL-2/Fc fusion proteins were expressed using the Expi293 expression system according to manufacturer instructions (Thermo Scientific). Proteins were as human IgG1 Fc fused at the N- or C-terminus to human IL-2 through a (G4S)4 linker. C-terminal fusions omitted the C-terminal lysine residue of human IgG1. The AviTag sequence GLNDIFEAQKIEWHE was included on whichever terminus did not contain IL-2. Fc mutations to prevent dimerization were introduced into the Fc sequence [@Ishino04242013]. Proteins were purified using MabSelect resin (GE Healthcare). Proteins were biotinylated using BirA enzyme (BPS Biosciences) according to manufacturer instructions, and extensively buffer-exchanged into phosphate buffered saline (PBS) using Amicon 10 kDa spin concentrators (EMD Millipore). The sequence of IL-2Rβ/γ Fc heterodimer was based on a reported active heterodimeric molecule (patent application US20150218260A1), with the addition of (G4S)2 linker between the Fc and each receptor ectodomain. The protein was expressed in the Expi293 system and purified on MabSelect resin as above. IL2-Rα ectodomain was produced with C-terminal 6xHis tag and purified on Nickel-NTA spin columns (Qiagen) according to manufacturer instructions. 

### Octet binding assays

Binding affinity was measured on an OctetRED384 (ForteBio). Briefly, biotinylated monomeric IL-2/Fc fusion proteins were uniformly loaded to Streptavidin biosensors (ForteBio) at roughly 10% of saturation point and equilibrated for 10 minutes in PBS + 0.1% bovine serum albumin (BSA). Association time was up to 40 minutes in IL-2Rβ/γ titrated in 2x steps from 400 nM to 6.25 nM, or IL-2Rα from 25 nM to 20 pM, followed by dissociation in PBS + 0.1% BSA. A zero-concentration control sensor was included in each measurement and used as a reference signal. Assays were performed in quadruplicate across two days. Binding to IL-2Rα did not fit to a simple binding model so equilibrium binding was used to determine the K~D~ within each assay. Binding to IL-2Rβ/γ fit a 1:1 binding model so on-rate (k~on~), off-rate (k~off~) and K~D~ were determined by fitting to the entire binding curve. Kinetic parameters and K~D~ were calculated for each assay by averaging all concentrations with detectable binding signal (typically 12.5 nM and above).
