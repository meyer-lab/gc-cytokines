## Model-data comparison

![**Model accurately predicts cell type-specific response across a panel of PBMC-derived cell types.** A) Comparison of two replicates measuring pSTAT5 response to a dose-response of IL-2/15, time course, and panel of PBMC-derived cell types. B–C) pSTAT5 response to IL-2 (B) or IL-15 (C) dose responses in NK, CD8+, and T~reg~ cells. D) Pearson correlation coefficients between model prediction and experimental measurements for all 10 cell populations (full data shown in [@fig:supp5]). E) Both experimentally-derived and model-predicted EC~50~s of dose response across IL-2/15 and all 10 cell types. EC~50~s are shown for 1 hr time point.](./Figures/figure4.svg){#fig:expData}

We evaluated whether our model could predict differences in the cell type-specificity of ligand treatment by comparing its predicts to IL-2 and IL-15 responses across a panel of PBMC-derived cell populations.



Overall, our model predictions of ligand pSTAT5 response closely matched experimental measurement ([@Fig:expData; @fig:supp5]). 




While our model accurately predicted our experimentally-measured responses overall, and specifically the sensitivities of the dose-response profiles, we noticed some discrepancy specifically at high ligand concentrations, longer times, and specific cell populations ([@Fig:expData; @fig:supp5]). For example, while CD8+ cells almost exactly match model predictions at 30 min, by 4 hrs we experimentally observed a fully biphasic response with respect to IL-2 concentration, and a plateau with IL-15 that decreased over time. This decreased signaling was most pronounced with the CD8+ cells, but could be observed to lesser extents with many of the other cell populations ([@fig:supp5]). We expect two possible explanations for this discrepancy: First, CD8+ populations are known to proteolytically shed IL-2Rα in an activity-responsive manner [@Junghans1587]. Our model also only uses a very simple sigmoidal relationship between active receptor and pSTAT5 signal. Other components of the Jak-STAT pathway surely influence its dynamic reponse (CITE). However, overall the model presented here remains useful for exploring the determinants of cell type-specific response, which originate at the receptor expression profile on the cell surface.

<!-- TODO: We could discuss the parameters of the sigmoidal fit, because it possibly suggests variation in Jak-STAT properties. -->
