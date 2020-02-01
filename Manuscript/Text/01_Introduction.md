---
title: Modeling Cell-Specific Dynamics and Regulation of the Common Gamma Chain Cytokines
author:
- name: Ali M. Farhat
  affilnum: a,c
- name: Adam C. Weiner
  affilnum: a,c
- name: Cori Posner
  affilnum: b
- name: Zoe S. Kim
  affilnum: a
- name: Brian Orcutt-Jahns
  affilnum: a
- name: Scott M. Carlson
  affilnum: b
- name: Aaron S. Meyer
  affilnum: a,d
keywords: [IL-2, IL-15, IL-21, IL-4, IL-7, IL-9, common gamma chain, cytokines, receptors, immunology, T cells, NK cells]
affiliation:
- name: Department of Bioengineering, Jonsson Comprehensive Cancer Center, Eli and Edythe Broad Center of Regenerative Medicine and Stem Cell Research; University of California, Los Angeles
  key: a
- name: Visterra, Inc., Waltham, MA
  key: b
- name: These authors contributed equally to this work
  key: c
- name: a@asmlab.org
  key: d
bibliography: ./Manuscript/References.bib
abstract: Many receptor families exhibit both pleiotropy and redundancy in their regulation, with multiple ligands, receptors, and responding cell populations. Any intervention, therefore, has multiple effects and is context specific, confounding intuition about how to carry out precise therapeutic manipulation. The common γ-chain cytokine receptor dimerizes with complexes of the cytokines interleukin (IL)-2, IL-4, IL-7, IL-9, IL-15, and IL-21 and their corresponding "private" receptors. These cytokines have existing uses and future potential as immune therapies due to their ability to regulate the abundance and function of specific immune cell populations. However, engineering cell specificity into a therapy is confounded by the complexity of the family across responsive cell types. Here, we build a binding-reaction model for the ligand-receptor interactions of common γ-chain cytokines enabling quantitative predictions of response. We show that accounting for receptor-ligand trafficking is essential to accurately model cell response. Using this model, we visualize regulation across the family and immune cell types by tensor factorization. This model accurately predicts ligand response across a wide panel of cell types under diverse experimental designs. Further, we can predict the effect and specificity of natural or engineered ligands across cell types. In total, these results present a more accurate model of ligand response validated across a panel of immune cell types, and demonstrate an approach for generating interpretable guidelines to manipulate the cell type-specific targeting of engineered ligands.
link-citations: true
csl: ./common/templates/nature.csl
---

# Summary points

- A dynamical model of the γ-chain cytokines accurately models responses to IL-2, IL-15, IL-4, and IL-7.
- Receptor trafficking is necessary for capturing ligand response. 
- Tensor factorization maps responses across cell populations, receptors, cytokines, and dynamics to visualize cytokine specificity.
- An activation model coupled with tensor factorization creates design specifications for engineering cell-specific responses.

# Introduction

<!-- Introduce the common gc family and its importance regulating the immune system.-->

Cytokines are cell signaling proteins responsible for cellular communication within the immune system. The common γ-chain (γ~c~) receptor cytokines, including interlukin (IL)-2, 4, 7, 9, 15, and 21, are integral for modulating both innate and adaptive immune responses. As such, they have existing uses and future potential as immune therapies [@Rochman_2009; @LEONARD2019832]. Each ligand binds to its specific private receptors before interacting with the common γ~c~ receptor to induce signaling [@Walsh2010]. γ~c~ receptor signaling induces lymphoproliferation, offering a mechanism for selectively expanding or repressing immune cell types [@Amorosi3304; @Vigliano2012]. Consequently, loss-of-function or reduced activity mutations in the γ~c~ receptor can cause severe combined immunodeficiency (SCID) due to insufficient T and NK cell maturation [@Wang9542]. Deletion or inactivating mutations in IL-2 or its private receptors leads to more selective effects, including diminished regulatory T cell (T~reg~) proliferation and loss of self-tolerance [@Sharma1251; @Sharfe3168; @Horak]. Deficiency in the IL-2 receptor IL-2Rα also causes hyperproliferation in CD8+ T cells, but diminished antigen response [@GOUDY2013248]. These examples show how γ~c~ receptor cytokines coordinate a dynamic balance of immune cell abundance and function.

<!--Complex gc receptor family with effects across many cell populations.-->

The γ~c~ cytokines’ ability to regulate lymphocytes can impact both solid and hematological tumors [@Pulliam2015]. IL-2 is an approved, effective therapy for metastatic melanoma, and the antitumor effects of IL-2 and -15 have been explored in combination with other treatments [@BentebibelCD; @ZhuSynergy]. Nonetheless, understanding these cytokines' regulation is stymied by their complex binding and activation mechanism [@Walsh2010]. Any intervention imparts effects across multiple distinct cell populations, with each population having a unique response defined by its receptor expression [@ring_mechanistic_2012; @Cotarira17]. These cytokines’ potency is largely limited by the severe toxicities, such as deadly vascular leakage with IL-2 [@Kreig:2010]. Finally, IL-2 and IL-15 are rapidly cleared by receptor-mediated endocytosis, limiting their half-life *in vivo* [@Konrad2009; @Bernett1595].

<!--Efforts in producing mutant ligands to induce specific responses.-->

To address the limitations of natural ligands, engineered proteins have been produced with potentially beneficial properties [@LEONARD2019832]. The most common approach has been to develop mutant ligands by modulating the binding kinetics for specific receptors [@Berndt1994; @Collins7709]. For example, mutant IL-2 forms with a higher binding affinity for IL-2Rβ, or reduced binding to IL-2Rα, induce greater cytotoxic T cell proliferation, antitumor responses, and proportionally less T~reg~ expansion [@Levin2012; @BentebibelCD]. This behavior can be understood through IL-2’s typical mode of action, in which T~reg~s are sensitized to IL-2 by expression of IL-2Rα [@ring_mechanistic_2012]. Bypassing this sensitization mechanism thus shifts cell-specificity [@Levin2012]. Conversely, mutants skewed toward IL-2Rα over IL-2Rβ binding selectively expand T~reg~ populations, over cytotoxic T cells and NK cells, compared to native IL-2 [@Bell_2015; @Peterson_2018].

<!--How previous computational models for this family performed.-->

The therapeutic potential and complexity of this family make computational models especially valuable for rational engineering. Early attempts at mathematically modeling the synergy between IL-2 and IL-4 in B and T cells successfully identified a phenomenological model that could capture the synergy between the two cytokines [@BURKE199742]. A cell population model has explained how T~reg~ IL-2 consumption suppresses effector T cell activation [@Feinerman437]. However, any model needs to incorporate the key regulatory features of a pathway to accurately predict cell response. With structural information that clarified the mechanism of cytokine binding, a model of IL-4, IL-7, and IL-21 binding revealed pathway cross-talk due to the relative γ~c~ receptor affinities [@Gonnordeaal1253]. Nevertheless, these models have failed to account for endosomal trafficking, and have not been constructed to model multiple immune cell types. IL-2 induces rapid endocytosis-mediated IL-2Rα and IL-2Rβ downregulation [@ring_mechanistic_2012; @Duprez15091988], and trafficking is known to be a potent regulatory mechanism for all members of the γ~c~ family [@Lamaze_2001]. Indeed, an early model showed the critical importance of IL-2 trafficking, but without the structural features of ligand binding enabling specific ligand engineering [@Fallon2000].

<!--Transition to our paper.-->

In this paper, we assemble a map of γ~c~ cytokine family regulation. We first built a family-wide mathematical model that incorporates both binding and trafficking kinetics. This more comprehensive model allows us to investigate emergent behavior, such as competition between cytokines. This cytokine family is inherently high dimensional—with multiple ligands, cognate receptors, and cells with distinct expression. Therefore, we use tensor factorization methods to visualize the family-wide regulation. This map helps visualize how native or engineered ligands are addressed to specific immune cell populations based on their receptor expression levels. The methods used here can similarly be used in experimental and computational efforts of decoding other complex signaling pathways such as Wnt, Hedgehog, Notch, and BMP/TGFβ [@Eubeleneaat1178; @Li543; @ANTEBI201716; @Antebi_BMP].
