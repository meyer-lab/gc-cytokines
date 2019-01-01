---
title: Tensor Factorization Maps Computation Performed by the Common Gamma Chain Receptors
author:
- name: Adam Weiner
  affilnum: a
- name: Ali Farhat
  affilnum: a
- name: Aaron S. Meyer
  affilnum: a,b
keywords: [IL2, IL15, IL21, IL4, IL7, IL9, common gamma chain, cytokines, receptors, immunology, T cells, NK cells]
affiliation:
- name: Department of Bioengineering, Jonsson Comprehensive Cancer Center, Eli and Edythe Broad Center of Regenerative Medicine and Stem Cell Research; University of California, Los Angeles
  key: a
- name: Contact info
  key: b
bibliography: ./Manuscript/References.bib
abstract: Many receptor families exhibit both pleitropy and redundancy in their regulation, with multiple ligands, receptors, and responding cell populations. The multivariate nature of these systems confounds intuition about therapeutic manipulation. The common γ-chain cytokine receptor dimerizes with complexes of the cytokines interleukin (IL)2, IL-4, IL-7, IL-9, IL-15, and IL-21 and their corresponding "private" receptors. These γ-chain cytokines have accrued broad interest as potential immune therapies because they potently modulate immune cell population types. Here, we build a reaction model for the diverse ligand-receptor interactions of common γ-chain cytokines enabling quantitative predictions of response. Using this quantitative model, we employ tensor factorization to build a quantitative map of regulation across the family. These results map the emergent behavior of the common γ-chain cytokines, and demonstrate an approach to generating interpretable guidelines for manipulation of complex receptor families.
link-citations: true
csl: ./Manuscript/Templates/nature.csl
---

# Summary points

- A mechanism-based binding model of the γ-chain cytokines successfully captures responses to IL-2, IL-15, IL-4, and IL-7
- Two
- Canonical polyadic decomposition maps responses across cell populations, receptors, cytokines, and dynamics

# Introduction

- The common gamma chain cytokine family is central to regulation of the immune system
    - Giving IL-15 doses to monkeys caused expansion of NK and T-eff cell lines, making it a potential therapy in cancers and immunodeficiencies. [@Sneller6845]
    - Transgenic mice that overexpressed IL-15 revealed that IL-15 selectively propagates memory T cells by blocking IL-2-induced T cell death. [@Marks-Konczalik11445]
    - gamma chain cytokines have accrued interest as cancer immunotherapeutics due to their ability to regulate lymphocytes. [@Pulliam2015]
    - Mutation in gamma chain causes XSCID. [@NOGUCHI1993147]
    - Cells must express gamma chain in order to undergo lymphoproliferation. [@Amorosi3304]
    - Gamma chain is tied to hematopoietic cell proliferation. High expression correlates with malignancies and knockouts slow the proliferation of these malignancies. [@Vigliano2012]
- Many receptor families are composed of multiple ligands, receptors, and cell populations
	- [@Antebi_BMP]
    - hGH affinities do not correlate well with cell stimulation levels because of endosomal trafficking. [@Haugh2004]
    - EGFRs are being targeted in cancer therapies due to their increased expression in tumor cells. [@CIARDIELLO20031348]
- Efforts to engineer gc behavior and/or gc treatments
    - [@ring_mechanistic_2012]
    - Specific IL-2 amino acid substitutions render it incapable from binding to IL2Ra. [@Collins7709]
    - IL-2 mutant library created and tested to see if any mutations increased binding affinity to receptors and therefore greater T cell activity. [@Berndt1994]
    - "in vitro evolution" was used to create an IL-2 mutant with increased affinity for IL-2Rb. [@Levin2012]
- Efforts to model behavior of cytokine families
    - Model that explains the positive and negative feedback loops of TCR antigen recognition [@FrancoisE888]
    - Modeling that doesn't account for trafficking
        - Model of T cell proliferative response to IL-2 and IL-4. [@BURKE199742]
        - High level model of how IL-2 feedback loops regulate T cell populations. Focus on helper T and regulatory T cells. [@Feinerman437], [@Garcia2012]. 
    - Modeling that accounts for trafficking
        - IL-2 stimulates downregulation of IL2Ra due to endocytosis of IL2R complexes. [@Duprez15091988]
        - Accounting for trafficking allows for more accurate ligand depletion and cell proliferation model in case of IL-2. [@Fallon2000]
        - Trafficking and TCR inhibition of pSTAT5 activation can be used to improve reaction model of IL-2... potentially get rid of this [@Tkach2014TCT]
    - Focused on competition for common gamma chain
        - Showed how high IL2Ra abundnaces lowered the IL-2 EC50 but dampened the signaling responses of IL-7 and IL-15. Their work was mostly experimental (flow cytometry)  [@Cotarira17]
        - Abundance of gamma chain on surface played role in IL-2 mediated activation. [@Ponce2016]
- Transition to paper

