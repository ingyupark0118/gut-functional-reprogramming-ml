# Analysis code, parameters and figure values (v2.0)

Supporting material for "Interpretable functional profiling of the gut microbiome
identifies divergent strain-level commensal adaptation across the
diabetes–cirrhosis spectrum" (revised version).

This archive provides the analysis code, the random seeds and parameter settings
used at each step of the pipeline, the values underlying the figure panels, and
the clinical characteristics of the four study groups. Changes from v1.0 are
listed in `CHANGES_from_v1.md`.

## Contents

| Folder | Content |
|---|---|
| `code/` | Analysis scripts, ordered by step (below) |
| `parameters/` | Cross-validation design, hyper-parameter grids and the values selected in each fold (`parameters.json`, `selected_hyperparameters_by_fold.csv`), filtering and significance thresholds (`thresholds.csv`), random seeds (`seeds.txt`) |
| `figure_values/` | Values underlying the figure panels, named by figure and panel |
| `clinical_summary/` | Clinical characteristics of the four study groups, with the number of participants having a recorded value for each variable |
| `reference_genomes/` | Public reference genomes used in the strain-level analyses (Supplementary Table 5) |
| `environment/` | Python and R package versions, external tool versions |

## Analysis steps (`code/`)

| Step | Folder | Analysis | Figures and tables |
|---|---|---|---|
| 1 | `01_nested_cross_validation/` | Feature matrices and nested cross-validation, including the clinical-plus-microbial models | Fig. 2, Fig. 3, Supplementary Fig. 8 |
| 2 | `02_batch_correction_sensitivity/` | ConQuR batch correction, as a sensitivity analysis | Supplementary Fig. 4 |
| 3 | `03_covariate_adjusted_models/` | Covariate-adjusted models of the candidate KOs and their taxonomic contribution | Fig. 4B, 4C |
| 4 | `04_directional_score/` | Directional functional shift score and its application to the external cohorts | Fig. 5 |
| 5 | `05_correlation_networks/` | Spearman and SparCC networks with size-matched sub-sampling | Fig. 4A, 4D, 4E, Supplementary Fig. 2 |
| 6 | `06_targeted_assembly/` | Read recruitment, assembly and contig filtering for *Lactobacillus crispatus* | Fig. 6 |
| 7 | `07_reference_genomes/` | Selection, filtering and habitat assignment of reference genomes | Fig. 6, Supplementary Table 5 |
| 8 | `08_phylogeny_and_ANI/` | Core-gene phylogenies and average nucleotide identity | Fig. 6A–C |
| 9 | `09_habitat_analysis/` | Habitat of the reference genomes nearest to each assembled genome | Fig. 6D |
| 10 | `10_permanova/` | PERMANOVA and ANOSIM of community structure | Supplementary Table 2 |

## Figure values

Each file is named after the figure and panel it belongs to; files ending in
`_aggregated` contain group-level summaries and test results only. Schematic
panels (Fig. 1A, Fig. 2A) have no underlying values, and the design numbers of
the cross-validation scheme are given in `Fig2A_nested_CV_schematic.csv`.

## Scope

Individual-level data are not included; access to the underlying data is
described in the Data availability statement of the paper.

SparCC (Friedman and Alm, PLoS Comput Biol 2012) is not redistributed; the
scripts that prepare its input, call it and aggregate its output are provided.

In the scripts, paths are neutral placeholders, sample lists are read from files
that are not distributed, genome names follow the labels used in the figures
(CDM-1 to CDM-4, Cirrhosis-1), and collection sites are coded Site 1–5.

## License

Code: MIT License (see `LICENSE`). Data files: Creative Commons Attribution 4.0
International (CC BY 4.0).