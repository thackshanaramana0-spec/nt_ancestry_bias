# arXiv Submission Metadata

## Title
Training-Set Diversity Does Not Eliminate Ancestry-Associated Reference Bias in DNA Foundation Models

## Authors
Thackshanaramana B

## Affiliations
Department of Computational Intelligence, College of Engineering and Technology,
SRM Institute of Science and Technology, Kattankulathur, Tamil Nadu, India

## Primary category
q-bio.GN (Genomics)

## Cross-list categories
cs.LG (Machine Learning)

## Abstract (148 words — paste as-is)
DNA foundation models are increasingly used for genomic variant interpretation,
yet whether diversity in pretraining data is sufficient to mitigate
ancestry-associated scoring bias remains unclear. Here, we examine reference
allele preference in Nucleotide Transformer using ancestry-enriched common
variants from gnomAD and compare models pretrained on the human reference genome
and on 3,202 genetically diverse human genomes. Both models exhibit a
significant ancestry-associated scoring asymmetry, with European-enriched
variants receiving greater reference allele preference than African-enriched
variants. Strikingly, the EUR-AFR bias gap changes by only 1.2% following
diverse-genome pretraining. Neither population allele frequency nor local
genomic variant density explains the observed difference, indicating that the
effect is not simply driven by variant prevalence or regional sequence variation.
These results show that substantially increasing genomic diversity during
pretraining does not eliminate ancestry-associated reference bias in Nucleotide
Transformer and point to a persistent reference-centered component in DNA
foundation-model scoring. More representative genomic representations, including
pangenome-aware approaches, may therefore be necessary to address
ancestry-associated asymmetries in genomic foundation models.

## Comments
8 pages, 3 figures, 4 tables. Code and data: https://github.com/thackshanaramana0-spec/nt_ancestry_bias

## MSC classes (optional)
92-10 (Biology and other natural sciences)

## ACM classes (optional)
J.3 (Life and Medical Sciences)

---

## Suggested GitHub repo name
nt-ancestry-bias

## Upload checklist
- [ ] paper_D_ieee_final.tex
- [ ] references.bib
- [ ] figures/fig1_violin.png
- [ ] figures/fig2_calibration.png
- [ ] figures/fig3_ranking_displacement.png
- [ ] Create GitHub repo `nt-ancestry-bias` and push
- [ ] Submit to arXiv (arxiv.org -> Submit -> q-bio.GN)
- [ ] After arXiv ID assigned: submit to Briefings in Bioinformatics
