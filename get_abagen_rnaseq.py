# %%
import numpy as np
import abagen
import pandas as pd
import os
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
from abagen.io import read_microarray, read_tpm, read_annotation, read_probes
from abagen.utils import efficient_corr
from abagen.datasets import fetch_rnaseq, fetch_microarray
from abagen.probes_ import _groupby_structure_id

##################
# %% GROUP AVG
##################

# load atlas info
atlas = nib.load('./data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz')
atlas_info = pd.read_csv('./data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv')

# check atlas and atlas_info
atlas = abagen.images.check_atlas(atlas, atlas_info)

# %%
# fetch rna and microarray data from abagen and compare sampled regions

# fetch RNAseq data
rnaseq = fetch_rnaseq(donors='all', data_dir='./data/')
donors = list(rnaseq.keys())

# fetch matching microarray data
microarray = fetch_microarray(donors=donors, data_dir='./data/')
peptides = pd.read_csv('./data/gene_list.csv')['Gene'].tolist()

corrs = []

for n, (donor, data) in enumerate(rnaseq.items()):
        expression_fn = microarray[donor]['microarray']
        annotation_fn = microarray[donor]['annotation']
        probes_fn = microarray[donor]['probes']
        
        # use abagen.io to read files
        expression = read_microarray(expression_fn)
        annotation = read_annotation(annotation_fn)
        probes = read_probes(probes_fn)
        
        # collapse (i.e., average) data across AHAB  anatomical regions
        micro = _groupby_structure_id(expression, annotation)
        rna = _groupby_structure_id(read_tpm(data['tpm']),
                                data['annotation'])

        # get rid of "constant" RNAseq genes
        rna = rna[np.logical_not(np.isclose(rna.std(axis=1, ddof=1), 0))]

        # get matching genes + strcutres between microarray + RNAseq
        regions = np.intersect1d(micro.columns, rna.columns)
        mask = np.isin(np.asarray(probes.gene_symbol),
                np.intersect1d(probes.gene_symbol, rna.index))
        genes = np.asarray(probes.loc[mask, 'gene_symbol'])
        
        # match with available peptides
        genes = np.intersect1d(genes, peptides)
        mask = np.isin(np.asarray(probes.gene_symbol), genes)
        
        # mask data accordingly
        micro, rna = micro.loc[mask, regions], rna.loc[genes, regions].T
        
        # average micro across probes that map to the same gene
        micro = micro.groupby(probes.loc[mask, 'gene_symbol']).mean().T
        
        # map structure id to region names
        id2name = dict(zip(annotation['structure_id'], annotation['structure_name']))
        micro.index = micro.index.map(id2name)
        rna.index = rna.index.map(id2name)

        # correlate expression values across regions for each gene
        corrs.append(efficient_corr(micro.rank(), rna.rank()))

corr_df = pd.DataFrame(corrs, index=['donor-{}'.format(d) for d in donors],
                       columns=genes)

# plot correlation distribution
sns.histplot(corr_df.T)
plt.xlabel('Spearman correlation')

# %%
# sort columns by mean correlation
corr_df = corr_df.reindex(corr_df.mean().sort_values(ascending=False).index, axis=1)

# keep only genes with mean correlation of at least > 0.2
stable_genes = corr_df.loc[:, corr_df.mean() > 0.2].columns
# alternatively, include all peptide genes
# stable_genes = corr_df.columns

abagen_genes = pd.read_csv('data/abagen_genes_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0).columns
relevant_genes = np.intersect1d(stable_genes, abagen_genes)

# load differential stability
diff_stability = pd.read_csv('data/abagen_genes_stability.csv', index_col=0).T
diff_stability = diff_stability[relevant_genes]

# show scatterplot of differential stability and mean gene expression correlation
sns.scatterplot(x=corr_df.loc[:,relevant_genes].mean(), y=diff_stability.values.flatten())
plt.xlabel('Microarray-RNAseq correlation (donor-average)')
plt.ylabel('Differential stability')
