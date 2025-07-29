# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sstats
from utils import index_structure

savefig = False

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         LOAD DATA
###############################################################################
# load all genes extracted from abagen
all_genes = pd.read_csv('./data/abagen_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
precursor_names = pd.read_csv('data/precursor_qc.csv', index_col=0).index

# select only precursor genes from all_genes
precursor_genes = all_genes[precursor_names]

# load colors
spectral = [color for i, color in enumerate(sns.color_palette('Spectral')) if i in [1,2,4]]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                    PRECURSOR PEPTIDE GENE DISTRIBUTION
###############################################################################

# split precursor into cortex, subcortex and hth. rename their index as ctx, sbctx and hth respectively
precursor_ctx = index_structure(precursor_genes, structure='CTX').rename(index=lambda x: 'CTX')
precursor_sbctx = index_structure(precursor_genes, structure='SBCTX').rename(index=lambda x: 'SBCTX')
precursor_hth = index_structure(precursor_genes, structure='HTH').to_frame().T  # type:ignore

# merge into one tall dataframe
precursor_genes = pd.concat((precursor_ctx, precursor_sbctx, precursor_hth)).reset_index() # type:ignore
precursor_genes = precursor_genes.melt(id_vars='index', var_name='gene', value_name='expression')

# average expression of each gene in each structure
precursor_genes = precursor_genes.groupby(['index', 'gene']).mean().reset_index()

# search for lowest expression in CTX index
lowest_ctx = precursor_genes[precursor_genes['index'] == 'CTX']['expression'].min()

# plot boxplot of precursor gene expression in cortex, subcortex and hypothalamus
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
sns.boxplot(data=precursor_genes, x='index', y='expression', ax=ax, palette=spectral[::-1], 
            saturation=1, whis=3, order=['CTX', 'HTH', 'SBCTX'])
ax.set_ylim(0, 1)
ax.set_title('Peptide precursor gene expression')
ax.set_ylabel('Gene expression')
ax.set_xlabel('Structure')
sns.despine(trim=True)

if savefig:
    plt.savefig('figs/precursor_gene_expression.pdf')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                 TEST SIGNIFICANCE BETWEEN STRUCTURES
###############################################################################
# median of each gene in each structure
cortex_median = precursor_ctx.median(axis=0)
subcortex_median = precursor_sbctx.median(axis=0)
hth_median = precursor_hth.median(axis=0)

# do significance test between structures
# test between hypothalamus and cortex
tstat, pval = sstats.ttest_ind(hth_median, cortex_median, alternative='two-sided', equal_var=False)
print(f'Hypothalamus vs Cortex: t-statistic = {tstat:0.2f}, p-value = {pval:0.4f}')

# test between subcortex and hypothalamus
tstat, pval = sstats.ttest_ind(hth_median, subcortex_median, alternative='two-sided', equal_var=False)
print(f'Hypothalamus vs Subcortex: t-statistic = {tstat:0.2f}, p-value = {pval:0.4f}')

# test between cortex and subcortex
tstat, pval = sstats.ttest_ind(cortex_median, subcortex_median, alternative='two-sided', equal_var=False)
print(f'Subcortex vs Cortex: t-statistic = {tstat:0.2f}, p-value = {pval:0.4f}')
