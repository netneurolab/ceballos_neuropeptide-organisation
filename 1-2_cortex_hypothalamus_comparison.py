# %%
import numpy as np
import abagen
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import index_structure

# %%
# load the genes
all_genes = pd.read_csv('./data/abagen_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)

# load atlas info
atlas_info = pd.read_csv('./data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv', index_col=0)
atlas_info['network'] = atlas_info['name'].str.split('_').str[0].str.split('-').str[0]
atlas_info['network'] = atlas_info['network'].str.replace('lAMY', 'AMY').str.replace('mAMY', 'AMY')
atlas_info['network'] = atlas_info['network'].str.replace('aGP', 'GP').str.replace('pGP', 'GP')

# create name to network mapper
mapper = dict(zip(atlas_info['name'], atlas_info['network']))

# load list of relevant genes
gene_list = pd.read_csv('data/gene_list.csv')

receptor_names = pd.read_csv('data/receptor_qc.csv', index_col=0).index
precursor_names = pd.read_csv('data/precursor_qc.csv', index_col=0).index

# keep only genes that are in all_genes
receptor_names =  receptor_names[receptor_names.isin(all_genes.columns)]
precursor_names = precursor_names[precursor_names.isin(all_genes.columns)]

# select overlapping genes from genes and gene_list
receptor_genes = all_genes[receptor_names]
precursor_genes = all_genes[precursor_names]

# split precursor into cortex, subcortex and hth. rename their index as ctx, sbctx and hth respectively
# precursor_ctx = index_structure(precursor_genes, structure='CTX').rename(index=lambda x: 'CTX')
# precursor_sbctx = index_structure(precursor_genes, structure='SBCTX').rename(index=lambda x: 'SBCTX')

# map the index to the network
precursor_ctx = index_structure(precursor_genes, structure='CTX').rename(index=mapper)
precursor_sbctx = index_structure(precursor_genes, structure='SBCTX').rename(index=mapper)
precursor_hth = index_structure(precursor_genes, structure='HTH').to_frame().T  # type:ignore

# merge into one tall dataframe
precursor_genes = pd.concat((precursor_ctx, precursor_sbctx, precursor_hth)).reset_index() # type:ignore
precursor_genes = precursor_genes.melt(id_vars='index', var_name='gene', value_name='expression')

# plot boxplot of precursor gene expression in cortex, subcortex and hypothalamus
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.boxplot(data=precursor_genes, x='index', y='expression', ax=ax)
ax.set_title('Peptide precursor gene expression')
ax.set_ylabel('Gene expression')
ax.set_xlabel('Structure')
plt.xticks(rotation=45)

if savefig:
    plt.savefig('figs/precursor_gene_expression.pdf')

# ax.set_xticklabels(['Cortex', 'Subcortex', 'Hypothalamus'])

# %%
# compare subcortex and cortex boxplots with hypothalamus as a dot
fig, ax = plt.subplots(2, 1, figsize=(20, 10))

# color first boxplot as in dimmed green and the second as in lightblue
sns.boxplot(data=subcortex_genes, ax=ax[0], color='lightgreen')
sns.boxplot(data=cortex_genes, ax=ax[1], color='lightblue')

for i in range(len(subcortex_genes.columns)):
    ax[0].scatter(i, hth_genes[subcortex_genes.columns[i]], color='red', label='Hypothalamus')
for i in range(len(cortex_genes.columns)):
    ax[1].scatter(i, hth_genes[cortex_genes.columns[i]], color='red', label='Hypothalamus')


ax[0].set_ylabel('Gene expression')
ax[1].set_ylabel('Gene expression')

ax[0].set_xticklabels(subcortex_genes.columns, rotation=90)
ax[1].set_xticklabels(cortex_genes.columns, rotation=90)

# increase whitespace between subplots
plt.subplots_adjust(hspace=0.3)

# legend with boxplot color and hypothalamus dot
# include hypothalamus as red dot the legend
ax[0].legend(['Subcortex', 'Hypothalamus'], frameon=False)
ax[1].legend(['Cortex', 'Hypothalamus'], frameon=False)

# change hypothalamus legend to red dot
ax[0].get_legend().legend_handles[1].set_color('red')
ax[1].get_legend().legend_handles[1].set_color('red')


# %%
# average gene expression in subcortex and cortex
subcortex_avg = subcortex_genes.mean(axis=0)
cortex_avg = cortex_genes.mean(axis=0)
hth_avg = hth_genes

# plot average gene expression in subcortex and cortex and compare with hypothalamus distribution
fig, ax = plt.subplots(1, 1, figsize=(5, 4))

# color first boxplot as in lightblue and the second as in lightgreen and third as lightcoral
sns.boxplot(data=[cortex_avg, subcortex_avg, hth_genes], ax=ax, palette=['lightblue', 'lightgreen', 'lightcoral'])
ax.set_title('Average prepropetide gene expression per tissue')
ax.set_ylabel('Gene expression')
ax.set_xticklabels(['Cortex', 'Subcortex', 'Hypothalamus'])

# %%
# do significance test between subcortex and cortex average and hypothalamus
from scipy.stats import ttest_ind

# test between subcortex and cortex
tstat, pval = ttest_ind(hth_genes, cortex_avg, alternative='greater', equal_var=False)
print(f'Hypothalamus vs Cortex: t-statistic = {tstat}, p-value = {pval}')

# test between subcortex and hypothalamus
tstat, pval = ttest_ind(hth_genes, subcortex_avg, alternative='greater', equal_var=False)
print(f'Hypothalamus vs Subcortex: t-statistic = {tstat}, p-value = {pval}')

# %%
