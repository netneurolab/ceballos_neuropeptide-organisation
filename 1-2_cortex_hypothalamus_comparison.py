# %%
import numpy as np
import abagen
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# load the genes
all_genes = pd.read_csv('./data/abagen_genes_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)

# load atlas info
atlas_info = pd.read_csv('./data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv')

# load list of relevant genes
gene_list = pd.read_csv('data/gene_list.csv')

# divide into subcortex and cortex
subcortex = atlas_info[atlas_info['structure'] == 'subcortex'].iloc[:-1] # remove hypothalamus
cortex = atlas_info[atlas_info['structure'] == 'cortex']

# find overlapping genes in genes columns and gene_list
overlapping_genes = gene_list[gene_list['Gene'].isin(all_genes.columns)]['Gene']

# select overlapping genes from genes and gene_list
genes = all_genes[overlapping_genes]
gene_list = gene_list[gene_list['Gene'].isin(overlapping_genes)]

receptor_labels = gene_list[gene_list['Description'].str.contains('receptor')]
receptor_labels = receptor_labels.append({'Gene': 'SORT1'}, ignore_index=True)
peptide_labels = gene_list[~gene_list['Gene'].isin(receptor_labels['Gene'])]
receptor_genes = all_genes[receptor_labels['Gene']]
peptide_genes = all_genes[peptide_labels['Gene']]

# %%
# split genes dataframe by cortex and subcortex
subcortex_genes = peptide_genes.loc[subcortex['name']]
cortex_genes = peptide_genes.loc[cortex['name']]
hth_genes = peptide_genes.loc['HTH']

# compare subcortex and cortex boxplots with hypothalamus as a dot
fig, ax = plt.subplots(2, 1, figsize=(20, 10))

# color first boxplot as in dimmed green and the second as in lightblue
sns.boxplot(data=subcortex_genes, ax=ax[0], color='lightgreen')
sns.boxplot(data=cortex_genes, ax=ax[1], color='lightblue')

for i in range(len(subcortex_genes.columns)):
    ax[0].scatter(i, hth_genes[subcortex_genes.columns[i]], color='red', label='Hypothalamus')
for i in range(len(cortex_genes.columns)):
    ax[1].scatter(i, hth_genes[cortex_genes.columns[i]], color='red', label='Hypothalamus')
ax[0].set_title('Prepropetide gene expression')

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
