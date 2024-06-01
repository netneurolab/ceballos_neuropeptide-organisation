# %%
import os
import numpy as np
import pandas as pd
import scipy.stats as sstats
import seaborn as sns
import matplotlib.pyplot as plt

# %%
bb_profiles = np.load("./data/annotations/bigbrain_intensity_profiles_Schaefer400.npy")

all_genes = pd.read_csv('data/abagen_genes_Schaefer2018_400.csv', index_col=0)
receptor_list = pd.read_csv('data/receptor_list.csv')['Gene']
overlapping_genes = all_genes.columns[all_genes.columns.isin(receptor_list)]
receptor_genes = all_genes[overlapping_genes]

n_genes = receptor_genes.shape[1]
n_profiles = bb_profiles.shape[1]

corr = np.zeros((n_genes, n_profiles))

# iterate through columns of receptor_genes and bb_profiles
for i in range(n_genes):
    for j in range(n_profiles):
        corr[i, j] = sstats.pearsonr(receptor_genes.iloc[:, i], bb_profiles[:, j])[0]

# %%
# plot correlation as lineplots where each line is a receptor
plt.figure(figsize=(12, 8))
for i in range(n_genes):
    plt.plot(corr[i], label=receptor_genes.columns[i])
# legend should be horizontally aligned
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), ncol=1)
plt.xlabel('Big Brain intensity profiles')
plt.ylabel('Pearson correlation')

# %%
g = sns.clustermap(corr, row_cluster=True, col_cluster=False, figsize=(10, 12), 
                   cmap='coolwarm', center=0, robust=True, 
                   cbar_kws={'label': 'Pearson correlation'})
# show only first and last xtick as 'white' and 'pial'
xticklabels = g.ax_heatmap.get_xticklabels()
g.ax_heatmap.set_xticks(np.arange(n_profiles) + 0.5)
g.ax_heatmap.set_xticklabels(['white'] + ['']*(n_profiles-2) + ['pial'])
g.ax_heatmap.set_xlabel('Big Brain intensity profiles')

# set yticklabels as receptor genes
# use reorderer dendrogram to reorder yticklabels
reordered_idx = g.dendrogram_row.reordered_ind
reordered_labels = receptor_genes.columns[reordered_idx]
g.ax_heatmap.set_yticks(np.arange(n_genes) + 0.5)
g.ax_heatmap.set_yticklabels(reordered_labels, rotation=0)

# %%
# """
# Plot heatmap of correlation between big brain intensity profiles and receptor genes
# Sort the heatmap by the similarity of the receptor genes correlation to the big brain intensity profiles
# """
# # calculate similarity between receptor genes and big brain intensity profiles
# similarity = np.corrcoef(corr)
# # cluster the similarity matrix
# g = sns.clustermap(similarity, row_cluster=True, col_cluster=True, figsize=(12, 12))
# g.ax_heatmap.set_xlabel('Receptor genes')
# g.ax_heatmap.set_ylabel('Receptor genes')
