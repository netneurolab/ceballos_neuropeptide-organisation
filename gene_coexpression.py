# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from netneurotools import plotting, modularity
# %%
gene_list = pd.read_csv('data/gene_list.csv')
receptor_labels = gene_list[gene_list['Description'].str.contains('receptor')]
receptor_labels = receptor_labels.append({'Gene': 'SORT1'}, ignore_index=True)
receptor_labels = receptor_labels['Gene'].values

# load gene expression data
genes = pd.read_csv('data/peptide_genes_ahba_Schaefer400.csv', index_col=0)
genes = genes[genes.columns[genes.columns.isin(gene_list['Gene'])]]
receptor_genes = genes[genes.columns[genes.columns.isin(receptor_labels)]]

# %%
gene_coexpression = genes.corr(method='spearman').values
np.fill_diagonal(gene_coexpression, 0)
gene_coexpression[gene_coexpression < 0] = 0

ci, Q, zrand = modularity.consensus_modularity(gene_coexpression, gamma=1, seed=1234)
plotting.plot_mod_heatmap(gene_coexpression, ci, cmap='coolwarm')
plt.title('Gene coexpression')

# %%
receptor_coexpression = receptor_genes.corr(method='spearman').values
np.fill_diagonal(receptor_coexpression, 0)
receptor_coexpression[receptor_coexpression < 0] = 0

ci, Q, zrand = modularity.consensus_modularity(receptor_coexpression, gamma=0.8, seed=1234)
plotting.plot_mod_heatmap(receptor_coexpression, ci, cmap='coolwarm')
plt.title('Receptor gene coexpression')

# %%
# load receptor_hth_coupling
receptor_df = pd.read_csv('results/receptor_hth_coupling.csv')

# add column with ci values
receptor_df['ci'] = ci

# plot violinplot of HTH_corr split by ci
fig, ax = plt.subplots(figsize=(5,6), dpi=150)
sns.violinplot(data=receptor_df, x='ci', y='HTH_corr', ax=ax)
plt.title('Receptor HTH coupling')
plt.xlabel('Module')

# %%
# check how the number of communities changes with gamma
communities = []
for g in np.arange(0, 2, 0.1):
    ci, Q, zrand = modularity.consensus_modularity(receptor_coexpression, gamma=g, seed=1234)
    n_communities = len(np.unique(ci))
    communities.append(n_communities)

sns.lineplot(x=np.arange(0, 2, 0.1), y=communities)
plt.title('Stability of communities')
plt.xlabel('gamma')
plt.ylabel('Number of communities')
plt.xticks(np.arange(0, 2.2, 0.2))
