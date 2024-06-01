# %%
import os
import numpy as np
import pandas as pd
import scipy.stats as sstats
import seaborn as sns
import matplotlib.pyplot as plt
from netneurotools.stats import get_dominance_stats

# %%
# load receptor names from data/annotations
meg_bands = np.load('data/annotations/meg_bands_Schaefer400.npy')

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
da_fn = 'results/da_meg_peptides_alldominance.npy'
if os.path.exists(da_fn):
    dom_global_list = np.load(da_fn, allow_pickle=True)
else:
    dom_global_list = []
    for name in receptor_genes.columns:
        X = sstats.zscore(meg_bands, ddof=1)
        y = sstats.zscore(receptor_genes[name], ddof=1)
        
        model_metrics, model_r_sq = get_dominance_stats(X, y, n_jobs=-1)
        dom_global_list.append((model_metrics, model_r_sq))
    np.save(da_fn, dom_global_list)

dom_global_total = [_[0]["total_dominance"] for _ in dom_global_list]
dom_global_total = np.array(dom_global_total)

# %%
# plot dominance in heatmap
# each row is a receptor, each column is an MEG bands
receptor_names = gene_list[gene_list['Gene'].isin(receptor_genes.columns)]['Description'].to_list()
band_names = ['delta', 'theta', 'alpha', 'beta', 'low-gamma', 'high-gamma']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 20), gridspec_kw={'width_ratios': [1, 2]})

sns.barplot(x=dom_global_total.sum(axis=1), y=receptor_names, ax=ax1, color='lightblue')
ax1.set_xlabel('$D_{total}$')

sns.heatmap(dom_global_total, xticklabels=band_names, yticklabels=receptor_genes.columns,
            cbar_kws={'label': 'Total dominance', 'shrink': 0.5}, ax=ax2)
ax2.set_title('Relative importance of MEG bands to predict receptor')
# %%
