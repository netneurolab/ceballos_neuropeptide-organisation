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
nt_densities = pd.read_csv('data/annotations/nt_receptor_densities.csv', index_col=None)

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
da_fn = 'results/da_nt_peptides_alldominance.npy'
if os.path.exists(da_fn):
    dom_global_list = np.load(da_fn, allow_pickle=True)
else:
    dom_global_list = []
    for name in receptor_genes.columns:
        X = sstats.zscore(nt_densities.values, ddof=1)
        y = sstats.zscore(receptor_genes[name], ddof=1)
        
        model_metrics, model_r_sq = get_dominance_stats(X, y, n_jobs=-1)
        dom_global_list.append((model_metrics, model_r_sq))
    np.save(da_fn, dom_global_list)

dom_global_total = [_[0]["total_dominance"] for _ in dom_global_list]
dom_global_total = np.array(dom_global_total)
dom_global_rel = dom_global_total / dom_global_total.sum(axis=1, keepdims=True)
del dom_global_list

# %%
# plot dominance in heatmap
# each row is a receptor, each column is a neurotransmitter
receptor_names = gene_list[gene_list['Gene'].isin(receptor_genes.columns)]['Description'].to_list()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 20), gridspec_kw={'width_ratios': [1, 3]})

sns.barplot(x=dom_global_total.sum(axis=1), y=receptor_names, ax=ax1, color='lightblue')
ax1.set_xlabel('$D_{total}$')

sns.heatmap(dom_global_rel, xticklabels=nt_densities.columns, yticklabels=receptor_genes.columns,
            cbar_kws={'label': 'Relative importance [%]', 'shrink': 0.5}, vmin=0,
            ax=ax2)
ax2.set_title('Relative importance of neurotransmitters to predict receptor')


# %% Load receptor classes
receptor_classes = pd.read_csv('data/annotations/receptor_classes.csv')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOT BY Gs/Gi/Gq CATEGORIES
###############################################################################
dom_global_rel_df = pd.DataFrame(dom_global_rel.T, index=nt_densities.columns, columns=receptor_genes.columns)
dom_global_rel_df['GPCR'] = dom_global_rel_df.index.map(receptor_classes.set_index('protein')['Gs/Gi/Gq'])
dom_global_rel_df = dom_global_rel_df.groupby('GPCR').mean()
dom_global_rel_df[dom_global_rel_df < 0] = 0

# do same for dom_global_total
dom_global_total_df = pd.DataFrame(dom_global_total.T, index=nt_densities.columns, columns=receptor_genes.columns)
dom_global_total_df['GPCR'] = dom_global_total_df.index.map(receptor_classes.set_index('protein')['Gs/Gi/Gq'])
dom_global_total_df = dom_global_total_df.groupby('GPCR').mean()

# plot in heatmap
plt.figure(figsize=(5, 10))
sns.heatmap(dom_global_rel_df.T,  cbar_kws={'label': 'Relative importance [%]', 'shrink': 0.5},
            yticklabels=receptor_genes.columns)
plt.tight_layout()

# %%
plt.figure(figsize=(5, 10))
sns.heatmap(dom_global_total_df.T,  cbar_kws={'label': 'Total dominance', 'shrink': 0.5},
            yticklabels=receptor_genes.columns)
plt.tight_layout()

# show distribution across GPCR classes
plt.figure(figsize=(4, 6))
sns.boxplot(data=dom_global_total_df.T)
plt.ylabel('Total dominance')
plt.tight_layout()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          PLOT BY M/I CATEGORIES
###############################################################################
# %% do same for Metab/Iono	
# create df for dom_global_rel
dom_global_rel_df = pd.DataFrame(dom_global_rel.T, index=nt_densities.columns, columns=receptor_genes.columns)
dom_global_rel_df['M/I'] = dom_global_rel_df.index.map(receptor_classes.set_index('protein')['Metab/Iono'])
dom_global_rel_df = dom_global_rel_df.groupby('M/I').mean()
dom_global_rel_df[dom_global_rel_df < 0] = 0

# do same for dom_global_total
dom_global_total_df = pd.DataFrame(dom_global_total.T, index=nt_densities.columns, columns=receptor_genes.columns)
dom_global_total_df['M/I'] = dom_global_total_df.index.map(receptor_classes.set_index('protein')['Metab/Iono'])
dom_global_total_df = dom_global_total_df.groupby('M/I').mean()

# plot in heatmap
plt.figure(figsize=(5, 10))
sns.heatmap(dom_global_rel_df.T,  cbar_kws={'label': 'Relative importance [%]', 'shrink': 0.5},
            yticklabels=receptor_genes.columns)
plt.tight_layout()

plt.figure(figsize=(4, 6))
sns.boxplot(dom_global_rel_df.T)
plt.ylabel('Relative importance [%]')
plt.tight_layout()

# %%
plt.figure(figsize=(5, 10))
sns.heatmap(dom_global_total_df.T,  cbar_kws={'label': 'Total dominance', 'shrink': 0.5},
            yticklabels=receptor_genes.columns)
plt.tight_layout()

# show distribution across GPCR classes
plt.figure(figsize=(4, 6))
sns.boxplot(data=dom_global_total_df.T)
plt.ylabel('Total dominance')
plt.tight_layout()
