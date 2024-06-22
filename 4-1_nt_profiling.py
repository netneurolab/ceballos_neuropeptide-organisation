# %%
import os
import numpy as np
import pandas as pd
import scipy.stats as sstats
import seaborn as sns
import matplotlib.pyplot as plt
from netneurotools.stats import get_dominance_stats
from plot_utils import divergent_green_orange

# %%
# load gene expression data
gene_list = pd.read_csv('data/gene_list.csv')
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0).iloc[:-1]

# load receptor names from data/annotations
nt_densities = pd.read_csv('data/annotations/nt_receptor_densities.csv', index_col=0).iloc[:-1]
# %%
da_fn = 'results/da_nt_peptides_alldominance.npy'
if os.path.exists(da_fn):
    dom_global_list = np.load(da_fn, allow_pickle=True)
else:
    dom_global_list = []
    for name in receptor_genes.columns:
        X = sstats.zscore(nt_densities.values, ddof=1)
        y = sstats.zscore(receptor_genes[name].values, ddof=1)
        
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

# prepare figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 20), gridspec_kw={'width_ratios': [1, 3]})

# barplot of total dominance
sns.barplot(x=dom_global_total.sum(axis=1), y=receptor_names, ax=ax1, 
            color=lightorange, alpha=0.6)
ax1.set_xlabel('Total dominance', fontsize=14)
sns.despine(ax=ax1)

# heatmap with each neurotransmitter as a column
sns.heatmap(dom_global_total, xticklabels=list(nt_densities.columns), 
            yticklabels=list(receptor_genes.columns), cbar_kws={'shrink': 0.5}, 
            ax=ax2, cmap=divergent_green_orange(), center=0, vmin=0,
            linecolor='white', linewidths=0.5)

cbar = ax2.collections[0].colorbar
cbar.set_label('Dominance', size=14)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, horizontalalignment='center');

# %% Load receptor classes
receptor_classes = pd.read_csv('data/annotations/receptor_classes.csv')
# discard transporters
receptor_classes = receptor_classes[~receptor_classes['Gs/Gi/Gq'].str.contains('trans')]

# plot frequency of receptor classes Gi/Gs/Gq
receptor_classes['Gs/Gi/Gq'].value_counts().plot(kind='bar')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOT BY Gs/Gi/Gq CATEGORIES
###############################################################################
# do same for dom_global_total
dom_global_total_df = pd.DataFrame(dom_global_total.T, index=nt_densities.columns, columns=receptor_genes.columns)
dom_global_total_df['GPCR'] = dom_global_total_df.index.map(receptor_classes.set_index('protein')['Gs/Gi/Gq'])
dom_global_total_df = dom_global_total_df.groupby('GPCR').sum()

# normalize dom_global_total_df by number of GPCRs in each class
dom_global_total_df = dom_global_total_df \
                      .div(receptor_classes['Gs/Gi/Gq'].value_counts(), axis=0)

# per column normalization
dom_global_rel_df = dom_global_total_df.div(dom_global_total_df.sum(axis=0), axis=1)

# plot in heatmap
plt.figure(figsize=(5, 10))
sns.heatmap(dom_global_rel_df.T,  cbar_kws={'label': 'Receptor colocalization [%]', 'shrink': 0.5},
            yticklabels=list(receptor_genes.columns), cmap=divergent_green_orange(), center=0, vmin=0,
            linecolor='white', linewidths=0.5)
plt.tight_layout()

# show promiscuity in G protein coupling
plt.figure(figsize=(4, 6))
sns.boxplot(dom_global_rel_df.T, color='orange')
plt.ylabel('Receptor colocalization [%]')
plt.tight_layout()
sns.despine(trim=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          PLOT BY M/I CATEGORIES
###############################################################################

# create df with M/I categories
dom_global_total_df = pd.DataFrame(dom_global_total.T, index=nt_densities.columns, columns=receptor_genes.columns)
dom_global_total_df['M/I'] = dom_global_total_df.index.map(receptor_classes.set_index('protein')['Metab/Iono'])
dom_global_total_df = dom_global_total_df.groupby('M/I').sum()

# average per category
dom_global_total_df = dom_global_total_df \
                      .div(receptor_classes['Metab/Iono'].value_counts(), axis=0)

# per column normalization
dom_global_rel_df = dom_global_total_df.div(dom_global_total_df.sum(axis=0), axis=1)

# plot in heatmap
plt.figure(figsize=(4, 10))
sns.heatmap(dom_global_rel_df.T,  cbar_kws={'label': 'Receptor colocalization [%]', 'shrink': 0.5},
            yticklabels=list(receptor_genes.columns), cmap=divergent_green_orange(), 
            center=0.2, vmin=0.2, vmax=0.8, linecolor='white', linewidths=0.5)
plt.tight_layout()

# test whether difference is significant
metab = dom_global_rel_df.loc['metabotropic']
iono = dom_global_rel_df.loc['ionotropic']
t, p = sstats.ttest_ind(metab, iono)

plt.figure(figsize=(3, 5))
sns.boxplot(dom_global_rel_df.T, color='orange')
plt.ylabel('Receptor colocalization [%]')
plt.tight_layout()
sns.despine(trim=True)
plt.title(f't={t:.2f} | p={p:.2f}')
