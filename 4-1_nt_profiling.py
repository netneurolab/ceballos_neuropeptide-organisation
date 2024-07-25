# %%
import os
import numpy as np
import pandas as pd
import scipy.stats as sstats
import seaborn as sns
import matplotlib.pyplot as plt
from neuromaps.stats import compare_images
from netneurotools.stats import get_dominance_stats
from plot_utils import divergent_green_orange

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         LOAD DATA
###############################################################################
# load gene expression data
gene_list = pd.read_csv('data/gene_list.csv')
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0).iloc[:-1]

# load receptor names from data/annotations
nt_densities = pd.read_csv('data/annotations/nt_receptor_densities.csv', index_col=0).iloc[:-1]

# Load colors
palette = divergent_green_orange(n_colors=9, return_palette=True)
bipolar = [palette[1], palette[-2]]
spectral = [color for i, color in enumerate(sns.color_palette('Spectral')) if i in [1,2,4]]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         DOMINANCE ANALYSIS
###############################################################################
# check if already computed
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
dom_global_rel = dom_global_total / dom_global_total.sum(axis=0)
del dom_global_list


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOT DOMINANCE HEATMAP
###############################################################################
# each row is a receptor, each column is a neurotransmitter
peptide_names = receptor_genes.columns.values
nt_names = nt_densities.columns.values

# create dataframe ordered by sum of dominance
df = pd.DataFrame(dom_global_total.T, columns=peptide_names, index=nt_names)
receptors_by_dominance = df.sum(axis=0).sort_values(ascending=True).index
idx = [df.columns.get_loc(_) for _ in receptors_by_dominance]
df = df.loc[:, receptors_by_dominance]

# prepare figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 10), gridspec_kw={'width_ratios': [1, 3]}, 
                               dpi=200)

# barplot of total dominance
orange_color = bipolar[1]
sns.barplot(x=df.sum(axis=0), y=receptors_by_dominance, ax=ax1, 
            color=orange_color)
ax1.set_xlabel('Total dominance', fontsize=14)
ax1.set_xlim(0, 1)
ax1.set_yticks([])
ax1.invert_xaxis()
sns.despine(ax=ax1, left=True)

# heatmap with each neurotransmitter as a column
sns.heatmap(dom_global_rel[idx], xticklabels=df.index, 
            yticklabels=receptors_by_dominance, cbar_kws={'shrink': 0.5}, 
            ax=ax2, cmap=divergent_green_orange(), center=0, vmin=0,
            linecolor='white', linewidths=0.5)

cbar = ax2.collections[0].colorbar
cbar.set_label('Dominance', size=14)
ax2.set_yticklabels(receptors_by_dominance)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, horizontalalignment='center')
plt.tight_layout()
plt.savefig('figs/colocalization_nt_peptides.pdf')


# %% VALIDATION
# correlate gene and PET map for kappa-opioid receptor
# load maps
kappa_gene = receptor_genes['OPRK1']
kappa_pet = nt_densities['KOR']

# pet nulls
kappa_pet_nulls = np.load('data/kor_nulls_Schaefer400_TianS4.npy')
r, p_pet = compare_images(kappa_pet, kappa_gene, metric='spearmanr', nulls=kappa_pet_nulls)

# gene nulls
idx = receptor_genes.columns.get_loc('OPRK1')
kappa_gene_nulls = np.load('data/gene_null_sets_Schaefer400_TianS4.npy')[:, :, idx].T
r, p_gene = compare_images(kappa_gene, kappa_pet, metric='spearmanr', nulls=kappa_gene_nulls)

# comparison plot
plt.figure(figsize=(5, 5))
sns.regplot(x=kappa_gene, y=kappa_pet, color='grey', ci=None)
plt.title(f'r={r:.2f}\n P$_{{SMASH}}$={p_pet:.4f} | P$_{{gene}}$={p_gene:.4f}')
sns.despine()
plt.savefig('figs/kappa_opioid_receptor_comparison.pdf')


# correlate gene and PET map for mu-opioid receptor
# load maps
mu_gene = receptor_genes['OPRM1']
mu_pet = nt_densities['MOR']

# pet nulls
mu_pet_nulls = np.load('data/mor_nulls_Schaefer400_TianS4.npy')
r, p_pet = compare_images(mu_pet, mu_gene, metric='spearmanr', nulls=mu_pet_nulls)

# gene nulls
idx = receptor_genes.columns.get_loc('OPRM1')
mu_gene_nulls = np.load('data/gene_null_sets_Schaefer400_TianS4.npy')[:, :, idx].T
r, p_gene = compare_images(mu_gene, mu_pet, metric='spearmanr', nulls=mu_gene_nulls)

# comparison plot
plt.figure(figsize=(5, 5))
sns.regplot(x=mu_gene, y=mu_pet, color='grey', ci=None)
plt.title(f'r={r:.2f}\n P$_{{SMASH}}$={p_pet:.4f} | P$_{{gene}}$={p_gene:.4f}')
plt.xlim(0,1)
sns.despine()
plt.savefig('figs/mu_opioid_receptor_comparison.pdf')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          PLOT BY M/I CATEGORIES
###############################################################################

# load receptor classes
receptor_classes = pd.read_csv('data/annotations/receptor_classes.csv')

# discard transporters
receptor_classes = receptor_classes[~receptor_classes['Gs/Gi/Gq'].str.contains('trans')]
receptor_classes.set_index('protein', inplace=True)

# create df with M/I categories
dom_global_total_df = pd.DataFrame(dom_global_total.T, index=nt_densities.columns, columns=receptor_genes.columns)
m_count, i_count = receptor_classes['Metab/Iono'].value_counts().to_numpy()
categories = receptor_classes['Metab/Iono'].copy().to_frame()
categories['count'] = categories['Metab/Iono'].map({'metabotropic': m_count, 'ionotropic': i_count})

# correct for uneven size of categories column by category count
dom_global_total_df = dom_global_total_df.div(categories['count'], axis=0)

# turn into percentage contribution
dom_global_rel_df = dom_global_total_df.div(dom_global_total_df.sum(axis=0), axis=1) * 100

# sum contribution by category
dom_global_rel_df['Metab/Iono'] = dom_global_rel_df.index.map(categories['Metab/Iono'])
dom_category_rel_df = dom_global_rel_df.groupby('Metab/Iono').sum()

# plot in heatmap
plt.figure(figsize=(4, 10))
sns.heatmap(dom_category_rel_df.T,  cbar_kws={'label': 'Receptor colocalization [%]', 'shrink': 0.5},
            yticklabels=list(receptor_genes.columns), cmap=divergent_green_orange(), 
            linecolor='white', linewidths=0.5)
plt.tight_layout()

# test whether difference is significant
metab = dom_category_rel_df.loc['metabotropic']
iono = dom_category_rel_df.loc['ionotropic']
t, p = sstats.ttest_ind(metab, iono)

plt.figure(figsize=(3, 5))
sns.boxplot(dom_category_rel_df.T, palette=bipolar)
plt.ylabel('Average colocalization [%]')
plt.tight_layout()
sns.despine(trim=True)
plt.title(f't={t:.2f} | p={p:.2f}')

plt.savefig('figs/ionotropic_metabotropic_receptors.pdf')


# %% VALIDATION
# drop OPRK1 and OPRM1 to test whether result remains significant
validation_df = dom_global_rel_df.drop(['OPRK1', 'OPRM1'], axis=1)

metab = validation_df.loc['metabotropic']
iono = validation_df.loc['ionotropic']
sstats.ttest_ind(metab, iono)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOT BY Gs/Gi/Gq CATEGORIES
###############################################################################
# plot distribution of GPCR categories
receptor_classes['Gs/Gi/Gq'].value_counts().plot(kind='bar')

# do same for dom_global_total
dom_global_total_df = pd.DataFrame(dom_global_total.T, index=nt_densities.columns, columns=receptor_genes.columns)
dom_global_total_df['GPCR'] = dom_global_total_df.index.map(receptor_classes['Gs/Gi/Gq'])
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
sns.boxplot(dom_global_rel_df.T, color=orange_color)
plt.ylabel('Receptor colocalization [%]')
plt.tight_layout()
sns.despine(trim=True)

