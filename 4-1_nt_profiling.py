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

savefig = False

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         LOAD DATA
###############################################################################
# load gene expression data
gene_list = pd.read_csv('data/gene_list.csv')
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0).iloc[:-1]

# load receptor names from data/annotations
nt_densities = pd.read_csv('data/annotations/nt_receptor_densities.csv', index_col=0)

# Load colors
palette = divergent_green_orange(n_colors=9, return_palette=True)
bipolar = [palette[1], palette[-2]]
spectral = [color for i, color in enumerate(sns.color_palette('Spectral')) if i in [1,2,4]]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         DOMINANCE ANALYSIS
###############################################################################
# check if already computed
da_fn = 'results/da_nt_peptides_total_dominance.npy'
if os.path.exists(da_fn):
    dom_total = np.load(da_fn)
else:
    dom_list = []
    for name in receptor_genes.columns:
        # standardize data
        X = sstats.zscore(nt_densities.values, ddof=1)
        y = sstats.zscore(receptor_genes[name].values, ddof=1)
        
        # dominance analysis
        model_metrics, model_r_sq = get_dominance_stats(X, y, n_jobs=-1)
        dom_list.append((model_metrics, model_r_sq))
    dom_total = [_[0]["total_dominance"] for _ in dom_list]
    dom_total = np.array(dom_total)
    del dom_list
    np.save(da_fn, dom_total)

# turn into relative dominance
dom_rel = dom_total / dom_total.sum(axis=0) * 100


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOT DOMINANCE HEATMAP
###############################################################################
# define names for plotting
peptide_names = receptor_genes.columns.values
nt_names = nt_densities.columns.values

# create dataframe ordered by sum of dominance
df = pd.DataFrame(dom_total.T, columns=peptide_names, index=nt_names)
receptors_by_dominance = df.sum(axis=0).sort_values(ascending=True).index
idx = [df.columns.get_loc(_) for _ in receptors_by_dominance]
df = df[receptors_by_dominance]

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
sns.heatmap(dom_rel[idx], xticklabels=df.index, 
            yticklabels=receptors_by_dominance, cbar_kws={'shrink': 0.5}, 
            ax=ax2, cmap=divergent_green_orange(), center=0, vmin=0,
            linecolor='white', linewidths=0.5)

cbar = ax2.collections[0].colorbar
cbar.set_label('Dominance', size=14)
ax2.set_yticklabels(receptors_by_dominance)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, horizontalalignment='center')
plt.tight_layout()

if savefig:
    plt.savefig('figs/colocalization_nt_peptides.pdf')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          PLOT BY M/I CATEGORIES
###############################################################################

# load receptor classes
receptor_classes = pd.read_csv('data/annotations/receptor_classes.csv')

# discard transporters
receptor_classes = receptor_classes[~receptor_classes['Metab/Iono'].str.contains('trans')]
receptor_classes.set_index('protein', inplace=True)

# create df with M/I categories
dom_total_df = pd.DataFrame(dom_total.T, index=nt_densities.columns, columns=receptor_genes.columns)

# correct for uneven size of categories
m_count, i_count = receptor_classes['Metab/Iono'].value_counts().to_numpy()
categories = receptor_classes['Metab/Iono'].copy().to_frame()
categories['count'] = categories['Metab/Iono'].map({'metabotropic': m_count, 'ionotropic': i_count})
dom_total_df = dom_total_df.div(categories['count'], axis=0)

# turn into percentage contribution
dom_rel_df = dom_total_df.div(dom_total_df.sum(axis=0), axis=1) * 100

# sum contribution by category
dom_rel_df['Metab/Iono'] = dom_rel_df.index.map(categories['Metab/Iono'])
dom_category_rel_df = dom_rel_df.groupby('Metab/Iono').sum()

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

if savefig:
    plt.savefig('figs/ionotropic_metabotropic_receptors.pdf')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                       GENE AND PET MAP CORRESPONDENCE
###############################################################################
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

if savefig:
    plt.savefig('figs/mu_opioid_receptor_comparison.pdf')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                 VALIDATION DOMINANCE WITH SPATIAL NULLS
###############################################################################
# load spatial nulls
spatial_nulls = np.load('data/nt_nulls_Schaefer400_TianS4.npy')
n_nulls = spatial_nulls.shape[-1]

# redo dominance analysis with spatial nulls
# check if already computed
da_null_fn = 'results/da_nt_nulls_peptides_total_dominance.npy'
if os.path.exists(da_null_fn):
    dom_total_nulls = np.load(da_null_fn, allow_pickle=True)
else:
    dom_total_nulls = []
    for i in range(n_nulls):
        dom_list = []
        for name in receptor_genes.columns:
            # standardize data
            X = sstats.zscore(spatial_nulls[..., i].T, ddof=1)
            y = sstats.zscore(receptor_genes[name].values, ddof=1)
            
            # dominance analysis
            model_metrics, model_r_sq = get_dominance_stats(X, y, n_jobs=-1)
            dom_list.append((model_metrics, model_r_sq))
        dom_total_null = [_[0]["total_dominance"] for _ in dom_list]
        dom_total_null = np.array(dom_total_null)
        dom_total_nulls.append(dom_total)
    dom_total_nulls = np.array(dom_total_nulls)
    np.save(da_null_fn, dom_total_nulls)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                     VALIDATION WITHOUT OPRK1 AND OPRM1
###############################################################################
# drop OPRK1 and OPRM1 to test whether result remains significant
validation_df = dom_rel_df.drop(['OPRK1', 'OPRM1'], axis=1)

metab = validation_df.loc['metabotropic']
iono = validation_df.loc['ionotropic']
val_t, val_p = sstats.ttest_ind(metab, iono)

print(f'Original: t={t:.2f} | p={p:.2f}')
print(f'Validation: t={val_t:.2f} | p={val_p:.2f}')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOT BY Gs/Gi/Gq CATEGORIES
###############################################################################
# plot distribution of GPCR categories
receptor_classes['Gs/Gi/Gq'].value_counts().plot(kind='bar')

# do same for dom_total
dom_total_df = pd.DataFrame(dom_total.T, index=nt_densities.columns, columns=receptor_genes.columns)
dom_total_df['GPCR'] = dom_total_df.index.map(receptor_classes['Gs/Gi/Gq'])
dom_total_df = dom_total_df.groupby('GPCR').sum()

# normalize dom_total_df by number of GPCRs in each class
dom_total_df = dom_total_df \
                      .div(receptor_classes['Gs/Gi/Gq'].value_counts(), axis=0)

# per column normalization
dom_rel_df = dom_total_df.div(dom_total_df.sum(axis=0), axis=1)

# plot in heatmap
plt.figure(figsize=(5, 10))
sns.heatmap(dom_rel_df.T,  cbar_kws={'label': 'Receptor colocalization [%]', 'shrink': 0.5},
            yticklabels=list(receptor_genes.columns), cmap=divergent_green_orange(), center=0, vmin=0,
            linecolor='white', linewidths=0.5)
plt.tight_layout()

# show promiscuity in G protein coupling
plt.figure(figsize=(4, 6))
sns.boxplot(dom_rel_df.T, color=orange_color)
plt.ylabel('Receptor colocalization [%]')
plt.tight_layout()
sns.despine(trim=True)

