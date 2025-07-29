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
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0).iloc[:-1]

# load receptor names from data/annotations
nt_densities = pd.read_csv('data/annotations/nt_receptor_densities_Schaefer400_TianS4_HTH.csv', index_col=0)

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
dom_rel = (dom_total / dom_total.sum(axis=0)) * 100

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOT DOMINANCE HEATMAP
###############################################################################
# define names for plotting
peptide_names = receptor_genes.columns.values
nt_names = nt_densities.columns.values

# create domaince dataframe 
df = pd.DataFrame(dom_total.T, columns=peptide_names, index=nt_names)

# order peptide receptors by sum of dominance, i.e. R squared
receptors_by_dominance = df.sum(axis=0).sort_values(ascending=True).index
pep_idx = [df.columns.get_loc(_) for _ in receptors_by_dominance]
df = df[receptors_by_dominance]

# order nt receptors by ionotropic and metabotropic
# load nt classes
nt_classes = pd.read_csv('data/annotations/nt_receptor_classes.csv', index_col=0)
mi = nt_classes['Metab/Iono'].loc[nt_names]

# split df into two dfs and concatenate
idf = df.loc[mi[mi == 'ionotropic'].index] #type:ignore
mdf = df.loc[mi[mi == 'metabotropic'].index] #type:ignore
df = pd.concat((idf, mdf), axis=0)
nt_idx = [np.where(nt_names == _)[0][0] for _ in df.index]

# create df with relative dominance and use df order
plot_df = pd.DataFrame(dom_rel[pep_idx], index=receptors_by_dominance, 
                       columns=nt_names)
# plot_df = plot_df.iloc[:, nt_idx] # uncomment to order plot by ionotropic/metabotropic

# prepare figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 10), gridspec_kw={'width_ratios': [1, 3]}, 
                               dpi=200)

# barplot of total dominance
orange_color = bipolar[1]
sns.barplot(x=df.sum(axis=0), y=receptors_by_dominance, ax=ax1, 
            color=orange_color)
ax1.set_xlabel('R$^2$', fontsize=14)
ax1.set_xlim(0, 1)
ax1.set_yticks([])
ax1.invert_xaxis()
sns.despine(ax=ax1, left=True)

# heatmap with each neurotransmitter as a column
max_val = plot_df.max().round(0).max()
sns.heatmap(plot_df, cbar_kws={'shrink': 0.5}, 
            ax=ax2, cmap=divergent_green_orange(), center=0, vmin=0, vmax=max_val,
            linecolor='white', linewidths=0.5)

cbar = ax2.collections[0].colorbar
cbar.set_label('Relative contribution (%)', size=14)
ax2.set_xticklabels(plot_df.columns, rotation=90, horizontalalignment='center')
plt.tight_layout()

if savefig:
    plt.savefig('figs/colocalization_nt_peptides.pdf')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                   PLOT DISTRIBUTION OF R^2
###############################################################################
# locate values for rows ['OPRK1', 'OPRM1'] and columns ['KOR', 'MOR']
r2_df = df.T[sorted(df.T.columns)]
kappa_opioid = r2_df.loc['OPRK1', 'KOR']
mu_opioid = r2_df.loc['OPRM1', 'MOR']
colocalization_distribution = r2_df.values.flatten()

# plot kde of colocalization distribution and mark kappa and mu opioid receptors
fig, ax = plt.subplots(figsize=(5, 3), dpi=200)
ax.axvline(kappa_opioid, color='blue', linestyle='--', label='OPRK1-KOR')
ax.axvline(mu_opioid, color='orange', linestyle='--', label='OPRM1-MOR')
sns.kdeplot(colocalization_distribution, color='grey', label='Other pairs', ax=ax)
# add legend
ax.legend(bbox_to_anchor=(1, 1), frameon=False)
plt.xlabel('Colocalization [R$^2$]')
sns.despine()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          PLOT BY M/I CATEGORIES
###############################################################################

# load receptor classes
receptor_classes = pd.read_csv('data/annotations/nt_receptor_classes.csv')

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
da_null_fn = 'results/da_nt_nulls_peptides_total_dominance_100.npy'
if os.path.exists(da_null_fn):
    dom_total_nulls = np.load(da_null_fn)
else:
    dom_total_nulls = []
    for i in range(n_nulls):
        dom_list = []
        for name in receptor_genes.columns:
            # standardize data
            X = sstats.zscore(spatial_nulls[..., i].T, ddof=1)
            y = sstats.zscore(receptor_genes[name].values, ddof=1)
            
            # dominance analysis
            model_metrics, model_r_sq = get_dominance_stats(X, y, n_jobs=32)
            dom_list.append((model_metrics, model_r_sq))
        dom_total_null = [_[0]["total_dominance"] for _ in dom_list]
        dom_total_null = np.array(dom_total_null)
        dom_total_nulls.append(dom_total_null)
    dom_total_nulls = np.array(dom_total_nulls)
    np.save(da_null_fn, dom_total_nulls)


# %%
# sumarize into average dominance
emp = dom_total.sum(axis=1).mean()
nulls = dom_total_nulls.sum(axis=2).mean(axis=1).mean()

print(f'Neuropeptide receptor variance explained by neurotransmitter receptors on average: {emp:.3f}')
print(f'Average variance explained by spatial nulls: {nulls:.3f}')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                     VALIDATION WITHOUT OPRK1 AND OPRM1
###############################################################################
# drop OPRK1 and OPRM1 to test whether result remains significant
validation_df = dom_rel_df.drop(['OPRK1', 'OPRM1'], axis=1)
validation_df = validation_df.groupby('Metab/Iono').sum()

metab = validation_df.loc['metabotropic']
iono = validation_df.loc['ionotropic']
val_t, val_p = sstats.ttest_ind(metab, iono)

print(f'Original: t={t:.2f} | p={p:.2f}')
print(f'Validation: t={val_t:.2f} | p={val_p:.2f}')
