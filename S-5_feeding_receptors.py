# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore, spearmanr
from pyls import behavioral_pls
from nilearn.maskers import NiftiLabelsMasker
from plot_utils import split_barplot

savefig = False
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                              LOAD DATA
###############################################################################
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)

# peptides of interest
peptides = ['APLNR', 'CALCRL','CCKBR','GALR1',
            'GLP2R','GRPR', 'MCHR2','NPR2',
            'NPY1R', 'NPY5R','NTSR1','OPRK1',
            'OPRM1', 'TACR2']

# regions of interest
rois = ['HTH', 'AMY', 'HIP', 'NAc', 'THA-VP']

# select rows that contain the rois
receptor_genes_filt = receptor_genes[receptor_genes.index.str.contains('|'.join(rois))]
feeding_receptors = receptor_genes_filt[peptides]
rois = receptor_genes_filt.index

# locate the indices of the rois in the original dataframe
gene_idx = [receptor_genes.columns.get_loc(gene_name) for gene_name in peptides]
roi_idx = [receptor_genes.index.get_loc(roi_name) for roi_name in rois]

masker = NiftiLabelsMasker('data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz')
eating = masker.fit_transform('data/eating_term.nii.gz')[0, roi_idx]
food = masker.fit_transform('data/food_term.nii.gz')[0, roi_idx]

# load gene nulls for receptor genes
nulls = np.load('data/gene_null_sets_Schaefer400_TianS4_HTH.npy')[:, roi_idx]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                              PLS
###############################################################################

# Define X and Y
X = zscore(np.c_[eating, food], ddof=1)
Y = zscore(receptor_genes_filt, ddof=1)
nlv = len(X.T) if len(X.T) < len(Y.T) else len(Y.T) # number of latent variables
lv = 0  # interested only in first latent variable
nperm = len(nulls)  # number of permutations

# behavioral PLS with gene nulls for Y
pls_result = behavioral_pls(X, Y, n_boot=nperm, n_perm=nperm, rotate=True, permsamples=nulls,
                            permindices=False, test_split=0, seed=0)

# check significance
cv = pls_result["singvals"]**2 / np.sum(pls_result["singvals"]**2)
null_singvals = pls_result['permres']['perm_singval']
cv_spins = null_singvals**2 / sum(null_singvals**2)
p = (1+sum(null_singvals[lv, :] > pls_result["singvals"][lv]))/(1+nperm)

plt.figure(figsize=(5, 5), dpi=150)
sns.boxplot(cv_spins.T * 100, color='lightgreen', fliersize=0, zorder=1, width=0.3)
sns.scatterplot(x=range(nlv), y=cv*100, s=50, color='orange', linewidth=1, edgecolor='black')
plt.ylabel("Covariance accounted for [%]")
plt.xlabel("Latent variable")
plt.title(f'LV{lv+1} accounts for {cv[lv]*100:.2f}% covariance | p = {p:.4f}');

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                              RECEPTOR LOADINGS
###############################################################################

receptor_names = receptor_genes.columns

# error bars are ci from bootstrapping
err = (pls_result["bootres"]["y_loadings_ci"][:, lv, 1]
      - pls_result["bootres"]["y_loadings_ci"][:, lv, 0]) / 2
receptors_df = pd.DataFrame({'receptor': receptor_names, 'loading': pls_result["y_loadings"][:, lv],
                        'err': err})
receptors_df['sign'] = np.sign(receptors_df['loading'])    
receptors_df = receptors_df.sort_values('loading', ascending=False)

fig, axes = split_barplot(receptors_df, x='loading', y='receptor', equal_scale=True,
                          figsize=(8, 5), dpi=200)
if savefig:
    plt.savefig('figs/feeding_receptor_loadings.pdf')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                              PLOT
###############################################################################

# plot correlation to eating and feeding in 4x4 grid
fig, ax = plt.subplots(4, 4, figsize=(10, 10), dpi=200)
for i, peptide in enumerate(peptides):
    gene = receptor_genes_filt[peptide].values
    eat_r, p = spearmanr(gene, eating)
    food_r, p = spearmanr(gene, food)
    
    sns.regplot(x=gene, y=eating, ax=ax.flatten()[i], color='lightblue', ci=None,
                scatter_kws={'s': 2})
    sns.regplot(x=gene, y=food, ax=ax.flatten()[i], color='orange', ci=None,
                scatter_kws={'s': 2})
    ax.flatten()[i].set_title(f'{peptide}\neating_r={eat_r:.2f} | food_r={food_r:.2f}', fontsize=8)
    sns.despine(ax=ax.flatten()[i])
    
# do not show last two plots
ax.flatten()[-1].axis('off')
ax.flatten()[-2].axis('off')
fig.suptitle(f'Correlation to eating and food terms')
# add legend for figure
fig.legend(handles = [plt.Line2D([0], [0], color='lightblue', lw=2),
                      plt.Line2D([0], [0], color='orange', lw=2)],
           labels=['eating', 'food'], bbox_to_anchor=(1.01, 1), frameon=False)
                                                                                   
plt.tight_layout()
# %%
# fronto-opercular-insular cortex
rois = ['FrOperIns']
# select rows that contain the rois
receptor_genes_filt = receptor_genes[receptor_genes.index.str.contains('|'.join(rois))]

# locate the indices of the rois in the original dataframe
idx = [receptor_genes.index.get_loc(name) for name in receptor_genes_filt.index]

masker = NiftiLabelsMasker('data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz')
eating = masker.fit_transform('data/tmp/eating_term.nii.gz')[0, idx]
food = masker.fit_transform('data/tmp/food_term.nii.gz')[0, idx]
peptides = ['APLNR','CALCRL','CCKBR','GALR1',
            'GLP2R','GRPR','MCHR2','NPR2',
            'NPY1R','NPY2R','NPY5R','NTSR1',
            'OPRK1','TACR2']
# plot correlation to eating and feeding in 4x4 grid
fig, ax = plt.subplots(4, 4, figsize=(10, 10), dpi=200)
for i, peptide in enumerate(peptides):
    gene = receptor_genes_filt[peptide].values
    sns.regplot(x=gene, y=eating, ax=ax.flatten()[i], color='lightblue', ci=None, lowess=True,
                scatter_kws={'s': 1})
    sns.regplot(x=gene, y=food, ax=ax.flatten()[i], color='orange', ci=None, lowess=True,
                scatter_kws={'s': 1})
    ax.flatten()[i].set_title(peptide)
    sns.despine(ax=ax.flatten()[i])
# do not show last two plots
ax.flatten()[-1].axis('off')
ax.flatten()[-2].axis('off')
fig.suptitle('Correlation to eating and feeding terms')
# add legend for figure
fig.legend(handles = [plt.Line2D([0], [0], color='lightblue', lw=2),
                      plt.Line2D([0], [0], color='orange', lw=2)],
           labels=['eating', 'food'], bbox_to_anchor=(0.65, 0.25), frameon=False)
                                                                                   
plt.tight_layout()


# %%
# summarize previous plot with heatmap of correlation values
corrs = np.zeros((len(peptides), 2))
for i, peptide in enumerate(peptides):
    gene = receptor_genes_filt[peptide].values
    eat_r, p = spearmanr(gene, eating)
    food_r, p = spearmanr(gene, food)
    corrs[i] = [eat_r, food_r]

fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
sns.heatmap(corrs, ax=ax, cmap='coolwarm', xticklabels=['eating', 'food'], yticklabels=peptides, 
            square=True, center=0, cbar_kws={'label': 'Spearman r'})
plt.xticks(rotation=45)

# %%
# load gene nulls
nulls = np.load('data/gene_null_sets_Schaefer400_TianS4_HTH.npy')[:, roi_idx, gene_idx]

# compare gene expression to eating and food terms
results = np.array([spearmanr(null.T, eating) for null in nulls])
eat_nulls = results[:, 0]

results = np.array([spearmanr(null.T, food) for null in nulls])
food_nulls = results[:, 0]

# calculate p-values
eat_p = np.sum(eat_nulls > eat_r) / len(eat_nulls)

# %%
from neuromaps.stats import compare_images
# compare gene expression to eating and food terms
eating_results = np.array([compare_images(receptor_genes_filt[peptide].values, eating, 
                                          nulls=nulls[:,roi_idx,i].T, metric='spearmanr') 
                            for i, peptide in enumerate(peptides)])

food_results = np.array([compare_images(receptor_genes_filt[peptide].values, food, 
                                        nulls=nulls[:,roi_idx,i].T, metric='spearmanr')
                        for i, peptide in enumerate(peptides)])

# %%
# put correlations together
corrs = np.zeros((len(peptides), 2))
for i in range(len(peptides)):
    corrs[i] = [eating_results[i, 0], food_results[i, 0]]



# plot heatmap of results and grey out regions where p > 0.05
fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=200)
sns.heatmap(corrs, ax=ax, cmap='coolwarm', xticklabels=['eating', 'food'], yticklabels=peptides, 
            square=True, center=0, cbar_kws={'label': 'Spearman r'})

# grey out regions where p > 0.05
for i in range(len(peptides)):
    if eating_results[i, 1] > 0.05:
        ax.add_patch(plt.Rectangle((0, i), 2, 1, fill=True, color='grey'))

for i in range(len(peptides)):
    if food_results[i, 1] > 0.05:
        ax.add_patch(plt.Rectangle((0, i), 2, 1, fill=True, color='grey'))
        
# %%
# plot p values
p_values = np.zeros((len(peptides), 2))
for i in range(len(peptides)):
    p_values[i] = [eating_results[i, 1], food_results[i, 1]]

fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
sns.heatmap(p_values, ax=ax, cmap='coolwarm', xticklabels=['eating', 'food'], yticklabels=peptides, 
            annot=True, center=0.05, cbar_kws={'label': 'p-values'})
plt.xticks(rotation=45)
