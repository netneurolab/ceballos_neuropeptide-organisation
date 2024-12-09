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

