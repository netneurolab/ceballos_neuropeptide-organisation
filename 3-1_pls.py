# %%
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, spearmanr
from pyls import behavioral_pls
from scipy.spatial.distance import pdist, squareform
from netneurotools.utils import get_centroids
from utils import gene_null_set
from plot_utils import divergent_green_orange, split_barplot

savefigs = False

# %% LOAD DATA
# Load neurosynth
ns = pd.read_csv('./data/neurosynth_Schaefer400_TianS4.csv', index_col=0)

# Load gene receptor data
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)

# Load colors
palette = divergent_green_orange(n_colors=9, return_palette=True)
colors = [palette[1], palette[-2]]

# %% Gene null sets
nperm = 10000

# Load gene nulls if existent otherwise generate
if os.path.exists('data/gene_null_sets_Schaefer400_TianS4_HTH.npy'):
    nulls = np.load('data/gene_null_sets_Schaefer400_TianS4_HTH.npy')
else:
    # load all genes from abagen and get non-peptide genes
    all_genes = pd.read_csv('data/abagen_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
    peptide_list = pd.read_csv('data/gene_list.csv')['Gene']
    non_peptide_genes = all_genes.T[~all_genes.columns.isin(peptide_list)].T

    # define distance matrix including hypothalamus
    centroids = get_centroids('data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz')
    distance = squareform(pdist(centroids))

    # generate nulls
    nulls = gene_null_set(receptor_genes, non_peptide_genes, distance, n_permutations=nperm, 
                          n_jobs=32, seed=0)

    nulls = np.array(nulls)
    np.save('data/gene_null_sets_Schaefer400_TianS4_HTH.npy', nulls)

# %% PLS
# Define X and Y
X = zscore(ns.values, ddof=1)
Y = zscore(receptor_genes.values, ddof=1)

pls_result_fn = 'results/pls_result_Schaefer400_TianS4_HTH.npy'
if os.path.exists(pls_result_fn):
    pls_result = np.load(pls_result_fn, allow_pickle=True).item()
else:
    # behavioral PLS with gene nulls for Y
    pls_result = behavioral_pls(X, Y, n_boot=nperm, n_perm=nperm, rotate=True, permsamples=nulls,
                                permindices=False, test_split=0, seed=0)
    np.save(pls_result_fn, pls_result)

# check significance
nlv = len(X.T) if len(X.T) < len(Y.T) else len(Y.T) # number of latent variables
lv = 0  # interest only in first latent variable

cv = pls_result["singvals"]**2 / np.sum(pls_result["singvals"]**2)
null_singvals = pls_result['permres']['perm_singval']
cv_spins = null_singvals**2 / sum(null_singvals**2)
p = (1+sum(null_singvals[lv, :] > pls_result["singvals"][lv]))/(1+nperm)

plt.figure(figsize=(10, 5))
sns.boxplot(cv_spins.T * 100, color='lightgreen', fliersize=0, zorder=1)
sns.scatterplot(x=range(nlv), y=cv*100, s=30, color='orange', linewidth=1, edgecolor='black')
plt.ylabel("Covariance accounted for [%]")
plt.xlabel("Latent variable")
plt.title(f'LV{lv+1} accounts for {cv[lv]*100:.2f}% covariance | p = {p:.4f}');

if savefigs:
    plt.savefig('figs/pls_covexp.pdf')

# %% Redo PLS for X
# Switch X and Y
X = zscore(receptor_genes.values, ddof=1)
Y = zscore(ns.values, ddof=1)

pls_result_X_fn = 'results/pls_result_X_Schaefer400_TianS4_HTH.pickle'
if os.path.exists(pls_result_X_fn):
    with open(pls_result_X_fn, 'rb') as f:
        pls_result_X = pickle.load(f)
else:
    nulls = np.load('data/neurosynth_nulls_Schaefer400_TianS4_HTH.npy')
    # transpose to have last dimension as first
    nulls = np.transpose(nulls, (2, 1, 0))
    pls_result_X = behavioral_pls(X, Y, n_boot=nperm, n_perm=nperm, rotate=True, permsamples=nulls,
                                  permindices=False, test_split=0, seed=0)
    with open(pls_result_X_fn, 'wb') as f:
        # Use protocol 4 as it's a large file
        pickle.dump(pls_result_X, f, protocol=4)

# check significance
nlv = len(X.T) if len(X.T) < len(Y.T) else len(Y.T) # number of latent variables
lv = 0  # interest only in first latent variable

cv = pls_result_X["singvals"]**2 / np.sum(pls_result_X["singvals"]**2)
null_singvals = pls_result_X['permres']['perm_singval']
cv_spins = null_singvals**2 / sum(null_singvals**2)
p = (1+sum(null_singvals[lv, :] > pls_result_X["singvals"][lv]))/(1+nperm)

print("p-value: {:.4f}".format(p))

# %% plot scores
xscore = pls_result["x_scores"][:, lv]
yscore = pls_result["y_scores"][:, lv]

# show linear fit with no CI
sns.regplot(x=xscore, y=yscore, color=colors[0], scatter_kws={'alpha': 0.5}, ci=None)
sns.despine()
plt.xlabel('Receptor scores')
plt.ylabel('Cognitive scores')
if savefigs:
    plt.savefig('figs/scores.pdf')

# %% Plot receptor loadings
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
# fig.set_tight_layout(True)
if savefigs:
    plt.savefig('figs/receptor_loadings.pdf')

# # Plot altogether
# plt.figure(figsize=(6,8))
# sns.barplot(x='loading', y='receptor', data=receptors_df, xerr=err, hue='sign', dodge=True, palette=colors)
# plt.legend([],[], frameon=False)
# plt.tight_layout()

# %% Plot cognitive loadings
cognitive_terms = ns.columns

# error bars are ci from bootstrapping
err = (pls_result_X["bootres"]["y_loadings_ci"][:, lv, 1]
      - pls_result_X["bootres"]["y_loadings_ci"][:, lv, 0]) / 2

cognition_df = pd.DataFrame({'cognitive_term': cognitive_terms, 'loading': pls_result_X["y_loadings"][:, lv],
                             'err': err})
cognition_df['sign'] = np.sign(cognition_df['loading'])
cognition_df = cognition_df.sort_values('loading', ascending=False)

fig, axes = split_barplot(cognition_df, x='loading', y='cognitive_term', top=20, figsize=(8, 5), dpi=200)
# fig.set_tight_layout(True)
if savefigs:
    plt.savefig('figs/cognitive_loadings.pdf')

# %% LOADINGS VS HTH FC SIMILARITY
# load hypothalamus-receptor FC
df = pd.read_csv('./results/receptor_hth_coupling_Schaefer400_TianS4.csv', index_col=0)
hth = df['hth_corr'].values
loadings = pls_result["y_loadings"][:, 0]
r, p = spearmanr(hth, loadings)
p = 'p<0.001' if p < 0.001 else f'p={p:.0e}'

plot_df = pd.DataFrame({'hth_corr': df['hth_corr'], 
                        'loading': pls_result["y_loadings"][:, lv], 
                        'module': df['ci']})
spectral = [color for i, color in enumerate(sns.color_palette('Spectral')) if i in [1,2,4]]

sns.regplot(x=hth, y=loadings, scatter_kws={'s': 0}, color='black', ci=None)
sns.scatterplot(data=plot_df, x='hth_corr', y='loading', hue='module', palette=spectral)
plt.legend(frameon=False, title='Module')
plt.xlabel('Receptor similarity to FC$_{Hypothalamus}$')
plt.ylabel('Receptor loadings on 1st PLS component')
plt.title(f'Spearman r={r:.2f} | {p}')
sns.despine()
if savefigs:
    plt.savefig('figs/hth_corr_vs_loadings.pdf')

# %%
from surfplot import Plot
from neuromaps.datasets import fetch_fslr
from brainspace.datasets import load_parcellation
from utils import index_structure
from plot_utils import divergent_green_orange

# turn pls scores into dataframe for plotting
plot_df = pd.DataFrame({'cognitive': pls_result["y_scores"][:, 0], 'receptor': pls_result["x_scores"][:, 0]})
plot_data = index_structure(plot_df, structure='CTX')
plot_df.to_csv('results/pls_scores_Schaefer400_TianS4_HTH.csv', index=False)

# %%
# load surface and parcellation
surfaces = fetch_fslr()
lh, rh = surfaces['inflated']
atlas = load_parcellation('schaefer', 400)
atlas = atlas[0] # only left hemisphere
regions = np.unique(atlas)[1:] # discard 0
atlas_values = atlas.copy()

# cognitive scores
for roi in regions:
    roi_value = plot_data['cognitive'].values[roi]
    layer_data = np.where(atlas==roi, roi_value, atlas_values)

p = Plot(lh, views=['lateral','medial'], zoom=1.2, size=(1200, 800), dpi=200, brightness=0.6)
p.add_layer(layer_data, cmap=divergent_green_orange(), tick_labels=['min', 'max'], cbar_label='Cognitive scores')

if savefigs:
    p.build(dpi=300, save_as=f'figures/cognitive_scores_brainmap.pdf');
else:
    p.build(dpi=300);

# receptor scores
for roi in regions:
    roi_value = plot_data['receptor'].values[roi]
    layer_data = np.where(atlas==roi, roi_value, atlas_values)

p = Plot(lh, views=['lateral','medial'], zoom=1.2, size=(1200, 800), dpi=200, brightness=0.6)
p.add_layer(layer_data, cmap=divergent_green_orange(), tick_labels=['min', 'max'], cbar_label='Receptor scores')

if savefigs:
    p.build(dpi=300, save_as=f'figures/receptor_scores_brainmap.pdf');
else:
    p.build(dpi=300);