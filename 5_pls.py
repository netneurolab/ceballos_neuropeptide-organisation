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
from utils import gene_null_set, index_structure, reorder_subcortex
from plot_utils import divergent_green_orange, split_barplot
from surfplot import Plot
from neuromaps.datasets import fetch_fslr
from brainspace.datasets import load_parcellation
from enigmatoolbox.plotting import plot_subcortical
from sklearn.model_selection import train_test_split

savefigs = False

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         LOAD DATA
###############################################################################
# Load neurosynth
ns = pd.read_csv('./data/neurosynth_Schaefer400_TianS4.csv', index_col=0)

# Load gene receptor data
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)

# Load colors
palette = divergent_green_orange(n_colors=9, return_palette=True)
bipolar = [palette[1], palette[-2]]
spectral = [color for i, color in enumerate(sns.color_palette('Spectral')) if i in [1,2,4]]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         GENE NULL SETS
###############################################################################
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                              PLS1
###############################################################################
# Define X and Y
X = zscore(ns.values, ddof=1)
Y = zscore(receptor_genes.values, ddof=1)
nlv = len(X.T) if len(X.T) < len(Y.T) else len(Y.T) # number of latent variables
lv = 0  # interested only in first latent variable

pls_result_fn = 'results/pls_result_Schaefer400_TianS4_HTH.npy'
if os.path.exists(pls_result_fn):
    pls_result = np.load(pls_result_fn, allow_pickle=True).item()
else:
    # behavioral PLS with gene nulls for Y
    pls_result = behavioral_pls(X, Y, n_boot=nperm, n_perm=nperm, rotate=True, permsamples=nulls,
                                permindices=False, test_split=0, seed=0)
    np.save(pls_result_fn, pls_result)

# check significance
cv = pls_result["singvals"]**2 / np.sum(pls_result["singvals"]**2)
null_singvals = pls_result['permres']['perm_singval']
cv_spins = null_singvals**2 / sum(null_singvals**2)
p = (1+sum(null_singvals[lv, :] > pls_result["singvals"][lv]))/(1+nperm)

plt.figure(figsize=(6, 5), dpi=200)
sns.boxplot(cv_spins.T * 100, color='lightgreen', zorder=1, width=0.4, linewidth=0.6,
            showfliers=False)
sns.scatterplot(x=range(nlv), y=cv*100, s=10, color='orange', linewidth=0.8, edgecolor='grey')
plt.ylabel("Covariance accounted for [%]")
plt.xlabel("Latent variables")
plt.xticks([])
plt.title(f'LV{lv+1} accounts for {cv[lv]*100:.2f}% covariance | p = {p:.4f}');

if savefigs:
    plt.savefig('figs/pls_cov_exp.pdf')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                              PLS2
###############################################################################
# Switch X and Y
X = zscore(receptor_genes.values, ddof=1)
Y = zscore(ns.values, ddof=1)
nlv = len(X.T) if len(X.T) < len(Y.T) else len(Y.T) # number of latent variables

# check if already computed
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
cv_X = pls_result_X["singvals"]**2 / np.sum(pls_result_X["singvals"]**2)
null_singvals_X = pls_result_X['permres']['perm_singval']
cv_spins_X = null_singvals_X**2 / sum(null_singvals_X**2)
p_X = (1+sum(null_singvals_X[lv, :] > pls_result_X["singvals"][lv]))/(1+nperm)

print(f"p-value: {p_X:.4f}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                              PLOT SCORES
###############################################################################
scores_fn = 'results/pls_scores_Schaefer400_TianS4_HTH.csv'

if os.path.exists(scores_fn):
    scores = pd.read_csv(scores_fn)
else:
    xscore = pls_result["x_scores"][:, lv]
    yscore = pls_result["y_scores"][:, lv]
    scores = pd.DataFrame({'networks': receptor_genes.index,
                           'receptor': xscore, 
                           'term': yscore})

atlas_info = pd.read_csv('data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv', index_col=0)
atlas_info['network'].iloc[-1] = 'Hypothalamus'
atlas_info['network'] = np.where(atlas_info['structure'] == 'cortex', 'Cortex', atlas_info['network'])
scores['network'] = atlas_info['network']

sns.regplot(data=scores, x='receptor', y='term', color='black', scatter_kws={'s': 0}, ci=None)
sns.scatterplot(data=scores, x='receptor', y='term', hue='network', palette=spectral,
                hue_order=['Cortex', 'Hypothalamus', 'Subcortex'], s=50)
sns.despine()
plt.legend(frameon=False, title='Network')
plt.xlabel('Receptor scores')
plt.ylabel('Term scores')
if savefigs:
    plt.savefig('figs/scores.pdf')
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          SCORES CROSS-VALIDATION
###############################################################################

# randomly split data into train and test 1000 times
train = np.zeros(1000)
test = np.zeros(1000)

for i in range(1000):
    # split data
    train_idx, test_idx = train_test_split(np.arange(X.shape[0]), test_size=0.7, random_state=i)
    Xtrain, Ytrain = X[train_idx], Y[train_idx]
    Xtest, Ytest = X[test_idx], Y[test_idx]

    # do PLS on train data
    pls_result_cv = behavioral_pls(Xtrain, Ytrain, n_boot=0, n_perm=0, test_split=0)

    # correlate score of cv model
    train[i], _ = spearmanr(pls_result_cv["x_scores"][:, lv], 
                         pls_result_cv["y_scores"][:, lv])
    test[i], _ = spearmanr(Xtest @ pls_result_cv["x_weights"][:, lv], 
                        Ytest @ pls_result_cv["y_weights"][:, lv])

fig, ax = plt.subplots(figsize=(3, 6), dpi=200)
sns.boxplot(data=[train, test], palette=bipolar, ax=ax)
ax.set_xticklabels(['Train', 'Test'])
ax.set_ylabel('Score correlation')
sns.despine(trim=True)

if savefigs:
    plt.savefig('figs/pls_cv_score_correlation.pdf')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          PLOT RECEPTOR LOADINGS
###############################################################################
receptor_names = receptor_genes.columns

# error bars are ci from bootstrapping
err = (pls_result["bootres"]["y_loadings_ci"][:, lv, 1]
      - pls_result["bootres"]["y_loadings_ci"][:, lv, 0]) / 2
receptors_df = pd.DataFrame({'receptor': receptor_names, 'loading': pls_result["y_loadings"][:, lv],
                        'err': err})
receptors_df['sign'] = np.sign(receptors_df['loading'])    
receptors_df = receptors_df.sort_values('loading', ascending=False)

fig, axes = split_barplot(receptors_df, x='loading', y='receptor', top=10,
                          figsize=(8, 5), dpi=200)

if savefigs:
    plt.savefig('figs/receptor_loadings.pdf')

# # Plot altogether
# plt.figure(figsize=(6,8))
# sns.barplot(x='loading', y='receptor', data=receptors_df, xerr=err, hue='sign', dodge=True, palette=bipolar)
# plt.legend([],[], frameon=False)
# plt.tight_layout()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          PLOT TERM LOADINGS
###############################################################################
terms = ns.columns

# error bars are ci from bootstrapping
err = (pls_result_X["bootres"]["y_loadings_ci"][:, lv, 1]
      - pls_result_X["bootres"]["y_loadings_ci"][:, lv, 0]) / 2

term_df = pd.DataFrame({'term': terms, 'loading': pls_result_X["y_loadings"][:, lv],
                             'err': err})
term_df['sign'] = np.sign(term_df['loading'])
term_df = term_df.sort_values('loading', ascending=False)

fig, axes = split_barplot(term_df, x='loading', y='term', top=10, 
                          figsize=(8, 5), dpi=200)

if savefigs:
    plt.savefig('figs/term_loadings.pdf')

# # Plot altogether
# plt.figure(figsize=(6,8))
# sns.barplot(x='loading', y='cognitive_term', data=term_df, xerr=err, hue='sign', dodge=True, palette=bipolar)
# plt.legend([],[], frameon=False)
# plt.tight_layout()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOT BRAINMAPS ON SURFACE
###############################################################################

scores_fn = 'results/pls_scores_Schaefer400_TianS4_HTH.csv'
if os.path.exists(scores_fn):
    plot_df = pd.read_csv(scores_fn)
else:
    # turn pls scores into dataframe for plotting
    plot_df = pd.DataFrame({'term': pls_result["y_scores"][:, lv], 'receptor': pls_result["x_scores"][:, lv]})
    plot_data = index_structure(plot_df, structure='CTX')
    # plot_df.to_csv('results/pls_scores_Schaefer400_TianS4_HTH.csv', index=False)

# %% CORTEX
# load surface and parcellation
surfaces = fetch_fslr()
lh, rh = surfaces['inflated']
atlas = load_parcellation('schaefer', 400)
atlas = atlas[0] # only left hemisphere
regions = np.unique(atlas)[1:] # discard 0
atlas_values = atlas.copy()

# term scores
for roi in regions:
    roi_value = plot_data['term'].values[roi]
    layer_data = np.where(atlas==roi, roi_value, atlas_values)

p = Plot(lh, views=['lateral','medial'], zoom=1.2, size=(1200, 800), dpi=200, brightness=0.6)
p.add_layer(layer_data, cmap=divergent_green_orange(), tick_labels=['min', 'max'], cbar_label='Term scores')

if savefigs:
    p.build(dpi=300, save_as=f'figures/term_scores_brainmap.pdf');
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
    
# %% SUBCORTEX
sbctx = index_structure(plot_df, structure='SBCTX')
sbctx = reorder_subcortex(sbctx, type='enigma')
plot_subcortical(sbctx['term'].values, ventricles=False, transparent_bg=True, size=(1200, 800),
                 interactive=True, embed_nb=False) 
# %%
plot_subcortical(sbctx['receptor'].values, ventricles=False, transparent_bg=True, size=(1200, 800),
                 interactive=True, embed_nb=False) 