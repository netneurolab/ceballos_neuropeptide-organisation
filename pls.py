# %%
from ensurepip import bootstrap
import os
import pyls
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from neuromaps.nulls import vasa
from neuromaps.images import annot_to_gifti, relabel_gifti
from netneurotools.datasets import fetch_schaefer2018

# %% Load atlas
schaefer_fs6 = fetch_schaefer2018(version='fsaverage6', data_dir='/poolz2/ceballos/')
atlas_fs = annot_to_gifti(schaefer_fs6['400Parcels7Networks'])
atlas_fs = relabel_gifti(atlas_fs)

# %% Load neurosynth
ns = pd.read_csv('./data/neurosynth_Schaefer2018_400.csv')

# %% Load gene receptor data
gene_list = pd.read_csv('data/gene_list.csv')
receptor_labels = gene_list[gene_list['Description'].str.contains('receptor')]
receptor_labels = receptor_labels.append({'Gene': 'SORT1'}, ignore_index=True)
receptor_labels = receptor_labels['Gene'].values

# load gene expression data
genes = pd.read_csv('data/peptide_genes_ahba_Schaefer400.csv', index_col=0)
genes = genes[genes.columns[genes.columns.isin(gene_list['Gene'])]]
receptor_genes = genes[genes.columns[genes.columns.isin(receptor_labels)]]

# %% PLS
X = zscore(ns.values, ddof=1)
Y = zscore(receptor_genes.values, ddof=1)

nperm = 10000
nspins = nperm

if os.path.exists('results/pls_result.npy'):
    pls_result = np.load('results/pls_result.npy', allow_pickle=True).item()
else:
    spins = vasa(None, atlas='fsaverage', density='41k', parcellation=atlas_fs,
                 n_perm=nspins, seed=0)

    pls_result = pyls.behavioral_pls(X, Y, n_boot=nperm, n_perm=nperm, rotate=True, permsamples=spins,
                                    test_split=0, seed=1234)
    
    np.save('results/pls_result.npy', pls_result)

# check significance
nlv = len(X.T) if len(X.T) < len(Y.T) else len(Y.T) # number of latent variables
lv = 0  # interest only in first latent variable

cv = pls_result["singvals"]**2 / np.sum(pls_result["singvals"]**2)
null_singvals = pls_result['permres']['perm_singval']
cv_spins = null_singvals**2 / sum(null_singvals**2)
p = (1+sum(null_singvals[lv, :] > pls_result["singvals"][lv]))/(1+nperm)

print("p-value: {:.4f}".format(p))

# %% X loadings of PLS
X = zscore(receptor_genes.values, ddof=1)
Y = zscore(ns.values, ddof=1)

nperm = 10000
nspins = nperm

if os.path.exists('results/pls_result_X.npy'):
    pls_result_X = np.load('results/pls_result_X.npy', allow_pickle=True).item()
else:
    spins = vasa(None, atlas='fsaverage', density='41k', parcellation=atlas_fs,
                 n_perm=nspins, seed=0)

    pls_result_X = pyls.behavioral_pls(X, Y, n_boot=nperm, n_perm=nperm, rotate=True, permsamples=spins,
                                       test_split=0, seed=1234)
    
    np.save('results/pls_result_X.npy', pls_result_X)

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
sns.regplot(x=xscore, y=yscore, scatter_kws={'alpha': 0.5}, ci=None)
sns.despine()
plt.xlabel('Receptor scores')
plt.ylabel('Cognitive scores')

# %%
lv = 0
receptor_names = receptor_genes.columns

# plot term loadings
err = (pls_result["bootres"]["y_loadings_ci"][:, lv, 1]
      - pls_result["bootres"]["y_loadings_ci"][:, lv, 0]) / 2
relidx = (abs(pls_result["y_loadings"][:, lv]) - err) > 0  # CI doesnt cross 0
sorted_idx = np.argsort(pls_result["y_loadings"][relidx, lv])
plt.figure(figsize=(10, 5))
plt.ion()
plt.bar(np.arange(sum(relidx)), 
        np.sort(pls_result["y_loadings"][relidx, lv]),
        yerr=err[relidx][sorted_idx])
plt.xticks(np.arange(sum(relidx)), 
           labels=[receptor_names[i] for i in sorted_idx],
           rotation='vertical')
plt.ylabel("Receptor loadings")
plt.tight_layout()


err = (pls_result_X["bootres"]["y_loadings_ci"][:, lv, 1]
      - pls_result_X["bootres"]["y_loadings_ci"][:, lv, 0]) / 2
relidx = (abs(pls_result_X["y_loadings"][:, lv]) - err) > 0  # CI doesnt cross 0
sorted_idx = np.argsort(pls_result_X["y_loadings"][relidx, lv])
plt.figure(figsize=(15, 5))
plt.bar(np.arange(sum(relidx)), 
        pls_result_X["y_loadings"][relidx, lv][sorted_idx],
        yerr=err[relidx])
plt.xticks(np.arange(sum(relidx)),
           labels=ns.columns[relidx][sorted_idx],
           rotation='vertical')
plt.ylabel("Cognitive term loadings")
plt.tight_layout()


# %%
# load hypothalamus-receptor FC
df = pd.read_csv('./results/receptor_hth_coupling.csv')
hth = df['HTH_corr'].values
loadings = pls_result["y_loadings"][:, 0]

sns.regplot(x=hth, y=loadings, scatter_kws={'alpha': 0.5})
plt.xlabel('Hypothalamus-receptor FC')
plt.ylabel('Receptor loadings in first PLS component')
