# %%
import numpy as np
import pandas as pd
from neuromaps.datasets import fetch_annotation, available_annotations
from neuromaps.images import relabel_gifti, annot_to_gifti
from netneurotools.datasets import fetch_schaefer2018
from neuromaps.parcellate import Parcellater

# %%
# check which maps are available
annotations = available_annotations()

# within available_annotations look for map that contains evo
evo_map_desc = [(map[0], map[1]) for map in annotations if 'evo' in map[1]]

# same for dev
dev_map_desc = [(map[0], map[1]) for map in annotations if 'scalinghcp' in map[1]]

# %%
# fetch evoexp from xu2020
evoexp = fetch_annotation(source=evo_map_desc[1][0], desc=evo_map_desc[1][1], data_dir='./data/')
devexp = fetch_annotation(source=dev_map_desc[0][0], desc=dev_map_desc[0][1], data_dir='./data/')

# %%
schaefer_fs6 = fetch_schaefer2018(version='fsaverage6', data_dir='./data/')
atlas_fs = annot_to_gifti(schaefer_fs6['400Parcels7Networks'])
atlas_fs = relabel_gifti(atlas_fs)
surf_mask = Parcellater(atlas_fs, 'fsaverage', resampling_target='parcellation')

# %%
evo_parc = surf_mask.fit_transform(evoexp, 'fslr', ignore_background_data=True)
dev_parc = surf_mask.fit_transform(devexp, 'civet', ignore_background_data=True)

# %%
receptor_list = pd.read_csv('data/receptor_list.csv')['Gene']
peptide_genes = pd.read_csv('data/peptide_genes_ahba_Schaefer400.csv', index_col=0)

overlapping_genes = peptide_genes.columns.intersection(receptor_list)
peptide_genes = peptide_genes[overlapping_genes]

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
pca = PCA(n_components=10)

peptide_genes = scaler.fit_transform(peptide_genes)
peptide_pcs = pca.fit_transform(peptide_genes)

# how much variance is explained by the first 10 components
varexp = pca.explained_variance_ratio_

# what are the loadings of the first 10 components
loadings = pca.components_

# %%

# correlation between pc1 and pc2 with evo and dev
from neuromaps.stats import compare_images
# load spins
spin_idx = np.load('data/vasa_schaefer400_fsaverage_spin_indices.npy')

evo_nulls = evo_parc[spin_idx]
dev_nulls = dev_parc[spin_idx]

evo1_r, evo1_p = compare_images(evo_parc, peptide_pcs[:,0], nulls=evo_nulls, metric='spearmanr')
print(f'Correlation between PC1 and Evo: {evo1_r:.3f} (p={evo1_p:.3f})')
evo2_r, evo2_p = compare_images(evo_parc, peptide_pcs[:,1], nulls=evo_nulls, metric='spearmanr')
print(f'Correlation between PC2 and Evo: {evo2_r:.3f} (p={evo2_p:.3f})')

dev1_r, dev1_p = compare_images(dev_parc, peptide_pcs[:,0], nulls=dev_nulls, metric='spearmanr')
print(f'Correlation between PC1 and Dev: {dev1_r:.3f} (p={dev1_p:.3f})')
dev2_r, dev2_p = compare_images(dev_parc, peptide_pcs[:,1], nulls=dev_nulls, metric='spearmanr')
print(f'Correlation between PC2 and Dev: {dev2_r:.3f} (p={dev2_p:.3f})')

# %%

import seaborn as sns
import matplotlib.pyplot as plt
sns.regplot(x=peptide_pcs[:,0], y=evo_parc, scatter_kws={'s': 1})

# %%
# correlate every peptide gene with evo and dev
# store as dictionary indivating which gene is correlated with evo and dev
from scipy.stats import spearmanr

evo_corrs = {}
dev_corrs = {}
for gene in peptide_genes.columns:
    evo_r, evo_p = spearmanr(peptide_genes[gene], evo_parc)
    dev_r, dev_p = spearmanr(peptide_genes[gene], dev_parc)
    evo_corrs[gene] = evo_r
    dev_corrs[gene] = dev_r
    
# %%
