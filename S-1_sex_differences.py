# %%
import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/poolz2/ceballos/miniconda3/envs/peptides/python3.11/site-packages/PyQt5/Qt5/plugins/platforms'

import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
from plot_utils import divergent_green_orange
from abagen.images import check_atlas
from abagen import get_expression_data

# %%
# load atlas info
atlas = nib.load('./data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz')
atlas_info = pd.read_csv('./data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv')

# load peptide receptor names
receptors = pd.read_csv('./data/receptor_filtered.csv')['gene'].values

# check atlas and atlas_info
atlas = check_atlas(atlas, atlas_info)
# %%
# get donor expression data
donor_expressions = get_expression_data(atlas, atlas_info, data_dir='./data/', ibf_threshold=0.2,
                                        norm_matched=False, missing='interpolate', lr_mirror='bidirectional',
                                        probe_selection='rnaseq', return_donors=True)

# separate data into male/ female
# make copy of donor_expressions
male_dict = donor_expressions.copy()

# donor 15496 is female
# extract female from dictionary
female_data = donor_expressions['15496'][receptors].values
# delete from male_dict
del male_dict['15496']

# create zscore based on male donors data
# create array of size (n_donors_male, n_regions, n_genes)
n_donors = len(donor_expressions) - 1
n_regions = len(atlas_info)
n_genes = len(receptors)

male_data = np.zeros((n_donors, n_regions, n_genes))

for i,key in enumerate(male_dict.keys()):
    male_data[i] = male_dict[key][receptors].values

# %%
# calculate mean and std of male data
mean = male_data.mean(axis=0)
std = male_data.std(axis=0)

# standardize male data
zscored_male_data = (male_data - mean) / std

# now female data
zscored_female_data = (female_data - mean) / std

# clip to -3, 3
zscored_female_data = np.clip(zscored_female_data, -3, 3)

# %%
clustermap = sns.clustermap(zscored_female_data, row_cluster=False, col_cluster=True,
               xticklabels=receptors, dendrogram_ratio=0.1,
               cbar=True, figsize=(10, 15), yticklabels=False);
reordered_ind = clustermap.dendrogram_col.reordered_ind

# %%
# find where the structure changes
# create new list of structures
structures = [s[1]['name'].split('_')[0] 
              if s[1]['structure'] == 'cortex' 
              else s[1]['structure'].capitalize()
              for s in atlas_info.iterrows()]
structures = np.array(structures)

# at which index is there a change in structures
change_ind = np.where(np.array(structures[:-1]) != np.array(structures[1:]))[0]

# %%
# cluster by gene and plot data
# have yticks only at change_ind
cmap = divergent_green_orange()
clustermap = sns.clustermap(zscored_female_data, cmap=cmap, row_cluster=False, col_cluster=True,
               xticklabels=receptors, dendrogram_ratio=0.1,
               cbar=True, figsize=(10, 15), yticklabels=False)

clustermap.ax_heatmap.set_yticks(change_ind)
clustermap.ax_heatmap.set_yticklabels(structures[change_ind]);

# draw lines at change_ind
for ind in change_ind:
    clustermap.ax_heatmap.axhline(ind, color='white', linewidth=3)
    clustermap.ax_heatmap.axhline(ind, color='black', linewidth=3, alpha=0.8)

plt.savefig('./figs/sex_differences.pdf', bbox_inches='tight')

# %%
from netneurotools.plotting import plot_fslr
from neuromaps.images import dlabel_to_gifti
oxtr = female_data[:, reordered_ind[0]][54:-1]
atlas_fslr = dlabel_to_gifti('data/parcellations/Schaefer2018_400_7N_space-fsLR_den-32k.dlabel.nii')

plot_fslr(oxtr, lhlabel=atlas_fslr[0], rhlabel=atlas_fslr[1], cmap='viridis', colorbar=True, cbar_kws={'label': 'Z-score'})

# plot GRPR, OXTR, LEPR, RAMP1, NPY2R
subcortex_info = atlas_info[:54]

# %%
