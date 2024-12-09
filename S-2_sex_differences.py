# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nibabel.loadsave import load as nii_load
from plot_utils import divergent_green_orange
from abagen.images import check_atlas
from abagen import get_expression_data

savefig = False

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                      LOAD ATLAS AND RECEPTOR NAMES
###############################################################################

# load atlas info
atlas = nii_load('./data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz')
atlas_info = pd.read_csv('./data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv')

# load peptide receptor names
receptors = pd.read_csv('./data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0).columns

# check atlas and atlas_info
atlas = check_atlas(atlas, atlas_info)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                      FETCH DONOR DATA SEPARATELY
###############################################################################
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#               ZSCORE MALE DATA AND PROJECT FEMALE DATA
###############################################################################
# calculate mean and std of male data
mean = male_data.mean(axis=0)
std = male_data.std(axis=0)

# standardize male data
zscored_male_data = (male_data - mean) / std

# now female data
zscored_female_data = (female_data - mean) / std

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                        PLOT DATA BY NETWORKS
###############################################################################
female_df = pd.DataFrame(zscored_female_data, columns=receptors, index=atlas_info['name'])

# load region names
atlas_regions = pd.read_csv('./data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv', index_col=0)

name_map = {'HIP': 'hippocampus',
            'THA': 'thalamus',
            'mAMY': 'amygdala',
            'lAMY': 'amygdala',
            'PUT': 'putamen',
            'aGP': 'globus-pallidus',
            'pGP': 'globus-pallidus',
            'CAU': 'caudate',
            'NAc': 'nucleus-accumbens',
            'HTH': 'hypothalamus'}

# change the name of regions in subcortex using name_map
atlas_regions['name'] = atlas_regions['name'].str.split('-').str[0].replace(name_map)
female_df.index = atlas_regions['id'].values

# group df first by structure, then network, and then name
atlas_regions = atlas_regions.groupby(['structure', 'network', 'name']).apply(lambda x: x).reset_index(drop=True)

# average by network
atlas_regions = atlas_regions.sort_values('id').reset_index()
atlas_regions['network_alt'] = atlas_regions['hemisphere'] + '_' + atlas_regions['name'].str.split('_').str[0].replace(name_map)

# group by network and average
networks = atlas_regions['network_alt'].unique()
network_genes = {network: female_df[(atlas_regions['network_alt'] == network).values].mean(axis=0) \
                 for network in networks}
network_genes = pd.DataFrame(network_genes).T

network_order = [f'L_{ctx_net}' for ctx_net in ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Cont', 'Default', 'Limbic' ]] + \
                [f'L_{sbctx_net}' for sbctx_net in ['amygdala', 'caudate', 'globus-pallidus', 'hippocampus', 'nucleus-accumbens', 'putamen', 'thalamus']] + \
                ['B_hypothalamus']

# reorder network_genes
network_genes = network_genes.loc[network_order].T

# plot as heatmap
fig, ax = plt.subplots(figsize=(5, 11), dpi=200)
sns.heatmap(network_genes, cmap=divergent_green_orange(), cbar_kws={'label': 'Z-score', 'shrink':0.5}, 
            linewidths=0.01, linecolor='white', square=True, ax=ax, vmin=-3, vmax=3)
if savefig:
    plt.savefig('./figs/sexdiff_networks.pdf', bbox_inches='tight')

