# %%
import numpy as np
import abagen
import pandas as pd
import os
from nibabel.loadsave import load as nii_load

##################
# %% GROUP AVG
##################

# load atlas info
atlas = nii_load('./data/parcellations/Cammoun2012_250_7N_Freesurfer_Subcortex_space-MNI152_den_1mm.nii.gz')
atlas_info = pd.read_csv('./data/parcellations/Cammoun2012_7N_Freesurfer_Subcortex_LUT.csv')
atlas_info = atlas_info[atlas_info['scale'] == 'scale250'].reset_index(drop=True)

# load quality control overview
overview = pd.read_csv('./data/receptor_overview.csv')

# check atlas and atlas_info
atlas = abagen.images.check_atlas(atlas, atlas_info)

# run abagen on the new atlas
genes = abagen.get_expression_data(atlas, atlas_info, data_dir='./data/', 
                                   norm_matched=False, missing='interpolate',
                                   probe_selection='rnaseq', ibf_threshold=0.2)

# drop brainstem
genes = genes.iloc[:-1]

# change index to region names + hemisphere from atlas_info
genes.index = atlas_info['hemisphere'].iloc[:-1] + '_' + atlas_info['label'].iloc[:-1]

# # move first 54 rows from subcortex to end of dataframe
# genes = pd.concat([genes.iloc[54:, :], genes.iloc[:54, :]], axis=0)

# which rows of atlas_info are subcortex
subcortex = atlas_info[atlas_info['structure'] == 'subcortex'].iloc[:-1]
subcortex = subcortex['hemisphere'] + '_' + subcortex['label']

# put subcortex regions at the start of the dataframe
genes = pd.concat([genes.loc[subcortex], genes.drop(subcortex, axis=0)], axis=0)

# keep columns of genes that are in the overview and have mean_RNAcorr > 0.2
receptor_names = overview['gene'][overview['mean_RNAcorr'] > 0.2].to_list()
receptors = genes[genes.columns.intersection(receptor_names)]


# save genes to csv
genes.to_csv('./data/abagen_gene_expression_Cammoun2012_250_7N_Freesurfer_Subcortex.csv')
receptors.to_csv('./data/receptor_gene_expression_Cammoun2012_250_7N_Freesurfer_Subcortex.csv')

##################
# %% INDIVIDUALS
##################
user_path = os.path.dirname(os.getcwd())
abagen_path = 'data/abagen_genes_Schaefer2018_400.csv'

donor_expressions = abagen.get_expression_data(atlas, atlas_info, data_dir=user_path,
                                               norm_matched=False, missing='interpolate', 
                                               probe_selection='rnaseq', ibf_threshold=0.2,
                                               return_donors=True)

# turn dictionary into list
donor_list = []
for donor in donor_expressions:
    donor_list.append(donor_expressions[donor])

# %%
# check differential stability of genes
genes, ds = abagen.keep_stable_genes(donor_list, threshold=0, percentile=False, return_stability=True)
