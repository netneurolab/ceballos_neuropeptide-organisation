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
atlas = nii_load('./data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz')
atlas_info = pd.read_csv('./data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv')

# load quality control overview
overview = pd.read_csv('./data/receptor_overview.csv')

# check atlas and atlas_info
atlas = abagen.images.check_atlas(atlas, atlas_info)

# run abagen on the new atlas
genes = abagen.get_expression_data(atlas, atlas_info, data_dir='./data/', 
                                   norm_matched=False, missing='interpolate',
                                   probe_selection='rnaseq', ibf_threshold=0.2)

# change index to region names from atlas_info
genes.index = atlas_info['name']

# # move first 54 rows from subcortex to end of dataframe
# genes = pd.concat([genes.iloc[54:, :], genes.iloc[:54, :]], axis=0)

# # put hypothalamus at the end
# hth = genes.loc['HTH']
# genes = genes.drop('HTH', axis=0)
# genes = pd.concat([genes, pd.DataFrame(hth).T], axis=0)

# keep columns of genes that are in the overview and have mean_RNAcorr > 0.2
receptor_names = overview['gene'][overview['mean_RNAcorr'] > 0.2].to_list()
receptors = genes[genes.columns.intersection(receptor_names)]

# save genes to csv
receptors.to_csv('./data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv')
genes.to_csv('./data/abagen_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv')

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
