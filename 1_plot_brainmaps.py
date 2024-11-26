# %%
import pandas as pd
import numpy as np
from utils import index_structure
from plot_utils import divergent_green_orange
from surfplot import Plot
from brainspace.datasets import load_parcellation
from neuromaps.datasets import fetch_fslr

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         LOAD DATA
###############################################################################
# load genes and gene-family correspondence
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
ctx_data = index_structure(receptor_genes, structure='CTX')
receptor_list = pd.read_csv('data/receptor_overview.csv')

# genes without a family are 'Unclassified'
receptor_list['family'] = receptor_list['family'].fillna('Unclassified')

# plot one receptor per family
receptors = [
    'ADIPOR2',
    'GRPR',
    'CALCRL',
    'CCKBR',
    'EDNRB',
    'NPY1R',
    'GALR1',
    'VIPR1',
    'RXFP1',
    'NTSR1',
    'NPR2',
    'OPRK1',
    'SSTR1',
    'OXTR',
    'GHR']

# # alternatively, plot all receptors
# receptors = receptor_genes.columns

family_map = {gene: family for gene, family in zip(receptor_list['gene'], receptor_list['family']) \
              if gene in receptors}

# print gene and family
for gene, family in family_map.items():
    print(f'{gene}: {family}')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOT SURFACE MAPS
###############################################################################

surfaces = fetch_fslr()
lh, rh = surfaces['inflated']
atlas = load_parcellation('schaefer', 400)
atlas = atlas[0] # only left hemisphere

for receptor in receptors:
    # family = family_map[receptor]
    parc_data = ctx_data[receptor].values
    full_name = receptor_list[receptor_list['gene'] == receptor]['description'].values[0]

    unique = np.unique(atlas)
    unique = unique[1:] # discard 0
    
    plot_data = atlas.copy()
    for i in range(unique.shape[0]):
        plot_data = np.where(plot_data==unique[i], parc_data[i], plot_data)

    # p = Plot(lh, views=['lateral','medial'], zoom=1.2, size=(1200, 800), brightness=0.6)
    # p.add_layer(plot_data, cmap=divergent_green_orange(), tick_labels=['min', 'max'])#, cbar_label=full_name)
    # p.build(dpi=300, save_as=f'figures/{receptor}_brainmap.pdf');

