# %% PLOT BRAIN MAPS
import pandas as pd
import numpy as np
from utils import index_structure
from plot_utils import divergent_green_orange
from surfplot import Plot
from brainspace.datasets import load_parcellation
from neuromaps.datasets import fetch_fslr

# load genes and gene list
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
receptor_list = pd.read_csv('data/receptor_list.csv')
ctx_data = index_structure(receptor_genes, structure='CTX')

surfaces = fetch_fslr()
lh, rh = surfaces['inflated']
atlas = load_parcellation('schaefer', 400)
atlas = atlas[0] # only left hemisphere

receptors = [
    'ADCYAP1R1',
    'CALCRL',
    'CCKBR',
    'EDNRA',
    'GALR1',
    'GIPR',
    'GLP2R',
    'GRPR',
    'LEPR',
    'NPY1R',
    'OPRK1',
    'OPRM1',
    'OXTR',
    'RXFP1',
    'SSTR1',
    'VIPR1']

for receptor in receptors[-2:]:
    parc_data = ctx_data[receptor].values
    full_name = receptor_list[receptor_list['Gene'] == receptor]['Description'].values[0]

    unique = np.unique(atlas)
    unique = unique[1:] # discard 0
    
    plot_data = atlas.copy()
    for i in range(unique.shape[0]):
        plot_data = np.where(plot_data==unique[i], parc_data[i], plot_data)

    p = Plot(lh, views=['lateral','medial'], zoom=1.2, size=(1200, 800), dpi=200, brightness=0.6)
    p.add_layer(plot_data, cmap=divergent_green_orange(), tick_labels=['min', 'max'])#, cbar_label=full_name)
    p.build(dpi=300, save_as=f'figures/{receptor}_brainmap.pdf');