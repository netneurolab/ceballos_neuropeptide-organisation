# %% 
import pandas as pd
import numpy as np
from utils import index_structure
from plot_utils import divergent_green_orange
from surfplot import Plot
from brainspace.datasets import load_parcellation
from neuromaps.datasets import fetch_fslr

# %% CORTEX PLOTS
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
    
# %% SUBCORTICAL PLOTS
import pandas as pd
from utils import index_structure
from enigmatoolbox.plotting import plot_subcortical

receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
receptor_genes = index_structure(receptor_genes, structure='SBCTX')

regions = receptor_genes.index.str.split('-').str[0]
hemi = receptor_genes.index.str.split('-').str[-1]

name_map = {'HIP': 'hippocampus',
            'THA': 'thalamus',
            'mAMY': 'amygdala',
            'lAMY': 'amygdala',
            'PUT': 'putamen',
            'aGP': 'pallidum',
            'pGP': 'pallidum',
            'CAU': 'caudate',
            'NAc': 'accumbens'}

hemi_map = {'lh': 'left',
            'rh': 'right'}

# map regions/ hemi to names
regions = regions.map(name_map)
hemi = hemi.map(hemi_map)

# join the two and rename index
receptor_genes.index = hemi + '-' + regions

# average same regions
receptor_genes = receptor_genes.groupby(receptor_genes.index).mean()

# separate df into two according to hemisphere
left = receptor_genes[receptor_genes.index.str.contains('left')]
right = receptor_genes[receptor_genes.index.str.contains('right')]

# add *-ventricles with NaN
left.loc['left-ventricles'] = np.nan
right.loc['right-ventricles'] = np.nan

# reorder alphabetically
left = left.sort_index()
right = right.sort_index()

# merge the two
receptor_genes = pd.concat([left, right])

# %%
plot_subcortical(receptor_genes['NPY2R'].values, ventricles=True, transparent_bg=True, size=(1200, 800), 
                 interactive=True, embed_nb=False, cmap='green_orange', color_range=(0, 1))
# %%
