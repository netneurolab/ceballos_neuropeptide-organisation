# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sstats
from utils import index_structure, reorder_subcortex, gene_null_set
from plot_utils import divergent_green_orange, split_barplot
from netneurotools import plotting, modularity
from neuromaps.stats import compare_images

savefig = False
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                 LOAD BRAINSTEM FUNCTIONAL CONNCETIVITY DATA
###############################################################################
hth_fc = pd.read_csv('data/hth_fc_Schaefer400_Fischl14.csv', index_col=0).values

# look up table from Hansen et al. 2023
region_info = pd.read_csv('data/parcellations/Schaefer2018_400_region_info_brainstem_subcortex.csv', index_col=0)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                   LOAD GENE DATA AND CREATE NULL SETS
###############################################################################
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
receptor_genes = index_structure(receptor_genes, structure='CTX-SBCTX')

# load all genes from abagen
all_genes = pd.read_csv('data/abagen_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
all_genes = index_structure(all_genes, structure='CTX-SBCTX')

# need distance matrix for null function
distance = np.load('data/template_parc-Schaefer400_TianS4_desc-distance.npy')
nulls_fn = 'data/gene_null_sets_Schaefer400_TianS4.npy'
if os.path.exists(nulls_fn):
    nulls = np.load(nulls_fn)
else:        
    # generate nulls
    nulls = gene_null_set(receptor_genes, non_peptide_genes, distance, n_permutations=10000, 
                        n_jobs=32, seed=0)

    nulls = np.array(nulls)
    np.save(nulls_fn, nulls)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                   REORDER GENE DATA TO MATCH HTH FC FORMAT
###############################################################################
# reorder null genes to match HTH FC 
# gene data needs to be reloaded
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
null_ro = []
for null in nulls:
    # add last row of NaNs for HTH
    null = np.vstack([null, np.nan*np.ones(null.shape[1])])
    null_df = pd.DataFrame(null, index=receptor_genes.index)
    null_ro.append(reorder_subcortex(null_df, type='freesurfer', region_info=region_info).values)
null_ro = np.array(null_ro)

# reorder receptor genes to match HTH FC
receptor_genes_rs = reorder_subcortex(receptor_genes, type='freesurfer', region_info=region_info).values

# iterate through genes and compare to the hth_fc distribution
results = np.array([compare_images(receptor_genes_rs[:, i], hth_fc, nulls=null_ro[:,:,i].T, metric='spearmanr') 
                    for i in range(null_ro.shape[2])])

fc_gene_corr, p_values = results[:, 0], results[:, 1]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                           PLOT
###############################################################################
# build dataframe for results
plot_df = pd.DataFrame({'gene': receptor_genes.columns, 'hth_corr': fc_gene_corr, 
                   'p_values': p_values})
plot_df['sign'] = np.where(plot_df['hth_corr'] < 0, 'neg', 'pos')
order = plot_df.sort_values('hth_corr', ascending=False)['gene']

# barplot
fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
sns.barplot(data=plot_df, x='gene', y='hth_corr', hue='sign', palette='Spectral', 
            order=order)
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1, 1), frameon=False)
plt.ylabel('Alignment with FC$_{Hypothalamus}$ (Spearman $\\rho$)')
sns.despine()
