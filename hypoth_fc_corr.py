# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sympy import plot
from utils import index_structure, reorder_subcortex, gene_null_set
from plot_utils import divergent_green_orange
from netneurotools import plotting, modularity
from neuromaps.stats import compare_images

# %% LOAD BRAINSTEM FUNCTIONAL CONNCETIVITY DATA
hth_fc = pd.read_csv('data/hth_fc_Schaefer400_Fischl14.csv', index_col=0).values#.flatten()
# region info from Hansen et al. 2023
region_info = pd.read_csv('data/parcellations/Schaefer2018_400_region_info_brainstem_subcortex.csv', index_col=0)

# %% GET EXPRESSION FROM MATCHING REGIONS AND CREATE NULL GENE SET
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
receptor_genes = index_structure(receptor_genes, structure='CTX-SBCTX')

# load all genes from abagen
all_genes = pd.read_csv('data/abagen_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
all_genes = index_structure(all_genes, structure='CTX-SBCTX')

# load list of peptide genes
peptide_list = pd.read_csv('data/gene_list.csv')['Gene']

# get non-peptide genes
non_peptide_genes = all_genes.T[~all_genes.columns.isin(peptide_list)].T

# need distance matrix for null function
distance = np.load('data/template_parc-Schaefer400_TianS4_desc-distance.npy')

# generate nulls
nulls = gene_null_set(receptor_genes, non_peptide_genes, distance, n_permutations=10000, 
                      n_jobs=32, seed=1234)

nulls = np.array(nulls)
np.save('data/gene_null_sets_Schaefer400_TianS4.npy', nulls)

# %%
# reorder null genes to match HTH FC
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
null_ro = []
for null in nulls:
    # add last row of NaNs for HTH
    null = np.vstack([null, np.nan*np.ones(null.shape[1])])
    null_df = pd.DataFrame(null, index=receptor_genes.index)
    null_ro.append(reorder_subcortex(null_df, type='freesurfer', region_info=region_info).values)
null_ro = np.array(null_ro)

# %%
# reorder receptor genes to match HTH FC
receptor_genes_rs = reorder_subcortex(receptor_genes, type='freesurfer', region_info=region_info).values

# iterate through genes and compare to the hth_fc distribution
results = np.array([compare_images(receptor_genes_rs[:, i], hth_fc, nulls=null_ro[:,:,i].T, metric='spearmanr') 
                    for i in range(null_ro.shape[2])])

fc_gene_corr, p_values = results[:, 0], results[:, 1]

# %%
# build dataframe for results
df = pd.DataFrame({'Gene': receptor_genes.columns, 'HTH_corr': fc_gene_corr, 'p': p_values})
df['Sign'] = df['HTH_corr'] > 0

fig, ax = plt.subplots(figsize=(5,6), dpi=200)
palette = divergent_green_orange(n_colors=9, return_palette=True)
colors = [palette[1], palette[-2]]
plot_df = df.sort_values('HTH_corr', ascending=False).copy()
sns.barplot(data=plot_df, x='HTH_corr', y='Gene', hue='Sign', dodge=False, ax=ax, palette=colors)
# add asterisks on the side if p-value < 0.05
for i, p in enumerate(plot_df['p']):
    if p < 0.05:
        ax.text(plot_df['HTH_corr'].max()+0.1, i+0.6, '*', fontsize=12, color='black')

ax.set(xlabel=r'Similarity ($\rho$)', ylabel='Gene', title='Aligment with FC$_{Hypothalamus}$')
ax.legend_.remove()
fig.set_tight_layout(True)


# %% GENE COEXPRESSION

receptor_coexpression = receptor_genes.corr(method='spearman').values
np.fill_diagonal(receptor_coexpression, 0)
receptor_coexpression[receptor_coexpression < 0] = 0

ci, Q, zrand = modularity.consensus_modularity(receptor_coexpression, gamma=1, repeats=250, seed=1234)
plotting.plot_mod_heatmap(receptor_coexpression, ci, cmap=divergent_green_orange())
plt.title('Peptide receptor gene coexpression')

# add module assignment to HTH_corr dataframe
df['ci'] = ci
# reset order
df = df.sort_values('Gene')

# plot violinplot of HTH_corr split by ci
fig, ax = plt.subplots(figsize=(5,6), dpi=200)
palette = divergent_green_orange(n_colors=9, return_palette=True)
colors = [palette[1], palette[4], palette[7]]
sns.violinplot(data=df, x='ci', y='HTH_corr', ax=ax, palette=colors)
plt.ylabel('Similarity to FC$_{Hypothalamus}$')
plt.xlabel('Module')