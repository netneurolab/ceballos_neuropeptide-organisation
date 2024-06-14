# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from utils import index_structure, reorder_subcortex, gene_null_set
from plot_utils import divergent_green_orange
from netneurotools import plotting, modularity

# %% LOAD BRAINSTEM FUNCTIONAL CONNCETIVITY DATA
# open brainstem data under data/brainstemfc_mean_corrcoeff_full.npy
bs = np.load('data/brainstemfc_mean_corrcoeff_full.npy')

# load region_info
region_info = pd.read_csv('data/parcellations/Schaefer2018_400_region_info_brainstem_subcortex.csv', index_col=0)

# find hypothalamus stored as HTH
hth_idx = region_info[region_info['labels'] == 'HTH'].index[0]

# find cerebellum index and drop from region_info
cerebellum_idx = region_info[region_info['labels'].str.contains('Cereb-Ctx')].index
region_info = region_info.drop(cerebellum_idx)

# find subcortex and cortex index and put them together
cortex_idx = region_info[region_info['structure'] == 'cortex'].index.values
subcortex_idx = region_info[region_info['structure'] == 'subcortex'].index.values
idx = np.concatenate([subcortex_idx, cortex_idx])

# store functional connectivity between hypothalamus and subcortex/cortex
hth_fc = bs[hth_idx, idx]


# %% GET EXPRESSION FROM MATCHING REGIONS
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
receptor_genes = reorder_subcortex(receptor_genes, type='freesurfer', region_info=region_info)

# %% 
# correlation between hypothalamus functional connectivity and gene expression
fc_gene_corr = np.array([spearmanr(hth_fc, receptor_genes[gene])[0] for gene in receptor_genes.columns])

# plot correlation values in descending order
df = pd.DataFrame(np.vstack([receptor_genes.columns, fc_gene_corr]).T, columns=['Gene', 'HTH_corr'])
df['HTH_corr'] = df['HTH_corr'].astype(float)
df['Sign'] = df['HTH_corr'] > 0
df = df.sort_values('HTH_corr', ascending=False)
df['Gene'] = df['Gene'].str.replace(' ', '\n')

fig, ax = plt.subplots(figsize=(5,6), dpi=200)
palette = divergent_green_orange(n_colors=9, return_palette=True)
colors = [palette[1], palette[-2]]
sns.barplot(data=df, x='HTH_corr', y='Gene', hue='Sign', dodge=False, ax=ax, palette=colors)
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


# %% CREATE NULL GENE SET

# load all genes from abagen
all_genes = pd.read_csv('data/abagen_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
all_genes = index_structure(all_genes, structure='CTX-SBCTX')
# load list of peptide genes
peptide_list = pd.read_csv('data/gene_list.csv')['Gene']
# get non-peptide genes
non_peptide_genes = all_genes.T[~all_genes.columns.isin(peptide_list)].T
# non_peptide_genes = reorder_subcortex(non_peptide_genes, type='freesurfer', region_info=region_info)

# load distance matrix
distance = np.load('data/template_parc-Schaefer400_TianS4_desc-distance.npy')
nulls = gene_null_set(receptor_genes, non_peptide_genes, distance, n_permutations=1000, seed=1234)
