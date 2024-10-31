# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pyls import behavioral_pls
from scipy.stats import zscore, spearmanr
from plot_utils import divergent_green_orange, split_barplot

savefig = False

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                              LOAD DATA
###############################################################################
# load genes and gene list
receptor_genes = pd.read_csv('data/receptor_gene_expression_Cammoun2012_250_7N_Freesurfer_Subcortex.csv', index_col=0)
receptor_names = receptor_genes.columns
receptor_list = pd.read_csv('data/receptor_list.csv')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                              RAW DATA PLOT
###############################################################################

# load region names
atlas_regions = pd.read_csv('./data/parcellations/Cammoun2012_7N_Freesurfer_Subcortex_LUT.csv')

# only select scale250 in column scale
atlas_regions = atlas_regions[atlas_regions['scale'] == 'scale250'].iloc[:-1]
# rename label in atlas_info to name and yeo_7 to network
atlas_regions = atlas_regions.rename(columns={'label': 'name', 'yeo_7': 'network'})
atlas_regions = atlas_regions.groupby(['structure', 'network', 'name']).apply(lambda x: x).reset_index(drop=True)
atlas_regions['name_alt'] = atlas_regions['hemisphere'] + '_' + atlas_regions['name']
atlas_regions['network_alt'] = atlas_regions.apply(lambda x: x['name'] if x['structure'] == 'subcortex' else x['network'], axis=1)
atlas_regions['network_alt'] = atlas_regions['hemisphere'] + '_' + atlas_regions['network_alt']


# reorder receptor_genes according to atlas_regions
receptor_genes = receptor_genes.loc[atlas_regions['name_alt']]

# average by network
networks = atlas_regions['network_alt'].unique()
network_genes = {network: receptor_genes.loc[(atlas_regions['network_alt'] == network).values].mean(axis=0) \
                    for network in networks}

# define network order
network_order = [f'L_{ctx_net}' for ctx_net in ['visual', 'somatomotor', 
                                                'dorsal attention', 'ventral attention', 
                                                'frontoparietal', 'default mode', 'limbic']] + \
                [f'L_{sbctx_net}' for sbctx_net in ['accumbensarea', 'amygdala', 'caudate', 'hippocampus', 
                                                    'pallidum', 'putamen', 'thalamusproper']]
network_genes = pd.DataFrame(network_genes).T
network_genes = network_genes.loc[network_order]

# load schaefer gene expression order
order = np.load('results/gene_expression_cluster_order.npy')

# swap columns in network_genes to match order
network_genes = network_genes.iloc[:, order]

# plot heatmap
plt.figure(figsize=(5, 11), dpi=200)
sns.heatmap(network_genes.T, cmap=divergent_green_orange(), cbar_kws={'label': 'gene expression', 'shrink': 0.5},
            xticklabels=True, yticklabels=True, square=True, linewidths=0.01, linecolor='white', vmin=0, vmax=1)

if savefig:
    plt.savefig('./figs/genes_heatmap_cammoun.pdf')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#       SUPPLEMENTARY: COMPARE PCA LOADINGS TO SCHAEFER 400 ATLAS
###############################################################################
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

# load all abagen genes
schaefer = pd.read_csv('data/abagen_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
cammoun = pd.read_csv('data/abagen_gene_expression_Cammoun2012_250_7N_Freesurfer_Subcortex.csv', index_col=0)

# standardize data columnwise
scaler = StandardScaler()
schaefer_scaled = scaler.fit_transform(schaefer.values)
cammoun_scaled = scaler.fit_transform(cammoun.values)

# fit PCA
schaefer_pca = PCA(n_components=2)
schaefer_pca.fit(schaefer_scaled)
cammoun_pca = PCA(n_components=2)
cammoun_pca.fit(cammoun_scaled)

# compare loadings of gene PCs
schaefer_loadings = schaefer_pca.components_
cammoun_loadings = cammoun_pca.components_


# do four subplots showing a scatterplot of the first two PCs of schaefer and cammoun
fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=200)
for i in range(2):
    for j in range(2):
        # use regplot to show correlation
        sns.regplot(x=schaefer_loadings[j], y=cammoun_loadings[i], ax=axs[i, j], color='gray', scatter_kws={'s': 1})
        r, p = spearmanr(schaefer_loadings[j], cammoun_loadings[i])
        axs[i, j].set_title(f'Loading correlation r={r:.2f}')
        axs[i, j].set_xlabel(f'Schaefer PC{j+1} loadings')
        axs[i, j].set_ylabel(f'Cammoun PC{i+1} loadings')
        
        
# set tight layout
plt.tight_layout()
sns.despine(trim=True)

if savefig:
    plt.savefig('./figs/cammoun_scahefer_genes_pca_loadings_correlation.pdf')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#       SUPPLEMENTARY: COMPARE RECEPTORS ONLY
###############################################################################
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# load all abagen genes
schaefer = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
cammoun = pd.read_csv('data/receptor_gene_expression_Cammoun2012_250_7N_Freesurfer_Subcortex.csv', index_col=0)

# standardize data columnwise
scaler = StandardScaler()
schaefer_scaled = scaler.fit_transform(schaefer.values)
cammoun_scaled = scaler.fit_transform(cammoun.values)

# fit PCA
schaefer_pca = PCA(n_components=10)
schaefer_pca.fit(schaefer_scaled)
cammoun_pca = PCA(n_components=10)
cammoun_pca.fit(cammoun_scaled)

# compare loadings of gene PCs
schaefer_loadings = schaefer_pca.components_
cammoun_loadings = cammoun_pca.components_

# plot loadings
from scipy.stats import spearmanr
# do four subplots showing a scatterplot of the first two PCs of schaefer and cammoun
fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=200)
for i in range(2):
    for j in range(2):
        axs[i, j].scatter(cammoun_loadings[j], schaefer_loadings[i], s=1)
        axs[i, j].set_xlabel(f'Schaefer PC{i+1}')
        axs[i, j].set_ylabel(f'Cammoun PC{j+1}')
        r, p = spearmanr(schaefer_loadings[i], cammoun_loadings[j])
        axs[i, j].set_title(f'Loading correlation r={r:.2f}')
        
# set tight layout
plt.tight_layout()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                   SUPPLEMENTARY: COMPARE PLS WEIGHTS
###############################################################################

# Load receptor gene expression
cammoun = pd.read_csv('data/receptor_gene_expression_Cammoun2012_250_7N_Freesurfer_Subcortex.csv', index_col=0)

# Load neurosynth
cammoun_ns = pd.read_csv('data/neurosynth/derivatives/Cammoun2012_7N_Freesurfer_Subcortex_neurosynth.csv', index_col=0)
cammoun_ns = cammoun_ns.iloc[:-1] # no brainstem

atlas_regions = pd.read_csv('data/parcellations/Cammoun2012_7N_Freesurfer_Subcortex_LUT.csv')
atlas_regions = atlas_regions[atlas_regions['scale'] == 'scale250']
atlas_regions.set_index('label', inplace=True)

# join hemisphere and label to create label_alt
atlas_regions['label_alt'] = atlas_regions['hemisphere'] + '_' + atlas_regions.index

# rename cammoun_ns index to label_alt
cammoun_ns.index = atlas_regions['label_alt'].iloc[:-1]

# order cammoun_ns according to cammoun index
cammoun_ns = cammoun_ns.loc[cammoun.index]

X = zscore(cammoun_ns.values, axis=0)
Y = zscore(cammoun.values, axis=0)

pls_results = behavioral_pls(X, Y, n_boot=0, n_perm=0, rotate=True, permsamples=None,
                                permindices=False, test_split=0, seed=0)

schaefer_pls_result = np.load('results/pls_result_Schaefer400_TianS4_HTH.npy', allow_pickle=True).item()

# compare PLS1 weights
r, p = spearmanr(pls_results['x_weights'][:, 0], schaefer_pls_result['x_weights'][:, 0])
plt.figure(figsize=(5,5),dpi=200)
sns.regplot(x=pls_results['x_weights'][:, 0], y=schaefer_pls_result['x_weights'][:, 0], color='gray', scatter_kws={'s': 10})
plt.title(f'Term maps\nSpearman r={r:.2f}')
plt.xlabel('Weights using anatomical Cammoun atlas')
plt.ylabel('Weights using functional Schaefer atlas')
sns.despine()

if savefig:
    plt.savefig('./figs/cammoun_schaefer_pls_weights_terms.pdf')


r, p = spearmanr(pls_results['y_weights'][:, 0], schaefer_pls_result['y_weights'][:, 0])
plt.figure(figsize=(5,5),dpi=200)
sns.regplot(x=pls_results['y_weights'][:, 0], y=schaefer_pls_result['y_weights'][:, 0], color='gray', scatter_kws={'s': 10})
plt.xlabel('Weights using anatomical Cammoun atlas')
plt.ylabel('Weights using functional Schaefer atlas')
plt.title(f'Receptor maps\nSpearman r={r:.2f}')
sns.despine()

if savefig:
    plt.savefig('./figs/cammoun_schaefer_pls_weights_receptor.pdf')
    
# %%
# create receptor_df with receptor names and y_loadings
receptor_df = pd.DataFrame({'receptor': receptor_names, 'loading': pls_results['y_loadings'][:, 0]})

# create term_df with term names and y_loadings of pls_result_X
pls_result_X = behavioral_pls(Y, X, n_boot=0, n_perm=0, rotate=True, permsamples=None,
                                permindices=False, test_split=0, seed=0)

term_df = pd.DataFrame({'term': cammoun_ns.columns, 'loading': pls_result_X['y_loadings'][:, 0]})



fig, axes = split_barplot(term_df, x='loading', y='term', top=15,
                          figsize=(8, 5), dpi=200)

fig, axes = split_barplot(receptor_df, x='loading', y='receptor', top=15,
                          figsize=(8, 5), dpi=200)
