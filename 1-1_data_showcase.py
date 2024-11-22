# %%
import pandas as pd
import numpy as np
from scipy import cluster
import seaborn as sns
import matplotlib.pyplot as plt
from plot_utils import divergent_green_orange

savefig = False

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                              LOAD DATA
###############################################################################
# load genes and gene list
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
receptor_names = receptor_genes.columns
receptor_list = pd.read_csv('data/receptor_list.csv')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                              RAW DATA PLOT
###############################################################################

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
receptor_genes.index = atlas_regions['id'].values

# group df first by structure, then network, and then name
atlas_regions = atlas_regions.groupby(['structure', 'network', 'name']).apply(lambda x: x).reset_index(drop=True)

# reorder receptor_genes according to atlas_regions
receptor_genes = receptor_genes.loc[atlas_regions['id']]

# map index back to name
receptor_genes.index = receptor_genes.index.map(atlas_regions.set_index('id')['name'])

# find where the structure changes
# create new list of structures
structures = [s[1]['name'].split('_')[0] 
              if s[1]['structure'] == 'cortex' 
              else s[1]['structure'].capitalize()
              for s in atlas_regions.iterrows()]
structures = np.array(structures)

# at which index is there a change in structures
change_ind = np.where(np.array(structures[:-1]) != np.array(structures[1:]))[0]

# plot clustermap and have the dendrogram on the same side of the xticks
clustermap = sns.clustermap(receptor_genes, cmap=divergent_green_orange(), col_cluster=True, row_cluster=False, 
                            xticklabels=True, cbar_kws={'label': 'gene expression'}, cbar_pos=None,
                            figsize=(14, 10))
                    
# create a map to assign each gene to a family
family_map = {gene: family for gene, family in zip(receptor_list['Gene'], receptor_list['Family']) \
              if gene in receptor_names}

# crate family-wise color map
families = set(family_map.values())
family_colors = sns.color_palette('tab20', n_colors=len(families))
family_color_map = {family: color for family, color in zip(families, family_colors)}

# color xticks by family in receptor_list
for i, label in enumerate(clustermap.ax_heatmap.get_xticklabels()):
    gene = label.get_text()
    family = family_map.get(gene)
    if family:
        label.set_color(family_color_map[family])

clustermap.ax_heatmap.set_yticks(change_ind)
clustermap.ax_heatmap.set_yticklabels(structures[change_ind]);

# draw lines at change_ind
for ind in change_ind:
    clustermap.ax_heatmap.axhline(ind, color='white', linewidth=3)
    clustermap.ax_heatmap.axhline(ind, color='black', linewidth=3, alpha=0.8)

if savefig:
    plt.savefig('./figs/genes_clustermap.pdf', bbox_inches='tight')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                            AVERAGE BY NETWORK
###############################################################################
# rename networks to differentiate between left and right hemisphere
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0).reset_index(drop=True)
atlas_regions = atlas_regions.sort_values('id').reset_index()
atlas_regions['network_alt'] = atlas_regions['hemisphere'] + '_' + atlas_regions['name'].str.split('_').str[0].replace(name_map)

# group by network and average
networks = atlas_regions['network_alt'].unique()
network_genes = {network: receptor_genes[atlas_regions['network_alt'] == network].mean(axis=0) \
                 for network in networks}
network_genes = pd.DataFrame(network_genes).T

# drop all right hemisphere networks
# network_genes = network_genes.loc[~network_genes.index.str.contains('R')]

# define order of networks alphabetically. start with cortex networks, then subcortex networks and then hypothalamus
network_order = [f'L_{ctx_net}' for ctx_net in ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Cont', 'Default', 'Limbic' ]] + \
                [f'L_{sbctx_net}' for sbctx_net in ['amygdala', 'caudate', 'globus-pallidus', 'hippocampus', 'nucleus-accumbens', 'putamen', 'thalamus']] + \
                ['B_hypothalamus']

# reorder network_genes
network_genes = network_genes.loc[network_order].T

# plot clustermap and have the dendrogram on the same side of the xticks
clustermap = sns.clustermap(network_genes, cmap=divergent_green_orange(), col_cluster=False, row_cluster=True, 
                            xticklabels=True, yticklabels=True, cbar_pos=None, figsize=(5, 11), 
                            linewidths=0.01, linecolor='white')
clustermap.figure.set_dpi(200)

# create a map to assign each gene to a family
family_map = {gene: family for gene, family in zip(receptor_list['Gene'], receptor_list['Family']) \
              if gene in receptor_names}

# create family-wise color map
# unique families
families = set(family_map.values())
family_colors = sns.color_palette('tab20', n_colors=len(families))
family_color_map = {family: color for family, color in zip(families, family_colors)}

# color xticks by family in receptor_list
for i, label in enumerate(clustermap.ax_heatmap.get_yticklabels()):
    gene = label.get_text()
    family = family_map.get(gene)
    if family:
        label.set_color(family_color_map[family])
        
# # show legend based on family_color_map
# legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=family, markerfacecolor=color, markersize=10) \
#                      for family, color in family_color_map.items()]
# clustermap.ax_heatmap.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1.05))
        
if savefig:
    plt.savefig('./figs/genes_network_clustermap.pdf')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                            BOXPLOT PER STRUCTURE
###############################################################################
# Reload data
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0).reset_index(drop=True)
atlas_regions = pd.read_csv('./data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv', index_col=0)
atlas_regions['structure'].iloc[-1] = 'hypothalamus'
receptor_genes['structure'] = atlas_regions['structure'].values

# create plot_df with long format. Should have structure and column names id variables
plot_df = receptor_genes.melt(id_vars='structure', var_name='gene', value_name='expression')

# create boxplot gene on y axis, expression on x axis and structure as hue
# use order from clustermap dendrogram
order_idx = clustermap.dendrogram_row.reordered_ind
order = receptor_genes.columns[order_idx]

# create copy with only hypothalamus
hth_df = plot_df[plot_df['structure'] == 'hypothalamus']
# drop hypothalamus from plot_df
plot_df = plot_df[plot_df['structure'] != 'hypothalamus']

# define colormap
green, orange = [color for i, color in enumerate(divergent_green_orange(n_colors=9, return_palette=True)) if i in [2, 4]]
yellow = sns.color_palette('Spectral')[2]
orange, yellow, green = [color for i, color in enumerate(sns.color_palette('Spectral')) if i in [1,2,4]]

fig, ax = plt.subplots(figsize=(4, 11), dpi=200)
sns.boxplot(data=plot_df, x='expression', y='gene', hue='structure', ax=ax, order=order, palette=[green, orange],
            hue_order=['cortex', 'subcortex'], showfliers=False, dodge=True, width=0.5, linewidth=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), frameon=False)

# add hypothalamus as pointplot between boxplots
sns.pointplot(data=hth_df, x='expression', y='gene', ax=ax, color=yellow, join=False, markers='o',
                scale=0.6, order=order)
sns.despine(ax=ax, trim=True)

if savefig:
    plt.savefig('./figs/expression_per_structure.pdf')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                            PLOT MEDIAN TRACE
###############################################################################
# median trace of the three structures
median = plot_df.groupby(['gene', 'structure']).median().reset_index()
median = median.pivot(index='gene', columns='structure', values='expression')
median['hypothalamus'] = hth_df['expression'].values
median = median.loc[order]

plt.figure(figsize=(11,1))
sns.lineplot(median, palette=[green, yellow, orange], linewidth=1, dashes=False)
plt.ylim([0,1])
plt.legend(bbox_to_anchor=(1.01, 1), frameon=False)
plt.xticks(rotation=90);
sns.despine(trim=True)

if savefig:
    plt.savefig('./figs/median_expression_trace.pdf')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#             SUPPLEMENTARY: RECEPTOR LOCATION IN PC SPACE
###############################################################################
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# load all abagen genes
all_genes = pd.read_csv('data/abagen_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)

# standardize data columnwise
scaler = StandardScaler()
genes_scaled = scaler.fit_transform(all_genes.values)

# fit PCA
pca = PCA(n_components=2)
pca.fit(genes_scaled)

# locate receptors in PCA space
ctx_idx = [i for i, gene in enumerate(all_genes.columns) if gene in receptor_names[order_idx][:16]]
sbctx_idx = [i for i, gene in enumerate(all_genes.columns) if gene in receptor_names[order_idx][16:]]
mchr_idx = [i for i, gene in enumerate(all_genes.columns) if gene=='MCHR1']
oxtr_idx = [i for i, gene in enumerate(all_genes.columns) if gene=='OXTR']

# plot first two PCs
plt.figure(figsize=(5,5))
sns.scatterplot(x=pca.components_[0], y=pca.components_[1], color='gray', linewidth=0, alpha=0.2)
sns.scatterplot(x=pca.components_[0][ctx_idx], y=pca.components_[1][ctx_idx], color=green, edgecolor='black')
sns.scatterplot(x=pca.components_[0][sbctx_idx], y=pca.components_[1][sbctx_idx], color=orange, edgecolor='black')
sns.scatterplot(x=pca.components_[0][mchr_idx], y=pca.components_[1][mchr_idx], color='red', edgecolor='black')
sns.scatterplot(x=pca.components_[0][oxtr_idx], y=pca.components_[1][oxtr_idx], color='blue', edgecolor='black')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(['all genes', 'cortical', 'subcortical'], bbox_to_anchor=(1.01, 1), frameon=False)
sns.despine()

if savefig:
    plt.savefig('./figs/pca.pdf')
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#             SUPPLEMENTARY: COMPARE PEPTIDE PCs TO GENE PCs
###############################################################################
# redo PCA for all genes
scaler = StandardScaler()
pca = PCA(n_components=3)

all_scaled = scaler.fit_transform(all_genes.values)
pca_all = pca.fit_transform(genes_scaled)[:, :3]

# PCA for peptide receptor genes
receptors__scaled = scaler.fit_transform(receptor_genes.drop('structure', axis=1).values)
pca_receptors = pca.fit_transform(receptors__scaled)[:,:3]

# correlate PCs between all genes and peptide receptor genes
# do it for all combinations of PCs
corrs = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        corrs[i,j] = np.corrcoef(pca_all[:,i], pca_receptors[:,j])[0,1]
        
# plot correlation matrix
plt.figure(figsize=(5,5),dpi=200)
sns.heatmap(corrs, annot=True, cmap=divergent_green_orange(), square=True, center=0)

# scatterplot of first PC for all genes and peptide receptor genes
network = atlas_regions['network']
plt.figure(figsize=(5,5),dpi=200)
sns.scatterplot(x=pca_all[:,0], y=pca_receptors[:,0], color='gray', alpha=0.5,
                hue=network, palette='tab10')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#               SUPPLEMENTARY: ALTERNATIVE CLUSTERMAP
###############################################################################
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import pdist
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
distance_matrix = pdist(receptor_genes.T.values, metric='euclidean')
Z = linkage(distance_matrix, method='average')
reordered_indices = leaves_list(Z)

receptors = receptor_genes.columns[reordered_indices]
family_map = {gene: family for gene, family in zip(receptor_list['Gene'], receptor_list['Family']) \
              if gene in receptors}

# print gene and family
for gene, family in family_map.items():
    print(f'{gene}: {family}')
