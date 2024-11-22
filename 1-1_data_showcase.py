# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot_utils import divergent_green_orange

savefig = False

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                              LOAD DATA
###############################################################################
# load genes and gene list
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0).reset_index(drop=True)
receptor_names = receptor_genes.columns
receptor_list = pd.read_csv('data/receptor_overview.csv')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                     ASSIGN ATLAS REGIONS TO NETWORKS
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

# group df first by structure, then network, and then name
atlas_regions = atlas_regions.groupby(['structure', 'network', 'name']).apply(lambda x: x).reset_index(drop=True)

# create network label with hemisphere
atlas_regions = atlas_regions.sort_values('id').reset_index()
atlas_regions['network_alt'] = atlas_regions['hemisphere'] + '_' + atlas_regions['name'].str.split('_').str[0].replace(name_map)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                            AVERAGE BY NETWORK
###############################################################################

# group by receptor data by network and average per network
networks = atlas_regions['network_alt'].unique()
network_genes = {network: receptor_genes[atlas_regions['network_alt'] == network].mean(axis=0) \
                 for network in networks}
network_genes = pd.DataFrame(network_genes).T

# drop all right hemisphere networks
network_genes = network_genes.loc[~network_genes.index.str.contains('R')]

# define order of networks
network_order = [f'L_{ctx_net}' for ctx_net in ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Cont', 'Default', 'Limbic' ]] + \
                [f'L_{sbctx_net}' for sbctx_net in ['amygdala', 'caudate', 'globus-pallidus', 'hippocampus', 'nucleus-accumbens', 'putamen', 'thalamus']] + \
                ['B_hypothalamus']

# reorder data and transpose to have genes as rows
network_genes = network_genes.loc[network_order].T

# plot clustermap and have the dendrogram on the same side of the xticks
clustermap = sns.clustermap(network_genes, cmap=divergent_green_orange(), col_cluster=False, row_cluster=True, 
                            xticklabels=True, yticklabels=True, cbar_pos=None, figsize=(5, 11), 
                            linewidths=0.01, linecolor='white')
clustermap.figure.set_dpi(200)

# create a map to assign each gene to a family
family_map = {gene: family for gene, family in zip(receptor_list['gene'], receptor_list['family']) \
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
