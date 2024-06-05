# %%
import pandas as pd
import numpy as np
import abagen as abg
import seaborn as sns
import matplotlib.pyplot as plt
from utils import index_structure
from plot_utils import divergent_green_orange

# load structural and functional connectivity
sc = np.load('data/template_parc-Schaefer400_TianS4_desc-SC.npy')
fc = np.load('data/template_parc-Schaefer400_TianS4_desc-FC.npy')

# load genes and gene list
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)

nrois = len(receptor_genes)
triu_indices = np.triu_indices(nrois, k=1)

# %% RAW DATA PLOT

# load receptor list
receptor_list = pd.read_csv('data/receptor_list.csv')

# crate family-wise color map
families = receptor_list['Family'].unique()
family_colors = sns.color_palette('tab20', n_colors=len(families))
family_color_map = {family: color for family, color in zip(families, family_colors)}

# plot genes in clustermap
# show all column ticklabels
ax = sns.clustermap(receptor_genes, cmap=divergent_green_orange(), col_cluster=True, row_cluster=False, 
                    xticklabels=True, cbar_kws={'label': 'gene expression'},
                    figsize=(14, 10))

# color xticks by family in receptor_list
for i, label in enumerate(ax.ax_heatmap.get_xticklabels()):
    gene = label.get_text()
    family = receptor_list[receptor_list['Gene'] == gene]['Family'].values
    if len(family) > 0:
        family = family[0]
        fcolor = family_color_map[family]
        label.set_color(fcolor)


# %% CGE PLOTS

# create SC nulls
n_nulls = 1000

if os.path.exists('data/sc_nulls_Schaefer400_TianS4.npy'):
    sc_nulls = np.load('data/sc_nulls_Schaefer400_TianS4.npy')
else:
    nulls = Parallel(n_jobs=20) \
            (delayed(strength_preserving_rand_sa)(sc, seed=i) 
            for i in range(n_nulls))

    sc_nulls = []
    for null in nulls:
        sc_nulls.append(null[0])
    sc_nulls = np.array(sc_nulls)
    np.save('data/sc_nulls_Schaefer400_TianS4.npy', sc_nulls)


# %% Structural connectivity
# create correlated gene expression (cge) matrix
cge_receptors = receptor_genes.T.corr(method='spearman').values

# nan diagonal
np.fill_diagonal(cge_receptors, np.nan)

# use sc to divide cge into sc and non-sc
# this is empirical
cge_sc = (cge_receptors * sc)[triu_indices]
cge_nsc = (cge_receptors * (1 - sc))[triu_indices]

# remove zeros and nans
cge_sc = cge_sc[cge_sc != 0]
cge_nsc = cge_nsc[cge_nsc != 0]
cge_sc = cge_sc[~np.isnan(cge_sc)]
cge_nsc = cge_nsc[~np.isnan(cge_nsc)]

cge_sc_nulls = []
cge_nsc_nulls = []

# divide cge into sc and non-sc using SC nulls
for null in sc_nulls:
    cge_sc_null = (cge_receptors * null)[triu_indices]
    cge_nsc_null = (cge_receptors * (1 - null))[triu_indices]

    cge_sc_nulls.append(cge_sc_null)
    cge_nsc_nulls.append(cge_nsc_null)


# convert to numpy and flatten
cge_sc_nulls = np.array(cge_sc_nulls).flatten()
cge_nsc_nulls = np.array(cge_nsc_nulls).flatten()

# remove zeros and nans
cge_sc_nulls = cge_sc_nulls[cge_sc_nulls != 0]
cge_nsc_nulls = cge_nsc_nulls[cge_nsc_nulls != 0]
cge_sc_nulls = cge_sc_nulls[~np.isnan(cge_sc_nulls)]
cge_nsc_nulls = cge_nsc_nulls[~np.isnan(cge_nsc_nulls)]

# plot comparison of cge_sc and cge_nsc for null and empirical sc in separate plots
plt.figure(figsize=(6, 5), dpi=150)
plt.subplot(1, 2, 1)
ax1 = sns.boxplot(data=[cge_sc, cge_nsc])
plt.title('empirical')
plt.ylabel('correlated gene expression')
plt.xticks([0, 1], ['sc', 'non-sc'])
sns.despine(ax=ax1)

plt.subplot(1, 2, 2)
ax2 = sns.boxplot(data=[cge_sc_nulls, cge_nsc_nulls])
plt.title('null')
plt.xticks([0, 1], ['sc', 'non-sc'])
plt.yticks([])
sns.despine(ax=ax2, left=True)


# %% Functional connectivity
# compare fc and cge in lmplot
np.fill_diagonal(fc, np.nan)
fc_flat = fc[triu_indices]
cger_flat = cge_receptors[triu_indices]
cgeo_flat = other_genes[triu_indices]


cger_fc_df = pd.DataFrame({'fc': fc_flat, 'cge': cger_flat, 'type': 'receptor'})
cgeo_fc_df = pd.DataFrame({'fc': fc_flat, 'cge': cgeo_flat, 'type': 'other'})

fc_df = pd.concat([cger_fc_df, cgeo_fc_df])

# drop nan and zero values
fc_df = fc_df.dropna()
fc_df = fc_df[fc_df['fc'] != 0]

# plot
plt.figure(figsize=(10, 5))
sns.lmplot(x='fc', y='cge', hue='type', data=fc_df, scatter_kws={'s': 2, 'alpha': 0.4})
plt.ylabel('correlated gene expression')

