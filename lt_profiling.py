# %%
import pandas as pd
import numpy as np
import scipy.stats as sstats
import glob

# %%
layers = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
layer_data = np.load('data/annotations/bigbrain_layers_Schaefer400.npy')
# %%
gene_list = pd.read_csv('data/gene_list.csv')
receptor_labels = gene_list[gene_list['Description'].str.contains('receptor')]
receptor_labels = receptor_labels['Gene'].to_list() + ['SORT1'] 
receptor_labels.sort()

# load gene expression data
genes = pd.read_csv('data/peptide_genes_ahba_Schaefer400.csv', index_col=0)
genes = genes[genes.columns[genes.columns.isin(gene_list['Gene'])]]
receptor_genes = genes[genes.columns[genes.columns.isin(receptor_labels)]]

# %%
# correlate columns of receptor_genes with each row of layer_data
corrs = []

for i in range(layer_data.shape[0]):
    tmp_df = pd.Series(layer_data[i, :])
    corrs.append(receptor_genes.corrwith(tmp_df, method='spearman'))
l_df = pd.concat(corrs, axis=1)
l_df.columns = layers

# long format
l_df = l_df.stack().reset_index()
l_df.columns = ['receptor', 'layer', 'correlation']

# %%
# plot distribution of correlations for each layer in one figure
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure()
sns.violinplot(data=l_df, x='layer', y='correlation')
plt.title('Correlation of receptor genes with layer data')
plt.ylabel('Spearman correlation')
plt.xlabel('Layer')

# %%
# show top 10 genes with highest correlation with each layer
top_genes = l_df.groupby('layer').apply(lambda x: x.nlargest(10, 'correlation')).reset_index(drop=True)
top_genes
# %%
# show correlateions of SSTR1 with each layer
l_df[l_df['receptor'] == 'SSTR1']