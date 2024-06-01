# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# %%
# open brainstem data under data/brainstemfc_mean_corrcoeff_full.npy
bs = np.load('data/brainstemfc_mean_corrcoeff_full.npy')
# load region_info
region_info = pd.read_csv('data/parcellations/Schaefer2018_400_region_info_brainstem_subcortex.csv', index_col=0)
# find hypothalamus stored as HTH
ht_idx = region_info[region_info['labels'] == 'HTH'].index[0]
# find cortex index
cortex_idx = region_info[region_info['structure'] == 'cortex'].index.values

# locate hypothalamus to cortex functional connectivity
ht_ctx_fc = bs[ht_idx, cortex_idx]

# %% 
# load gene_list
gene_list = pd.read_csv('data/gene_list.csv')
receptor_labels = gene_list[gene_list['Description'].str.contains('receptor')]
receptor_labels = receptor_labels.append({'Gene': 'SORT1'}, ignore_index=True)
receptor_labels = receptor_labels['Gene'].values

# load gene expression data
genes = pd.read_csv('data/peptide_genes_ahba_Schaefer400.csv', index_col=0)
# select columns that are in receptor_labels
receptor_genes = genes[genes.columns[genes.columns.isin(receptor_labels)]]

# correlation between hypothalamus to cortex functional connectivity and gene expression
fc_gene_corr = np.array([spearmanr(ht_ctx_fc, receptor_genes[gene])[0] for gene in receptor_genes.columns])

# %%
# plot correlation values from top 20 and bottom 20 as barplot, split by positive and negative correlations
top = np.argsort(fc_gene_corr)[-20:]
bottom = np.argsort(fc_gene_corr)[:20]

df = pd.DataFrame(np.vstack([receptor_genes.columns, fc_gene_corr]).T, columns=['Gene', 'HTH_corr'])
df = df.loc[np.concatenate([top, bottom])]
df['HTH_corr'] = df['HTH_corr'].astype(float)
df['Sign'] = df['HTH_corr'] > 0
df = df.sort_values('HTH_corr', ascending=False)
df['Gene'] = df['Gene'].str.replace(' ', '\n')

fig, ax = plt.subplots(figsize=(5,6), dpi=150)
sns.barplot(data=df, x='HTH_corr', y='Gene', hue='Sign', dodge=False, ax=ax)
ax.set(xlabel='Spearman correlation', ylabel='Gene', title='Top and bottom 20 gene-fc correlations')
ax.legend_.remove()
fig.set_tight_layout(True)
