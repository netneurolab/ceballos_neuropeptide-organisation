# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from plot_utils import divergent_green_orange

savefig = False 

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                              LOAD HPA DATA
###############################################################################
# load hpa whole brain and keep only gene region and ntpm columns
hpa = pd.read_csv('data/hpa_whole-brain.tsv', sep='\t')
hpa = hpa.rename(columns={'Gene name': 'gene', 'Subregion': 'region', 'nTPM': 'ntpm'})
hpa = hpa[['gene', 'region', 'ntpm']]

# clean up hpa and keep only peptide genes
gene_list = pd.read_csv('data/gene_list.csv')
hpa = hpa[hpa['gene'].isin(gene_list['Gene'])]
hpa = hpa[~hpa['region'].str.contains('white matter')]

# pivot df to have genes as columns and regions as rows
hpa = pd.pivot(hpa, columns='gene', index='region', values='ntpm')
hpa = np.log10(hpa + 1)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#               MAP HPA TO DESTRIEUX AND MATCH WITH AHBA DATA
###############################################################################
# load destrieux_labels in data
destrieux_labels = pd.read_csv('data/parcellations/destrieux_labels.csv')
destrieux_labels = destrieux_labels[destrieux_labels['hemisphere'] == 'R']

# read hpa to destrieux mapping
hpa_map = pd.read_csv('data/parcellations/hpa_destrieux_map.csv')
hpa_map['Destrieux id'] = hpa_map['Destrieux region'].map(destrieux_labels.set_index('label')['id'].to_dict())

# load ahba genes
ahba = pd.read_csv('data/abagen_genes_Destrieux.csv', index_col=0)

# index matching regions in hpa and ahba
hpa = hpa.loc[hpa_map['HPA region']]
ahba = ahba.loc[hpa_map['Destrieux id']]

# keep only genes found in both datasets
overlapping_genes = hpa.columns.intersection(ahba.columns)
hpa = hpa[overlapping_genes]
ahba = ahba[overlapping_genes]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                PLOT SCATTERPLOT OF HPA AND AHBA VALUES
###############################################################################
palette = divergent_green_orange(n_colors=9, return_palette=True)
bipolar = [palette[2], palette[-3]]

# do a five by five grid where each cell is a scatterplot of hpa and ahba values in a region
fig, axes = plt.subplots(5, 5, figsize=(25, 25), dpi=150)
plt.rcParams.update({'font.size': 15})
for i, ax in enumerate(axes.flat):
    r, p = spearmanr(hpa.iloc[i], ahba.iloc[i])
    if p > 0.05:
        color = bipolar[0]
    else:
        color = bipolar[1]
    sns.regplot(x=hpa.iloc[i].values, y=ahba.iloc[i].values, ax=ax, color=color)
    ax.set_title(f'{hpa.index[i]}\nr={r:.2f}, p={p:.2f}')
    ax.set_xlabel('HPA')
    ax.set_ylabel('AHBA')
fig.set_tight_layout(True)

if savefig:
    plt.savefig('figs/hpa_ahba_scatterplot.pdf')

