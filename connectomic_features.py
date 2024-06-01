# %%
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from netneurotools import modularity
from bct import participation_coef

plot = True

# load structural and functional connectivity
sc = np.load('data/template_parc-Schaefer400_desc-SC_wei.npy')
fc = np.load('data/template_parc-Schaefer400_desc-FC.npy')

# load schaefer 2018 atlas info
atlas_info = pd.read_pickle('data/parcellations/Schaefer2018_400_LUT.pkl')

# read agaben schaefer 2018 gene expression data
all_genes = pd.read_csv('data/abagen_genes_Schaefer2018_400.csv')
# load receptor list
receptor_list = pd.read_csv('data/receptor_list.csv')
receptor_labels = receptor_list['Gene'].to_list()
receptor_genes = all_genes[all_genes.columns[all_genes.columns.isin(receptor_labels)]]


# %% create features
# participation coefficient
corr = fc.copy()
corr[corr < 0] = 0
ci, Q, zrand =  (ci, Q, zrand) if 'ci' in globals() else \
                modularity.consensus_modularity(corr, gamma=1, seed=1234, repeats=100)
pc = participation_coef(fc, ci)

# hubs
hubs_fc = fc.sum(axis=0)
hubs_sc = sc.sum(axis=0)

# distance
dist = np.load('data/template_parc-Schaefer400_desc-distance.npy')
sc_bin = np.load('data/template_parc-Schaefer400_desc-SC.npy')
dist = (dist * sc_bin)
dist[dist == 0] = np.nan
dist = np.nanmean(dist, axis=0)

# %% Compare features with receptor gene data
# calculate correlation between receptor genes and features
pc_corr = receptor_genes.corrwith(pd.Series(pc), method='spearman', axis=0)
hubs_fc_corr = receptor_genes.corrwith(pd.Series(hubs_fc), method='spearman', axis=0)
hubs_sc_corr = receptor_genes.corrwith(pd.Series(hubs_sc), method='spearman', axis=0)
dist_corr = receptor_genes.corrwith(pd.Series(dist), method='spearman', axis=0)

corr_df = pd.DataFrame({'PartCoeff': pc_corr.values, 
                        'HubsFC': hubs_fc_corr.values, 
                        'HubsSC': hubs_sc_corr.values, 
                        'Distance': dist_corr.values})
corr_df.index = dist_corr.index

families = receptor_list['Family']
families.index = receptor_list['Gene']
corr_df['Family'] = families[corr_df.index]


# plot distributions of correlations by family
# sort boxplots from highest to lowest median correlation
if plot:
    plt.figure()
    order = corr_df.groupby('Family')['PartCoeff'].median().sort_values(ascending=False).index
    sns.boxplot(data=corr_df, x='Family', y='PartCoeff', order=order, color='skyblue')
    plt.title('Correlation between receptor genes and participation coefficient')
    plt.ylabel('Correlation')
    plt.xticks(rotation=90)
    
    plt.figure()
    order = corr_df.groupby('Family')['HubsFC'].median().sort_values(ascending=False).index
    sns.boxplot(data=corr_df, x='Family', y='HubsFC', order=order, color='skyblue')
    plt.title('Correlation between receptor genes and functional connectivity hubs')
    plt.ylabel('Correlation')
    plt.xticks(rotation=90)
    
    plt.figure()
    order = corr_df.groupby('Family')['HubsSC'].median().sort_values(ascending=False).index
    sns.boxplot(data=corr_df, x='Family', y='HubsSC', order=order, color='skyblue')
    plt.title('Correlation between receptor genes and structural connectivity hubs')
    plt.ylabel('Correlation')
    plt.xticks(rotation=90)
    
    plt.figure()
    order = corr_df.groupby('Family')['Distance'].median().sort_values(ascending=False).index
    sns.boxplot(data=corr_df, x='Family', y='Distance', order=order, color='skyblue')
    plt.title('Correlation between receptor genes and distance')
    plt.ylabel('Correlation')
    plt.xticks(rotation=90)

# %% long range connections
# find long range connections
lengths = (dist * sc_bin)
lengths = np.where(lengths > 0, lengths, np.nan)
plt.figure()
sns.histplot(lengths.flatten(), kde=True, bins=20)

cge = receptor_genes.T.corr(method='spearman').values
long_range_corr = cge[lengths > np.nanpercentile(lengths, 90)]
long_range_corr = long_range_corr[~np.isnan(long_range_corr)]

short_range_corr = cge[lengths <= np.nanpercentile(lengths, 90)]
short_range_corr = short_range_corr[~np.isnan(short_range_corr)]

# compare in violin plot
if plot:
    plt.figure()
    sns.violinplot(data=[long_range_corr, short_range_corr])
    plt.title('Correlation between receptor genes')
    plt.ylabel('Correlated gene expression')
    plt.xticks([0, 1], ['Long-range connections\n(top 10%)', 'Rest'])

# %%
