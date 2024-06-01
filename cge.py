# %%
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from netneurotools import modularity

plot = False

# load schaefer 2018 atlas info
atlas_info = pd.read_pickle('data/parcellations/Schaefer2018_400_LUT.pkl')

# read agaben schaefer 2018 gene expression data
all_genes = pd.read_csv('data/abagen_genes_Schaefer2018_400.csv')
# load gene list
gene_list = pd.read_csv('data/receptor_list.csv')

# %%
# find overlapping genes between gene list and abagen data
available_peptides = all_genes.columns[all_genes.columns.isin(gene_list['Gene'])]
n_genes = len(available_peptides)
peptide_genes = all_genes[available_peptides]

# calculate correlation between peptide genes
cge = peptide_genes.T.corr(method='spearman').values
non_peptides = all_genes.columns[~all_genes.columns.isin(gene_list['Gene'])]

# %% Community detection
corr = cge.copy()
corr[corr < 0] = 0

ci, Q, zrand = modularity.consensus_modularity(corr, gamma=1, seed=1234, repeats=100)

# see which genes are in which module according to ci
networks = atlas_info['network']
modules = pd.DataFrame({'module': ci, 'network': networks})

# plot module assignment using bar chart
modules = modules.groupby(['module', 'network']).size().reset_index(name='count')
modules = modules.pivot(index='module', columns='network', values='count')
modules = modules.div(modules.sum(axis=1), axis=0)
# fill nan with 0
modules = modules.fillna(0)

# plot the average network distribution in each module as bar chart
if plot:
    plt.figure()
    modules = modules[networks.unique()]
    modules.plot(kind='bar', stacked=True)
    plt.legend(title='Network', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Network distribution in modules (peptide receptors)')
    plt.xlabel('Module')

####################
# %% Random genes  #
####################

# cge from randomly selected genes
n_permutations = 1000
random_cge = []
for _ in range(n_permutations):
    random_genes = all_genes[non_peptides].sample(n=n_genes, axis=1, random_state=1234)
    random_cge.append(random_genes.T.corr(method='spearman').values)

# %% Contrast with community detection of random gene expression data
rnd_ci_fn = 'results/clustering_random_ci.npy'

def module_assignment(mat):
    corr = mat.copy()
    corr[corr < 0] = 0
    ci, _, _ = modularity.consensus_modularity(corr, gamma=1, seed=1234, repeats=100)
    return ci

from joblib import Parallel, delayed

if os.path.exists(rnd_ci_fn):
    random_cis = np.load(rnd_ci_fn)
else:
    random_cis = Parallel(n_jobs=-1)(delayed(module_assignment)(random_corr) for random_corr in random_cge)
    random_cis = np.array(random_cis)
    np.save(rnd_ci_fn, random_cis)
    

# %%
# compare the network distribution of the real gene expression data with the random gene expression data
random_modules = []
for ci in random_cis:
    random_modules.append(pd.DataFrame({'module': ci, 'network': networks}))

random_modules = pd.concat(random_modules)

# average count of network in each module
random_modules = random_modules.groupby(['module', 'network']).size().reset_index(name='count')
random_modules = random_modules.pivot(index='module', columns='network', values='count')
random_modules = random_modules.div(random_modules.sum(axis=1), axis=0)
# fill nan with 0
random_modules = random_modules.fillna(0)

# plot the average network distribution in each module as bar chart
# keep the same network order as the real gene expression data
if plot:
    plt.figure()
    random_modules = random_modules[networks.unique()]
    random_modules.plot(kind='bar', stacked=True)
    plt.legend(title='Network', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Network distribution in modules (random gene expression data)')
    plt.xlabel('Module')

#########################
# %% Spin RSN  labels   #
#########################

n_permutations = 1000
spins = np.load('data/vasa_schaefer400_fsaverage_spin_indices.npy')[:, :n_permutations]

# get correlated gene expression from receptor genes
corr = cge.copy()
corr[corr < 0] = 0

ci, Q, zrand = modularity.consensus_modularity(corr, gamma=1, seed=1234, repeats=100)


# see which genes are in which module according to ci
networks = atlas_info['network']

# spin network labels
spun_modules = []
for spin in spins.T:
    spun_networks = networks[spin].values
    spun_modules.append(pd.DataFrame({'module': ci, 'network': spun_networks}))

spun_modules = pd.concat(spun_modules)

# plot module assignment using bar chart
spun_modules = spun_modules.groupby(['module', 'network']).size().reset_index(name='count')
spun_modules = spun_modules.pivot(index='module', columns='network', values='count')
spun_modules = spun_modules.div(spun_modules.sum(axis=1), axis=0)
# fill nan with 0
spun_modules = spun_modules.fillna(0)

# plot the average network distribution in each module as bar chart
# keep the same network order as the real gene expression data
if plot:
    plt.figure()
    spun_modules = spun_modules[networks.unique()]
    spun_modules.plot(kind='bar', stacked=True)
    plt.legend(title='Network', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Network distribution in modules (randomized network labels)')
    plt.xlabel('Module')

