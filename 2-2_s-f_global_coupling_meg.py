# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sstats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from utils import index_structure, non_diagonal_elements, communication_measures
from plot_utils import divergent_green_orange

savefig = False

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                              LOAD DATA
###############################################################################
# load template matrices
delta_mat = np.load("data/annotations/meg_delta_mat_Schaefer400.npy")
theta_mat = np.load("data/annotations/meg_theta_mat_Schaefer400.npy")
alpha_mat = np.load("data/annotations/meg_alpha_mat_Schaefer400.npy")
beta_mat = np.load("data/annotations/meg_beta_mat_Schaefer400.npy")
lgamma_mat = np.load("data/annotations/meg_lgamma_mat_Schaefer400.npy")
hgamma_mat = np.load("data/annotations/meg_hgamma_mat_Schaefer400.npy")
meg_bands = np.stack([delta_mat, theta_mat, alpha_mat, beta_mat, lgamma_mat, hgamma_mat])

dist_mat = np.load("data/template_parc-Schaefer400_TianS4_desc-distance.npy")[54:,54:]
sc = np.load("data/template_parc-Schaefer400_TianS4_desc-SC.npy")[54:,54:]
sc_neglog = -1 * np.log(sc / (np.max(sc) + 1))

# load peptide-receptor pairs from Zhang et al. 2021 PNAS
pairs_available = pd.read_csv('data/peptide_receptor_ligand_pairs.csv', index_col=0)

# load receptor and peptide from qc
receptor_genes = pd.read_csv('data/receptor_filtered.csv', index_col=0).index.to_list()
precursor_genes = pd.read_csv('data/precursor_qc.csv', index_col=0).index.to_list()
genes_of_interest = receptor_genes + precursor_genes

# keep only pairs that are in the gene library
pairs_available = pairs_available[pairs_available['Peptide'].isin(genes_of_interest)]
pairs_available = pairs_available[pairs_available['Receptor'].isin(genes_of_interest)]

# # create list of unique gene names in pairs_available
# peptides_list = pairs_available['Peptide'].unique().tolist()
# receptor_list = pairs_available['Receptor'].unique().tolist()
# genes_of_interest = peptides_list + receptor_list

# load genes of interest from all genes in gene library
all_genes = pd.read_csv('data/abagen_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)

# in all_genes column names, select only ones that are in genes_of_interest
genes = all_genes[all_genes.columns.intersection(genes_of_interest)]
genes = index_structure(genes, structure='CTX')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                           PLOT CONNECTOMES
###############################################################################
cmap = divergent_green_orange()

fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=200)
plot_mat = alpha_mat - np.diag(np.diag(alpha_mat))
sns.heatmap(plot_mat, ax=ax, cmap=cmap, xticklabels=False, yticklabels=False, cbar=False, square=True)
sns.despine(ax=ax, left=True, bottom=True)

if savefig:
    plt.savefig('figs/alpha_connectome.pdf')
    
# %%
# plot gene distribution of receptor-ligand pair, e.g., NPY and NPY5R
fig, ax = plt.subplots(1, 2, figsize=(1, 5), dpi=200)
sns.heatmap(genes['NPY'].values[:,np.newaxis], ax=ax[0], cmap=cmap, 
            xticklabels=False, yticklabels=False, cbar=False, square=True)   
sns.heatmap(genes['NPY5R'].values[:,np.newaxis], ax=ax[1], cmap=cmap,
            xticklabels=False, yticklabels=False, cbar=False, square=True)
sns.despine(ax=ax[0], left=True, bottom=True)
sns.despine(ax=ax[1], left=True, bottom=True)

if savefig:
    plt.savefig('figs/npy_npy5r_heatmap.pdf')
    
# plot outer product of receptor-ligand pair
mat = np.outer(genes['NPY'], genes['NPY5R'])
fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
sns.heatmap(mat, ax=ax, cmap=cmap,
            xticklabels=False, yticklabels=False, cbar=False, square=True)
sns.despine(ax=ax, left=True, bottom=True)

if savefig:
    plt.savefig('figs/npy_npy5r_outer_product.pdf')
    
# mask the outer product with SC
fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
sns.heatmap(mat * sc, ax=ax, cmap=cmap, mask=np.invert(sc.astype(bool)), vmin=-0.2,
            xticklabels=False, yticklabels=False, cbar=False, square=True)
sns.despine(ax=ax, left=True, bottom=True)

if savefig:
    plt.savefig('figs/npy_npy5r_outer_product_masked.pdf')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                S-F COUPLING WITH STRUCTURAL CONNECTIVITY
###############################################################################
# derive communication measures
sc_comm_mats = communication_measures(sc, sc_neglog, dist_mat)
sc_comm_names = ["spl", "npe", "sri", "cmc", "dfe"]
sc_comm = dict(zip(sc_comm_names, sc_comm_mats))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                  S-F COUPLING WITH SIGNALLING WEIGHTS
###############################################################################
# derive new sc based on peptide-receptor pairs matrix
all_mat = []
pair_names = []
pep_comm = []
for index, row in pairs_available.iterrows():
    peptide = genes[row['Peptide']].values[:, np.newaxis]
    receptor = genes[row['Receptor']].values[:, np.newaxis]
    
    # make sure peptide and receptor have no zero values
    # if so, add minimal value to avoid division by zero
    if np.any(peptide == 0):
        peptide[np.where(peptide == 0)] = 1e-10
    if np.any(receptor == 0):
        receptor[np.where(receptor == 0)] = 1e-10
    
    mat = peptide @ receptor.T
    all_mat.append(mat)
    
    pair_names.append(f'{row["Receptor"]}-{row["Peptide"]}')
    
    s = sc * mat
    s_neglog = -1 * np.log(s / (np.max(s) + 1))
    pep_comm.append(communication_measures(s, s_neglog, dist_mat))

# %%
# predicatbility of fc using peptide communication
all_r2s = []
for band in meg_bands:
    y = non_diagonal_elements(band)
    band_r2s = []
    for comm in pep_comm:
        reg = LinearRegression(fit_intercept=True, n_jobs=-1)
        X = sstats.zscore(comm, ddof=1, axis=1).T
        reg_res = reg.fit(X, y)
        yhat = reg.predict(X)
        SS_Residual = sum((y - yhat) ** 2)
        SS_Total = sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (float(SS_Residual)) / SS_Total
        num = (1 - r_squared) * (len(y) - 1)
        denom = len(y) - X.shape[1] - 1
        adjusted_r_squared = 1 -  num / denom
        band_r2s.append(adjusted_r_squared)
    all_r2s.append(band_r2s)

# %%
# create a dataframe with the results
pred_df = pd.DataFrame([], index=pair_names)

# load results from fc predictability
fc_pred = pd.read_csv('results/fc_pred_peptide_ligand_pairs.csv', index_col=0)

# map FC predictability to pred_df
pred_df['R2_FC'] = pred_df.index.map(fc_pred['R2_FC'])

# locate no annotation predictability and drop from df
adjusted_r_squared_fc = fc_pred.loc['no_annotation']
# pred_df = fc_pred.drop('no_annotation')

# MEG band names
band_names = ['delta', 'theta', 'alpha', 'beta', 'lgamma', 'hgamma']

no_annt_r2s = list(adjusted_r_squared_fc)
for name, band, band_r2s in zip(band_names, meg_bands, all_r2s):
    # predict band with no annotation
    y = non_diagonal_elements(band)
    reg = LinearRegression(fit_intercept=True, n_jobs=-1)
    X = sstats.zscore(sc_comm_mats, ddof=1, axis=1).T
    reg_res = reg.fit(X, y)
    yhat = reg.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    num = (1 - r_squared) * (len(y) - 1)
    denom = len(y) - X.shape[1] - 1
    adjusted_r_squared = 1 -  num / denom
    no_annt_r2s.append(adjusted_r_squared)

    # add that band's R2s to pred_df
    pred_df[f'R2_{name}'] = band_r2s



# subtract no annotation predictability to get relative predictability
for i,name in enumerate(['FC'] + band_names):
    pred_df[f'R2_{name}'] = pred_df[f'R2_{name}'] - no_annt_r2s[i]


# add last row with index name no_annotation
# pred_df.loc[len(pred_df)] = ['no_annotation', *no_annt_r2s]
# sort by alpha predictability
# pred_df = pred_df.sort_values('R2_alpha', ascending=False).reset_index(drop=True)
# store pred_df
# pred_df.to_csv('results/fc_pred_peptide_ligand_pairs_alpha.csv', index=False)

# plot
sns.clustermap(data=pred_df, col_cluster=False, row_cluster=True, cmap=divergent_green_orange(), center=0)

# %%
nneg_df = pred_df[pred_df > 0.01].copy()

nneg_df.index = nneg_df.index.str.split('-').str[0]
nneg_df = nneg_df.sort_index()

# rename R2_FC to functional connectivity
nneg_df = nneg_df.rename(columns={'R2_FC': 'functional connectivity', 'R2_alpha': 'alpha oscillations',
                                  'R2_beta': 'beta oscillations', 'R2_delta': 'delta oscillations',
                                  'R2_hgamma': 'high gamma oscillations', 'R2_lgamma': 'low gamma oscillations',
                                  'R2_theta': 'theta oscillations'})

# find indices of non-zero values using np.where
indices = np.where(nneg_df.values != np.nan)

combos = []

for i, j in zip(*indices):
    combos.append((f'{nneg_df.index[i]} and {nneg_df.columns[j]}'))
    

# %%
# Create a lollipop plot
fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
ax.stem(pred_df['Pair'], pred_df['R2_alpha'],)
ax.set_xticklabels(pred_df['Pair'], rotation=90)
ax.set_ylabel('Adjusted R$^2$')
# ax.set_yticks(np.arange(0, 0.21, 0.1))
ax.axhline(adjusted_r_squared, color='r', linestyle='--', label='Simple connectome')
sns.despine(trim=True)

if savefig:
    plt.savefig('figs/fc_pred_peptide_ligand_pairs.pdf')

# %%
pred_df = pd.read_csv('results/fc_pred_peptide_ligand_pairs_alpha.csv')

# load fc results
fc_pred = pd.read_csv('results/fc_pred_peptide_ligand_pairs.csv', index_col=0)
# map FC predictability to pred_df
pred_df['R2_FC'] = pred_df['Pair'].map(fc_pred['R2_FC'])


# %%
# plot predictability of FC against alpha
fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
sns.scatterplot(data=pred_df, x='R2_alpha', y='R2_FC', ax=ax)
ax.axvline(pred_df.loc[pred_df['Pair'] == 'no_annotation', 'R2_alpha'].values[0], color='lightblue')
ax.axhline(pred_df.loc[pred_df['Pair'] == 'no_annotation', 'R2_FC'].values[0], color='lightblue')

# add names of pairs
for i, txt in enumerate(pred_df['Pair']):
    ax.annotate(txt, (pred_df['R2_alpha'][i], pred_df['R2_FC'][i]), fontsize=6)
    
if savefig:
    plt.savefig('figs/structure_function_fc_meg.pdf')


# %% 
# list of peptide-receptor pairs with predictability higher than SC
pred_df[pred_df['R squared'] > adjusted_r_squared].sort_values('R squared', ascending=False).reset_index(drop=True)