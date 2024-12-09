# %%
import brainconn as bc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sstats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from utils import index_structure, communication_measures, non_diagonal_elements
from plot_utils import divergent_green_orange

savefig = False

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                              LOAD DATA
###############################################################################
# load template matrices
fc_cons = np.load("data/template_parc-Schaefer400_TianS4_desc-FC.npy")
dist_mat = np.load("data/template_parc-Schaefer400_TianS4_desc-distance.npy")
sc = np.load("data/template_parc-Schaefer400_TianS4_desc-SC.npy")
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
genes = index_structure(genes, structure='CTX-SBCTX')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                           PLOT CONNECTOMES
###############################################################################
cmap = divergent_green_orange()

# separate cmap into two cmaps split at half
green = cmap(np.linspace(0, 0.5, 128))
orange = cmap(np.linspace(0.5, 1, 128))

green = sns.color_palette(green[::-1])
orange = sns.color_palette(orange)


fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
sns.heatmap(fc_cons, ax=ax[0], cmap=cmap, xticklabels=False, yticklabels=False, cbar=False, square=True,
            vmin=-0.2, vmax=0.5)
sns.despine(ax=ax[0], left=True, bottom=True)

ax[1].spy(sc, markersize=0.1)
sns.despine(ax=ax[1], left=True, bottom=True, top=True, right=True)
ax[1].set_yticks([])
ax[1].set_xticks([]);

if savefig:
    plt.savefig('figs/fc_sc_connectomes.pdf')
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                  PLOT CONNECTOME WITH ANNOTATED WEIGHTS
###############################################################################
# compute outer product of receptor-ligand pair, e.g., NPY and NPY5R    
mat = np.outer(genes['NPY'], genes['NPY5R'])

# plot receptor-ligand pair correspondence
fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
sns.heatmap(mat, ax=ax, cmap=cmap,
            xticklabels=False, yticklabels=False, cbar=False, square=True)
sns.despine(ax=ax, left=True, bottom=True)

if savefig:
    plt.savefig('figs/npy_npy5r_outer_product.pdf')
    
# mask the matrix with SC
fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
sns.heatmap(mat * sc, ax=ax, cmap=cmap, mask=np.invert(sc.astype(bool)), vmin=-0.2,
            xticklabels=False, yticklabels=False, cbar=False, square=True)
sns.despine(ax=ax, left=True, bottom=True)

if savefig:
    plt.savefig('figs/npy_npy5r_outer_product_masked.pdf')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                 ANNOTATE CONNECTOME WITH ALL PAIRS AVAILABLE
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
    
    pair_names.append(f'{row["Peptide"]}-{row["Receptor"]}')
    
    s = sc * mat
    s_neglog = -1 * np.log(s / (np.max(s) + 1))
    pep_comm.append(communication_measures(s, s_neglog, dist_mat))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                 S-F COUPLING WITH ANNOTATED CONNECTOME
###############################################################################
# predicatbility of fc using peptide communication
all_rsq = []
y = non_diagonal_elements(fc_cons)
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
    all_rsq.append(adjusted_r_squared)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#               S-F COUPLING WITHOUT ANNOTATED WEIGHTS
###############################################################################
# derive communication measures
sc_comm_mats = communication_measures(sc, sc_neglog, dist_mat)
sc_comm_names = ["spl", "npe", "sri", "cmc", "dfe"]
sc_comm = dict(zip(sc_comm_names, sc_comm_mats))

for key, value in sc_comm.items():
    print(f'{key}: {np.mean(value)}')
    
# predicatbility of fc using simple connectome
y = non_diagonal_elements(fc_cons)
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

# add to pred_df
pred_df = pd.DataFrame({'Pair': pair_names, 'R2_FC': all_rsq})
pred_df.loc[len(pred_df)] = ['no_annotation', adjusted_r_squared]
pred_df = pred_df.sort_values('R2_FC', ascending=False).reset_index(drop=True)
# store pred_df
pred_df.to_csv('results/fc_pred_peptide_ligand_pairs.csv', index=False)
