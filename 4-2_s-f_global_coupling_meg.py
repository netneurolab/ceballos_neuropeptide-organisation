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
# load all MEG bands
delta_mat = np.load("data/annotations/meg_delta_mat_Schaefer400.npy")
theta_mat = np.load("data/annotations/meg_theta_mat_Schaefer400.npy")
alpha_mat = np.load("data/annotations/meg_alpha_mat_Schaefer400.npy")
beta_mat = np.load("data/annotations/meg_beta_mat_Schaefer400.npy")
lgamma_mat = np.load("data/annotations/meg_lgamma_mat_Schaefer400.npy")
hgamma_mat = np.load("data/annotations/meg_hgamma_mat_Schaefer400.npy")
meg_bands = np.stack([delta_mat, theta_mat, alpha_mat, beta_mat, lgamma_mat, hgamma_mat])
band_names = ['delta', 'theta', 'alpha', 'beta', 'lgamma', 'hgamma']

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

# load genes of interest from all genes in gene library
all_genes = pd.read_csv('data/abagen_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)

# in all_genes column names, select only ones that are in genes_of_interest
genes = all_genes[all_genes.columns.intersection(genes_of_interest)]
genes = index_structure(genes, structure='CTX')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#             ANNOTATE CONNECTOME WITH PEPTIDE-RECEPTOR PAIRS
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                S-F COUPLING WITH ANNOTATED CONNECTOME
###############################################################################
# outer loop is for each band in MEG bands
# inner loop is for each peptide-receptor pair
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                       PUTTING IT ALL TOGETHER
###############################################################################
# create a dataframe with the results
pred_df = pd.DataFrame([], index=pair_names)

# load results from haemodynamic fc predictability
fc_pred = pd.read_csv('results/fc_pred_peptide_ligand_pairs.csv', index_col=0)

# store predictability from haemodynamic connectivity
pred_df['R2_FC'] = pred_df.index.map(fc_pred['R2_FC'])

# locate no annotation predictability from fc_pred and store for later
adjusted_r_squared_fc = fc_pred.loc['no_annotation']
no_annt_r2s = list(adjusted_r_squared_fc)

# derive structural communication from unannotated connectome
sc_comm_mats = communication_measures(sc, sc_neglog, dist_mat) 

# for each band in MEG bands, predict FC using no annotation 
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

    # store previously computed R2s for this band in pred_df
    pred_df[f'R2_{name}'] = band_r2s

# pred_df.to_csv('results/comm_pred_peptide_ligand_pairs.csv')

# plot predictability of FC against alpha
fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
sns.scatterplot(data=pred_df, x='R2_alpha', y='R2_FC', ax=ax)

# add names of pairs
for i, txt in enumerate(pred_df.index):
    ax.annotate(txt, (pred_df['R2_alpha'][i], pred_df['R2_FC'][i]), fontsize=6)

# contrast line with no annotation predictability
no_annt_fc_r2 = no_annt_r2s[0]
no_annt_alpha_r2 = no_annt_r2s[3]
ax.axvline(no_annt_alpha_r2, color='lightblue')
ax.axhline(no_annt_fc_r2, color='lightblue')
    
if savefig:
    plt.savefig('figs/structure_function_annotated_connectome.pdf')

# %%
# subtract predictability of having no annotation from predictability of each band
diff_df = pd.DataFrame([], index=pair_names)
for i,name in enumerate(['FC'] + band_names):
    diff_df[f'R2_{name}'] = pred_df[f'R2_{name}'] - no_annt_r2s[i]

# diff_df.to_csv('results/comm_pred_diff_peptide_ligand_pairs.csv')

# plot relative predictability difference against baseline
fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
pair_names = pred_df.index
sns.heatmap(diff_df, ax=ax, cmap=divergent_green_orange(), xticklabels=True, yticklabels=True, cbar=True, square=True,
            cbar_kws={'label': 'Adjusted R$^2$ - No annotation'}, center=0)
