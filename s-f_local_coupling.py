# %%
import brainconn as bc
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sstats
import seaborn as sns
from netneurotools.stats import get_dominance_stats
from sklearn.linear_model import LinearRegression
from utils import navigation_wu, search_information, communicability_wei

# load data
fc_cons = np.load("data/template_parc-Schaefer400_desc-FC.npy")
dist_mat = np.load("data/template_parc-Schaefer400_desc-distance.npy")
sc = np.load("data/template_parc-Schaefer400_desc-SC.npy")
sc_neglog = -1 * np.log(sc / (np.max(sc) + 1))
nrois = fc_cons.shape[0]

pairs_available = pd.read_csv('data/peptide_receptor_ligand_pairs-locality.csv')
all_genes = pd.read_csv('data/abagen_genes_Schaefer2018_400.csv', index_col=0)

peptides_list = pairs_available['Peptide'].unique()
receptor_list = pairs_available['Receptor'].unique()
gene_list = pd.DataFrame(np.concatenate([peptides_list, receptor_list]), columns=['Gene'])

overlapping_genes = gene_list[gene_list['Gene'].isin(all_genes.columns)]['Gene']
genes = all_genes[overlapping_genes]

# derive communication measures
def non_diagonal_elements(matrix):
    mat = matrix.copy()
    rows, cols = np.triu_indices(matrix.shape[0], k=1)
    return mat[rows, cols].flatten()

def communication_measures(sc, sc_neglog, dist_mat):
    spl_mat, sph_mat, _ = bc.distance.distance_wei_floyd(sc_neglog)
    nsr, nsr_n, npl_mat_asym, nph_mat_asym, nav_paths = navigation_wu(dist_mat, sc)
    npe_mat_asym = 1 / npl_mat_asym
    npe_mat = (npe_mat_asym + npe_mat_asym.T) / 2
    sri_mat_asym = search_information(sc, sc_neglog)
    sri_mat = (sri_mat_asym + sri_mat_asym.T) / 2
    cmc_mat = communicability_wei(sc)
    mfpt_mat_asym = bc.distance.mean_first_passage_time(sc)
    dfe_mat_asym = 1 / mfpt_mat_asym
    dfe_mat = (dfe_mat_asym + dfe_mat_asym.T) / 2

    sc_comm_mats = [spl_mat, npe_mat, sri_mat, cmc_mat, dfe_mat]

    return sc_comm_mats

sc_comm_mats = communication_measures(sc, sc_neglog, dist_mat)
comm_names = ["spl", "npe", "sri", "cmc", "dfe"]
sc_comm = dict(zip(comm_names, sc_comm_mats))

# %% Compare communication measures of structural connectivity and peptide-receptor pairs
# derive new sc based on peptide-receptor pairs matrix

all_mat = []
pair_names = []
pep_comm = []
for index, row in pairs_available.iterrows(): #.head(5).iterrows():
    peptide = genes[row['Peptide']].values[:, np.newaxis]
    receptor = genes[row['Receptor']].values[:, np.newaxis]
    mat = peptide @ receptor.T
    all_mat.append(mat)
    
    pair_names.append(f'{row["Peptide"]}-{row["Receptor"]}')
    
    s = sc * mat
    s_neglog = -1 * np.log(s / (np.max(s) + 1))
    pep_comm_mats = communication_measures(s, s_neglog, dist_mat)
    pep_comm.append(dict(zip(comm_names, pep_comm_mats)))

# %%
# predicatbility of fc using peptide communication
all_pairs = []
for comm in pep_comm:
    regional_rsq = []
    for i in range(nrois):
        reg = LinearRegression(fit_intercept=True)
        X_no_diag = np.delete(
            np.c_[comm["spl"][:, i],
                  comm["npe"][:, i],
                  comm["sri"][:, i],
                  comm["cmc"][:, i],
                  comm["dfe"][:, i]
            ],
            i,
            axis=0,
        )
        X = sstats.zscore(X_no_diag, ddof=1)
        y = np.delete(fc_cons[:, i], i, axis=0)
        reg_res = reg.fit(X, y)
        yhat = reg.predict(X)
        SS_Residual = sum((y - yhat) ** 2)
        SS_Total = sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (float(SS_Residual)) / SS_Total
        num = (1 - r_squared) * (len(y) - 1)
        denom = len(y) - X.shape[1] - 1
        adjusted_r_squared = 1 -  num / denom
        regional_rsq.append(adjusted_r_squared)
    regional_rsq = np.array(regional_rsq)
    all_pairs.append(regional_rsq)

all_pairs = np.array(all_pairs).T

# # create dataframe with peptide-receptor pairs and their predictability
pred_df = pd.DataFrame(all_pairs, columns=pair_names)

# %%
# using structural connectivity
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

# compare distribution of adjusted r squared
plt.figure()
sns.histplot(all_rsq, kde=True, label='peptide weighted', bins=10)
plt.axvline(adjusted_r_squared, color='r', label='SC weighted')
plt.xlabel('Adjusted R squared')
plt.legend()
plt.title('Predictability of FC using communication measures')

# %% 
# list of peptide-receptor pairs with predictability higher than SC
pred_df[pred_df['R squared'] > adjusted_r_squared]
