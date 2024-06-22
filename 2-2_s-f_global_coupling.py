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
from utils import index_structure, navigation_wu, search_information, communicability_wei

# load data
fc_cons = np.load("data/template_parc-Schaefer400_TianS4_desc-FC.npy")
dist_mat = np.load("data/template_parc-Schaefer400_TianS4_desc-distance.npy")
sc = np.load("data/template_parc-Schaefer400_TianS4_desc-SC.npy")
sc_neglog = -1 * np.log(sc / (np.max(sc) + 1))

pairs_available = pd.read_csv('data/peptide_receptor_ligand_pairs.csv')
# keep only Peptide and Receptor columns
pairs_available = pairs_available[['Peptide', 'Receptor']]
pairs_available = pairs_available.dropna()
# for Receptor column, separate multiple receptors into different columns
pairs_available = pairs_available.assign(Receptor=pairs_available['Receptor'].str.split(';')).explode('Receptor')
pairs_available = pairs_available.drop_duplicates().reset_index(drop=True)

all_genes = pd.read_csv('data/abagen_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)

peptides_list = pairs_available['Peptide'].unique()
receptor_list = pairs_available['Receptor'].unique()
gene_list = pd.DataFrame(np.concatenate([peptides_list, receptor_list]), columns=['Gene'])

overlapping_genes = gene_list[gene_list['Gene'].isin(all_genes.columns)]['Gene']
pairs_available = pairs_available[pairs_available['Peptide'].isin(overlapping_genes) & 
                                  pairs_available['Receptor'].isin(overlapping_genes)]
genes = all_genes[overlapping_genes]
genes = index_structure(genes, structure='CTX-SBCTX')

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
    sc_comm_mats = list(map(lambda x: non_diagonal_elements(x), sc_comm_mats))

    return sc_comm_mats

sc_comm_mats = communication_measures(sc, sc_neglog, dist_mat)
sc_comm_names = ["spl", "npe", "sri", "cmc", "dfe"]
sc_comm = dict(zip(sc_comm_names, sc_comm_mats))

# %% Compare communication measures of structural connectivity and peptide-receptor pairs
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

# %%
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

# create dataframe with peptide-receptor pairs and their predictability
pred_df = pd.DataFrame({'Pair': pair_names, 'R squared': all_rsq})

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
sns.histplot(all_rsq, kde=True, label='SC with peptide weights', bins=10)
plt.axvline(adjusted_r_squared, color='r', label='Binary SC')
plt.xlabel('Adjusted R$^2$')
plt.legend(frameon=False)
plt.title('Predictability of FC using communication measures')
sns.despine()
plt.savefig('figs/predictability_fc_communication.pdf', dpi=300)

# %% 
# list of peptide-receptor pairs with predictability higher than SC
pred_df[pred_df['R squared'] > adjusted_r_squared].sort_values('R squared', ascending=False).reset_index(drop=True)