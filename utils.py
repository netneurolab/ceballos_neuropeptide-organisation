import itertools
import numpy as np
import pandas as pd
from brainconn.distance import distance_wei_floyd, mean_first_passage_time, retrieve_shortest_path
from sklearn.linear_model import LinearRegression
from scipy.sparse.linalg import expm
from scipy.stats import ks_2samp, zscore
from joblib import Parallel, delayed

def morans_i(dist, y, normalize=False, local=False, invert_dist=True):
    """
    Calculates Moran's I from distance matrix `dist` and brain map `y`.
    Taken from Markello et al 2020 (Neuroimage).

    Parameters
    ----------
    dist : (N, N) array_like
        Distance matrix between `N` regions / vertices / voxels / whatever
    y : (N,) array_like
        Brain map variable of interest
    normalize : bool, optional
        Whether to normalize rows of distance matrix prior to calculation.
        Default: False
    local : bool, optional
        Whether to calculate local Moran's I instead of global. Default: False
    invert_dist : bool, optional
        Whether to invert the distance matrix to generate a weight matrix.
        Default: True

    Returns
    -------
    i : float
        Moran's I, measure of spatial autocorrelation
    """
    # Ensure dist and y are numpy arrays
    dist = np.array(dist)
    y = np.array(y)

    # convert distance matrix to weights
    if invert_dist:
        with np.errstate(divide='ignore'):
            dist = 1 / dist
    np.fill_diagonal(dist, 0)

    # normalize rows, if desired
    if normalize:
        dist /= dist.sum(axis=-1, keepdims=True)

    # calculate Moran's I
    z = y - y.mean()
    if local:
        with np.errstate(all='ignore'):
            z /= y.std()

    zl = np.squeeze(dist @ z[:, np.newaxis])
    den = (z * z).sum()

    if local:
        return (len(y) - 1) * z * zl / den

    return len(y) / dist.sum() * (z * zl).sum() / den 


def get_gene_null(gene_set, gene_set_mi, non_overlapping_set, distance, null_set_size=100, seed=1234):
    """
    Creates a null set of gene expression data by randomly selecting genes from a non-overlapping set
    and matching the spatial autocorrelation and value distribution of the original gene set.
    Moran's I is used to match the spatial autocorrelation. Kolmogorov-Smirnov test is used to match 
    the value distribution.
    
    Parameters
    ----------
    gene_set : pandas.DataFrame
        Gene expression data to be matched
    gene_set_mi : numpy.ndarray
        Spatial autocorrelation of gene expression data, derived from Moran's I. Should be equal to the
        number of genes in gene_set
    non_overlapping_set : pandas.DataFrame
        Gene expression data to be used for null set sampling
    distance : numpy.ndarray
        Distance matrix for spatial autocorrelation
    null_set_size : int
        Number of genes to sample for null set
    seed : int
        Random seed for reproducibility

    Returns
    -------
    null : numpy.ndarray
        Array containing gene expression data matching the spatial autocorrelation and value distribution
        of the original gene set
    """
    # sample genes
    random_genes = non_overlapping_set.sample(n=null_set_size, axis=1, random_state=seed)
    # get the spatial autocorrelation of each gene in the random gene set
    random_mi_values = []
    random_ks_values = []
    for _, random_gene in random_genes.items():
        random_mi_values.append(morans_i(distance, random_gene.values))
        random_ks_values.append([ks_2samp(gene, random_gene.values).statistic for _, gene in gene_set.items()])
    random_mi_values = np.array(random_mi_values)
    random_ks_values = np.array(random_ks_values)

    # for each gene in gene_set, calculate which moran_i is most similar among the random genes
    comparison_mi = [np.abs(gene_set_mi - random_mi_values[i]) for i in range(null_set_size)]
    rank_mi = np.argsort(comparison_mi, axis=0)

    # get the value distribution of each gene in the gene set
    rank_ks = np.argsort(random_ks_values, axis=0)  

    # select the gene with the highest average rank
    rank = (rank_mi + rank_ks) / 2
    indices = np.argmin(rank, axis=0)
    selected_genes = random_genes.iloc[:, indices]
    null = selected_genes.values
    return null


def gene_null_set(gene_set, non_overlapping_set, distance, null_set_size=100, n_permutations=1000, 
                  n_jobs=1, seed=1234):
    """
    Creates a null set of gene expression data by randomly selecting genes from a non-overlapping set
    and matching the spatial autocorrelation and value distribution of the original gene set.
    Moran's I is used to match the spatial autocorrelation. Kolmogorov-Smirnov test is used to match 
    the value distribution.
    
    Parameters
    ----------
    gene_set : pandas.DataFrame
        Gene expression data to be matched
    non_overlapping_set : pandas.DataFrame
        Gene expression data to be used for null set
    distance : numpy.ndarray
        Distance matrix for spatial autocorrelation
    n_permutations : int
        Number of permutations to create
    seed : int
        Random seed for reproducibility

    Returns
    -------
    nulls : list
        List of numpy.ndarray containing null gene expression data
    """
    rng = np.random.default_rng(seed)
    # seeds = [rng.integers(0, 2**32-1) for _ in range(n_permutations)]

    # get the spatial autocorrelation of each gene in the gene set
    moran_i = []
    for gene in gene_set.columns:
        moran_i.append(morans_i(distance, gene_set[gene].values))
    moran_i = np.array(moran_i)

    # parallelize using joblib
    null_set = Parallel(n_jobs=n_jobs,)(delayed(get_gene_null)(gene_set, moran_i, non_overlapping_set, distance, 
                                                              null_set_size=null_set_size, seed=i)
                                      for i in range(n_permutations))
        
    return null_set


def navigation_wu(nav_dist_mat, sc_mat, show_progress=False):
    from tqdm import tqdm
    nav_paths = []  # (source, target, distance, hops, path)
    for src in tqdm(range(len(nav_dist_mat)), disable=not show_progress):
        for tar in range(len(nav_dist_mat)):
            curr_pos = src
            curr_path = [src]
            curr_dist = 0
            while curr_pos != tar:
                neig = np.where(sc_mat[curr_pos, :] != 0)[0]
                if len(neig) == 0:
                    curr_path = []
                    curr_dist = np.inf
                    break
                neig_dist_to_tar = nav_dist_mat[neig, tar]
                min_dist_idx = np.argmin(neig_dist_to_tar)

                new_pos = neig[min_dist_idx]
                if new_pos in curr_path:
                    curr_path = []
                    curr_dist = np.inf
                    break
                else:
                    curr_path.append(new_pos)
                    curr_dist += nav_dist_mat[curr_pos, new_pos]
                    curr_pos = new_pos
            nav_paths.append((src, tar, curr_dist, len(curr_path) - 1, curr_path))

    nav_sr = len([_ for _ in nav_paths if _[3] != -1]) / len(nav_paths)

    nav_sr_node = []
    for k, g in itertools.groupby(
        sorted(nav_paths, key=lambda x: x[0]), key=lambda x: x[0]
    ):
        curr_path = list(g)
        nav_sr_node.append(len([_ for _ in curr_path if _[3] != -1]) / len(curr_path))

    nav_path_len, nav_path_hop = np.zeros_like(nav_dist_mat), np.zeros_like(
        nav_dist_mat
    )
    for nav_item in nav_paths:
        i, j, length, hop, _ = nav_item
        if hop != -1:
            nav_path_len[i, j] = length
            nav_path_hop[i, j] = hop
        else:
            nav_path_len[i, j] = np.inf
            nav_path_hop[i, j] = np.inf

    return nav_sr, nav_sr_node, nav_path_len, nav_path_hop, nav_paths


def search_information(W, L, has_memory=False):
    import brainconn as bc
    N = len(W)

    if np.allclose(W, W.T):
        flag_triu = True
    else:
        flag_triu = False
    try:
        T = np.linalg.solve(np.diag(np.sum(W, axis=1)), W)
    except np.linalg.LinAlgError:
        # singular matrix! solve for x using pseudo-inverse
        T = np.linalg.pinv(np.diag(np.sum(W, axis=1))) @ W
        
    _, hops, Pmat = distance_wei_floyd(L, transform=None)

    SI = np.zeros((N, N))
    SI[np.eye(N) > 0] = np.nan

    for i in range(N):
        for j in range(N):
            if (j > i and flag_triu) or (not flag_triu and i != j):
                try:
                    path = retrieve_shortest_path(i, j, hops, Pmat)
                except ValueError as e:
                    print(f"Error retrieving path from {i} to {j}: {e}")
                    continue
                lp = len(path) - 1
                if flag_triu:
                    if np.any(path):
                        pr_step_ff = np.zeros(lp)
                        pr_step_bk = np.zeros(lp)
                        if has_memory:
                            pr_step_ff[0] = T[path[0], path[1]]
                            pr_step_bk[lp - 1] = T[path[lp], path[lp - 1]]
                            for z in range(1, lp):
                                pr_step_ff[z] = T[path[z], path[z + 1]] / (
                                    1 - T[path[z - 1], path[z]]
                                )
                                pr_step_bk[lp - z - 1] = T[
                                    path[lp - z], path[lp - z - 1]
                                ] / (1 - T[path[lp - z + 1], path[lp - z]])
                        else:
                            for z in range(lp):
                                pr_step_ff[z] = T[path[z], path[z + 1]]
                                pr_step_bk[z] = T[path[z + 1], path[z]]

                        prob_sp_ff = np.prod(pr_step_ff)
                        prob_sp_bk = np.prod(pr_step_bk)
                        SI[i, j] = -np.log2(prob_sp_ff)
                        SI[j, i] = -np.log2(prob_sp_bk)
                else:
                    if np.any(path):
                        pr_step_ff = np.zeros(lp)
                        if has_memory:
                            pr_step_ff[0] = T[path[0], path[1]]
                            for z in range(1, lp):
                                pr_step_ff[z] = T[path[z], path[z + 1]] / (
                                    1 - T[path[z - 1], path[z]]
                                )
                        else:
                            for z in range(lp):
                                pr_step_ff[z] = T[path[z], path[z + 1]]

                        prob_sp_ff = np.prod(pr_step_ff)
                        SI[i, j] = -np.log2(prob_sp_ff)
                    else:
                        SI[i, j] = np.inf

    return SI


def non_diagonal_elements(matrix):
    mat = matrix.copy()
    rows, cols = np.triu_indices(matrix.shape[0], k=1)
    return mat[rows, cols].flatten()


def communication_measures(sc, sc_neglog, dist_mat, symmetric=True):
    spl_mat, sph_mat, _ = distance_wei_floyd(sc_neglog)
    nsr, nsr_n, npl_mat_asym, nph_mat_asym, nav_paths = navigation_wu(dist_mat, sc)
    npe_mat_asym = 1 / (npl_mat_asym + np.finfo(float).eps)
    npe_mat = (npe_mat_asym + npe_mat_asym.T) / 2 if symmetric else npe_mat_asym
    sri_mat_asym = search_information(sc, sc_neglog)
    sri_mat = (sri_mat_asym + sri_mat_asym.T) / 2 if symmetric else sri_mat_asym
    cmc_mat = communicability_wei(sc)
    mfpt_mat_asym = mean_first_passage_time(sc)
    dfe_mat_asym = 1 / (mfpt_mat_asym + np.finfo(float).eps)
    dfe_mat = (dfe_mat_asym + dfe_mat_asym.T) / 2 if symmetric else dfe_mat_asym


    sc_comm_mats = [spl_mat, npe_mat, sri_mat, cmc_mat, dfe_mat]
    sc_comm_mats = list(map(lambda x: non_diagonal_elements(x), sc_comm_mats))

    return sc_comm_mats


def group_by_index(val_List, idx_list):
    result = []
    for _ in sorted(set(idx_list)):
        result.append([val_List[it] for it, idx in enumerate(idx_list) if idx == _])
    return result


def communicability_wei(adjacency):
    """
    Computes the communicability of pairs of nodes in `adjacency`
    Parameters
    ----------
    adjacency : (N, N) array_like
        Weighted, direct/undirected connection weight/length array
    Returns
    -------
    cmc : (N, N) numpy.ndarray
        Symmetric array representing communicability of nodes {i, j}
    References
    ----------
    Crofts, J. J., & Higham, D. J. (2009). A weighted communicability measure
    applied to complex brain networks. Journal of the Royal Society Interface,
    6(33), 411-414.
    Examples
    --------
    >>> from netneurotools import metrics
    >>> A = np.array([[2, 0, 3], [0, 2, 1], [0.5, 0, 1]])
    >>> Q = metrics.communicability_wei(A)
    >>> Q
    array([[0.        , 0.        , 1.93581903],
           [0.07810379, 0.        , 0.94712177],
           [0.32263651, 0.        , 0.        ]])
    """

    # negative square root of nodal degrees
    row_sum = adjacency.sum(1)
    row_sum[row_sum == 0] = 1e-10 # add small value to avoid division by zero
    neg_sqrt = np.power(row_sum, -0.5)
    square_sqrt = np.diag(neg_sqrt)

    # normalize input matrix
    for_expm = square_sqrt @ adjacency @ square_sqrt

    # calculate matrix exponential of normalized matrix
    cmc = expm(for_expm)
    cmc[np.diag_indices_from(cmc)] = 0

    return cmc


def index_structure(df, structure='CTX-SBCTX'):
    """
    Indexes the dataframe by the specified structure for parcellated data with
    the combined Schaefer 400 + Tian S4 + HTH atlas
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to be indexed
    structure : str
        Structure to index by. Can be CTX, SBCTX, HTH, CTX-SBCTX, CTX-HTH or 
        SBCTX-HTH
    Returns
    -------
    indexed_df : pandas.DataFrame
        Dataframe indexed by the specified structure
    """
    if structure == 'CTX':
        indexed_df = df.iloc[54:-1]
    elif structure == 'SBCTX':
        indexed_df = df.iloc[:54]
    elif structure == 'HTH':
        indexed_df = df.iloc[-1]
    elif structure == 'CTX-SBCTX':
        indexed_df = df.iloc[:-1]
    elif structure == 'CTX-HTH':
        indexed_df = df.iloc[54:]
    elif structure == 'SBCTX-HTH':
        indexed_df = pd.concat((df.iloc[:54], df.iloc[-1]))
    else:
        raise ValueError('Invalid structure. Must be CTX, SBCTX, HTH, CTX-SBCTX, ' +
                         'CTX-HTH or SBCTX-HTH')
    return indexed_df


def reorder_subcortex(gene_df, type='freesurfer', region_info=None):
    """
    Reorders the gene expression data for Tian subcortex regions to match the freesurfer or enigma atlas.
    Parameters
    ----------
    gene_df : pandas.DataFrame
        Gene expression data to be reordered
    type : str
        Type of atlas to reorder to. Can be 'freesurfer' or 'enigma'
    region_info : pandas.DataFrame, optional
        Region information dataframe to reorder to freesurfer atlas
    Returns
    -------
    gene_df : pandas.DataFrame
        Reordered gene expression data
    """

    # is it freesurfer or enigma?
    if type not in ['freesurfer', 'enigma']:
        raise ValueError("Invalid type. Must be 'freesurfer' or 'enigma'")
    
    ctx_genes = index_structure(gene_df, structure='CTX')
    sbctx_genes = index_structure(gene_df, structure='CTX-SBCTX')

    regions = sbctx_genes.index.str.split('-').str[0]
    hemi = sbctx_genes.index.str.split('-').str[-1]

    name_map = {'HIP': 'Hippocampus' if type == 'freesurfer' else 'hippocampus',
                'THA': 'Thalamus-Proper' if type == 'freesurfer' else 'thalamus',
                'mAMY': 'Amygdala' if type == 'freesurfer' else 'amygdala',
                'lAMY': 'Amygdala' if type == 'freesurfer' else 'amygdala',
                'PUT': 'Putamen' if type == 'freesurfer' else 'putamen',
                'aGP': 'Pallidum' if type == 'freesurfer' else 'pallidum',
                'pGP': 'Pallidum' if type == 'freesurfer' else 'pallidum',
                'CAU': 'Caudate' if type == 'freesurfer' else 'caudate',
                'NAc': 'Accumbens-area' if type == 'freesurfer' else 'accumbens',}

    hemi_map = {'lh': 'L' if type == 'freesurfer' else 'left',
                'rh': 'R' if type == 'freesurfer' else 'right'}

    # map regions/ hemi to names
    regions = regions.map(name_map)
    hemi = hemi.map(hemi_map)

    # join the two and rename index
    sbctx_genes.index = hemi + '-' + regions

    # average same regions
    sbctx_genes = sbctx_genes.groupby(sbctx_genes.index).mean()
    if type == 'freesurfer':
        if region_info is None:
            raise ValueError("'region_info' must be provided for freesurfer reordering")
        # order according to region_info
        filtered_labels = [label for label in region_info['labels'] if label in sbctx_genes.index]
        sbctx_genes = sbctx_genes.loc[filtered_labels]

    # concatenate subcortex and cortex
    gene_df = pd.concat([sbctx_genes, ctx_genes])
    return gene_df

def linear_fit(X, y):
    """
    Computes the adjusted R2 of a linear regression model.
    Parameters
    ----------
    X : np.ndarray
        Independent variable (predictor) matrix.
    y : np.ndarray
        Dependent variable (response) vector.
    Returns
    -------
    adjusted_r_squared : float
        Adjusted R2 value of the linear regression model.
    """
    
    reg = LinearRegression(fit_intercept=True)
    reg_res = reg.fit(X, y)
    yhat = reg.predict(X)

    SS_Residual = np.sum((y - yhat) ** 2)
    SS_Total = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (SS_Residual / SS_Total)

    num = (1 - r_squared) * (len(y) - 1)
    denom = len(y) - X.shape[1] - 1

    adjusted_r_squared = 1 - (num / denom)
    return adjusted_r_squared

def fit_to_connectivity(peptide, receptor, sc, dist_mat, y):
    """
    Computes the structure-function coupling for a given peptide and receptor pair.
    Parameters
    ----------
    peptide : np.ndarray
        Regional estimates of peptide ligand.
    receptor : np.ndarray
        Regional estimates of peptide receptor.
    sc : np.ndarray
        Structural connectivity matrix.
    dist_mat : np.ndarray
        Distance matrix.
    y : np.ndarray
        Non-diagonal elements of the functional connectivity matrix.
    Returns
    -------
    r2 : float
        Goodness of fit (adjusted R2) of peptide-receptor pair to functional connectivity.
    """
    
    if peptide.ndim == 1:
        peptide = peptide[:, np.newaxis]
    if receptor.ndim == 1:
        receptor = receptor[:, np.newaxis]
    
    # make sure the distribution is positive
    peptide = peptide - np.min(peptide) + 1e-10
    receptor = receptor - np.min(receptor) + 1e-10
    
    mat = peptide @ receptor.T
    s = sc * mat
    s_neglog = np.log1p(s / (np.max(s) + 1))
    pep_comm = communication_measures(s, s_neglog, dist_mat)

    X = zscore(pep_comm, ddof=1, axis=1).T
    r2 = linear_fit(X, y)
    return r2

def scaled_robust_sigmoid(x):
    """
    Apply Scaled Robust Sigmoid normalization to a 1D numpy array.
    
    Parameters:
    x : np.ndarray or list
        Input array to be normalized.

    Returns:
    ----
    normalized: np.ndarray
        Normalized array with values between 0 and 1.
    """
    x = np.asarray(x, dtype=np.float64)
    median = np.median(x)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    
    if iqr == 0:
        raise ValueError("IQR is zero. SRS normalization cannot be applied.")

    # Step 1: Robust sigmoid transformation
    transformed = 1 / (1 + np.exp(-(x - median) / (iqr / 1.35)))

    # Step 2: Min-max scaling to [0, 1]
    min_val = np.min(transformed)
    max_val = np.max(transformed)
    if max_val == min_val:
        return np.zeros_like(transformed)  # or np.ones_like(transformed), depending on context
    normalized = (transformed - min_val) / (max_val - min_val)
    
    return normalized


def compute_dfc_var(time_series, window_size=60, step_size=30):
    """
    Compute variability of dynamic FC (sliding window FC) within a session.

    Parameters:
    - time_series: np.ndarray of shape (T, R), time series per region
    - window_size: int, number of time points in the sliding window
    - step_size: int, how much the window moves at each step

    Returns:
    - dFC_var: np.ndarray of shape (R, R), variance of dynamic FC
    """
    T, R = time_series.shape
    all_dFC = []
    for start in range(0, T - window_size + 1, step_size):
        window = time_series[start:start + window_size]
        window = zscore(window, axis=0)  # Normalize each region
        dFC = np.corrcoef(window, rowvar=False)
        all_dFC.append(dFC)
    dFC_var = np.stack(all_dFC).var(axis=0)
    
    return dFC_var