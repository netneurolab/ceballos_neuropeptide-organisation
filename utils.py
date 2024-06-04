import itertools
import brainconn as bc
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm



def navigation_wu(nav_dist_mat, sc_mat, show_progress=True):
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
    N = len(W)

    if np.allclose(W, W.T):
        flag_triu = True
    else:
        flag_triu = False

    T = np.linalg.solve(np.diag(np.sum(W, axis=1)), W)
    _, hops, Pmat = bc.distance.distance_wei_floyd(L, transform=None)

    SI = np.zeros((N, N))
    SI[np.eye(N) > 0] = np.nan

    for i in range(N):
        for j in range(N):
            if (j > i and flag_triu) or (not flag_triu and i != j):
                path = bc.distance.retrieve_shortest_path(i, j, hops, Pmat)
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
    neg_sqrt = np.power(row_sum, -0.5)
    square_sqrt = np.diag(neg_sqrt)

    # normalize input matrix
    for_expm = square_sqrt @ adjacency @ square_sqrt

    # calculate matrix exponential of normalized matrix
    cmc = scipy.sparse.linalg.expm(for_expm)
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
        Structure to index by. Can be CTX, SBCTX, CTX-SBCTX
    Returns
    -------
    indexed_df : pandas.DataFrame
        Dataframe indexed by the specified structure
    """
    if structure == 'CTX':
        indexed_df = df.iloc[54:-1]
    elif structure == 'SBCTX':
        indexed_df = df.iloc[:54]
    elif structure == 'CTX-SBCTX':
        indexed_df = df.iloc[:-1]
    else:
        raise ValueError('Invalid structure. Must be CTX, SBCTX, or CTX-SBCTX')
    return indexed_df