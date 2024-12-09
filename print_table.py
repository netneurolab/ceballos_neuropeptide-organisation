# %%
import pandas as pd

# %%
# load all peptide names
names = pd.read_csv('data/precursor_list.csv', index_col=0)

# load precursor_qc
precursor_qc = pd.read_csv('data/precursor_qc.csv', index_col=0)
# swap second and third columns
precursor_qc = precursor_qc.iloc[:, [0, 2, 1]]

# add name description to precursor_qc by mapping indices to precursor_qc
precursor_qc['description'] = precursor_qc.index.map(names['Description'])

# %%
# load pairings
pairings = pd.read_csv('data/peptide_receptor_ligand_pairs.csv', index_col=0)
# load receptor_overview
receptor_overview = pd.read_csv('data/receptor_overview.csv', index_col=0)
# map family to pairings
pairings['family'] = pairings['Receptor'].map(receptor_overview['family'])

# turn Peptide into index
pairings = pairings.set_index('Peptide')

# map family to precursor_qc
precursor_qc['family'] = precursor_qc.index.map(pairings['family'].to_dict())

# replace NaN with 'Unclassified'
precursor_qc['family'] = precursor_qc['family'].fillna('Unclassified')

# %%
# load abagen gene expression
all_genes = pd.read_csv('data/abagen_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)

# only precursor
precursor_genes = all_genes[all_genes.columns.intersection(precursor_qc.index)]

# average across first 54 rows for sbctx expression per gene
sbctx = precursor_genes.iloc[:54].mean(axis=0)

# cortex is 54:-1
ctx = precursor_genes.iloc[54:-1].mean(axis=0)

# hth is the last row
hth = precursor_genes.iloc[-1]

# add values to precursor_qc
precursor_qc['sbctx'] = sbctx
precursor_qc['ctx'] = ctx
precursor_qc['hth'] = hth


# %%
# swap order of columns: description, family, mean_intensity, mean_RNAcorr, diff_stability, ctx, hth, sbctx
precursor_qc = precursor_qc.iloc[:, [3, 4, 0, 1, 2, 6, 7, 5]]
#  save as precursor_overview
precursor_qc.to_csv('data/precursor_overview.csv')

# save also as latex table
precursor_qc.to_latex('data/precursor_overview.tex')
# %%
