# %%
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns

# %%
enrichment_type = 'brainspan'
folder = f'./results/GO/{enrichment_type}/csv/'
files = sorted(glob.glob(folder + '*.csv'))

# load and concatenate all files
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# find the unique names in column cLabel
names = df['cLabel'].unique()
names.sort()

prenatal = [name for name in names if name.startswith('prenatal')]
infant = [name for name in names if name.startswith('infant')]
child = [name for name in names if name.startswith('child')]
adolescent = [name for name in names if name.startswith('adolescent')]
adult = [name for name in names if name.startswith('adult')]

# join together
names = prenatal + infant + child + adolescent + adult

# %%
"""
Create a dataframe where each row is a file 
and each column is an element of names. If
the current file doesn't contain a name, add 
a NaN value. Else use the value from pValPermCorr.
Then, create another dataframe with the same 
procedure but now using values from cScorePheno
"""
# create a dataframe with pValPermCorr
df_pValPermCorr = pd.DataFrame()
for f in files:
    temp = pd.read_csv(f)
    temp = temp.set_index('cLabel')
    temp = temp.reindex(names)
    df_pValPermCorr = pd.concat([df_pValPermCorr, temp['pValPermCorr']], axis=1)

# create a dataframe with cScorePheno
df_cScorePheno = pd.DataFrame()
for f in files:
    temp = pd.read_csv(f)
    temp = temp.set_index('cLabel')
    temp = temp.reindex(names)
    df_cScorePheno = pd.concat([df_cScorePheno, temp['cScorePheno']], axis=1)
    

# %%
# transpose dataframe and delete index
df_pValPermCorr = df_pValPermCorr.T
df_pValPermCorr.reset_index(drop=True, inplace=True)

df_cScorePheno = df_cScorePheno.T
df_cScorePheno.reset_index(drop=True, inplace=True)

# load receptor names from older file
df_receptors = pd.read_csv('./results/receptor_hth_coupling.csv')
receptor_names = df_receptors['Gene'].values

# use receptor names as new index
df_pValPermCorr.index = receptor_names
df_cScorePheno.index = receptor_names

# save to csv
df_pValPermCorr.to_csv(f'./results/GO/{enrichment_type}/pValPermCorr.csv')
df_cScorePheno.to_csv(f'./results/GO/{enrichment_type}/cScorePheno.csv')

# %%
# combine both dataframes into tall format
df_pValPermCorr = df_pValPermCorr.reset_index()
df_cScorePheno = df_cScorePheno.reset_index()

df_pValPermCorr = df_pValPermCorr.melt(id_vars='index', value_name='pValPermCorr')
df_cScorePheno = df_cScorePheno.melt(id_vars='index', value_name='cScorePheno')

# concatenate columns
df_GO = pd.concat([df_pValPermCorr, df_cScorePheno['cScorePheno']], axis=1)

# rename column index to receptor and variable to cell
df_GO = df_GO.rename(columns={'index': 'receptor', 'variable': 'cell'})


# save as csv
df_GO.to_csv(f'./results/GO/{enrichment_type}/GO.csv', index=False)

# %%
"""
Create a heatmap with where the size of a circle is based on the 
values of pValPermCorr and the color is based on the values of
cScorePheno. The heatmap should have the cell types in the x-axis
and the receptors in the y-axis.
"""
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

df = pd.read_csv(f'./results/GO/{enrichment_type}/GO.csv')
cscore = pd.read_csv(f'./results/GO/{enrichment_type}/cScorePheno.csv', index_col=0)
pval = pd.read_csv(f'./results/GO/{enrichment_type}/pValPermCorr.csv', index_col=0)
cscore = cscore.fillna(0)
pval = pval.fillna(1)


receptor_names = pval.index.values.tolist()
categories = pval.columns.values.tolist()

N = len(receptor_names)
M = len(categories)

xlabels = categories
ylabels = receptor_names

x, y = np.meshgrid(np.arange(M), np.arange(N))
s = -np.log10(pval.values + 1e-4)
c = cscore.values

fig, ax = plt.subplots(figsize=(25, 10), dpi=150)

R = s/s.max()/2 * 0.8
# plot circles and set to zero if value is NaN 
circles = [plt.Circle((j,(N-1)-i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]
col = PatchCollection(circles, array=c.flatten(), cmap="inferno")
ax.add_collection(col)

ax.set(xticks=np.arange(M), yticks=np.arange(N))
ax.set_yticklabels(ylabels[::-1]) # so that the first receptor is at the top
ax.set_xticklabels(xlabels, rotation=90)
ax.set_xticks(np.arange(M+1)-0.5, minor=True)
ax.set_yticks(np.arange(N+1)-0.5, minor=True)
ax.grid(which='minor', linewidth=1, alpha=0.2)
ax.set_aspect('equal', adjustable='box')

fig.colorbar(col, shrink=0.2, label='cScore', pad=0.01)
plt.tight_layout()

