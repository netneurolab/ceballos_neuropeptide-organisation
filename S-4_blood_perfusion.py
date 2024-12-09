# %%
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
from neuromaps.stats import compare_images
from utils import index_structure

savefig = False

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         CORRELATE GENE & CBF DATA
###############################################################################

genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
genes = index_structure(genes, structure='CTX-SBCTX')
nulls = np.load('data/receptor_spatial_nulls_Schaefer400_TianS4.npy')

cbf = np.load('data/annotations/cbf_hcpavg_Schaefer400_TianS4.npy')

# compare each gene to cbf map and get p-value from null distribution
corrs = []
pvals = []
for i, gene in enumerate(genes.columns):
    corr, pval = compare_images(genes[gene].values, cbf, nulls=nulls[i], metric='spearmanr')
    corrs.append(corr)
    pvals.append(pval)

# create plot_df
plot_df = pd.DataFrame({'gene': genes.columns, 'corr': corrs, 'pval': pvals})
plot_df['significant'] = plot_df['pval'] < 0.05
plot_df['p_fdr'] = pg.multicomp(plot_df['pval'], method='fdr_bh')[1]
plot_df['fdr_significant'] = plot_df['p_fdr'] < 0.05


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOT CORRELATION TO ALL GENES
###############################################################################

# plot correlation and p-values
# order by correlation
order = np.argsort(corrs)[::-1]
order = genes.columns[order]

fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=200)
sns.barplot(x='gene', y='corr', data=plot_df, color='lightblue', hue='significant', 
            ax=ax, order=order, dodge=False)
ax.set_xticklabels(order, rotation=90)

fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=200)
sns.barplot(x='gene', y='pval', data=plot_df, color='orange', ax=ax, order=order)
ax.set_xticklabels(order, rotation=90)
# draw line at 0.05
ax.axhline(0.05, color='grey', linestyle='--')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#          CORRESPONDENCE BETWEEN ENDOTHELIN RECEPTORS AND CBF
###############################################################################

# select endothelin receptors
ednra = genes['EDNRA'].values
ednrb = genes['EDNRB'].values

# compute correlation and retrieve FDR-corrected p-value
corr_ednra = compare_images(ednra, cbf, metric='spearmanr')
corr_ednrb = compare_images(ednrb, cbf, metric='spearmanr')
p_ednra = plot_df.loc[plot_df['gene'] == 'EDNRA', 'pval'].values[0]
p_ednrb = plot_df.loc[plot_df['gene'] == 'EDNRB', 'pval'].values[0]

# load atlas for regional references
atlas = pd.read_csv('data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv', index_col=0)
atlas = index_structure(atlas, structure='CTX-SBCTX')

# plot scatterplot of EDNRA and EDNRB against cbf
fig, ax = plt.subplots(1, 2, figsize=(15, 6), dpi=200, sharey=True)
sns.scatterplot(x=ednra, y=cbf, ax=ax[0], hue=atlas['network'])
sns.regplot(x=ednra, y=cbf, ax=ax[0], scatter=False, color='grey', ci=None)
sns.scatterplot(x=ednrb, y=cbf, ax=ax[1], hue=atlas['network'])
sns.regplot(x=ednrb, y=cbf, ax=ax[1], scatter=False, color='grey', ci=None)

ax[0].set_title(f'EDNRA\nr={corr_ednra:.2f}, p={p_ednra:.4f}')
ax[1].set_title(f'EDNRB\nr={corr_ednrb:.2f}, p={p_ednrb:.4f}')
ax[0].set_xlabel('EDNRA expression')
ax[1].set_xlabel('EDNRB expression')
ax[0].set_ylabel('Cerebral Blood Flow (mL/100g/min)')
ax[0].get_legend().remove()
ax[1].legend(title='Network', frameon=False)
sns.despine()

if savefig:
    fig.savefig('figs/endothelin_receptors_cbf.pdf')

