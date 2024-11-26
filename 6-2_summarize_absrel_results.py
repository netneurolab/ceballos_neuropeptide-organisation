# %%
import glob
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
from plot_utils import divergent_green_orange

savefig = False
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         LOAD DATA
###############################################################################
files = glob.glob('./data/evo/06absrel_results/*.json')
branches = ['pmarinus',
            'ccarcharias',
            'drerio',
            'xtropicalis',
            'ggallus',
            'oanatinus',
            'sharrisii',
            'dnovemcinctus',
            'btaurus',
            'mmusculus',
            'mmulatta',
            'ptroglodytes',
            'hsapiens']

nt_classes = pd.read_csv('./data/annotations/nt_receptor_gene_classes.csv')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                      SUMMARIZE ABSREL RESULTS
###############################################################################
# create a dataframe to store results for each gene and branch
col_names = ['gene', 'type', 'branch', 'omega_rate', 'lrt', 'dn/ds', 'p', 'p_bonferroni']
df = pd.DataFrame(np.nan, index=range(len(files)*len(branches)), columns=col_names)
p_fdr = []
for i,file in enumerate(files):
    # what is the gene name?
    gene = file.split('/')[-1].split('_')[0]
    # is it a neurotransmitter or a peptide?
    if np.isin(gene, nt_classes['gene']):
        gene_type = nt_classes[nt_classes['gene']==gene]['type'].values[0]
        gene_type = str(gene_type).lower()
    else:
        gene_type = 'peptide'
    # fetch data from hyphy json files
    with open(file, 'r') as f:
        data = json.load(f)
        for j, branch in enumerate(branches):
            idx = i*len(branches) + j
            try:
                omega_rate = data['branch attributes']['0'][branch]['Rate classes']
                rate_distribution = [value for value in data['branch attributes']['0'][branch]['Rate Distributions'] 
                                     if value != 1]
                dn_ds = np.max(rate_distribution)
                dn_ds = np.log10(dn_ds) if dn_ds != 0 else 0
                dn_ds = 7 if dn_ds > 7 else dn_ds
                lrt = data['branch attributes']['0'][branch]['LRT']
                p = data['branch attributes']['0'][branch]['Uncorrected P-value']
                p_bonferroni = data['branch attributes']['0'][branch]['Corrected P-value']
            except KeyError:
                omega_rate = np.nan
                dn_ds = np.nan
                p = np.nan
                p_bonferroni = np.nan
            # store in dataframe
            df.loc[idx] = [gene, gene_type, branch, omega_rate, lrt, dn_ds, p, p_bonferroni]
    # correct p-values of gene with fdr
    p_fdr.append(pg.multicomp(df[df['gene']==gene]['p'], method='fdr_bh')[1])

order = df['branch'].unique().tolist()
df['p_fdr'] = np.concatenate(p_fdr)
# df.to_csv('results/absrel_results.csv', index=False)

# test for differences in dn/ds between signaling types
pg.friedman(data=df, dv='dn/ds', within='type', subject='branch', method='chisq')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                      PLOT ABSREL RESULTS
###############################################################################
# prepare df and colormap
plot_df = df.copy()
palette = divergent_green_orange(n_colors=9, return_palette=True)
colors = [palette[i] for i in [1, 4, 7]]

plt.figure(figsize=(6, 8))
bp = sns.boxplot(data=plot_df, x='dn/ds', y='branch', hue='type', showfliers=False, 
                 palette='Spectral', hue_order=['ionotropic', 'metabotropic', 'peptide'])
plt.legend(frameon=False)
plt.xlabel('Max. dn/ds distribution (log-transformed)')
sns.despine()

if savefig:
    plt.savefig('figs/dn_ds_boxplot.pdf', dpi=300, bbox_inches='tight')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                PLOT MEDIAN VALUES PER BRANCH AND TYPE
###############################################################################
medians = plot_df.groupby(['branch', 'type'])['dn/ds'].median().reset_index()
medians = medians.pivot(index='branch', columns='type', values='dn/ds')
medians = medians.sort_values('branch')
medians = medians.reindex(index=order)

plt.figure(figsize=(4, 8))
sns.heatmap(medians, cmap=divergent_green_orange(), center=0, vmax=0.5, linewidths=0.5, 
            linecolor='white', cbar_kws={'shrink': 0.5, 'label': 'Median dn/ds (log-transformed)'})
# plt.xticks(rotation=45)
if savefig:
    plt.savefig('figs/median_dn_ds_heatmap.pdf', dpi=300, bbox_inches='tight')