# %%
from operator import le
from turtle import color
import numpy as np
import pandas as pd
from pyparsing import col
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# read brainspan data in ./data
brainspan = pd.read_csv('./data/brainspan_raw.csv', index_col=0)

# %%
# load receptor list
receptor_list = pd.read_csv('data/receptor_list.csv')
# select rows in test if they are in receptor_list['Gene']
overlapping_genes = brainspan.columns[brainspan.columns.isin(receptor_list['Gene'])]

abagen_genes = pd.read_csv('data/abagen_genes_Schaefer2018_400.csv', index_col=0).columns
overlapping_genes = overlapping_genes[overlapping_genes.isin(abagen_genes)]

# select overlapping genes from brainspan
receptor_genes = brainspan[overlapping_genes]

# %%
# store receptor_genes indices
sites = receptor_genes.index
sites = sites.str.split('_').str[1]
sites = sites.unique()

# %%
# developmental stages
stages = receptor_genes.index
stages = stages.str.split('_').str[0]
stages = stages.unique()
# %%
"""
create long form dataframe for sns.lineplot
"""
# create empty dataframe
long_form = pd.DataFrame(columns=['Gene', 'Site', 'Stage', 'Expression'])

# iterate through columns of receptor_genes
for i in range(receptor_genes.shape[1]):
    # iterate through rows of receptor_genes
    for j in range(receptor_genes.shape[0]):
        # what is the site
        current_site = receptor_genes.index[j].split('_')[1]
        # what is the stage
        current_stage = receptor_genes.index[j].split('_')[0]


        # append each row to long_form
        long_form = long_form.append({'Gene': receptor_genes.columns[i], 
                                      'Site': current_site, 
                                      'Stage': current_stage, 
                                      'Expression': receptor_genes.iloc[j, i]}, 
                                     ignore_index=True)
        
# # normalize values in Expression using IQR
# long_form['Expression'] = long_form.groupby(['Gene'])['Expression'].transform(lambda x: (x - x.quantile(0.25)) / (x.quantile(0.75) - x.quantile(0.25)))

# log transform values in Expression
long_form['Expression'] = np.log1p(long_form['Expression'])

# set NaN values to 0
long_form['Expression'] = long_form['Expression'].fillna(0)

# create dict with mapping of receptors to family using receptor_list
receptor_family = receptor_list.set_index('Gene')['Family'].to_dict()

# add family column to long_form
long_form['Family'] = long_form['Gene'].map(receptor_family)

# %%
# plot expression of receptor genes in different sites
# each site is a separate plot
# each gene is a line
# use a color palette for each family

# create a color palette for each family
family_palette = sns.color_palette('tab10', n_colors=len(long_form['Family'].unique()))

# create a figure with 16 subplots
fig, axs = plt.subplots(4, 4, figsize=(20, 10))

for i, ax in enumerate(axs.flat):
    # plot each gene in a site
    for j, gene in enumerate(long_form['Gene'].unique()):
        # select rows with site == sites[i] and gene == gene
        data = long_form[(long_form['Site'] == sites[i]) & (long_form['Gene'] == gene)]
        # plot line with color corresponding to family
        # which family is gene in
        family = data['Family'].unique()[0]
        # find index of family in family_palette
        j = list(long_form['Family'].unique()).index(family)
        sns.lineplot(data=data, x='Stage', y='Expression', ax=ax, color=family_palette[j], label=gene)
    ax.set_title(sites[i])
    ax.set_ylabel('LogExpression')
    ax.set_xlabel('Stage')


# fig, axs = plt.subplots(4, 4, figsize=(20, 10))
# for i, ax in enumerate(axs.flat):
#     sns.lineplot(data=long_form[long_form['Site'] == sites[i]], x='Stage', y='Expression', hue='Gene', ax=ax)
#     ax.set_title(sites[i])
#     ax.set_ylabel('LogExpression')
#     ax.set_xlabel('Stage')


# delete all legends except last one
for i, ax in enumerate(axs.flat):
    if i != 15:
        ax.get_legend().remove()
# move legend to the right
plt.legend(bbox_to_anchor=(1.05, 6))

# increase horizontal space between subplots
plt.subplots_adjust(hspace=0.5)


# %%
# average gene expression per family
family_avg = long_form.groupby(['Family', 'Stage', 'Site'])['Expression'].mean().reset_index()

# arrange Stage using custom order
family_avg['Stage'] = pd.Categorical(family_avg['Stage'], stages)

# plot average gene expression per family for each site
# each site is a separate plot

fig, axs = plt.subplots(4, 4, figsize=(20, 10))
for i, ax in enumerate(axs.flat):
    sns.lineplot(data=family_avg[family_avg['Site'] == sites[i]], x='Stage', y='Expression', hue='Family', ax=ax, palette=family_palette)
    ax.set_title(sites[i])
    ax.set_ylabel('LogExpression')
    ax.set_xlabel('Stage')

for i, ax in enumerate(axs.flat):
    ax.get_legend().remove()
plt.legend(bbox_to_anchor=(1.7, 5.5))
plt.subplots_adjust(hspace=0.5)