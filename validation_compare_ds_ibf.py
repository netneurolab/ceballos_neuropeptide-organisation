# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from abagen.io import read_probes, read_pacall
from abagen.datasets import fetch_microarray

# %%
# fetch matching microarray data
microarray = fetch_microarray(donors='all', data_dir='./data')
donor_gene_intensities = []

for n, (donor, data) in enumerate(microarray.items()):
        probes_fn = microarray[donor]['probes']
        pacall_fn = microarray[donor]['pacall']
        
        probes = read_probes(probes_fn)
        pacall = read_pacall(pacall_fn)
        
        pa_mean = pacall.mean(axis=1)
        intensity = pa_mean.groupby(probes['gene_symbol']).mean().values

        donor_gene_intensities.append(intensity)
        
genes_from_probes = pa_mean.groupby(probes['gene_symbol']).mean().index.values
mean_intensity = pd.DataFrame(np.mean(donor_gene_intensities, axis=0), index=genes_from_probes, columns=['mean_intensity'])
diff_stability = pd.read_csv('./data/abagen_genes_stability.csv')

# %% get diff stability for all genes
import nibabel as nib
from abagen.images import check_atlas
from abagen import get_expression_data, keep_stable_genes

# load atlas info
atlas = nib.load('./data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz')
atlas_info = pd.read_csv('./data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv')

# check atlas and atlas_info
atlas = check_atlas(atlas, atlas_info)

donor_expressions = get_expression_data(atlas, atlas_info, data_dir='./data/', ibf_threshold=0,
                                        norm_matched=False, missing='interpolate', 
                                        probe_selection='rnaseq', return_donors=True)

# turn dictionary into list
donor_list = []
for donor in donor_expressions:
    donor_list.append(donor_expressions[donor])

genes, ds = keep_stable_genes(donor_list, threshold=0, percentile=False, return_stability=True)
genes_in_abagen = donor_expressions['9861'].columns.values
genes_in_probes = mean_intensity.index.values

genes_in_abagen, idx, _ = np.intersect1d(genes_in_abagen, genes_in_probes, return_indices=True)

# get matching genes for mean intensity and diff stability
mean_intensity = mean_intensity[mean_intensity.index.isin(genes_in_abagen)]
ds = ds[idx]

# %%
gene_summary = pd.DataFrame({'mean_intensity': mean_intensity['mean_intensity'], 'diff_stability': ds}, 
                            index=genes_in_abagen)
gene_summary.to_csv('./data/gene_qc.csv')

# %%
# plot mean intensity vs diff stability
sns.regplot(data=gene_summary, x='mean_intensity', y='diff_stability', scatter_kws={'s': 0.5})
plt.xlabel('Mean gene detectability across probes')
plt.ylabel('Differential stability')

# %%
from scipy.stats import spearmanr
spearmanr(gene_summary['mean_intensity'], gene_summary['diff_stability'])
