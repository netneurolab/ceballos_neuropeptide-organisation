# %%
import nibabel as nib
import numpy as np
import pandas as pd
from neuromaps.nulls import burt2020
from scipy.ndimage import zoom
from joblib import Parallel, delayed
from utils import index_structure

# %%
def downsample_nifti(input_file, output_file, target_voxel_size):
    # Load the original NIfTI file
    img = nib.load(input_file)
    data = img.get_fdata()
    affine = img.affine

    # Get the original voxel size from the affine matrix
    original_voxel_size = np.abs(np.diag(affine)[:3])

    # Calculate the zoom factors
    zoom_factors = original_voxel_size / target_voxel_size

    # Downsample the image data using nearest-neighbor interpolation (order=0)
    downsampled_data = zoom(data, zoom_factors, order=0)  # order=0 for nearest-neighbor interpolation

    # Ensure the downsampled data is integer type
    downsampled_data = downsampled_data.astype(np.int32)

    # Create a new affine matrix for the downsampled image
    new_affine = affine.copy()
    new_affine[:3, :3] = np.diag(target_voxel_size)

    # Create and save the new NIfTI image
    new_img = nib.Nifti1Image(downsampled_data, new_affine)
    nib.save(new_img, output_file)
    print(f"Downsampled NIfTI file saved to: {output_file}")

# Define input and output file paths
input_file = './data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz'
output_file = './data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-3mm.nii.gz'
target_voxel_size = [3.0, 3.0, 3.0]  # Target voxel size in mm

# Downsample the NIfTI file
downsample_nifti(input_file, output_file, target_voxel_size)

# %% NEUROSYNTH MAPS
# Load files
parcellation = 'data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-3mm.nii.gz'
neurosynth_maps = pd.read_csv('data/neurosynth_Schaefer400_TianS4.csv', index_col=0).values

# Generate nulls in parallel using joblib
nulls = Parallel(n_jobs=32, verbose=1)(delayed(burt2020)(nmap, atlas='MNI152', density='3mm', parcellation=parcellation,
                                               n_perm=10000, seed=0, n_jobs=1) for nmap in neurosynth_maps.T)

nulls = np.array(nulls)
np.save('data/neurosynth_nulls_Schaefer400_TianS4.npy', nulls)



# %% 
def downsample_nifti(input_file, output_file, target_voxel_size):
    # Load the original NIfTI file
    img = nib.load(input_file)
    data = img.get_fdata()
    affine = img.affine

    # Get the original voxel size from the affine matrix
    original_voxel_size = np.abs(np.diag(affine)[:3])

    # Calculate the zoom factors
    zoom_factors = original_voxel_size / target_voxel_size

    # Downsample the image data using nearest-neighbor interpolation (order=0)
    downsampled_data = zoom(data, zoom_factors, order=0)  # order=0 for nearest-neighbor interpolation

    # Ensure the downsampled data is integer type
    downsampled_data = downsampled_data.astype(np.int32)

    # Create a new affine matrix for the downsampled image
    new_affine = affine.copy()
    new_affine[:3, :3] = np.diag(target_voxel_size)

    # Create and save the new NIfTI image
    new_img = nib.Nifti1Image(downsampled_data, new_affine)
    nib.save(new_img, output_file)
    print(f"Downsampled NIfTI file saved to: {output_file}")

# Define input and output file paths
input_file = './data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_noHTH_space-MNI152_den-1mm.nii.gz'
output_file = './data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_noHTH_space-MNI152_den-3mm.nii.gz'
target_voxel_size = [3.0, 3.0, 3.0]  # Target voxel size in mm]

# Downsample the NIfTI file
downsample_nifti(input_file, output_file, target_voxel_size)


# %% OPIOID RECEPTOR PET MAPS
# Load files
parcellation = 'data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_noHTH_space-MNI152_den-3mm.nii.gz'
nt_densities = pd.read_csv('data/annotations/nt_receptor_densities.csv', index_col=0).iloc[:-1]

# KOR nulls
kor = nt_densities['KOR']
kor_nulls = burt2020(kor, atlas='MNI152', density='3mm', parcellation=parcellation, n_perm=10000, seed=0, n_jobs=1)
np.save('data/kor_nulls_Schaefer400_TianS4.npy', kor_nulls)

# MOR nulls
mor = nt_densities['MOR']
mor_nulls = burt2020(mor, atlas='MNI152', density='3mm', parcellation=parcellation, n_perm=10000, seed=0, n_jobs=1)
np.save('data/mor_nulls_Schaefer400_TianS4.npy', mor_nulls)



# %% ALL NEUROTRANSITTERS PET MAPS

# Load files
parcellation = 'data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_noHTH_space-MNI152_den-3mm.nii.gz'
nt_densities = pd.read_csv('data/annotations/nt_receptor_densities.csv', index_col=0)

# Generate nulls in parallel using joblib
nulls = Parallel(n_jobs=32, verbose=1)(delayed(burt2020)(nt_densities[nt], atlas='MNI152', density='3mm', parcellation=parcellation,
                                               n_perm=1000, seed=0, n_jobs=1) for nt in nt_densities.columns)
nulls = np.array(nulls)

# store in dictionary with keys as neurotransmitter names
nulls_dict = dict(zip(nt_densities.columns, nulls))

# save dictionary
np.save('data/nt_nulls_Schaefer400_TianS4.npy', nulls_dict)

# %% PEPTIDE RECEPTOR MAPS
# Load files
parcellation = 'data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_noHTH_space-MNI152_den-3mm.nii.gz'
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)

# only CTX-SBCTX
receptor_genes = index_structure(receptor_genes, structure='CTX-SBCTX').values

# Generate nulls in parallel using joblib
nulls = Parallel(n_jobs=32, verbose=1)(delayed(burt2020)(rgene, atlas='MNI152', density='3mm', parcellation=parcellation,
                                               n_perm=10000, seed=0, n_jobs=1) for rgene in receptor_genes.T)

nulls = np.array(nulls)
np.save('data/receptor_spatial_nulls_Schaefer400_TianS4.npy', nulls)

# %% INCLUDING HTH IN PEPTIDE RECEPTOR MAPS
# Load files
parcellation = 'data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-3mm.nii.gz'
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0).values

# Generate nulls in parallel using joblib
nulls = Parallel(n_jobs=32, verbose=1)(delayed(burt2020)(rgene, atlas='MNI152', density='3mm', parcellation=parcellation,
                                               n_perm=10000, seed=0, n_jobs=1) for rgene in receptor_genes.T)

nulls = np.array(nulls)
np.save('data/receptor_spatial_nulls_Schaefer400_TianS4_HTH.npy', nulls)
