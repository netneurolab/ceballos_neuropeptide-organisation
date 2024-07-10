# %%
import nibabel as nib
import numpy as np
import pandas as pd
from neuromaps.nulls import burt2020
from scipy.ndimage import zoom
from joblib import Parallel, delayed

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

# %%
# Load files
parcellation = 'data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-3mm.nii.gz'
neurosynth_maps = pd.read_csv('data/neurosynth_Schaefer400_TianS4.csv', index_col=0).values

# Generate nulls in parallel using joblib
nulls = Parallel(n_jobs=32, verbose=1)(delayed(burt2020)(nmap, atlas='MNI152', density='3mm', parcellation=parcellation,
                                               n_perm=10000, seed=0, n_jobs=1) for nmap in neurosynth_maps.T)

nulls = np.array(nulls)
np.save('data/neurosynth_nulls_Schaefer400_TianS4.npy', nulls)

