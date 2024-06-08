# %%
import numpy as np
from neuromaps.nulls import burt2020

# %%
# Load atlas file
parcellation = 'data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz'

# Create numpy array from 1 to 455
array = np.arange(1, 456)

# Generate 10,000 nulls based on the array
nulls = burt2020(array, atlas='MNI152', density='1mm', parcellation=parcellation, n_perm=1, seed=0,
                 n_jobs=32)
