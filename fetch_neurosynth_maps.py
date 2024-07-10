# %%
import contextlib
import json
import os
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import requests
from nilearn.input_data import NiftiLabelsMasker
from nilearn._utils import check_niimg
from nibabel.loadsave import save as niisave

from nimare.dataset import Dataset
from nimare.meta.cbma.ale import ALE
from nimare.correct import FDRCorrector
from nimare.io import convert_neurosynth_to_dataset
from nimare.extract import fetch_neurosynth

from urllib.request import urlopen
import sys

# /sigh
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

atlas = './data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz'
labels = pd.read_csv('./data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv')['name'].to_list()

# %%
# this is where the raw and parcellated data will be stored
NSDIR = Path('./data/neurosynth/raw').resolve()
PARDIR = Path('./data/neurosynth/derivatives').resolve()

# these are the images from the neurosynth analyses we'll save
# can add 'uniformity-test_z' plus more, if desired
IMAGES = 'z_corr-FDR_method-indep'


def download(path='.', url=None, unpack=False):
    """ Download the latest data files.
    Args:
        path (str): Location to save the retrieved data files. Defaults to
            current directory.
        unpack (bool): If True, unzips the data file post-download.
    """

    if url is None:
        url = 'https://github.com/neurosynth/neurosynth-data/blob/e8f27c4/current_data.tar.gz?raw=true'
    if os.path.exists(path) and os.path.isdir(path):
        basename = os.path.basename(url).split('?')[0]
        filename = os.path.join(path, basename)
    else:
        filename = path

    f = open(filename, 'wb')

    u = urlopen(url)
    file_size = int(u.headers["Content-Length"][0])
    print("Downloading the latest Neurosynth files: {0} bytes: {1}".format(
        url, file_size))

    bytes_dl = 0
    block_size = 8192
    while True:
        buffer = u.read(block_size)
        if not buffer:
            break
        bytes_dl += len(buffer)
        f.write(buffer)
        p = float(bytes_dl) / file_size
        status = r"{0}  [{1:.2%}]".format(bytes_dl, p)
        status = status + chr(8) * (len(status) + 1)
        sys.stdout.write(status)

    f.close()

    if unpack:
        import tarfile
        tarfile.open(filename, 'r:gz').extractall(os.path.dirname(filename))



def fetch_ns_data(directory):
    """ Fetches NeuroSynth database + features to `directory`
    Paramerters
    -----------
    directory : str or os.PathLike
        Path to directory where data should be saved
    Returns
    -------
    database, features : PathLike
        Paths to downloaded NS data
    """

    directory = Path(directory)

    # if not already downloaded, download the NS data and unpack it
    database = directory / 'neurosynth_dataset.pkl.gz'
    if not database.exists():
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            neurosynth_db = fetch_neurosynth(data_dir=directory)
            # download(path=directory, unpack=True)

        neurosynth_db = neurosynth_db[0]
        neurosynth_dset = convert_neurosynth_to_dataset(
                        coordinates_file=neurosynth_db["coordinates"],
                        metadata_file=neurosynth_db["metadata"],
                        annotations_files=neurosynth_db["features"])
        neurosynth_dset.save(database)
    else:
        neurosynth_dset = Dataset.load(database)
    
    return neurosynth_dset


def get_cogatlas_concepts(url=None):
    """ Fetches list of concepts from the Cognitive Atlas
    Parameters
    ----------
    url : str
        URL to Cognitive Atlas API
    Returns
    -------
    concepts : set
        Unordered set of terms
    """

    if url is None:
        url = 'https://cognitiveatlas.org/api/v-alpha/concept'

    req = requests.get(url)
    req.raise_for_status()
    concepts = set([f.get('name') for f in json.loads(req.content)])

    return concepts


def run_meta_analyses(dset, use_features=None, outdir=None, sample_size=30):
    """
    Runs NiMARE-style meta-analysis based on `database` and `features`
    Parameters
    ----------
    dset: nimare.dataset.Dataset or os.PathLike
        Dataset object or path to dataset file
    use_features : list, optional
        List of features on which to run NiMARE meta-analyses; if not supplied all
        terms in `features` will be used
    outdir : str or os.PathLike
        Path to output directory where derived files should be saved
    Returns
    -------
    generated : list of str
        List of filepaths to generated term meta-analysis directories
    """

    # check outdir
    if outdir is None:
        outdir = NSDIR
    outdir = Path(outdir)
    
    # ensure 'sample_sizes' field exists
    if 'sample_sizes' not in dset.metadata.columns:
        dset.metadata['sample_sizes'] = sample_size

    # if we only want a subset of the features take the set intersection
    if use_features is not None:
        labels = dset.get_labels()
        
        # get only last part of the label as matching term
        terms = [l.split('__')[-1] for l in labels]
        terms = set(terms) & set(use_features)

        # find matching labels
        features = set([l for l in labels if l.split('__')[-1] in terms])
    else:
        features = set(dset.get_labels())
    pad = max([len(f) for f in features])

    generated = []
    for word in sorted(features):
        msg = f'Running meta-analysis for term: {word:<{pad}}'
        print(msg, end='\r', flush=True)

        # run meta-analysis + save specified outputs (only if they don't exist)
        path = outdir / word.split('__')[-1]
        path.mkdir(exist_ok=True)
        if not (path / f'{IMAGES}.nii.gz').exists():
            # find studies with term
            ids = dset.get_studies_by_label(word)
            subset_dset = dset.slice(ids)
            
            # run meta-analysis
            ma = ALE()
            result = ma.fit(subset_dset)
            corrector = FDRCorrector(alpha=0.01, method='indep')
            corrected_results = corrector.transform(result)
            nii = corrected_results.get_map(IMAGES)
            nii.to_filename(path / f'{IMAGES}.nii.gz')

        # store MA path
        generated.append(path)

    print(' ' * len(msg) + '\b' * len(msg), end='', flush=True)

    return generated


def parcellate_meta(outputs, annots, fname, regions):
    # empty dataframe to hold our parcellated data
    data = pd.DataFrame(index=regions)
    mask = NiftiLabelsMasker(annots, resampling_target='data')

    for outdir in outputs:
        cdata = []
        mgh = outdir / 'z_corr-FDR_method-indep.nii.gz'

        cdata.append(mask.fit_transform(
            check_niimg(mgh.__str__(), atleast_4d=True)).squeeze())

        # aaaand store it in the dataframe
        data = data.assign(**{outdir.name: np.hstack(cdata)})

    # now we save the dataframe! wooo data!
    data.to_csv(fname, sep=',')
    return fname

# %%
if __name__ == '__main__':
    NSDIR.mkdir(parents=True, exist_ok=True)
    PARDIR.mkdir(parents=True, exist_ok=True)

    # get concepts from CogAtlas and run relevant NS meta-analysess,
    dset = fetch_ns_data(NSDIR)
    generated = run_meta_analyses(dset, get_cogatlas_concepts(),
                                  outdir=PARDIR)
    
    # parcellate data and save to directory
    parcellate_meta(generated, atlas,
                    PARDIR / 'Schaefer2018_400_7N_Tian_Subcortex_S4_neurosynth.csv',
                    regions=labels)

