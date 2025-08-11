# cryosbi_manifolds

## Installation Guide.
- The project requires python3.12 or newer, so please install such a version.
- We recommend installing the project in a virtual environment, such as a python `venv`. An example script for creating a venv `cryosbi_manifolds` in a parent directory `VENVS_DIR`, and then activating the environment, is
```
python -m venv VENVS_DIR/cryosbi_manifolds
source VENVS_DIR/cryosbi_manifolds/bin/activate
```
- After activating a virtual environment, installation requires: cloning the directory, installing the package with pip, and installing megaman from lib. This can be be done via:
```
git clone https://github.com/ovmurad/cryosbi_manifolds.git
cd cryosbi_manifolds
python -m pip install .
python -m pip install lib/megaman-0.1.0-py3-none-any.whl
```
- After that,  you should be all set!


The main analysis from the manuscript [''Cryo-em images are intrinsically low dimensional''](https://arxiv.org/abs/2504.11249) is carried out in:
- `scripts/analysis_igg.py` for the simulated igg data. Data available in this repo on `data/igg_data`, from the file `igg_data.zip` from the Zenodo below
- `scripts/analysis_hem.py` for the simulated and experimental hemagglutinin data. Needs `hemagglutinin_data.zip` from the Zenodo below, imported as a folder `hemagglutinin_data` to the `data/` folder.

The data to use in the above scripts can be downloaded from [https://zenodo.org/records/15733579](https://zenodo.org/records/15733579).

Visualization of the results can be found from the routines in the scripts:
- `scripts/visualize_embeddings.py`
- `scripts/visualize_intrinsic_dim.py`
- `scripts/visualize_tslasso.py`

---

The `analysis`  scripts have many hyperparameters can be quickly adjusted. These include:
- `EMB_NPTS`: this sets the size of the dataset after it is subsampled to be approximately uniform density, and corresponding to the size of $\mathcal{X}_{unif}$ in the manuscript. In our case, N=100,000 for the datasets, and we subsample to `EMB_NPTS`=20,000 We recommend to subsample to at least 1/3 smaller than the original data.
- `SPLIT_NPTS`: this sets the training and test split. By default, this is set to 60,000, two-thirds of the original 100,000 points. 

The variables `IES_SUBSAMPLE_SIZE`, `TSLASSO_SUBSAMPLE_SIZE`, `GRAD_ESTIM_SUBSAMPLE_SIZE`, `EIGENGAP_ESTIM_SUBSAMPLE_SIZE` in the `analysis` scripts all set subsampling sizes to ease computational time of processes, and can be increased for more accuracy at the expense of computational time.

The radii and knn ranges for manifold learning and dimension estimation are adjustable as well in the scripts, although these can require more care than the above, and further updates to the repo will outline how to select these for more general datasets. 


