# cryosbi_manifolds

## Installation Guide.
- The project requires python3.12 or newer, so please install such a version.
- Installation requires: cloning the directory, installing the package with pip, and installing megaman from lib. This can be all be done via:
```
git clone https://github.com/ovmurad/cryosbi_manifolds.git
cd ensemble_sim
python -m pip install .
python -m pip install lib/megaman-0.1.0-py3-none-any.whl
```
- After that,  you should be all set!


The main analysis from the manuscript [''Cryo-em images are intrinsically low dimensional''](https://arxiv.org/abs/2504.11249) is carried out in:
- `scripts/analysis_igg.py` for the simulated igg data. Data available in this repo on `data/igg_data`, from the file `igg_data.zip` from the Zenodo below.
- `scripts/analysis_hem.py` for the simulated and experimental hemagglutinin data. Needs `hemagglutinin_data.zip` from the Zenodo below.

The data to use in the above scripts can be downloaded from [https://zenodo.org/records/15733579](https://zenodo.org/records/15733579).
