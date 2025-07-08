# cryosbi_manifolds

## Installation Guide.

- Install Poetry.
- The project requires python3.12 or newer, so please install such a version.
- From your terminal with Poetry environment corresponding to the cryosbi_manifold project loaded run: `poetry install`.
- From the terminal, assuming you are in the project directory, run: `pip install lib/megaman-0.1.0-py3-none-any.whl`.
- You should be all set!

The main analysis from the manuscript [``Cryo-em images are intrinsically low dimensional''](https://arxiv.org/abs/2504.11249) is carried out in:
- `scripts/analysis_hem.py` for the simulated and experimental hemagglutinin data
- `scripts/analysis_igg.py` for the simulated igg data

The data to use in the above scripts can be downloaded from [https://zenodo.org/records/15733579](https://zenodo.org/records/15733579).
