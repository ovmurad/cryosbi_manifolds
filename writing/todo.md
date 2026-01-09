Luke 01/09
=========
I tried to re-write one of the `analysis` scripts with minimal API, just to see how different the process would be using the Arrays directly instead of the `Database` dictionaries.
The file is [here](https://github.com/ovmurad/cryosbi_manifolds/blob/main/scripts/analysis_ethanol_new.py).

After looking through, there are some things I think that could be in a tutorial that aren't included in the previous scripts (maybe these are in other notebooks contributed here that I haven't looked at yet):
- Choosing the range for `MIN_R, MAX_R` and step-size
  - E.g, a few notebook cell with several sub-optimal choices, and picking the one based on properties of the histogram of nearest neighbors
- Choosing `RADIUS` for the embedding.
  - E.g, a few notebook cell with several sub-optimal choices, and picking the one based on some property (I'm not sure what it is here)

Some other todos for this script above:
    - TODO: check that slicing is correct in the sub-sampling.  

      - In this new script above, I am slicing `CSR` arrays a few times, and I'm not sure if I'm respecting the index-slicing rules correctly. I think this is done implicitly in `Database` by the `name|str_str` but I'm not sure where this happens.

    - TODO: include `IES`, `TSLASSO`

    - TODO: verify that the above script is working approximately right for the ethanol dataset
      - e.g: is it getting expected intrinsic dims and embeddings?

Some future todos:
  - TODO: I/O for passing data from `analysis` to `visualizing`
    - currently, the plotting, or embedding or dim. estimation scripts, e.g [here](https://github.com/ovmurad/cryosbi_manifolds/blob/main/scripts/visualize_intrinsic_dim.py), have dataset names and expected for very specific datasets
    - sub-TODO: make a `generic` script for a "simple" dataset
      - "simple": no `sim`/`exp` to pass in, or `params`, i.e. just doing the manifold learning for a dataset, which initally is passed in as one `num_data_points x num_features` array, that isn't necessarily from SBI, since many parts of this library are useful general manifold learning

  - TODO: transition to jupyter notebooks for tutorial
    - Not fun, BUT it's ubiquitous for tutorials, and will be easiest to tell if stuff could be usable `live'.

MMP 12/23
=========
* broke the presentation into modules, i.e. module-<something>.tex
* I added my tex slides from the last tutorial (edinburgh, very similar to IMSI).
* to make the new presentation compile, I added mmp-commands.tex to macros/
* and I added figures/figures-MMP
* These additions can/should be merged later in the tutorial structure, they are just temporary
* i also added a \graphicspath command
* with all these, compilation is a pain, and I don't think all the figures should be copied anyways. So currently the presentation has a lot of compilation errors. It's obvious that re-typing the slides is faster than making the originals compile, therefore I also added the pdf of the edinburgh talk, in tutorial/
Sorry for the relative mess, I think it can be fixed easily by re-typing each module and inputting the figures. Then we can delete my original modules. 


Luke
====
This is my (luke's) log of some things to adjust in the code for adding clarity to the tutorial.

I am making a script based off of one of the `analysis` scripts, and basically denoting where things are unclear to me right now, this is still informal.

### Variables/names a user needs to fill in
- `PARAMS`
- `SUBSAMPLE_KEY_PAIRS`
- `DATASET_NAME`
- `DATA_NAME`
- `TRAIN_NAME`
- `CLEAN_NAME`
- `UNIFORM_NAME`

These above relate to files in the `params` and `points` folders

- `EMB_NPTS`
- `SPLIT_NPTS`
- `IES_SUBSAMPLE_SIZE`
- `TSLASSO_SUBSAMPLE_SIZE`
- `GRAD_ESTIM_SUBSAMPLE_SIZE`
- `EIGENGAP_ESTIM_SUBSAMPLE_SIZE`


### Ranges/values that should be changed per dataset
- `MIN_K, MAX_K`, maybe?
- `MIN_R, MAX_R`
- `RADIUS`
- `DS`?


### Naming conventions, splitting strings
It is hard to figure out how to name folders with respect to `wc` for well-connected points, or related. These are `_` names that indicate a mask that should be applied.

Something that is pretty tricky right now is these functions:
- `tslasso`
- `local_grad_estimation`
have a a `sample` field that interacts with the arrays input in `x_pts`, `affs`, `grads`, but currently it is difficult to figure out what array sizes are correct, and to track down `IndexError` commands
