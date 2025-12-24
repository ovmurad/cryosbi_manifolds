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
