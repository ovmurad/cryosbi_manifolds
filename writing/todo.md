This is my (luke's) log of some things to adjust in the code for adding clarity to the tutorial.

I am making a script based off of one of the `analysis` scripts, and basically denoting where things are unclear to me right now, this is still informal.

### Variables a user needs to fill in
- `PARAMS'
- `SUBSAMPLE_KEY_PAIRS'
- `DATASET_NAME'
- `DATA_NAME' 
- `TRAIN_NAME'
- `CLEAN_NAME'
- `UNIFORM_NAME'

These relate to files in the `params` and `points` folders


### Naming conventions, splitting strings
It is hard to figure out how to name folders with respect to `wc` for well-connected points, or related.

Something that is pretty tricky right now is these functions:
- `tslasso`
- `local_grad_estimation`
have a a `sample` field that interacts with the arrays input in `x_pts`, `affs`, `grads`, but currently it is difficult to figure out what array sizes are correct, and to track down `IndexError` commands
