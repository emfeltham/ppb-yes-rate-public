# PPB replication package

Analysis code for *The Yes Rate Revisited: PPB as a Distribution-Free Complement
to Youden's J*, by Eric M. Feltham. PPB is the sum of the hit and false alarm
rates; Youden's J is their difference.

## Reproduce the results

Requirements: Julia 1.12 (the pinned environment was generated with 1.12.5),
Python 3, and internet access for the initial package and data downloads.
Quarto with Typst is needed only to render the manuscripts.

From the extracted package or cloned repository root:

```sh
python3 code/fetch_data.py
julia +1.12 --project=code --startup-file=no -e 'using Pkg; Pkg.instantiate()'
julia +1.12 --project=code --startup-file=no code/make.jl
```

`make.jl` regenerates the tables and figures and checks the reported numerical
results against the manuscript and supplement. It stops at the first failure.
Use `--no-figures` to regenerate tables and run the checks without plotting.
Simulation seeds are specified in the analysis code.

For the PPB point estimate and confidence intervals in base R, run
`source("code/ppb.R")`. Usage examples are in that file's header. The optional
`code/check_ppb_r.jl` checks the R intervals against the Julia implementation;
it requires `Rscript` and is separate from the main replication pipeline.

To render the manuscripts after regeneration:

```sh
quarto render ppb_paper.qmd --to typst
quarto render ppb_supplement.qmd --to typst
```

See [code/README.md](code/README.md) for the analysis entry points and
[data/README.md](data/README.md) for the original datasets. The downloader checks
input SHA-256 fingerprints recorded in `data/inputs.json`; `--check-only`
verifies local inputs without downloading them. Original datasets are obtained
from their owners and are not redistributed in this package.

## Package contents

- Manuscript and supplement sources, bibliography, and citation style.
- Code and the pinned `code/Project.toml` and `code/Manifest.toml` environment.
- Generated tables and figures for comparison with a fresh run.
- Input download URLs and fingerprints, and `SHA256SUMS` for the packaged files.

`python3 code/package_replication.py` builds `ppb_replication.zip` from an
explicit list of files. It excludes original datasets, internal notes, unrelated
analyses, and local configuration. Extract this archive to obtain the files
intended for the public repository. Rendered PDFs are prepared separately from
the replication package.
