# Running the analysis code

Run everything from the **project root** (not from `code/`), because scripts write to
`figures/` and `tables/` by relative path:

```
julia +1.12 --project=code code/<script>.jl
```

For a fresh download, first obtain the original datasets and instantiate the
pinned environment:

```
python3 code/fetch_data.py
julia +1.12 --project=code --startup-file=no -e 'using Pkg; Pkg.instantiate()'
julia +1.12 --project=code --startup-file=no code/make.jl
```

The downloader uses Python 3's standard library, checks every input against its
recorded SHA-256 fingerprint, and leaves matching existing files in place. It
stops rather than overwriting an existing file with a different fingerprint.
See `data/README.md` for sources and manual download instructions. The original
datasets are downloaded from their owners; they are not bundled in the release.

**Julia 1.12 is required.** Julia 1.13 cannot precompile the CairoMakie version pinned in
`Manifest.toml` (its Unitful dependency fails to parse), so `figures.jl` and `empirical.jl`
will not load. Install it with `juliaup add 1.12`. Do not re-resolve the Manifest to work
around this: the pinned versions are what the reported numbers were computed with.

**After any change to code, data or text, run:**

```
julia +1.12 --project=code code/make.jl            # ~50 s
julia +1.12 --project=code code/make.jl --no-figures   # tables + analyses + verify
```

`make.jl` regenerates `tables/*.md` (`tables.jl`, `inference.jl`, `invariance.jl`) and the three figures (`figures.jl`,
`figure_warp.jl`), then runs `verify_numbers.jl`, and exits 1 at the first failure.

The tables are Markdown pipe tables. Both manuscript sources pull them in with
`{{< include tables/....md >}}`. The invariance contrast table includes its caption;
the other tables receive captions and `#tbl-` labels in the manuscript sources.
Edit empirical cells in `tables.jl`, interval cells in `inference.jl`, and contrast
cells and their caption in `invariance.jl`.

`verify_numbers.jl` reads the value of every checked number **from the `.qmd` text** (each
check names the passage, with `#` where the number sits) and compares it with what the code
computes, at the precision printed. It fails when a number in the text disagrees with the
code, when a checked passage has been reworded (update the check's passage to match), or when
`tables/*.md` differs from what `tables.jl` now produces. `--all` lists every check.
When you add a number to the text, add a `cite(...)` for it.

The verifier also runs `regression_repetition_summary()` in `simulation.jl` with seeds 1–200
and checks the supplementary means and standard deviations of slopes and slope ratios,
plus the main-text summary. The figure retains its original seed-789 data set.

Shared code paths, so figures, tables and text cannot drift apart:

- `simulation.jl` `boundary_panel()` — Figure 2A and its text. Exact expectations by
  enumerating every binomial outcome; no Monte Carlo error.
- `simulation_gaps.jl` `panel_b_aggregation()` — Figure 2B and its text.
- `colloff.jl` — the Colloff et al. loader and the choosing-rate separation quartiles (ties in $J$ split
  proportionally, so the quartiles do not depend on row order), used by `tables.jl` and
  the verifier.
- `colloff_wixted.jl` — loader and showup-minus-simultaneous-showup contrast (PPB, $J$, log-linear $c$) for
  Experiments 1 to 3 of Colloff and Wixted (2020). Each participant has one trial of each type, so every
  participant is a boundary case; used by the verifier.

Scripts that only print (no files written): `simulation.jl` and `simulation_gaps.jl` (the
full simulation grids), `empirical.jl`.

`inference.jl` defines the Clopper–Pearson sum interval and the adjusted (add-one) interval, enumerates their exact coverage and width against the normal interval, and generates the interval coverage table. `invariance.jl` computes the
Snodgrass and Corwin tests and generates paired/Welch strength-contrast intervals from
`data/layher2020/` and `data/measuring_memory/`. Raw J is a primary discrimination index.

`load_colloff()` defaults to any identification versus rejection in both trial types;
`load_colloff(response=:identification)` uses the alternative perpetrator-only response coding.
The latter sum is not twice a common yes rate. Paired own-race contrasts match participants
by identifier and retain within-participant covariance. Sparse individual intervals use
simultaneous Clopper–Pearson intervals; participant-population intervals use t methods.

Saving a figure directly into this iCloud-synced directory can fail with
`SystemError: close: Operation timed out`, and the failed write **deletes** the target file.
`figures.jl` therefore renders to a temp file and copies it in (`safe_save`); do the same in
any new plotting script.

`ppb.R` is a base-R port of the PPB point estimate and the four intervals in `inference.jl` (`ppb_interval`, `wald_interval`, `adjusted_ppb_interval`, `ppb_difference_interval`), with usage in its header. `check_ppb_r.jl` confirms agreement with the Julia functions on a grid that includes boundary counts (requires Rscript; not run by `make.jl`).
