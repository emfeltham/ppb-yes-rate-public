# Data sources

The files required by `code/make.jl` can be downloaded from the original sources
with `python3 code/fetch_data.py`, run from the project root. The downloader
checks SHA-256 fingerprints listed in `data/inputs.json`. Use `--check-only`
to verify local files without downloading. The GitHub inputs are pinned to
pyWitness commit `06b9eb43828e3ffdd422900cdc6c4877db2e5f6b`; the OSF fingerprints
match the hashes reported by the source projects. These original files are not
bundled in the replication package.

## Colloff et al. Experiment 1 used in the paper

Download `Exp1_osf_data.csv` from the
[pyWitness published-data collection](https://github.com/lmickes/pyWitness/tree/main/data/published/2020_Colloff_Flowe_Smith_etal)
and place it at
`data/pyWitness/data/published/2020_Colloff_Flowe_Smith_etal/Exp1_osf_data.csv`.
The loader retains rows with `Include == "yes"`. Its primary response is any
identification versus rejection in both target-present and target-absent trials.
The perpetrator-only coding is a separate sensitivity analysis. The supplementary
response table records all included raw counts.

## Colloff and Wixted (2020), used in the paper

The three experiment files and their codebooks are in the pyWitness collection at
`data/pyWitness/data/published/2020_Colloff_Wixted/` (`Colloff_Wixted_Exp1.csv` to `Exp3.csv`).
The loader keeps rows with `include == "yes"`. `subjectNo` repeats within Experiment 1, so a participant is
identified by `subjectNo` (`subjectID` in Experiments 2 and 3) plus the `verification` code.
Only the three experiment CSVs are required to run the analyses.

## `layher2020/` — Layher, Dixit & Miller (2020)

- Citation: Layher, E., Dixit, A., & Miller, M. B. (2020). Who gives a criterion shift? A
  uniquely individualistic cognitive trait. *Journal of Experimental Psychology: Learning,
  Memory, and Cognition, 46*(11), 2075–2105. https://doi.org/10.1037/xlm0000951
- Source: OSF project https://osf.io/4k2hb/ , folder `data/`.
- Downloaded: 2026-09-23 (MD5 of every file matches the hash OSF reports).
- Files:
  - `layher20_exp1.csv` — Exp 1 (strength × payoff), session-level counts
    (`sub, ses, cCond, dCond, ht, ms, cr, fa`); MD5 dd8417f49815de68584c9b981fc9b882
  - `layher20_exp2.csv` — Exp 2 (strength × base rate), same columns;
    MD5 9514d3e1946f200c4fcda5a78aa8607c
  - `layher20_exp1_trial.csv` — Exp 1 trial-level data (18 MB); its counts reproduce
    `layher20_exp1.csv` exactly; MD5 7b5b96d1925ac72cf5cd5a3e6bff5586
  - `layher20_data_README` — the authors' column definitions;
    MD5 b387c82a5b890122f1098c7d32faf8d9
- Licence: none declared on the OSF project.

## `measuring_memory/` — Measuring Memory Project (Starns et al., 2019)

- Citation: Starns, J. J., Cataldo, A. M., Rotello, C. M., et al. (2019). Assessing
  theoretical conclusions with blinded inference to investigate a potential inference
  crisis. *Advances in Methods and Practices in Psychological Science, 2*(4), 335–349.
  https://doi.org/10.1177/2515245919869583
- Source: OSF project https://osf.io/92ahy/ , folder `Data/`.
- Downloaded: 2026-09-23 (MD5 matches OSF).
- Files:
  - `data_full.csv` — trial-level data, 459 participants × 100 trials (50 targets, 50
    lures), binary old/new response plus confidence; MD5 f1360179e9f7fabe8e11c3d5bf6d9b6b
  - `dataKey_full.pdf` — column definitions and the condition key (1–9 = study
    repetitions 1×/3×/2× × bias instruction none/conservative/liberal);
    MD5 bb26abf3f4edea3f6ce55c334f0e063c. Note that `*.pdf` is in `.gitignore`.
- Licence: none declared on the OSF project.

The automated download includes the two Layher session-level files and the
Measuring Memory trial-level file and condition key. The Layher trial-level
file and README are available from the source but are not needed by `make.jl`.
