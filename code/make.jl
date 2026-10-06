# make.jl
# ─────────────────────────────────────────────────────────────────────────────
# Regenerate every file the paper includes, then verify every checked number.
# Run from the project root after any change to code, data or text:
#
#     julia +1.12 --project=code code/make.jl
#
# Steps (each in a fresh process, stopping at the first failure):
#   tables.jl        empirical, quartile, response and paired-comparison tables
#   inference.jl     tables/tbl_interval_coverage.md
#   invariance.jl    tables/tbl_invariance_contrasts.md
#   figures.jl       figures/fg_advantages.pdf, figures/fg_conceptual.pdf
#   figure_warp.jl   figures/fg_warp.pdf
#   verify_numbers.jl
#
# The other scripts (simulation.jl, simulation_gaps.jl, empirical.jl) write no files; verify_numbers.jl reruns the parts the text
# cites. Pass --no-figures to skip the two figure scripts (about a minute).
# ─────────────────────────────────────────────────────────────────────────────

const JULIA = Base.julia_cmd()
const PROJECT = @__DIR__

steps = ["tables.jl", "inference.jl", "invariance.jl", "figures.jl", "figure_warp.jl", "verify_numbers.jl"]
"--no-figures" in ARGS && filter!(s -> !startswith(s, "figure"), steps)

for s in steps
    println("\n━━━ $s ━━━")
    ok = success(pipeline(`$JULIA --project=$PROJECT $(joinpath(PROJECT, s))`; stdout, stderr))
    ok || (println("\n$s failed; stopping."); exit(1))
end
println("\nAll outputs regenerated and verified.")
