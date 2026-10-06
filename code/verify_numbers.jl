# verify_numbers.jl
# ─────────────────────────────────────────────────────────────────────────────
# Checks the numbers in ppb_paper.qmd and ppb_supplement.qmd against the code
# and data that produce them. Exits 1 if any check fails.
#
# The value each check compares against is READ FROM THE .qmd TEXT, not typed
# into this file. A check names a passage with `#` where the number sits, e.g.
#     cite("...", :paper, "reaching # (@fig-advantages", computed)
# and fails if the passage is missing, if its number disagrees with `computed`
# at the precision printed, or if the passage occurs more than once with
# different numbers. So an edit to a number in the text, or to the code behind
# it, cannot pass silently; an edit that rewords a checked passage fails with
# "passage not found" until the check is updated to the new wording.
#
# It also expands included tables and fails if tables/*.md differ from their
# generators in tables.jl, inference.jl, or invariance.jl.
#
# Simulation checks go through the same functions figures.jl calls
# (Sim.boundary_panel, SimGaps.panel_b_aggregation, ...), so figures and text
# cannot drift apart.
#
# Run from the project root:
#     julia +1.12 --project=code code/verify_numbers.jl          (failures only)
#     julia +1.12 --project=code code/verify_numbers.jl --all    (every check)
# About 20 seconds once packages are precompiled.
# ─────────────────────────────────────────────────────────────────────────────

using CSV, DataFrames, Distributions, Statistics, Random

module Sim
    include("simulation.jl")
end

module SimGaps
    include("simulation_gaps.jl")
end

module Inv
    include("invariance.jl")
end

module Inference
    include("inference.jl")
end

module Tables
    include("tables.jl")
end

include("colloff.jl")
include("colloff_wixted.jl")

# ─── The documents ─────────────────────────────────────────────────────────
# Math delimiters are dropped and \% becomes %, so passages are written as
# they read: "F(2, 76) = # and #", "attenuated ... by #% to #%".

const ROOT = joinpath(@__DIR__, "..")
normtext(s) = replace(s, "\\%" => "%", "\$" => "", "−" => "-")
const RAW_DOCS = Dict(
    :paper => normtext(read(joinpath(ROOT, "ppb_paper.qmd"), String)),
    :supp  => normtext(read(joinpath(ROOT, "ppb_supplement.qmd"), String)),
)

# Included tables are part of the document for cell and caption checks.
# Retain the unexpanded sources as well, so a removed include cannot pass just
# because its generated file still exists on disk.
function expand_includes(source)
    return replace(source, r"\{\{< include ([^>]+?) >\}\}" => directive -> begin
        relative = only(match(r"\{\{< include ([^>]+?) >\}\}", directive).captures)
        path = joinpath(ROOT, strip(relative))
        isfile(path) ? normtext(read(path, String)) : directive
    end)
end
const DOCS = Dict(doc => expand_includes(source) for (doc, source) in RAW_DOCS)

const NUM = raw"([-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))"
escape_re(s) = replace(s, r"([\\.^$|?*+()\[\]{}])" => s"\\\1")

"""Regex for a passage: literal text, `#` for each number, any run of
whitespace matching any run of whitespace."""
passage_regex(t) = Regex(join((replace(escape_re(p), r"\s+" => raw"\s+") for p in split(t, '#')), NUM))

parse_num(s) = parse(Float64, replace(s, "," => ""))
decimals(s) = occursin('.', s) ? length(split(s, '.')[2]) : 0

# ─── Check harness ─────────────────────────────────────────────────────────

const RESULTS = Tuple{String, String, String, Bool}[]
record(label, paper, computed, ok) = (push!(RESULTS, (label, paper, computed, ok)); ok)

"""Compare each `#` in the passage `template` of `doc` with `computed`, at the
precision the text prints (half the last printed digit). `nth` selects one
occurrence when the same passage legitimately recurs with different numbers."""
function cite(label, doc::Symbol, template, computed::Real...; nth = nothing)
    ms = collect(eachmatch(passage_regex(template), DOCS[doc]))
    isempty(ms) && return record(label, "passage not found: \"$template\"", "", false)
    if nth !== nothing
        nth > length(ms) && return record(label, "occurrence $nth not found", "", false)
        ms = [ms[nth]]
    end
    length(ms[1].captures) == length(computed) ||
        return record(label, "template has $(length(ms[1].captures)) numbers, check gives $(length(computed))", "", false)
    ok = true
    for m in ms, (s, v) in zip(m.captures, computed)
        ok &= isfinite(v) && abs(parse_num(s) - v) <= 0.5 * 10.0^(-decimals(s)) + 1e-9
    end
    shown = join(ms[1].captures, ", ")
    length(unique(m.captures for m in ms)) > 1 && (ok = false; shown *= " (differs between occurrences)")
    comp = join((string(round(v; digits = decimals(s) + 3)) for (s, v) in zip(ms[1].captures, computed)), ", ")
    return record(label, shown, comp, ok)
end

"""A claim stated in words: the passage must be present and `ok` must hold."""
function claim(label, doc::Symbol, phrase, ok::Bool, detail = "")
    found = occursin(phrase, DOCS[doc])
    return record(label, found ? "\"$phrase\"" : "passage not found: \"$phrase\"", detail, found && ok)
end

"""A check with no passage in the text (code-level invariants)."""
invariant(label, ok::Bool, detail = "") = record(label, "(invariant)", detail, ok)

"""Rows of the pipe table whose caption carries `{#label}`, header and rule
excluded, each split into trimmed cells."""
function table_rows(doc::Symbol, label)
    lines = split(DOCS[doc], '\n')
    k = findfirst(l -> occursin("{#$label}", l), lines)
    if k === nothing
        invariant("$doc table $label has a caption", false)
        return Vector{String}[]
    end
    i = k - 1
    while i > 0 && isempty(strip(lines[i])); i -= 1; end
    rows = Vector{String}[]
    while i > 0 && startswith(strip(lines[i]), "|")
        pushfirst!(rows, String.(strip.(split(strip(lines[i]), '|')[2:end-1])))
        i -= 1
    end
    invariant("$doc table $label contains data", length(rows) >= 3)
    return length(rows) >= 2 ? rows[3:end] : rows
end
table_header(doc, label) = begin
    lines = split(DOCS[doc], '\n')
    k = findfirst(l -> occursin("{#$label}", l), lines)
    k === nothing && return ""
    i = k - 1
    while i > 0 && isempty(strip(lines[i])); i -= 1; end
    i == 0 && return ""
    while i > 1 && startswith(strip(lines[i - 1]), "|"); i -= 1; end
    lines[i]
end

"""Compare a printed table cell with a computed value at the cell's precision."""
function cell(label, s, v)
    m = match(Regex(NUM), s)
    m === nothing && return record(label, "no number in cell \"$s\"", "", false)
    p = m.captures[1]
    ok = isfinite(v) && abs(parse_num(p) - v) <= 0.5 * 10.0^(-decimals(p)) + 1e-9
    return record(label, p, string(round(v; digits = decimals(p) + 3)), ok)
end

function report_checks(show_all)
    nfail = count(r -> !r[4], RESULTS)
    println()
    println("=" ^ 100)
    println(show_all ? "ALL CHECKS" : "FAILED CHECKS  (run with --all to list every check)")
    println("=" ^ 100)
    for (label, paper, computed, ok) in RESULTS
        (show_all || !ok) || continue
        println(rpad(ok ? "PASS" : "FAIL", 6), rpad(label, 52), " text=", rpad(paper, 14), " computed=", computed)
    end
    println("-" ^ 100)
    println("$(length(RESULTS)) checks: $(length(RESULTS) - nfail) passed, $nfail failed.")
    if nfail > 0
        println("\nFAILED — the text and the code disagree. Do not submit until resolved.")
        exit(1)
    end
    println("All checks passed.")
end

# ─── Checks ────────────────────────────────────────────────────────────────

function check_panel_a()
    P = Sim.boundary_panel()
    cfgs = Sim.PANEL_A_CONFIGS
    cite("Methods: Panel A configurations", :paper,
         "(H, F, n) = (#, #, #), (#, #, #), (#, #, #), and (#, #, #)", Iterators.flatten(cfgs)...)

    pct = [r.pct_boundary for r in P]
    ll = [abs(r.c_ll_bias) for r in P]
    hn = [abs(r.c_hn_bias) for r in P]
    cite("Panel A boundary %, range", :paper,
         "Across configurations in which #% to #% of subjects produced a boundary estimate",
         minimum(pct), maximum(pct))
    claim("Panel A: PPB exactly unbiased", :paper, "PPB remained unbiased",
          all(abs(r.ppb_bias) < 1e-12 for r in P))
    sym = findfirst(r -> r.H == 0.95 && r.F == 0.05, P)
    claim("Panel A: corrections cancel only at H=.95, F=.05", :paper,
          "nearly unbiased only in the configuration with F = 1 - H (H = 0.95, F = 0.05)",
          ll[sym] < 1e-6 && hn[sym] < 1e-6 && all(ll[i] > 0.02 for i in eachindex(P) if i != sym))
    cite("Panel A max |bias|, log-linear and 1/2N", :paper,
         "the log-linear correction yielded bias of as much as #, and the 1/2N correction produced greater bias, reaching #",
         maximum(ll), maximum(hn))
    cite("Abstract: boundary-correction bias", :paper, "boundary corrections biased c by up to #", maximum(hn))
    claim("Panel A: 1/2N worse wherever c != 0", :paper, "the 1/2N correction produced greater bias",
          all(hn[i] > ll[i] for i in eachindex(P) if i != sym))
    rel = [100 * b / abs(r.c_true) for r in P if abs(r.c_true) > 1e-9 for b in (abs(r.c_ll_bias), abs(r.c_hn_bias))]
    cite("Panel A attenuation of c, range", :paper,
         "the corrections attenuated the population value of c by #% to #%", minimum(rel), maximum(rel))
    claim("Fig 2 caption: Panel A values are exact", :paper, "exact expectations over all binomial outcomes", true)
    cite("Supp: n convention", :supp, "n = 20 is n_s = n_n = #", 10)

    # Follow-up (correction, then aggregation; exclusion): exact estimands
    f = Sim.exact_boundary_bias(0.95, 0.20, 20)
    cite("Follow-up: config", :paper, "At H = #, F = #, and n = #, averaging", 0.95, 0.20, 20)
    cite("Follow-up: log-linear then average", :paper,
         "averaging log-linear-corrected individual values yielded bias of # in c", abs(f.c_ll_bias))
    cite("Follow-up: excluding boundary subjects", :paper,
         "increased the bias to # through selection", abs(f.c_excl_bias))
    cite("Discussion: trials per type and max correction bias", :paper,
         "At # to # trials of each type, boundary corrections introduced bias of up to #",
         minimum(n ÷ 2 for (_, _, n) in cfgs), maximum(n ÷ 2 for (_, _, n) in cfgs), maximum(hn))
    cite("Discussion: excluding boundary observers", :paper,
         "excluding them biased c by # in the follow-up simulation", abs(f.c_excl_bias))
end

function check_aggregation()
    agg = SimGaps.panel_b_aggregation()        # same call figures.jl Panel B makes
    t = agg[1]
    @assert t.μH == 0.90 && t.μF == 0.15
    c_pop = criterion_c(t.μH, t.μF)
    br_pop = b_r(t.μH, t.μF)

    claim("Aggregation: PPB zero in every scenario", :paper,
          "The aggregation discrepancy for PPB was zero to floating-point precision in every configuration",
          all(abs(r.ppb_discrepancy) < 1e-12 for r in agg))
    cite("Aggregation: scenario", :paper, "At high accuracy (\\mu_H = #, \\mu_F = #)", t.μH, t.μF)
    cite("Aggregation: c", :paper, "the discrepancy for c reached #,", t.c_discrepancy_mean)
    claim("Abstract: aggregation discrepancy nearly half of c", :paper,
          "the mean of individual c values differed from c at the mean rates by nearly half the latter’s magnitude",
          40 < 100 * abs(t.c_discrepancy_mean) / abs(c_pop) < 50,
          string(round(100 * abs(t.c_discrepancy_mean) / abs(c_pop); digits = 1)) * "%")
    cite("Aggregation: B'', B''_D, B_r", :paper,
         "The corresponding discrepancies were # for B'', # for B''_D, and # for B_r",
         t.bpp_discrepancy_mean, t.bppd_discrepancy_mean, t.br_discrepancy_mean)
    claim("Discussion: aggregation discrepancy nearly half of c", :paper,
          "In the high-accuracy aggregation simulation, the two group summaries of c, the mean of individual values and the value at the mean rates, differed by nearly half the latter’s magnitude",
          40 < 100 * abs(t.c_discrepancy_mean) / abs(c_pop) < 50,
          string(round(100 * abs(t.c_discrepancy_mean) / abs(c_pop); digits = 1)) * "%")
    cite("Aggregation: relative to B_r and c", :paper,
         "the discrepancy represented #% of B_r and #% of c",
         100 * abs(t.br_discrepancy_mean) / br_pop, 100 * abs(t.c_discrepancy_mean) / abs(c_pop))
    se = maximum(max(r.c_discrepancy_sd, r.bpp_discrepancy_sd, r.bppd_discrepancy_sd, r.br_discrepancy_sd)
                 for r in agg) / sqrt(2000)
    cite("Fig 2B caption: MC error bound", :paper, "Monte Carlo error less than #)", 0.001)
    invariant("Fig 2B: MC error actually below 0.001", se < 0.001, string(round(se; digits = 5)))
    cite("Fig 2B caption: design", :paper, "under three scenarios (# subjects, # replications", 100, 2000)
    cite("Methods: aggregation design", :paper,
         "compared the mean of # individual estimates with the estimate computed from the mean rates over # replications",
         100, 2000)

    # Supplement §3 worked example
    dnorm(x) = pdf(Normal(), x)
    zq(p) = quantile(Normal(), p)
    probit2(mu) = zq(mu) / dnorm(zq(mu))^2
    Ht = probit2(0.90) * 0.08^2
    Ft = probit2(0.15) * 0.06^2
    cite("Supp §3: H-term", :supp, "H-term: # \\times #^2 = #", probit2(0.90), 0.08, Ht)
    cite("Supp §3: F-term", :supp, "F-term: # \\times #^2 = #", probit2(0.15), 0.06, Ft)
    cite("Supp §3: approximation", :supp, "\\approx -\\tfrac{1}{4}(# - #) = #", Ht, -Ft, -(Ht + Ft) / 4)
    cite("Supp §3: c at mean rates", :supp, "\\Phi^{-1}(0.15)] \\approx #", c_pop)
    cite("Supp §3: simulated discrepancy", :supp, "underestimates the simulated discrepancy (#)", t.c_discrepancy_mean)
    cite("Supp §3: F-term offsets", :supp, "the F-term offsets only #% of the H-term", 100 * abs(Ft) / Ht)
    for row in table_rows(:supp, "tbl-second-deriv")
        cell("Supp second derivative at $(row[1])", row[2], probit2(parse_num(row[1])))
    end
end

function check_regression()
    res = Sim.sim_regression_artifact()
    hi, mod = res[1], res[2]
    ratio = mod.slope_ppb / hi.slope_ppb
    cite("Regression: PPB slopes", :paper,
         "a PPB slope of # among high-accuracy observers and # among moderate-accuracy observers",
         hi.slope_ppb, mod.slope_ppb)
    claim("Regression: threefold", :paper, "a threefold difference", round(ratio) == 3 && abs(ratio - 3) < 0.05,
          string(round(ratio; digits = 3)))
    cite("Regression: c ratio", :paper, "slopes for c differed by a factor of #",
         abs(mod.slope_c_loglin) / abs(hi.slope_c_loglin))
    cite("Regression: undefined c", :paper, "In addition, # of the # high-accuracy subjects", hi.n_nan, hi.n_subjects)
    cite("Methods: regression design", :paper, "coefficient (\\beta = #)", hi.β_bias)
    cite("Methods: regression n", :paper, "(# subjects per group, # trials per subject)", hi.n_subjects, hi.n_trials)
    cite("Fig 2C caption: design", :paper, "\\beta = # on the logit scale, shown separately", hi.β_bias)
end

function check_group_comparison()
    s = Sim.sim_group_comparison()[3]
    HA, FA, HB, FB = 0.95, 0.08, 0.65, 0.40     # the scenario's population rates (simulation.jl)
    invariant("Group comparison: rates match simulation.jl",
              isapprox(s.ppb_A_true, HA + FA) && isapprox(s.ppb_B_true, HB + FB))

    # The analytic example kept in the main text
    cA, cB = criterion_c(HA, FA), criterion_c(HB, FB)
    cite("Group comparison, main text: groups", :paper,
         "Consider Group A with H = # and F = # and Group B with H = # and F = #", HA, FA, HB, FB)
    cite("Group comparison, main text: PPB, J and c", :paper,
         "Their values of PPB are # and #, but J is # and # and c is # and #",
         HA + FA, HB + FB, HA - FA, HB - FB, cA, cB)
    claim("Group comparison, main text: opposite rankings", :paper,
          "PPB ranks Group B as the more liberal, and c ranks Group A as the more liberal",
          (HA + FA) < (HB + FB) && cA < cB)
    cite("Group comparison, main text: rate changes", :paper,
         "H falls by # and F rises by #, so PPB rises by #", HA - HB, FB - FA, (HB + FB) - (HA + FA))
    cite("Group comparison, main text: simulation size", :paper, "simulates this comparison with # subjects per group", s.K)
    invariant("Group comparison: rankings opposed in population",
              (HA + FA) < (HB + FB) && s.c_A_true < s.c_B_true)
    cite("Group comparison, supplement: population c", :supp,
         "computed at the group mean rates, are # and #", s.c_A_true, s.c_B_true)

    # The simulation, in the supplement
    cite("Group comparison, supplement: design", :supp,
         "The simulation generated two groups of # subjects, with # replications", s.K, s.n_replications)
    src = read(joinpath(ROOT, "code", "simulation.jl"), String)
    m = match(r"0\.95, 0\.08, ([\d.]+), ([\d.]+),[^\n]*\n\s*0\.65, 0\.40, ([\d.]+), ([\d.]+)\)", src)
    m === nothing && error("Group-comparison scenario not found in simulation.jl")
    sds = parse.(Float64, m.captures)
    invariant("Group comparison: spread equal across rates within a group", sds[1] == sds[2] && sds[3] == sds[4])
    cite("Group comparison, supplement: beta standard deviations", :supp,
         "standard deviations of # (Group A) and # (Group B)", sds[1], sds[3])
    cite("Group comparison, supplement: groups", :supp,
         "Group A has H = #, F = #, PPB = #, and J = #; Group B has H = #, F = #, PPB = #, and J = #",
         HA, FA, HA + FA, HA - FA, HB, FB, HB + FB, HB - FB, nth = 1)
    cite("Group comparison, supplement: disagreement", :supp, "Observed rankings differed in #% of replications", 100 * s.disagree_rate)
    cite("Fig S caption: groups", :supp,
         "Group A has H = #, F = #, PPB = #, and J = #; Group B has H = #, F = #, PPB = #, and J = #",
         HA, FA, HA + FA, HA - FA, HB, FB, HB + FB, HB - FB, nth = 2)
    cite("Fig S caption: population c", :supp, "Population c equals # in Group A and # in Group B", s.c_A_true, s.c_B_true)
    cite("Fig S caption: disagreement", :supp, "their observed rankings differ in #% of replications at this configuration", 100 * s.disagree_rate)
    cite("Fig S caption: design", :supp, "replications with # subjects per group", s.K)
    # figures.jl plots every step-th replication, step = length ÷ n_show
    n_show = let src = read(joinpath(ROOT, "code", "figures.jl"), String)
        parse(Int, only(match(r"n_show\s*=\s*(\d+)", src).captures))
    end
    shown = length(1:max(1, length(s.ppb_diffs) ÷ n_show):length(s.ppb_diffs))
    cite("Fig S caption: replications displayed", :supp, "The panel displays # of # replications", shown, s.n_replications)
    # The text says the disagreement rate approaches 100% as group size increases
    big = Sim.sim_group_comparison(n_replications = 1000, K = 1600)[3]
    claim("Group comparison, supplement: approaches 100% with group size", :supp,
          "it approaches 100% as group size increases", big.disagree_rate > 0.99,
          string(round(100 * big.disagree_rate; digits = 1), "% at 1,600 per group"))
end

function check_distribution_shape()
    res6 = SimGaps.sim6_nongaussian()
    fam = filter(r -> r.target_ppb == res6.target_ppb_3, res6.part1)
    Js = [r.H - r.F for r in fam]
    cs = [r.c_val for r in fam]
    rate(f) = [100 * f(r) for r in res6.part3]
    ppb_r, c_r = rate(r -> r.ppb_detection_rate), rate(r -> r.c_detection_rate)
    bpp_r, br_r = rate(r -> r.bpp_detection_rate), rate(r -> r.br_detection_rate)

    cite("Sim 6: common PPB", :supp, "set PPB to # in two groups", res6.target_ppb_3)
    cite("Sim 6: J range", :supp, "J ranged from # for the", minimum(Js))
    cite("Sim 6: J max and spread", :supp, "to # for the equal-variance Gaussian distribution, a spread of #",
         maximum(Js), maximum(Js) - minimum(Js))
    cite("Sim 6: population c range", :supp, "the corresponding population values of c ranged from # to #",
         maximum(cs), minimum(cs))
    claim("Sim 6: PPB holds nominal rate", :supp, "Tests based on PPB maintained the nominal 5% rejection rate",
          maximum(ppb_r) < 6.0, string(round(maximum(ppb_r); digits = 2)))
    cite("Sim 6: rejection ranges", :supp,
         "c detected a difference in #% to #% of replications, B_r in #% to #%, and B'' in #% to #%",
         minimum(c_r), maximum(c_r), minimum(br_r), maximum(br_r), minimum(bpp_r), maximum(bpp_r))

    JA = let r = first(filter(r -> r.label == "EVSDT (Gaussian)", fam)); r.H - r.F end
    dJ = [abs(let q = first(filter(q -> q.label == r.label, fam)); q.H - q.F end - JA) for r in res6.part3]
    ord = sortperm(dJ)
    for (nm, v) in (("c", c_r), ("B''", bpp_r), ("B_r", br_r))
        claim("Sim 6: $nm rejection follows |dJ|", :supp, "these rejection rates followed the magnitude of the difference in J",
              all(diff(v[ord]) .>= -0.05), string(round.(v[ord]; digits = 1)))
    end
    cite("Fig S caption: MC error", :supp, "maximum Monte Carlo error of # percentage points", 100 * sqrt(0.25 / 2000))
    cite("Methods: Sim 6 design", :supp, "at \\alpha = 0.05 (# subjects per group, # replications)", 200, 2000)
    cite("Methods: Sim 6 location shift", :supp, "Every distribution had a location shift of # between", res6.d_prime)

    labels = [r.label for r in res6.part3]
    invariant("Methods: Sim 6 comparison families are sigma 1.5, 2.0, logistic, t(3), t(5)",
              Set(labels) == Set(["Unequal variance (σ=1.5)", "Unequal variance (σ=2.0)", "Logistic",
                                  "Heavy-tailed (t, df=3)", "Heavy-tailed (t, df=5)"]), join(labels, "; "))
    cite("Methods: Sim 6 sigma values", :supp, "\\sigma \\in \\{#, #\\}", 1.5, 2.0)
    cite("Methods: Sim 6 t degrees of freedom", :supp, "distribution with # or # degrees of freedom", 3, 5)
end

function check_empirical()
    subj, cond = load_colloff()
    n, nb = nrow(subj), sum(subj.at_boundary)
    nc, nbc = nrow(cond), sum(cond.at_boundary)
    bsub = subj[subj.at_boundary, :]
    dis_b = maximum(abs.(bsub.c_ll .- bsub.c_hn))
    isub = subj[.!subj.at_boundary, :]
    dis_i = maximum(abs.(isub.c_ll .- isub.c_hn))

    cite("Abstract: boundary rate", :paper,
         "#% of participants had a rate of 0 or 1 (Colloff", 100 * nb / n)
    cite("Discussion: undefined c", :paper, "In the eyewitness data, c was undefined for #% of participants", 100 * nb / n)
    cite("Results: boundary participants", :paper, "At the subject level, # of # participants (#%)", nb, n, 100 * nb / n)
    cite("Results: condition level", :paper, "At the condition level, #% of observations required correction", 100 * nbc / nc)
    cite("Results: condition level, own-race paragraph", :paper, "depends on corrections for #% of the condition-level", 100 * nbc / nc)
    cite("Table 1 caption: condition level", :paper, "because #% of its observations lie on a boundary", 100 * nbc / nc)
    cite("Results: corrections disagree, boundary", :paper,
         "Among the # affected participants, estimates of c under the log-linear and 1/2N corrections differed by as much as #.", nb, dis_b)
    cite("Results: corrections disagree, interior", :paper, "differed by as much as # among interior rates", dis_i)
    nbr, nbrc = sum(subj.br_undefined), sum(cond.br_undefined)
    cite("Results: B_r undefined", :paper,
         "for # participants (#%) and # of the # condition-level observations (#%)",
         nbr, 100 * nbr / n, nbrc, nc, 100 * nbrc / nc)
    cite("Data: participants", :paper, "The study included # participants", n)
    cite("Data: subject level", :paper, "four signal and four noise trials for each of # participants", n)
    cite("Data: condition level", :paper, "in each of # participant-by-condition observations", nc)
    invariant("Data: 4 + 4 trials per participant", all(subj.n_tp .== 4) && all(subj.n_ta .== 4))
    invariant("Data: 2 + 2 trials per condition", all(cond.n_tp .== 2) && all(cond.n_ta .== 2))

    Q = Tables.separation_quartiles(subj)
    cite("Results: quartile boundary %", :paper, "across J quartiles: #%, #%, #%, and #%",
         (q.boundary_pct for q in Q)...)
    invariant("Quartile boundary % increases monotonically", issorted([q.boundary_pct for q in Q]))

    c_means = criterion_c(mean(subj.H_ll), mean(subj.F_ll))
    c_ind = mean(subj.c_ll)
    cite("Results: empirical aggregation", :paper,
         "The value of c computed from mean corrected rates was #, whereas the mean of individual c values was #. The resulting discrepancy was #.",
         c_means, c_ind, c_ind - c_means)
    invariant("Empirical PPB aggregates exactly", abs(mean(subj.ppb_val) - ppb(mean(subj.H), mean(subj.F))) < 1e-15)

    own = filter(:OwnRace => ==("ownRace"), cond)
    oth = filter(:OwnRace => ==("otherRace"), cond)
    cite("Results: own- vs other-race", :paper,
         "PPB_{\\text{own}} = #, PPB_{\\text{other}} = #; c_{\\text{own}} = #, c_{\\text{other}} = #",
         mean(own.ppb_val), mean(oth.ppb_val), mean(own.c_ll), mean(oth.c_ll))
    claim("Two-trial corrected c is affine in PPB", :paper,
          "log-linear c is an affine decreasing function of PPB",
          all(isapprox.(cond.c_ll, quantile(Normal(), 5 / 6) .* (1 .- cond.ppb_val); atol=1e-13)))

    raw = load_colloff_raw()
    count_response(tp, response) = count(r -> r.is_tp == tp && r.IDResponse == response, eachrow(raw))
    cite("Worked example: lineup response counts", :paper,
         "Across # target-present trials there were # perpetrator choices, # foil choices, and # rejections; across # target-absent trials there were # foil choices and # rejections",
         sum(raw.is_tp), count_response(true, "perpetrator"), count_response(true, "foil"),
         count_response(true, "reject"), sum(.!raw.is_tp), count_response(false, "foil"), count_response(false, "reject"))
    H, F = sum(raw.is_hit) / sum(raw.is_tp), sum(raw.is_fa) / sum(.!raw.is_tp)
    cite("Worked example: choosing rates and indexes", :paper,
         "Under the stated choosing response, H=#, F=#, J=#, and PPB =#.", H, F, H - F, H + F)
    cite("Worked example: choosing fraction", :paper, "PPB/2 =# is the choosing fraction", (H + F) / 2)
    invariant("Choosing PPB/2 equals observed choosing fraction", isapprox((H + F) / 2, mean(raw.is_positive)))
    p = paired_ownrace_summary(cond)
    cite("Worked example: paired PPB contrast", :paper,
         "The paired own-minus-other PPB difference was #, with a 95% t-based confidence interval of [#, #]",
         p.mean, p.lo, p.hi)
    cite("Worked example: paired yes-probability contrast", :paper,
         "gives a choosing-probability difference of #, with interval [#, #]",
         p.mean / 2, p.lo / 2, p.hi / 2)
    invariant("Own-race pairing retains all participants",p.n == n && p.df == n - 1)

    legacy_subj, legacy_cond = load_colloff(; response=:identification)
    legacy_p = paired_ownrace_summary(legacy_cond)
    cite("Sensitivity: pooled sum", :paper,
         "The perpetrator-only sensitivity coding gave a pooled sum of # instead of #", mean(legacy_subj.ppb_val), H + F)
    cite("Sensitivity: paired contrast", :paper,
         "Its own-minus-other sum difference was # with a 95% paired interval of [#, #]",
         legacy_p.mean, legacy_p.lo, legacy_p.hi)
    invariant("Sensitivity changes only TP foil numerator", isapprox(
              mean(subj.ppb_val) - mean(legacy_subj.ppb_val), count_response(true, "foil") / sum(raw.is_tp)))
end

"""Check generated content and its inclusion in the intended manuscript."""
function generated_table(name, content, doc; script)
    path = joinpath(ROOT, "tables", name)
    invariant("tables/$name is current (rerun $script)", isfile(path) && read(path, String) == content)
    invariant("$doc includes tables/$name", occursin("{{< include tables/$name >}}", RAW_DOCS[doc]))
end

function check_generated_tables()
    current = Tables.all_tables()
    invariant("all_tables covers every declared empirical table", Set(keys(current)) == Set(Tables.TABLE_FILES))
    for (name, content) in current
        doc = name in ("tbl_empirical.md", "tbl_quartiles.md") ? :paper : :supp
        generated_table(name, content, doc; script="code/tables.jl")
    end
end

function check_intervals()
    grid = Inference.coverage_grid()
    featured = Inference.featured_coverage()
    invariant("Finite-binomial interval boundary and coverage checks", Inference.check_inference(grid))
    panel_a = [Inference.interval_coverage(H, F, n ÷ 2, n ÷ 2) for (H, F, n) in Sim.PANEL_A_CONFIGS]
    cite("Intervals: featured normal-approximation coverage range", :paper,
         "nominal 95% normal-approximation coverage ranged from #% to #%",
         100minimum(r.wald_coverage for r in panel_a), 100maximum(r.wald_coverage for r in panel_a))
    r = only(filter(r -> r.H == 0.8 && r.F == 0.05 && r.ns == r.nn == 10, featured))
    cite("Intervals: expected widths", :paper,
         "expected widths were # for the simultaneous interval and # for the normal-approximation interval", r.cp_width, r.wald_width)
    minimum_coverage = minimum(r.cp_coverage for r in grid)
    cite("Intervals: minimum coverage, main", :paper,
         "the minimum simultaneous coverage was #%", 100minimum_coverage)
    G = Inference.grid_summary(grid)
    cite("Intervals: adjusted width at the featured point", :paper,
         "The adjusted interval had expected width # at that operating point", r.adj_width)
    cite("Intervals: adjusted coverage across the grid, main", :paper,
         "its coverage averaged #% with a minimum of #% and fell below 94% in #% of configurations, and its mean expected width was #, against # for the simultaneous interval",
         100G.adj.mean, 100G.adj.minimum, 100G.adj.below, G.adj.width, G.cp.width)
    cite("Intervals: normal-approximation coverage across the grid, main", :paper,
         "The normal-approximation interval averaged #% coverage, with a minimum of #%", 100G.wald.mean, 100G.wald.minimum)
    cite("Intervals: adjusted coverage, limitation", :paper,
         "its coverage fell as low as #% on the grid", 100G.adj.minimum)
    amin = grid[argmin([x.adj_coverage for x in grid])]
    wmin = grid[argmin([x.wald_coverage for x in grid])]
    cite("Intervals: adjusted coverage across the grid, supplement", :supp,
         "coverage averaged #% with a minimum of #% (at H=#, F=#, n_s=#, n_n=#) and was below 94% in #% of configurations; its mean expected width was #, against # for the conservative interval and # for the normal-approximation interval",
         100G.adj.mean, 100G.adj.minimum, amin.H, amin.F, amin.ns, amin.nn, 100G.adj.below, G.adj.width, G.cp.width, G.wald.width)
    cite("Intervals: normal-approximation coverage across the grid, supplement", :supp,
         "The normal-approximation interval averaged #% coverage with a minimum of #%, at H=F=# and four trials per class",
         100G.wald.mean, 100G.wald.minimum, wmin.H)
    invariant("Intervals: normal minimum occurs at H = F and four trials per class", wmin.H == wmin.F && wmin.ns == wmin.nn == 4)
    cite("Intervals: denominator grid", :supp, "n_s,n_n \\in \\{#,#,#\\}", sort(unique(r.ns for r in grid))...)
    cite("Intervals: probability grid", :supp, "H,F \\in \\{#,#,#,#,#,#,#,#,#\\}", sort(unique(r.H for r in grid))...)
    cite("Intervals: grid size", :supp, "giving # configurations", length(grid))
    cite("Intervals: minimum coverage, supplement", :supp,
         "Minimum coverage for the conservative interval was #%", 100minimum_coverage)
    rows = table_rows(:supp, "tbl-interval-coverage")
    invariant("Interval table has every featured configuration", length(rows) == length(featured))
    for (i, (row, r)) in enumerate(zip(rows, featured))
        for (j, value) in enumerate((r.H, r.F, r.ns, r.nn, 100r.wald_coverage, 100r.cp_coverage, 100r.adj_coverage,
                                     r.wald_width, r.cp_width, r.adj_width))
            cell("Interval table row $i column $j", row[j], value)
        end
    end
    generated_table("tbl_interval_coverage.md", Inference.interval_coverage_table(), :supp; script="code/inference.jl")
end

function check_properties()
    cite("Properties: range at J = 0.87", :paper, "when J = #, PPB must lie between # and #", 0.87, 0.87, 2 - 0.87)
    evs(d, c) = (cdf(Normal(), d / 2 - c), cdf(Normal(), -d / 2 - c))
    Ha, Fa = evs(1.0, -0.5)
    Hb, Fb = evs(2.0, -0.5)
    cite("Properties: EVSDT example rates", :paper,
         "maintains c = # has (H, F) = (#, #) at d' = # and (H, F) = (#, #) at d' = #",
         -0.5, Ha, Fa, 1, Hb, Fb, 2)
    cite("Properties: EVSDT example PPB", :paper, "PPB consequently falls from # to #", Ha + Fa, Hb + Fb)
    rect(q, v) = (q + (1 - q) * v, (1 - q) * v)
    H1, F1 = rect(0.3, 0.4)
    H2, F2 = rect(0.6, 0.4)
    cite("Properties: 2HT example", :paper, "guess rate of B_r = # raises PPB from # to # as J increases from # to #",
         0.4, H1 + F1, H2 + F2, H1 - F1, H2 - F2)
    # Fig 1 caption grids, read from the plotting script that draws them
    warp = read(joinpath(ROOT, "code", "figure_warp.jl"), String)
    grid(name) = parse.(Float64, strip.(split(only(match(Regex(name * raw"\s*=\s*\[([^\]]*)\]"), warp).captures), ",")))
    cite("Fig 1 caption: d' grid", :paper, "d' \\in \\{#, #, #, #\\}", grid("dvals")...)
    cite("Fig 1 caption: c grid", :paper, "c \\in \\{#, #, #, #, #\\}", grid("cvals")...)
    cite("Fig 1 caption: example point", :paper, "H = # and F = #, corresponding to J = # and PPB = #",
         0.80, 0.30, 0.80 - 0.30, 0.80 + 0.30)
end

function check_supplement()
    dnorm(x) = pdf(Normal(), x)
    zq(p) = quantile(Normal(), p)

    # §1.3 worked example: four families, mu = 2, theta solving H + F = 1.20
    fams = [("Gaussian EV | \\sigma_s = \\sigma_n = 1", x -> ccdf(Normal(), x)),
            ("Logistic | s = 1", x -> ccdf(Logistic(), x)),
            ("Gaussian UV | \\sigma_s = 2, \\sigma_n = 1", nothing),
            ("t(3) (scaled) | \\text{Var} = 1", x -> ccdf(TDist(3), x * sqrt(3)))]
    Js, cs = Float64[], Float64[]
    for (lab, S) in fams
        Hf(th) = S === nothing ? ccdf(Normal(), (th - 2) / 2) : S(th - 2)
        Ff(th) = S === nothing ? ccdf(Normal(), th) : S(th)
        lo, hi = -5.0, 5.0                         # H + F decreases in theta
        for _ in 1:200
            mid = (lo + hi) / 2
            Hf(mid) + Ff(mid) > 1.20 ? (lo = mid) : (hi = mid)
        end
        th = (lo + hi) / 2
        H, F = Hf(th), Ff(th)
        push!(Js, H - F); push!(cs, criterion_c(H, F))
        cite("Supp §1.3 row: $(split(lab, " |")[1])", :supp, "| $lab | # | # | # | # | # | # |",
             th, H, F, H + F, H - F, criterion_c(H, F))
    end
    cite("Supp §1.3: J and c ranges", :supp, "J ranges from # to # and computed c from # to #",
         minimum(Js), maximum(Js), minimum(cs), maximum(cs))

    # §2 boundary table and amplification table
    for row in table_rows(:supp, "tbl-boundary")
        Hs, ns = parse_num(row[1]), Int(parse_num(row[2]))
        cell("Supp P(boundary) H*=$(row[1]), n_s=$ns", row[3], Hs^ns + (1 - Hs)^ns)
    end
    base = 1 / dnorm(0)
    for row in table_rows(:supp, "tbl-amplification")
        x = parse_num(row[1])
        cell("Supp amplification (Phi^-1)'($(row[1]))", row[2], 1 / dnorm(zq(x)))
        x == 0.5 || cell("Supp amplification ratio at $(row[1])", row[3], (1 / dnorm(zq(x))) / base)
    end
    cite("Supp amplification caption", :supp,
         "near H = 0.01 is amplified #-fold in z(H) against #-fold near H = 0.50, a relative factor of #",
         1 / dnorm(zq(0.01)), base, (1 / dnorm(zq(0.01))) / base)

    # §2 sign reversal of the log-linear correction with unequal trial counts
    shrinkage = -0.5 * ((0.5 - 0.95) / (11 * dnorm(zq(0.95))) + (0.5 - 0.20) / (11 * dnorm(zq(0.20))))
    cite("Supp §2: shrinkage approximation versus exact bias", :supp,
         "it gives #, whereas exact expected bias is #", shrinkage, Sim.exact_boundary_bias(0.95, 0.20, 20).c_ll_bias)
    H, F, ns, nn = cdf(Normal(), 2.0), cdf(Normal(), -1.5), 1000, 10
    ec = sum(pdf(Binomial(ns, H), h) * pdf(Binomial(nn, F), f) *
             criterion_c((h + 0.5) / (ns + 1), (f + 0.5) / (nn + 1)) for h in 0:ns, f in 0:nn)
    first_order = criterion_c(H, F) - 0.5 * ((0.5 - H) / ((ns + 1) * dnorm(zq(H))) + (0.5 - F) / ((nn + 1) * dnorm(zq(F))))
    cite("Supp §2: sign reversal example", :supp,
         "the correction moves c from # to # in expectation (# to first order)",
         criterion_c(H, F), ec, first_order)

    # §4 covariate example
    sp(p) = p * (1 - p)
    for (k, p) in enumerate((0.95, 0.75))
        cite("Supp §4 example: dPPB/dx (TPR = $p)", :supp, "\\partial \\text{PPB}/\\partial x = 2 \\times # \\times 0.2 = #",
             sp(p), 2 * sp(p) * 0.2; nth = k)
        cite("Supp §4 example: dc/dx (TPR = $p)", :supp, "\\partial c / \\partial x = -0.5 \\times 2 \\times (#/#) \\times 0.2 = #",
             sp(p), dnorm(zq(p)), -0.5 * 2 * (sp(p) / dnorm(zq(p))) * 0.2; nth = k)
    end
    cite("Supp §4 example: ratios", :supp, "The PPB effect is #/# \\approx #x larger",
         2 * sp(0.75) * 0.2, 2 * sp(0.95) * 0.2, sp(0.75) / sp(0.95))
    dc(p) = (sp(p) / dnorm(zq(p))) * 0.2
    cite("Supp §4 example: c ratio", :supp, "The c effect is #/# \\approx #x larger",
         dc(0.75), dc(0.95), dc(0.75) / dc(0.95))

    # §5 Monte Carlo SE remark
    cite("Supp §5: MC SE", :supp, "/ 5000} = #", sqrt((0.80 * 0.20 / 10 + 0.05 * 0.95 / 10) / 5000))

    # Snodgrass & Corwin Table 3 means (external inputs: their H, F and c values)
    sc_c = Dict("Conservative" => (0.363, 0.485), "Neutral" => (0.032, 0.085), "Liberal" => (-0.161, -0.162))
    for row in table_rows(:supp, "tbl-sc-invariance")
        Hh, Hl = parse_num.(split(row[2], ", "))
        Fh, Fl = parse_num.(split(row[3], ", "))
        Ph, Pl = strip.(split(row[4], ", "))
        cell("S&C $(row[1]) PPB, high imagery", Ph, Hh + Fh)
        cell("S&C $(row[1]) PPB, low imagery", Pl, Hl + Fl)
        cell("S&C $(row[1]) c_rect gap", row[5], ((1 - (Hh + Fh)) - (1 - (Hl + Fl))) / 2)
        ch, cl = sc_c[row[1]]
        cell("S&C $(row[1]) c gap", row[6], ch - cl)
    end
end

function check_invariance()
    inv = Inv.run_all()
    L1, L2, MM = inv[:layher1], inv[:layher2], inv[:mm]
    passes(o, y) = Inv.verdict(o.res[y], y) == "pass"
    r1, rm = L1.res, MM.res

    cite("Invariance: Layher Exp 1", :paper,
         "(N = #), payoffs affected both indexes, while overall memory-strength and interaction effects were not significant: payoff F(#, #) = # and #, strength p = # and #, interaction p = # and #",
         length(unique(L1.data.sub)), r1[:PPB].bias.df1, r1[:PPB].bias.df2, r1[:PPB].bias.F, r1[:c].bias.F,
         r1[:PPB].strength.p, r1[:c].strength.p, r1[:PPB].interaction.p, r1[:c].interaction.p)
    invariant("Layher Exp 1: PPB and c pass, payoff p < .001",
              passes(L1, :PPB) && passes(L1, :c) && max(r1[:PPB].bias.p, r1[:c].bias.p) < 0.001)
    cite("Invariance: Measuring Memory", :paper,
         "(N = #), instructions affected both indexes, while overall memory-strength and interaction effects were not significant: instruction F(#, #) = # and #, strength p = # and #, interaction p = # and #",
         nrow(MM.data), rm[:PPB].bias.df1, rm[:PPB].bias.df2, rm[:PPB].bias.F, rm[:c].bias.F,
         rm[:PPB].strength.p, rm[:c].strength.p, rm[:PPB].interaction.p, rm[:c].interaction.p)
    invariant("Measuring Memory: PPB and c pass, instruction p < .001",
              passes(MM, :PPB) && passes(MM, :c) && max(rm[:PPB].bias.p, rm[:c].bias.p) < 0.001)
    claim("Layher Exp 2: every index fails", :paper, "Under the base-rate manipulation of Layher et al., every index failed to meet these criteria",
          count(y -> passes(L2, y), Inv.PRIMARY) == 0)
    claim("Layher Exp 1: J payoff and interaction effects", :paper,
          "Raw J also showed payoff and interaction effects in Layher Experiment 1",
          r1[:J].bias.p < 0.05 && r1[:J].interaction.p < 0.05)
    claim("Measuring Memory: J meets ANOVA criteria", :paper,
          "the Measuring Memory analysis detected a strength effect for J but no instruction or interaction effect", passes(MM, :J))
    dmax = maximum(maximum(Inv.cell_means(o.ds, o.data, :dprime)) for o in (L1, L2))
    cite("Invariance: d' bound", :paper, "poorly discriminated faces (d' \\le #)", 1.1)
    invariant("Layher max cell-mean d' <= 1.1", dmax <= 1.1, string(round(dmax; digits = 3)))
    cite("Data: Layher trials", :paper, "with # trials per participant and cell", only(unique(L1.data.n_old .+ L1.data.n_new)))
    invariant("Layher Exp 2 trials per cell = Exp 1", only(unique(L2.data.n_old .+ L2.data.n_new)) == 1000)
    cite("Data: MM trials", :paper, "the Measuring Memory Project [@starns_assessing_2019], with # per participant",
         only(unique(MM.data.n_old .+ MM.data.n_new)))
    claim("Correction sensitivity preserves ANOVA criteria outcomes", :supp, "did not change whether the indexes met the ANOVA criteria",
          count(o -> any(((y, y0),) -> Inv.verdict(o.res[y], y0) != Inv.verdict(o.res[y0], y0),
                         ((:PPB_ll, :PPB), (:c_hn, :c), (:Br, :Br_ll))), (L1, L2, MM)) == 0)

    # Supplement ANOVA tables: every F, p and verdict
    labels = Dict("PPB" => :PPB, "c" => :c, "J" => :J, "d'" => :dprime, "B_r" => :Br_ll,
                  "PPB, log-linear" => :PPB_ll, "J, log-linear" => :J_ll, "c, 1/2N" => :c_hn, "B_r, raw" => :Br)
    fmt_df(e) = "($(e.df1), $(e.df2))"
    for (tbl, o) in (("tbl-inv-layher1", L1), ("tbl-inv-layher2", L2), ("tbl-inv-mm", MM))
        hdr = table_header(:supp, tbl)
        e = o.res[:PPB]
        invariant("$tbl header df", occursin("Strength F $(fmt_df(e.strength))", hdr) &&
                  occursin("Bias F $(fmt_df(e.bias))", hdr) && occursin("Interaction F $(fmt_df(e.interaction))", hdr), hdr)
        rows = table_rows(:supp, tbl)
        invariant("$tbl has rows", !isempty(rows))
        checked_rows = Set{Tuple{Symbol, Bool}}()
        for row in rows
            lab = strip(replace(row[1], r"\\textsuperscript\{\w\}|\^\w\^" => ""))   # footnote marks
            type2 = endswith(lab, ", Type II")
            key = get(labels, replace(lab, ", Type II" => ""), nothing)
            key === nothing && (invariant("$tbl row \"$lab\" is known", false); continue)
            push!(checked_rows, (key, type2))
            r = type2 ? o.res_type2[key] : o.res[key]
            for (j, eff) in enumerate((:strength, :bias, :interaction))
                ef = getfield(r, eff)
                m = match(r"^([\d.]+) \((<?)([\d.]+)\)$", row[j + 1])
                m === nothing && (invariant("$tbl $lab $eff cell parses", false, row[j + 1]); continue)
                cell("$tbl $lab $eff F", m.captures[1], ef.F)
                invariant("$tbl $lab $eff p", m.captures[2] == "<" ? ef.p < 0.001 :
                          abs(parse_num(m.captures[3]) - ef.p) <= 0.0005 + 1e-9,
                          string(round(ef.p; digits = 4)))
            end
            invariant("$tbl $lab verdict", row[5] == Inv.verdict(r, key), Inv.verdict(r, key))
        end
        expected_rows = Set((y, false) for y in (Inv.PRIMARY..., Inv.SENSITIVITY...))
        o.ds.design == :between && union!(expected_rows, ((y, true) for y in (:PPB, :c, :dprime, :Br_ll)))
        invariant("$tbl includes every primary and sensitivity row once",
                  checked_rows == expected_rows && length(rows) == length(expected_rows))
    end
    cite("Supp: MM raw B_r exclusion", :supp, "are excluded (N = #; error df #)", MM.res[:Br].n, MM.res[:Br].bias.df2)

    # Cell-mean tables
    for (tbl, o) in (("tbl-inv-means-layher1", L1), ("tbl-inv-means-layher2", L2), ("tbl-inv-means-mm", MM))
        P = Inv.cell_means(o.ds, o.data, :PPB)
        C = Inv.cell_means(o.ds, o.data, :c)
        rows = table_rows(:supp, tbl)
        invariant("$tbl has $(size(P, 1)) rows", length(rows) == size(P, 1))
        for (i, row) in enumerate(rows), j in 1:3
            cell("$tbl row $i PPB col $j", row[1 + j], P[i, j])
            cell("$tbl row $i c col $j", row[4 + j], C[i, j])
        end
    end
    P1 = Inv.cell_means(L1.ds, L1.data, :PPB)
    dP = P1[2, :] .- P1[1, :]
    cite("Supp: Layher Exp 1 PPB strength gaps", :supp, "the strength differences in PPB are #, # and #", dP...)
    cite("Supp: Layher Exp 1 PPB interaction p", :supp, "the interaction is not significant (p = #)", r1[:PPB].interaction.p)
    cite("Supp: d' payoff F", :supp, "because payoff also moves it (F(2, 76) = #)", r1[:dprime].bias.F)
    dF = maximum(abs(getfield(L2.res[:PPB_ll], e).F - getfield(L2.res[:PPB], e).F) for e in (:strength, :bias, :interaction))
    cite("Supp: Exp 2 log-linear PPB change", :supp, "these statistics change by at most #", dF)

    cp = L1.contrasts[:PPB].con_minus_lib
    cite("Invariance: Layher PPB strength interaction interval", :paper,
         "The Layher Experiment 1 conservative-minus-liberal difference in PPB strength effects was #, with a paired 95% interval of [#, #]",
         cp.estimate, cp.lower, cp.upper)
    cp, cc = MM.contrasts[:PPB].simple["con"], MM.contrasts[:c].simple["con"]
    cite("Invariance: MM conservative PPB strength interval", :paper,
         "the conservative-instruction strength effect on PPB was #, with interval [#, #]", cp.estimate, cp.lower, cp.upper)
    cite("Invariance: MM conservative c strength interval", :paper,
         "the corresponding effect on c was #, with interval [#, #]", cc.estimate, cc.lower, cc.upper)
    claim("Invariance: MM pointwise departures", :supp,
          "the condition-specific confidence intervals for both PPB and c exclude zero even though the overall ANOVA tests detect no departure",
          cp.lower > 0 && cc.upper < 0 && passes(MM, :PPB) && passes(MM, :c))

    # Verify every estimate and endpoint in the included contrast table, as
    # well as its complete generated content and design-specific uncertainty.
    contrast_rows = table_rows(:supp, "tbl-inv-contrasts")
    invariant("Invariance contrast table has eight specified rows", length(contrast_rows) == 8)
    row_index = 0
    for o in (L1, MM), b in (o.ds.blevels..., "con_minus_lib")
        row_index += 1
        for (j, y) in enumerate((:PPB, :c, :J))
            ci = b == "con_minus_lib" ? o.contrasts[y].con_minus_lib : o.contrasts[y].simple[b]
            if row_index <= length(contrast_rows)
                ms = collect(eachmatch(Regex(NUM), contrast_rows[row_index][j + 2]))
                invariant("Contrast row $row_index $y has estimate and endpoints", length(ms) == 3)
                for (m, value) in zip(ms, (ci.estimate, ci.lower, ci.upper))
                    cell("Contrast row $row_index $y value", m.match, value)
                end
            end
            if o.ds.design == :within
                se = std(ci.subject_differences) / sqrt(ci.n)
                invariant("Contrast row $row_index $y retains participant covariance", isapprox(ci.se, se) && ci.df == ci.n - 1)
            else
                components = [cell.weight^2 * variance / n for (cell, variance, n) in zip(ci.cells, ci.cell_variances, ci.n_cells)]
                expected_df = sum(components)^2 / sum(components .^ 2 ./ (ci.n_cells .- 1))
                invariant("Contrast row $row_index $y uses independent-cell variances", isapprox(ci.se^2, sum(components)) && isapprox(ci.df, expected_df))
            end
        end
    end
    cite("Contrast caption: Layher sample and df", :supp, "(# participants; # df)", length(unique(L1.data.sub)), length(unique(L1.data.sub)) - 1)
    cell_n(b, strength) = count(r -> r.bias == b && r.strength == strength, eachrow(MM.data))
    cite("Contrast caption: Measuring Memory sample sizes", :supp,
         "sample sizes are #/# with no instruction, #/# with conservative instruction, and #/# with liberal instruction",
         (cell_n(b, s) for b in ("none", "con", "lib") for s in ("1x", "3x"))...)
    mktempdir() do temporary_dir
        path = Inv.write_contrast_table(inv; path=joinpath(temporary_dir, "contrasts.md"))
        generated_table("tbl_invariance_contrasts.md", read(path, String), :supp; script="code/invariance.jl")
    end
end

function check_colloff_wixted()
    R = [showup_contrast(e) for e in 1:3]
    cite("Colloff-Wixted: PPB differences and intervals", :paper,
         "by # (95% CI: # to #; N = #), # (# to #; N = #), and # (# to #; N = #)",
         (v for r in R for v in (r.ppb.diff, r.ppb.lo, r.ppb.hi, r.n))...)
    cite("Colloff-Wixted: percentage points", :paper,
         "which is #, #, and # percentage points in choosing probability",
         (100 * r.ppb.diff / 2 for r in R)...)
    cite("Colloff-Wixted: J differences", :paper, "No difference in J was detected (estimated differences: #, #, and #;",
         (r.J.diff for r in R)...)
    claim("Colloff-Wixted: every J interval includes zero", :paper, "every interval includes zero",
          all(r.J.lo < 0 < r.J.hi for r in R))
    claim("Colloff-Wixted: c undefined for every participant", :paper,
          "c is undefined for all of them", all(r.boundary_pct == 100 for r in R))
    claim("Colloff-Wixted: 1/2N gives c = 0 for everyone", :paper, "returns c = 0 for every participant",
          all(r.c_hn_values == [0.0] for r in R))
    claim("Colloff-Wixted: log-linear c has the same test as PPB", :paper,
          "yields the same test statistics",
          all(isapprox(r.c_ll.p, r.ppb.p; rtol = 1e-6) && isapprox(abs(r.c_ll.diff) / (r.c_ll.hi - r.c_ll.lo),
              abs(r.ppb.diff) / (r.ppb.hi - r.ppb.lo); rtol = 1e-3) for r in R),
          join((string(round(r.c_ll.p; sigdigits = 3), "/", round(r.ppb.p; sigdigits = 3)) for r in R), ", "))
    claim("Colloff-Wixted: showup PPB higher in all three", :paper, "Showups raised PPB",
          all(r.ppb.diff > 0 && r.ppb.lo > 0 for r in R))
    cite("Abstract: Colloff-Wixted percentage-point range", :paper,
         "showups raised identification probability, averaged across target-present and target-absent trials, by # to # percentage points relative to simultaneous showups",
         minimum(100 * r.ppb.diff / 2 for r in R), maximum(100 * r.ppb.diff / 2 for r in R))
    claim("Abstract: Colloff-Wixted c undefined for every participant", :paper,
          "c was undefined for every participant", all(r.boundary_pct == 100 for r in R))

    # Supplement §8
    z75 = quantile(Normal(), 0.75)
    cite("Supplement CW: z_.75", :supp, "PPB rescaled by z_{.75} = #", z75)
    affine = all(begin
                     d = load_colloff_wixted(e)
                     all(isapprox.([criterion_c(loglinear_correct(h, 1, f, 1)...) for (h, f) in zip(d.h, d.f)],
                                   z75 .* (1 .- d.ppb_val); atol = 1e-12))
                 end for e in 1:3)
    claim("Supplement CW: log-linear c = z_.75 (1 - PPB) for every participant", :supp,
          "z_{.75}\\left(1 - \\text{PPB}_i\\right)", affine)
    claim("Supplement CW: 1/2N gives c = 0 for everyone", :supp, "gives c_i = 0 for every participant",
          all(r.c_hn_values == [0.0] for r in R))
    claim("Supplement CW: every participant is a boundary case", :supp, "so every participant is a boundary case",
          all(r.boundary_pct == 100 for r in R))
    claim("Supplement CW: affine relation gives identical p values", :supp, "identical p values",
          all(isapprox(r.c_ll.p, r.ppb.p; rtol = 1e-6) for r in R))
end

function main()
    show_all = "--all" in ARGS
    println("Verifying numbers in ppb_paper.qmd and ppb_supplement.qmd ...")
    check_panel_a()
    check_aggregation()
    check_regression()
    check_group_comparison()
    check_distribution_shape()
    check_empirical()
    check_colloff_wixted()
    check_generated_tables()
    check_intervals()
    check_properties()
    check_supplement()
    check_invariance()
    report_checks(show_all)
end

main()
