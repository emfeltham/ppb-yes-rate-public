# invariance.jl
# ─────────────────────────────────────────────────────────────────────────────
# Snodgrass & Corwin (1988) independence test on two open datasets that cross a
# discriminability manipulation with a bias manipulation.
#
#   Layher, Dixit & Miller (2020), Exp 1: strength (1x / 5x study) x payoff
#     (conservative / neutral / liberal), within subjects, N = 39.
#   Layher, Dixit & Miller (2020), Exp 2: strength x base rate (25 / 50 / 75%
#     old), within subjects, N = 39.
#   Starns, Cataldo, Rotello et al. (2019), Measuring Memory Project: strength
#     (1x / 2x / 3x study) x bias instruction (none / conservative / liberal),
#     between subjects, N = 459.
#
# Indexes per subject x cell: PPB = H + F and J = H - F on raw rates;
# c and d' on log-linear
# corrected rates, (x + 0.5)/(N + 1); B_r = F/(1 - H + F) on log-linear rates,
# because raw B_r is 0/0 at H = 1, F = 0, which occurs in the Measuring Memory
# data. Sensitivity: PPB on log-linear rates (the rule Snodgrass & Corwin apply
# to every index, and the rule of the preliminary Python analysis); J on
# log-linear rates; c with the
# 1/2N correction; B_r on raw rates, dropping participants where it is 0/0
# (within-subjects: dropping the whole subject).
#
# Pass rule (Snodgrass & Corwin, 1988, p. 43), alpha = .05: a bias index needs
# a significant bias-factor effect, no discriminability-factor effect and no
# interaction; a discrimination index needs the reverse pattern on the two
# main effects and no interaction.
#
# ANOVAs: within-subjects designs use the textbook sums-of-squares
# decomposition (subject, A, B, AxB and the three subject-by-effect error
# terms), with no sphericity correction, as in Snodgrass & Corwin. The
# between-subjects design (unequal n, 48-53 per cell) uses Type III sums of
# squares from sum-to-zero coded least squares.
# The historical rule is retained for numerical replication. Non-rejection
# is described as no departure detected, not affirmative invariance. Specified
# strength contrasts use paired participant t intervals for Layher and Welch
# intervals for independent participant groups in Measuring Memory. Their
# pointwise 95% intervals have no multiplicity adjustment or equivalence margin.
#
# Run from the project root: julia +1.12 --project=code code/invariance.jl
# ─────────────────────────────────────────────────────────────────────────────

using CSV, DataFrames, Distributions, Statistics, LinearAlgebra, Printf

const INV_DATA = joinpath(@__DIR__, "..", "data")
const ALPHA = 0.05

# ─── Rates and indexes ─────────────────────────────────────────────────────

zq(p) = quantile(Normal(), p)
loglin(x, n) = (x + 0.5) / (n + 1)
function halfn(x, n)
    p = x / n
    p == 0 && return 1 / (2n)
    p == 1 && return 1 - 1 / (2n)
    return p
end
crit(H, F) = (0 < H < 1 && 0 < F < 1) ? -0.5 * (zq(H) + zq(F)) : NaN
dprime(H, F) = (0 < H < 1 && 0 < F < 1) ? zq(H) - zq(F) : NaN
function br(H, F)
    d = 1 - H + F
    return d == 0 ? NaN : F / d
end

"""Add rates and indexes to a table with columns ht, ms, fa, cr."""
function add_indexes!(d::DataFrame)
    d.n_old = d.ht .+ d.ms
    d.n_new = d.fa .+ d.cr
    d.H = d.ht ./ d.n_old
    d.F = d.fa ./ d.n_new
    d.H_ll = loglin.(d.ht, d.n_old)
    d.F_ll = loglin.(d.fa, d.n_new)
    d.H_hn = halfn.(d.ht, d.n_old)
    d.F_hn = halfn.(d.fa, d.n_new)
    d.boundary = (d.H .== 0) .| (d.H .== 1) .| (d.F .== 0) .| (d.F .== 1)
    d.PPB = d.H .+ d.F
    d.PPB_ll = d.H_ll .+ d.F_ll
    d.J = d.H .- d.F
    d.J_ll = d.H_ll .- d.F_ll
    d.c = crit.(d.H_ll, d.F_ll)
    d.c_hn = crit.(d.H_hn, d.F_hn)
    d.c_raw = crit.(d.H, d.F)
    d.dprime = dprime.(d.H_ll, d.F_ll)
    d.Br = br.(d.H, d.F)
    d.Br_ll = br.(d.H_ll, d.F_ll)
    return d
end

# ─── Data ──────────────────────────────────────────────────────────────────

const PAYOFF_LEVELS = ["con", "neu", "lib"]
const STRENGTH_LAYHER = ["low", "mod"]

"""Layher et al. (2020) Exp 1 or 2: sum the session-level counts to one row per
subject x strength x bias cell."""
function load_layher(exp::Int)
    raw = CSV.read(joinpath(INV_DATA, "layher2020", "layher20_exp$(exp).csv"), DataFrame)
    d = combine(groupby(raw, [:sub, :dCond, :cCond]),
        :ht => sum => :ht, :ms => sum => :ms, :fa => sum => :fa, :cr => sum => :cr,
        :ses => (s -> join(sort(unique(s)), ",")) => :sessions)
    d.strength = String.(d.dCond)
    d.bias = String.(d.cCond)
    return add_indexes!(d)
end

const MM_STRENGTH = Dict(1 => "1x", 2 => "1x", 3 => "1x", 4 => "3x", 5 => "3x",
                         6 => "3x", 7 => "2x", 8 => "2x", 9 => "2x")
const MM_BIAS = Dict(1 => "none", 2 => "con", 3 => "lib", 4 => "none", 5 => "con",
                     6 => "lib", 7 => "none", 8 => "con", 9 => "lib")

"""Measuring Memory Project full data (Starns et al., 2019): one row per
participant. IDs are unique only within condition (data key)."""
function load_mm()
    raw = CSV.read(joinpath(INV_DATA, "measuring_memory", "data_full.csv"), DataFrame)
    raw.old = raw.binary_response .== "old"
    raw.tgt = raw.stimulus_type .== "target"
    d = combine(groupby(raw, [:condition, :ID]),
        [:tgt, :old] => ((t, o) -> sum(t .& o)) => :ht,
        [:tgt, :old] => ((t, o) -> sum(t .& .!o)) => :ms,
        [:tgt, :old] => ((t, o) -> sum(.!t .& o)) => :fa,
        [:tgt, :old] => ((t, o) -> sum(.!t .& .!o)) => :cr,
        nrow => :n_trials)
    d.strength = [MM_STRENGTH[k] for k in d.condition]
    d.bias = [MM_BIAS[k] for k in d.condition]
    return add_indexes!(d)
end

# ─── ANOVA ─────────────────────────────────────────────────────────────────

struct Effect
    name::String
    SS::Float64
    df1::Int
    df2::Int
    F::Float64
    p::Float64
end

ftest(name, SSe, df, SSerr, dferr) = begin
    Fv = (SSe / df) / (SSerr / dferr)
    Effect(name, SSe, df, dferr, Fv, ccdf(FDist(df, dferr), Fv))
end

"""Two-way repeated-measures ANOVA, both factors within subjects.
Returns (discriminability effect, bias effect, interaction, n subjects)."""
function rm_anova(d::DataFrame, y::Symbol; alevels, blevels)
    subs = sort(unique(d.sub))
    n, a, b = length(subs), length(alevels), length(blevels)
    Y = fill(NaN, n, a, b)
    si = Dict(s => k for (k, s) in enumerate(subs))
    ai = Dict(l => k for (k, l) in enumerate(alevels))
    bi = Dict(l => k for (k, l) in enumerate(blevels))
    for r in eachrow(d)
        Y[si[r.sub], ai[r.strength], bi[r.bias]] = r[y]
    end
    any(!isfinite, Y) && error("rm_anova: missing or non-finite $y cell")
    g = mean(Y)
    ms = dropdims(mean(Y; dims = (2, 3)); dims = (2, 3))
    ma = dropdims(mean(Y; dims = (1, 3)); dims = (1, 3))
    mb = dropdims(mean(Y; dims = (1, 2)); dims = (1, 2))
    msa = dropdims(mean(Y; dims = 3); dims = 3)
    msb = dropdims(mean(Y; dims = 2); dims = 2)
    mab = dropdims(mean(Y; dims = 1); dims = 1)
    SS_S = a * b * sum((ms .- g) .^ 2)
    SS_A = n * b * sum((ma .- g) .^ 2)
    SS_B = n * a * sum((mb .- g) .^ 2)
    SS_AB = n * sum((mab .- ma .- mb' .+ g) .^ 2)
    SS_AS = b * sum((msa .- ms .- ma' .+ g) .^ 2)
    SS_BS = a * sum((msb .- ms .- mb' .+ g) .^ 2)
    SS_T = sum((Y .- g) .^ 2)
    SS_ABS = SS_T - SS_S - SS_A - SS_B - SS_AB - SS_AS - SS_BS
    eA = ftest("strength", SS_A, a - 1, SS_AS, (a - 1) * (n - 1))
    eB = ftest("bias", SS_B, b - 1, SS_BS, (b - 1) * (n - 1))
    eAB = ftest("interaction", SS_AB, (a - 1) * (b - 1), SS_ABS, (a - 1) * (b - 1) * (n - 1))
    return (strength = eA, bias = eB, interaction = eAB, n = n)
end

"""Sum-to-zero (deviation) coding of a factor: levels-1 columns."""
function sumcode(x, levels)
    k = length(levels)
    M = zeros(length(x), k - 1)
    for (i, v) in enumerate(x)
        j = findfirst(==(v), levels)
        if j == k
            M[i, :] .= -1
        else
            M[i, j] = 1
        end
    end
    return M
end

rss(X, y) = (r = y - X * (X \ y); sum(abs2, r))

"""Two-way between-subjects ANOVA. Type III sums of squares by default;
`sstype = 2` gives Type II (each main effect adjusted for the other, not for
the interaction)."""
function between_anova(d::DataFrame, y::Symbol; alevels, blevels, sstype::Int = 3)
    v = d[!, y]
    all(isfinite, v) || error("between_anova: non-finite $y")
    A = sumcode(d.strength, alevels)
    B = sumcode(d.bias, blevels)
    AB = hcat([A[:, i] .* B[:, j] for i in axes(A, 2) for j in axes(B, 2)]...)
    one_ = ones(length(v))
    Xf = hcat(one_, A, B, AB)
    rf = rss(Xf, v)
    dfe = length(v) - size(Xf, 2)
    if sstype == 3
        SSA = rss(hcat(one_, B, AB), v) - rf
        SSB = rss(hcat(one_, A, AB), v) - rf
    else
        rmain = rss(hcat(one_, A, B), v)
        SSA = rss(hcat(one_, B), v) - rmain
        SSB = rss(hcat(one_, A), v) - rmain
    end
    eA = ftest("strength", SSA, size(A, 2), rf, dfe)
    eB = ftest("bias", SSB, size(B, 2), rf, dfe)
    eAB = ftest("interaction", rss(hcat(one_, A, B), v) - rf, size(AB, 2), rf, dfe)
    return (strength = eA, bias = eB, interaction = eAB, n = length(v))
end

# ─── Verdicts ──────────────────────────────────────────────────────────────

const DISCRIM_INDEXES = (:dprime, :J, :J_ll)

"""Snodgrass & Corwin's rule. Bias index: bias p < .05, strength p >= .05,
interaction p >= .05. Discrimination index: strength p < .05, bias p >= .05,
interaction p >= .05."""
function verdict(res, index::Symbol)
    if index in DISCRIM_INDEXES
        ok = res.strength.p < ALPHA && res.bias.p >= ALPHA && res.interaction.p >= ALPHA
    else
        ok = res.bias.p < ALPHA && res.strength.p >= ALPHA && res.interaction.p >= ALPHA
    end
    return ok ? "pass" : "fail"
end

"""Interpret the historical rule without treating non-rejection as equivalence.
`target_detected` concerns the intended main effect; `departure_detected`
concerns the other main effect or interaction. A pass is 'no departure
detected' only when the intended manipulation has a detected effect. No
equivalence tolerance is imposed or tested."""
function rule_assessment(res, index::Symbol)
    discrimination = index in DISCRIM_INDEXES
    target = discrimination ? res.strength : res.bias
    other = discrimination ? res.bias : res.strength
    target_detected = target.p < ALPHA
    departure_detected = other.p < ALPHA || res.interaction.p < ALPHA
    status = departure_detected ? "departure detected" :
             target_detected ? "no departure detected" : "target effect not detected"
    return (status = status, target_detected = target_detected,
            departure_detected = departure_detected, historical_verdict = verdict(res, index))
end

"""Verdict with the Greenhouse-Geisser lower-bound correction (every effect
tested on 1 and n - 1 df), the most conservative sphericity correction. Used
only to show that the within-subjects verdicts do not depend on sphericity."""
function verdict_lowerbound(res, index::Symbol)
    n = res.n
    lb(e) = ccdf(FDist(1, n - 1), e.F)
    r = (strength = (p = lb(res.strength),), bias = (p = lb(res.bias),),
         interaction = (p = lb(res.interaction),))
    return verdict(r, index)
end

# ─── Datasets ──────────────────────────────────────────────────────────────

const DATASETS = (
    (key = :layher1, label = "Layher Exp 1 (strength x payoff)", design = :within,
     alevels = STRENGTH_LAYHER, blevels = PAYOFF_LEVELS, load = () -> load_layher(1)),
    (key = :layher2, label = "Layher Exp 2 (strength x base rate)", design = :within,
     alevels = STRENGTH_LAYHER, blevels = PAYOFF_LEVELS, load = () -> load_layher(2)),
    (key = :mm, label = "Measuring Memory (strength x instruction)", design = :between,
     alevels = ["1x", "2x", "3x"], blevels = ["none", "con", "lib"], load = load_mm),
)

const PRIMARY = (:PPB, :c, :J, :dprime, :Br_ll)
const SENSITIVITY = (:PPB_ll, :J_ll, :c_hn, :Br)

"""ANOVA for index `y`. Rows (between) or subjects (within) with a
non-finite value are dropped; only raw B_r can produce one."""
function run_anova(ds, d, y)
    if ds.design == :within
        bad = unique(d.sub[.!isfinite.(d[!, y])])
        dd = d[.!in.(d.sub, Ref(Set(bad))), :]
        return rm_anova(dd, y; alevels = ds.alevels, blevels = ds.blevels)
    else
        dd = d[isfinite.(d[!, y]), :]
        return between_anova(dd, y; alevels = ds.alevels, blevels = ds.blevels)
    end
end

"""Cell means of an index, strength (rows) x bias (columns)."""
function cell_means(ds, d, y)
    M = [mean(d[(d.strength .== a) .& (d.bias .== b), y]) for a in ds.alevels, b in ds.blevels]
    return M
end

cell_counts(ds, d) =
    [count((d.strength .== a) .& (d.bias .== b)) for a in ds.alevels, b in ds.blevels]

# ─── Specified strength contrasts and uncertainty ──────────────────────────

"""Two-sided t interval and test for a participant-population mean contrast.
The interval is pointwise, without a multiplicity adjustment. This is not a
binomial interval for an individual participant's rates."""
function contrast_interval(estimate, se, df; level = 0.95)
    0 < level < 1 || throw(ArgumentError("level must be between 0 and 1"))
    isfinite(estimate) && isfinite(se) && se >= 0 ||
        throw(ArgumentError("contrast estimate and nonnegative SE must be finite"))
    df > 0 || throw(ArgumentError("contrast degrees of freedom must be positive"))
    dist = isinf(df) ? Normal() : TDist(df)
    margin = quantile(dist, (1 + level) / 2) * se
    t = se == 0 ? (estimate == 0 ? 0.0 : sign(estimate) * Inf) : estimate / se
    p = se == 0 ? (estimate == 0 ? 1.0 : 0.0) : 2 * ccdf(dist, abs(t))
    return (estimate = Float64(estimate), se = Float64(se), df = Float64(df),
            lower = estimate - margin, upper = estimate + margin,
            t = t, p = p, level = level)
end

"""Paired contrast across subject-by-cell summaries.
`cells` contains named tuples `(strength, bias, weight)`. Every selected cell
must contain exactly one finite value for the same set of participants. The
weighted contrast is first computed within each participant; its observed
standard deviation determines the interval, preserving all within-person
covariances. Returns interval fields, `n`, `subjects`, and
`subject_differences` for independent verification."""
function paired_cell_contrast(d::DataFrame, y::Symbol, cells; level = 0.95)
    isempty(cells) && throw(ArgumentError("a contrast needs at least one cell"))
    maps = map(cells) do cell
        rows = d[(d.strength .== cell.strength) .& (d.bias .== cell.bias), :]
        values = Dict(r.sub => Float64(r[y]) for r in eachrow(rows))
        length(values) == nrow(rows) || error("paired contrast: duplicate participant cell")
        all(isfinite, Base.values(values)) || error("paired contrast: non-finite $y")
        values
    end
    subject_set = Set(keys(first(maps)))
    all(m -> Set(keys(m)) == subject_set, maps) ||
        error("paired contrast: selected cells must have the same participants")
    subjects = sort(collect(subject_set))
    n = length(subjects)
    n >= 2 || error("paired contrast: at least two participants are required")
    differences = [sum(cell.weight * values[s] for (cell, values) in zip(cells, maps))
                   for s in subjects]
    ci = contrast_interval(mean(differences), std(differences) / sqrt(n), n - 1; level)
    return merge(ci, (method = :paired_t, n = n, n_cells = fill(n, length(cells)),
                      cells = cells, subjects = subjects, subject_differences = differences))
end

"""Contrast of independent participant groups, with a Welch t interval.
`cells` contains named tuples `(strength, bias, weight)`. For a difference of
strength differences, all four cells are independent in the between-person
design: variance is the sum of weighted mean variances and df use the
Welch-Satterthwaite approximation. Returns group means, variances and sample
sizes for verification. Do not use this helper for repeated participants."""
function independent_cell_contrast(d::DataFrame, y::Symbol, cells; level = 0.95)
    isempty(cells) && throw(ArgumentError("a contrast needs at least one cell"))
    selected = [(cell.strength, cell.bias) for cell in cells]
    allunique(selected) || throw(ArgumentError("independent contrast cells must be distinct"))
    groups = [Float64.(d[(d.strength .== cell.strength) .& (d.bias .== cell.bias), y])
              for cell in cells]
    all(g -> length(g) >= 2 && all(isfinite, g), groups) ||
        error("independent contrast: each cell needs at least two finite $y values")
    n_cells = length.(groups)
    means, variances = mean.(groups), var.(groups)
    weights = [cell.weight for cell in cells]
    mean_variances = weights .^ 2 .* variances ./ n_cells
    variance = sum(mean_variances)
    df = variance == 0 ? Inf : variance^2 / sum(mean_variances .^ 2 ./ (n_cells .- 1))
    ci = contrast_interval(sum(weights .* means), sqrt(variance), df; level)
    return merge(ci, (method = :welch_t, n = sum(n_cells), n_cells = n_cells,
                      cells = cells, cell_means = means, cell_variances = variances))
end

"""Uniformly specified contrasts for PPB, c and J, without selection by p.
The high-minus-low strength contrast is computed at every bias level; the
conservative-minus-liberal contrast compares those two strength differences.
Layher uses paired t intervals; Measuring Memory uses independent-group
Welch intervals. Returns `Dict(index => (simple = Dict(bias => interval),
con_minus_lib = interval, low, high))`. Each interval also carries its data
summaries and design information."""
function planned_strength_contrasts(ds, d; indexes = (:PPB, :c, :J), level = 0.95)
    low, high = first(ds.alevels), last(ds.alevels)
    helper = ds.design == :within ? paired_cell_contrast : independent_cell_contrast
    simple_cells(b) = ((strength = high, bias = b, weight = 1.0),
                       (strength = low, bias = b, weight = -1.0))
    difference_cells = (simple_cells("con")...,
                        (strength = high, bias = "lib", weight = -1.0),
                        (strength = low, bias = "lib", weight = 1.0))
    return Dict(y => (simple = Dict(b => helper(d, y, simple_cells(b); level)
                                    for b in ds.blevels),
                      con_minus_lib = helper(d, y, difference_cells; level),
                      low = low, high = high) for y in indexes)
end

"""Run every dataset and index. Returns Dict key =>
`(ds, data, res, res_type2, contrasts)`. See `planned_strength_contrasts` for
the participant-level interval fields in `contrasts`."""
function run_all()
    out = Dict{Symbol, Any}()
    for ds in DATASETS
        d = ds.load()
        res = Dict(y => run_anova(ds, d, y) for y in (PRIMARY..., SENSITIVITY...))
        res2 = ds.design == :between ?
            Dict(y => between_anova(d, y; alevels = ds.alevels, blevels = ds.blevels,
                                    sstype = 2) for y in PRIMARY) :
            Dict{Symbol, Any}()
        contrasts = planned_strength_contrasts(ds, d)
        out[ds.key] = (ds = ds, data = d, res = res, res_type2 = res2, contrasts = contrasts)
    end
    return out
end

# ─── Printing ──────────────────────────────────────────────────────────────

fmtp(p) = p < 0.001 ? "<.001" : string(round(p; digits = 3))
fmtF(F) = string(round(F; digits = 2))
cellstr(e) = rpad("F($(e.df1),$(e.df2))=$(fmtF(e.F)), p=$(fmtp(e.p))", 26)

function print_table(out)
    println("=" ^ 100)
    println("SNODGRASS & CORWIN ANOVA CRITERIA  (alpha = .05; F(df1, df2), p)")
    println("=" ^ 100)
    for ds in DATASETS
        o = out[ds.key]
        d = o.data
        nsub = ds.design == :within ? length(unique(d.sub)) : nrow(d)
        println("\n$(ds.label)   N = $nsub   boundary subject-cells = $(count(d.boundary)) of $(nrow(d))")
        println(rpad("index", 9), rpad("strength", 26), rpad("bias", 26), rpad("interaction", 26), "assessment")
        for y in (PRIMARY..., SENSITIVITY...)
            r = o.res[y]
            lbv = ds.design == :within ? "  lower-bound: $(verdict_lowerbound(r, y))" : ""
            println(rpad(string(y), 9), cellstr(r.strength), cellstr(r.bias),
                    cellstr(r.interaction), rule_assessment(r, y).status,
                    "  (ANOVA criteria: $(verdict(r, y)); n = $(r.n))", lbv)
        end
        for (y, r) in sort(collect(o.res_type2); by = first)
            println(rpad(string(y) * "/II", 9), cellstr(r.strength), cellstr(r.bias),
                    cellstr(r.interaction), rule_assessment(r, y).status,
                    "  (ANOVA criteria: $(verdict(r, y)); Type II SS)")
        end
        println("cell n (rows strength $(ds.alevels), cols bias $(ds.blevels)): ",
                cell_counts(ds, d))
        if ds.design == :within
            println("trials per subject-cell: old ", sort(unique(d.n_old)),
                    ", new ", sort(unique(d.n_new)))
            println("sessions by strength: ",
                    Dict(s => unique(d.sessions[d.strength .== s]) for s in ds.alevels))
        else
            println("trials per participant: old ", sort(unique(d.n_old)),
                    ", new ", sort(unique(d.n_new)))
        end
        for y in (:PPB, :c, :J, :dprime, :H, :F)
            M = round.(cell_means(ds, d, y); digits = 3)
            println("mean $(rpad(string(y), 7)) ", join([join(M[i, :], "  ") for i in axes(M, 1)], "  |  "))
        end
    end
end

"""Generate the compact, reproducible contrast table for the supplement.
The full-precision values remain accessible through `run_all()[key].contrasts`."""
function write_contrast_table(out; path = joinpath(@__DIR__, "..", "tables", "tbl_invariance_contrasts.md"),
                              datasets = (:layher1, :mm))
    fmtci(ci) = @sprintf("%.3f [%.3f, %.3f]", ci.estimate, ci.lower, ci.upper)
    labels = Dict(:layher1 => "Layher Exp. 1", :layher2 => "Layher Exp. 2", :mm => "Measuring Memory")
    biaslabels = Dict("con" => "Conservative", "neu" => "Neutral", "lib" => "Liberal", "none" => "No instruction")
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "| Dataset | Strength contrast | PPB difference [95% CI] | c difference [95% CI] | J difference [95% CI] |")
        println(io, "|---|---|---|---|---|")
        for key in datasets
            o = out[key]
            for b in o.ds.blevels
                values = [fmtci(o.contrasts[y].simple[b]) for y in (:PPB, :c, :J)]
                println(io, "| $(labels[key]) | $(biaslabels[b]) | $(join(values, " | ")) |")
            end
            values = [fmtci(o.contrasts[y].con_minus_lib) for y in (:PPB, :c, :J)]
            println(io, "| $(labels[key]) | Conservative minus liberal | $(join(values, " | ")) |")
        end
        println(io)
        println(io, ": Specified strength contrasts, high minus low (5 minus 1 presentations for Layher; 3 minus 1 for Measuring Memory), and conservative-minus-liberal differences between those strength contrasts. PPB and J use raw rates; c uses log-linear rates. Layher intervals are t-based confidence intervals computed from participant-level contrasts (39 participants; 38 df), preserving the covariance among cells. Measuring Memory intervals use independent group means and Welch-Satterthwaite df; variances from all four independent cells enter the difference of strength differences. The 1-presentation/3-presentation sample sizes are 53/48 with no instruction, 52/48 with conservative instruction, and 53/50 with liberal instruction. Each comparison has a nominal 95% confidence interval, without adjustment for multiple comparisons. These intervals describe population-level contrasts across participants; individual binomial uncertainty is addressed separately. Strength is confounded with practice in Layher Experiment 1. No equivalence tolerance is specified, so intervals containing zero do not establish invariance. {#tbl-inv-contrasts}")
    end
    return path
end

function main()
    out = run_all()
    print_table(out)
    println("\nContrast table: ", write_contrast_table(out))
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
