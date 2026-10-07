# colloff.jl
# ─────────────────────────────────────────────────────────────────────────────
# Shared loader for Experiment 1 of Colloff et al. (2022), used by tables.jl,
# empirical.jl and verify_numbers.jl. Primary coding is any lineup identification
# versus rejection in BOTH target-present and target-absent trials.
#
# include() this file; it defines functions only.
# ─────────────────────────────────────────────────────────────────────────────

using CSV, DataFrames, Distributions, Statistics

const COLLOFF_PATH = joinpath(@__DIR__, "..", "data", "pyWitness", "data", "published",
    "2020_Colloff_Flowe_Smith_etal", "Exp1_osf_data.csv")

Φinv(p) = quantile(Normal(), p)
ppb(H, F) = H + F

function criterion_c(H, F)
    (H <= 0 || H >= 1 || F <= 0 || F >= 1) && return NaN
    return -0.5 * (Φinv(H) + Φinv(F))
end

function b_r(H, F)
    denom = 1 - H + F
    denom == 0 && return NaN
    return F / denom
end

function b_double_prime(H, F)
    denom = H * (1 - H) + F * (1 - F)
    denom == 0 && return NaN
    H == F && return 0.0
    return sign(H - F) * (H * (1 - H) - F * (1 - F)) / denom
end

function loglinear_correct(hits, n_signal, fa, n_noise)
    H = (hits + 0.5) / (n_signal + 1)
    F = (fa + 0.5) / (n_noise + 1)
    return (H, F)
end

function half_n_correct(hits, n_signal, fa, n_noise)
    H = hits / n_signal
    F = fa / n_noise
    if H == 0.0; H = 1 / (2 * n_signal); elseif H == 1.0; H = 1 - 1 / (2 * n_signal); end
    if F == 0.0; F = 1 / (2 * n_noise); elseif F == 1.0; F = 1 - 1 / (2 * n_noise); end
    return (H, F)
end

"""Included trials with an explicit response coding.

`:choose` (primary): choosing any lineup member is positive, whether the selected
member is the perpetrator or a foil; rejection is negative in both TP and TA.
Thus J measures TP–TA separation in choosing rates, not culprit identification
accuracy. `:identification` (alternative coding): TP perpetrator identifications
are hits and TA foil identifications are false alarms; TP foil choices are
negative. Its H+F is a descriptive sum of differently defined response rates,
not the primary binary choosing measure.
"""
function load_colloff_raw(; response = :choose)
    response in (:choose, :identification) ||
        throw(ArgumentError("response must be :choose or :identification"))
    raw = CSV.read(COLLOFF_PATH, DataFrame)
    filter!(:Include => ==("yes"), raw)
    all(x -> x in ("yes", "no"), raw.TargetPresent) || error("Unknown target status")
    all(x -> x in ("perpetrator", "foil", "reject"), raw.IDResponse) ||
        error("Unknown lineup response")
    raw.is_tp = raw.TargetPresent .== "yes"
    any(.!raw.is_tp .& (raw.IDResponse .== "perpetrator")) &&
        error("Perpetrator response on a target-absent trial")
    raw.is_positive = response == :choose ? raw.IDResponse .!= "reject" :
        ifelse.(raw.is_tp, raw.IDResponse .== "perpetrator", raw.IDResponse .== "foil")
    raw.is_hit = raw.is_tp .& raw.is_positive
    raw.is_fa = .!raw.is_tp .& raw.is_positive
    raw.is_miss = raw.is_tp .& .!raw.is_positive
    raw.is_cr = .!raw.is_tp .& .!raw.is_positive
    return raw
end

"""Aggregate coded trials, using the same measures for tables and reports."""
function aggregate_colloff(raw, groupcols)
    d = combine(groupby(raw, groupcols),
        :is_tp => sum => :n_tp,
        :is_hit => sum => :hits,
        :is_miss => sum => :misses,
        :is_tp => (tp -> sum(.!tp)) => :n_ta,
        :is_fa => sum => :fa,
        :is_cr => sum => :cr,
    )
    d.H = d.hits ./ d.n_tp
    d.F = d.fa ./ d.n_ta
    d.ppb_val = ppb.(d.H, d.F)
    d.J = d.H .- d.F
    d.at_boundary = (d.H .== 0) .| (d.H .== 1) .| (d.F .== 0) .| (d.F .== 1)
    d.br_undefined = (d.H .== 1) .& (d.F .== 0)
    d.c_raw = criterion_c.(d.H, d.F)
    d.br_raw = b_r.(d.H, d.F)
    d.bpp = b_double_prime.(d.H, d.F)

    ll = loglinear_correct.(d.hits, d.n_tp, d.fa, d.n_ta)
    d.H_ll = first.(ll)
    d.F_ll = last.(ll)
    d.c_ll = criterion_c.(d.H_ll, d.F_ll)
    d.br_ll = b_r.(d.H_ll, d.F_ll)
    d.bpp_ll = b_double_prime.(d.H_ll, d.F_ll)

    hn = half_n_correct.(d.hits, d.n_tp, d.fa, d.n_ta)
    d.H_hn = first.(hn)
    d.F_hn = last.(hn)
    d.c_hn = criterion_c.(d.H_hn, d.F_hn)
    return d
end

"""Subject-level (4 + 4 trials) and condition-level (2 + 2 trials) data frames."""
function load_colloff(; response = :choose)
    raw = load_colloff_raw(; response)
    subj = aggregate_colloff(raw, [:ParticipantId, :SubjectRace])
    cond = aggregate_colloff(raw, [:ParticipantId, :SubjectRace, :PerpRace, :OwnRace])
    return subj, cond
end

"""Participant-paired own-minus-other mean and two-sided Student-t interval.

Joining by ParticipantId avoids relying on row order. Set scale=0.5 for PPB/2.
The interval describes uncertainty across participants, using df = n - 1.
"""
function paired_ownrace_summary(cond; measure = :ppb_val, scale = 1.0, level = 0.95)
    own = select(filter(:OwnRace => ==("ownRace"), cond), :ParticipantId, measure => :own)
    other = select(filter(:OwnRace => ==("otherRace"), cond), :ParticipantId, measure => :other)
    pairs = innerjoin(own, other; on = :ParticipantId, validate = (true, true))
    nrow(pairs) == nrow(own) == nrow(other) || error("Incomplete own/other participant pairs")
    d = scale .* (pairs.own .- pairs.other)
    n = length(d)
    n > 1 || throw(ArgumentError("At least two participant pairs are required"))
    0 < level < 1 || throw(ArgumentError("level must be between zero and one"))
    estimate = mean(d)
    se = std(d) / sqrt(n)
    margin = quantile(TDist(n - 1), (1 + level) / 2) * se
    return (; n, df = n - 1, mean = estimate, se, lo = estimate - margin, hi = estimate + margin)
end

"""Membership weights for `k` equal-size groups ranked by `x`, with ties split
proportionally: a block of m tied values occupying ranks a+1..a+m gives each
member the share of those ranks that falls in each group. This equals the
expected membership under random tie-breaking, so the result does not depend on
row order. Returns an n × k matrix whose columns each sum to n/k."""
function quantile_weights(x; k = 4)
    n = length(x)
    width = n / k
    ord = sortperm(x)
    W = zeros(n, k)
    i = 1
    while i <= n
        j = i
        while j < n && x[ord[j + 1]] == x[ord[i]]
            j += 1
        end
        m = j - i + 1
        for q in 1:k
            overlap = max(0.0, min(j, q * width) - max(i - 1, (q - 1) * width))
            W[ord[i:j], q] .= overlap / m
        end
        i = j + 1
    end
    return W
end

wmean(x, w) = sum(w .* x) / sum(w)

"""TP–TA response-rate separation quartiles (by J), ties split proportionally.
One named tuple per quartile, lowest first."""
function separation_quartiles(subj)
    W = quantile_weights(subj.J)
    return [(; n = sum(W[:, q]),
               H = wmean(subj.H, W[:, q]),
               F = wmean(subj.F, W[:, q]),
               J = wmean(subj.J, W[:, q]),
               boundary_pct = 100 * wmean(subj.at_boundary, W[:, q]),
               ppb = wmean(subj.ppb_val, W[:, q]),
               c_ll = wmean(subj.c_ll, W[:, q]))
            for q in 1:4]
end

# Kept for scripts written before the primary response was made explicit.
accuracy_quartiles(subj) = separation_quartiles(subj)
