# colloff_wixted.jl
# ─────────────────────────────────────────────────────────────────────────────
# Loader and showup contrast for Experiments 1-3 of Colloff and Wixted (2020),
# used by verify_numbers.jl. Each included participant made one target-present
# and one target-absent decision in a single identification condition, so H_i
# and F_i are 0 or 1 and c is undefined for every participant.
#
# Response: a "Present" answer is positive. For showup and simultaneous-showup
# conditions that is "yes" to "Is the highlighted suspect the perpetrator?"; in
# the standard-lineup condition of Experiment 3 it is any face choice, filler
# included, so the control condition is not used here.
#
# include() this file after colloff.jl (it uses criterion_c, half_n_correct and
# loglinear_correct); it defines functions only.
# ─────────────────────────────────────────────────────────────────────────────

using CSV, DataFrames, Distributions, Statistics

const CW_DIR = joinpath(@__DIR__, "..", "data", "pyWitness", "data", "published", "2020_Colloff_Wixted")

"""One row per included participant: condition, H and F (each 0 or 1), PPB and J.

Participant identifiers are not unique within an experiment (`subjectNo` repeats
in Experiment 1), so a participant is identified by `subjectNo`/`subjectID`
together with the random `verification` code. Each participant must contribute
exactly one target-present and one target-absent trial."""
function load_colloff_wixted(exp::Int)
    exp in 1:3 || throw(ArgumentError("exp must be 1, 2 or 3"))
    d = CSV.read(joinpath(CW_DIR, "Colloff_Wixted_Exp$exp.csv"), DataFrame; normalizenames = true)
    idcol = hasproperty(d, :subjectID) ? :subjectID : :subjectNo
    vcol = Symbol(only(filter(n -> startswith(n, "verification"), names(d))))
    trim(x) = strip(string(coalesce(x, "")))
    d = d[trim.(d.include) .== "yes", :]
    all(trim.(d.SaidAbsentOrPresent) .∈ Ref(("Present", "Absent"))) || error("Unknown response code")
    all(trim.(d.targetLabel) .∈ Ref(("present", "absent"))) || error("Unknown target status")
    d.pos = trim.(d.SaidAbsentOrPresent) .== "Present"
    d.tp = trim.(d.targetLabel) .== "present"
    p = combine(groupby(d, [idcol, vcol, :treatmentLabel]),
                nrow => :n, [:pos, :tp] => ((x, t) -> sum(x[t])) => :h,
                [:pos, :tp] => ((x, t) -> sum(x[.!t])) => :f,
                :tp => sum => :n_tp)
    all(p.n .== 2) && all(p.n_tp .== 1) ||
        error("Expected one target-present and one target-absent trial per participant")
    p.condition = String.(strip.(p.treatmentLabel))
    p.ppb_val = Float64.(p.h .+ p.f)
    p.J = Float64.(p.h .- p.f)
    p.at_boundary = trues(nrow(p))      # H and F are each 0 or 1 by construction
    return select(p, :condition, :h, :f, :ppb_val, :J, :at_boundary)
end

"""Welch difference (a minus b) with a two-sided 95% interval and p value."""
function welch_diff(a, b; level = 0.95)
    va, vb = var(a) / length(a), var(b) / length(b)
    dm = mean(a) - mean(b)
    se = sqrt(va + vb)
    df = (va + vb)^2 / (va^2 / (length(a) - 1) + vb^2 / (length(b) - 1))
    m = quantile(TDist(df), (1 + level) / 2) * se
    return (; diff = dm, lo = dm - m, hi = dm + m, p = 2 * ccdf(TDist(df), abs(dm / se)),
              n_a = length(a), n_b = length(b))
end

"""Showup minus simultaneous showup for PPB, J, and log-linear c, computed on
participant-level values (so the two trials of a participant are not treated as
independent). Also returns the numbers of participants for whom c is defined
without correction and under the 1/2N correction (a constant)."""
function showup_contrast(exp::Int)
    p = load_colloff_wixted(exp)
    s = p[p.condition .== "showup", :]
    q = p[p.condition .== "simultaneousShowup", :]
    cl(d) = [criterion_c(loglinear_correct(h, 1, f, 1)...) for (h, f) in zip(d.h, d.f)]
    chn(d) = [criterion_c(half_n_correct(h, 1, f, 1)...) for (h, f) in zip(d.h, d.f)]
    return (; n = nrow(s) + nrow(q), n_showup = nrow(s), n_simul = nrow(q),
            boundary_pct = 100 * mean(p.at_boundary),
            ppb = welch_diff(s.ppb_val, q.ppb_val),
            J = welch_diff(s.J, q.J),
            c_ll = welch_diff(cl(s), cl(q)),
            c_hn_values = unique(round.(vcat(chn(s), chn(q)); digits = 12)))
end

"""Participants, mean H and F, PPB and J for each condition of an experiment,
in the order showup, simultaneous showup, standard lineup (Experiment 3 only)."""
function condition_summary(exp::Int)
    p = load_colloff_wixted(exp)
    order = ["showup", "simultaneousShowup", "control"]
    return [(; condition = c, n = nrow(d), H = mean(d.h), F = mean(d.f),
             ppb = mean(d.ppb_val), J = mean(d.J))
            for c in order for d in (p[p.condition .== c, :],) if nrow(d) > 0]
end
