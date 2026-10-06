# inference.jl
# Finite-binomial intervals for PPB and exact coverage by count enumeration.
# Run: julia +1.12 --project=code code/inference.jl
# Including this file defines functions only; the script checks the calculations
# and writes tables/tbl_interval_coverage.md when run directly.
#
# These are intervals for fixed binomial probabilities, not a replacement for
# participant-level inference with heterogeneity, pairing, or clustered trials.

using Distributions
using Printf
using Statistics

function _check_alpha(alpha::Real)
    isfinite(alpha) && 0 < alpha < 1 ||
        throw(ArgumentError("alpha must be finite and strictly between zero and one"))
    return nothing
end

function _check_counts(x::Integer, n::Integer)
    n > 0 || throw(ArgumentError("trial denominator must be positive"))
    0 <= x <= n || throw(ArgumentError("count must lie between zero and its denominator"))
    return nothing
end

"""
    clopper_pearson_interval(x, n; alpha=0.05)

Equal-tailed Clopper–Pearson interval for a binomial probability, with
coverage at least `1-alpha`. The returned named tuple has `lower` and `upper`
fields. Boundary endpoints are defined explicitly, avoiding invalid beta
shape parameters. Here `alpha` is the component interval's total error rate.
"""
function clopper_pearson_interval(x::Integer, n::Integer; alpha::Real=0.05)
    _check_counts(x, n)
    _check_alpha(alpha)
    lower = x == 0 ? 0.0 : quantile(Beta(x, n - x + 1), alpha / 2)
    upper = x == n ? 1.0 : cquantile(Beta(x + 1, n - x), alpha / 2)
    return (; lower, upper)
end

"""
    ppb_interval(hits, ns, false_alarms, nn; alpha=0.05)

Conservative finite-binomial interval for `H + F`. Construct one
Clopper–Pearson interval at confidence `1-alpha/2` for each probability and
add their lower and upper endpoints. The Bonferroni union bound gives coverage
at least `1-alpha` whenever each component has a valid binomial sampling
model; independence between the two counts is not required for this bound.

This interval leaves the raw point estimate `hits/ns + false_alarms/nn`
unchanged. Its endpoints remain in `[0,2]`, including at observed boundaries.
Repeated or heterogeneous responses need an appropriate sampling model;
arbitrary pooled participant counts need not have binomial margins.
"""
function ppb_interval(hits::Integer, ns::Integer, false_alarms::Integer,
                      nn::Integer; alpha::Real=0.05)
    _check_alpha(alpha)
    h = clopper_pearson_interval(hits, ns; alpha=alpha / 2)
    f = clopper_pearson_interval(false_alarms, nn; alpha=alpha / 2)
    return (lower=h.lower + f.lower, upper=h.upper + f.upper)
end

"""
    wald_interval(hits, ns, false_alarms, nn; alpha=0.05)

Untruncated large-sample Wald interval for `H + F`, using the independent
binomial plug-in variance. This is the manuscript's comparison procedure,
not the recommended procedure for sparse binomial counts. Endpoints may lie
outside `[0,2]`, and two observed boundary rates give a point interval.
"""
function wald_interval(hits::Integer, ns::Integer, false_alarms::Integer,
                       nn::Integer; alpha::Real=0.05)
    _check_counts(hits, ns)
    _check_counts(false_alarms, nn)
    _check_alpha(alpha)
    h, f = hits / ns, false_alarms / nn
    half_width = cquantile(Normal(), alpha / 2) *
                 sqrt(h * (1 - h) / ns + f * (1 - f) / nn)
    return (lower=h + f - half_width, upper=h + f + half_width)
end

"""
    adjusted_ppb_interval(hits, ns, false_alarms, nn; alpha=0.05)

Less conservative interval for `H + F`. Because `H + F = H - (1 - F) + 1`,
it is Agresti and Caffo's add-one-success-and-one-failure interval for the
difference `H - (1 - F)` of two independent proportions, shifted by one.
Each rate is estimated as `(x + 1)/(n + 2)` and the interval is the Wald
interval around the sum of those adjusted rates, with the same adjusted
variance, truncated to `[0,2]`. Truncation cannot lower coverage because
`H + F` lies in `[0,2]`.

This interval has no coverage guarantee. Its coverage is examined by exact
enumeration (`interval_coverage`). It is centered on the adjusted estimate,
not on the raw `hits/ns + false_alarms/nn`, and is never a point interval.
"""
function adjusted_ppb_interval(hits::Integer, ns::Integer, false_alarms::Integer,
                               nn::Integer; alpha::Real=0.05)
    _check_counts(hits, ns)
    _check_counts(false_alarms, nn)
    _check_alpha(alpha)
    h, f = (hits + 1) / (ns + 2), (false_alarms + 1) / (nn + 2)
    half_width = cquantile(Normal(), alpha / 2) *
                 sqrt(h * (1 - h) / (ns + 2) + f * (1 - f) / (nn + 2))
    return (lower=max(0.0, h + f - half_width), upper=min(2.0, h + f + half_width))
end

"""
    ppb_difference_interval(hits1, ns1, false_alarms1, nn1,
                            hits2, ns2, false_alarms2, nn2; alpha=0.05)

Conservative interval for `(H1 + F1) - (H2 + F2)`, suitable for comparing
two independent samples when their four component counts have valid
binomial sampling models. Each Clopper–Pearson component interval has
confidence `1-alpha/4`. The difference interval is
`[Lh1 + Lf1 - Uh2 - Uf2, Uh1 + Uf1 - Lh2 - Lf2]` and has coverage at least
`1-alpha` by the union bound. The bound itself does not require independence
among components. It does require validity of all four marginal binomial
models; it does not justify treating paired or clustered participants as
independent binomial trials.
"""
function ppb_difference_interval(hits1::Integer, ns1::Integer,
                                 false_alarms1::Integer, nn1::Integer,
                                 hits2::Integer, ns2::Integer,
                                 false_alarms2::Integer, nn2::Integer;
                                 alpha::Real=0.05)
    _check_alpha(alpha)
    h1 = clopper_pearson_interval(hits1, ns1; alpha=alpha / 4)
    f1 = clopper_pearson_interval(false_alarms1, nn1; alpha=alpha / 4)
    h2 = clopper_pearson_interval(hits2, ns2; alpha=alpha / 4)
    f2 = clopper_pearson_interval(false_alarms2, nn2; alpha=alpha / 4)
    return (lower=h1.lower + f1.lower - h2.upper - f2.upper,
            upper=h1.upper + f1.upper - h2.lower - f2.lower)
end

# The tolerance only absorbs floating-point error at mathematically equal
# endpoints; it is far smaller than the reported precision of any result.
_interval_contains(ci, target) = ci.lower - 1e-12 <= target <= ci.upper + 1e-12

function _interval_tables(ns::Integer, nn::Integer; alpha::Real=0.05)
    _check_counts(0, ns)
    _check_counts(0, nn)
    _check_alpha(alpha)
    h_intervals = [clopper_pearson_interval(x, ns; alpha=alpha / 2) for x in 0:ns]
    f_intervals = [clopper_pearson_interval(y, nn; alpha=alpha / 2) for y in 0:nn]
    cp = [(lower=h.lower + f.lower, upper=h.upper + f.upper)
          for h in h_intervals, f in f_intervals]
    wald = [wald_interval(x, ns, y, nn; alpha) for x in 0:ns, y in 0:nn]
    adj = [adjusted_ppb_interval(x, ns, y, nn; alpha) for x in 0:ns, y in 0:nn]
    return (; cp, wald, adj)
end

function _interval_coverage(H::Real, F::Real, ns::Integer, nn::Integer, intervals)
    isfinite(H) && 0 <= H <= 1 || throw(ArgumentError("H must lie in [0,1]"))
    isfinite(F) && 0 <= F <= 1 || throw(ArgumentError("F must lie in [0,1]"))
    ph = pdf.(Binomial(ns, H), 0:ns)
    pf = pdf.(Binomial(nn, F), 0:nn)
    target = H + F
    cp_coverage = wald_coverage = adj_coverage = 0.0
    cp_width = wald_width = adj_width = mass = zero_variance = 0.0
    for x in 0:ns, y in 0:nn
        probability = ph[x + 1] * pf[y + 1]
        cp, wald, adj = intervals.cp[x + 1, y + 1], intervals.wald[x + 1, y + 1],
                        intervals.adj[x + 1, y + 1]
        cp_coverage += probability * _interval_contains(cp, target)
        wald_coverage += probability * _interval_contains(wald, target)
        adj_coverage += probability * _interval_contains(adj, target)
        cp_width += probability * (cp.upper - cp.lower)
        wald_width += probability * (wald.upper - wald.lower)
        adj_width += probability * (adj.upper - adj.lower)
        mass += probability
        zero_variance += probability * ((x == 0 || x == ns) && (y == 0 || y == nn))
    end
    return (; H, F, ns, nn, cp_coverage, wald_coverage, adj_coverage,
            cp_width, wald_width, adj_width, zero_variance, mass)
end

"""
    interval_coverage(H, F, ns, nn; alpha=0.05)

Enumerate every pair of independent binomial counts to obtain exact
repeated-sampling coverage and expected width for `ppb_interval`, the
untruncated `wald_interval`, and `adjusted_ppb_interval`. No Monte Carlo
sampling is used. Results include `cp_coverage`, `wald_coverage`,
`adj_coverage`, the matching `*_width` fields, `zero_variance` (the
probability of two boundary rates), and total probability `mass`.

Independence is an assumption of this enumeration, even though the
Bonferroni coverage guarantee only needs valid component intervals.
"""
function interval_coverage(H::Real, F::Real, ns::Integer, nn::Integer; alpha::Real=0.05)
    intervals = _interval_tables(ns, nn; alpha)
    return _interval_coverage(H, F, ns, nn, intervals)
end

"""
    coverage_grid(; trial_counts=(4,10,20),
                    probabilities=(.01,.05,.1,.2,.5,.8,.9,.95,.99), alpha=.05)

Exact independent-binomial coverage and expected widths at every combination
of signal denominator, noise denominator, H, and F. The default grid has
729 configurations, including unequal denominators. Intervals are computed
once per denominator pair, then reused across operating points.
"""
function coverage_grid(; trial_counts=(4, 10, 20),
                        probabilities=(0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99),
                        alpha::Real=0.05)
    rows = NamedTuple[]
    for ns in trial_counts, nn in trial_counts
        intervals = _interval_tables(ns, nn; alpha)
        for H in probabilities, F in probabilities
            push!(rows, _interval_coverage(H, F, ns, nn, intervals))
        end
    end
    return rows
end

"""Minimum and mean coverage, share of configurations below `threshold`, and
mean expected width for each interval over a grid from `coverage_grid`."""
function grid_summary(grid=coverage_grid(); threshold::Real=0.94)
    summarize(cov, wid) = (minimum = minimum(getproperty.(grid, cov)),
                           mean = mean(getproperty.(grid, cov)),
                           below = mean(getproperty.(grid, cov) .< threshold),
                           width = mean(getproperty.(grid, wid)))
    return (cp = summarize(:cp_coverage, :cp_width),
            adj = summarize(:adj_coverage, :adj_width),
            wald = summarize(:wald_coverage, :wald_width))
end

"""Featured sparse configurations, followed by two unequal-denominator cases."""
function featured_coverage(; alpha::Real=0.05)
    configurations = ((0.80, 0.05, 10, 10), (0.95, 0.20, 10, 10),
                      (0.95, 0.05, 10, 10), (0.95, 0.10, 20, 20),
                      (0.50, 0.50, 4, 4), (0.80, 0.05, 4, 20),
                      (0.95, 0.20, 20, 4))
    return [interval_coverage(H, F, ns, nn; alpha) for (H, F, ns, nn) in configurations]
end

"""Generated pipe table, without a caption, for inclusion in the supplement."""
function interval_coverage_table(; alpha::Real=0.05)
    io = IOBuffer()
    println(io, "| \$H\$ | \$F\$ | \$n_s\$ | \$n_n\$ | Normal coverage | CP coverage | Adjusted coverage | Normal width | CP width | Adjusted width |")
    println(io, "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in featured_coverage(; alpha)
        @printf(io, "| %.2f | %.2f | %d | %d | %.2f%% | %.2f%% | %.2f%% | %.3f | %.3f | %.3f |\n",
                r.H, r.F, r.ns, r.nn, 100r.wald_coverage, 100r.cp_coverage,
                100r.adj_coverage, r.wald_width, r.cp_width, r.adj_width)
    end
    return String(take!(io))
end

"""
    check_inference(grid=coverage_grid())

Check boundary formulas, invalid counts, label symmetries, selected exact
Wald benchmarks, and the nominal coverage guarantee on the entire grid.
Also enumerate a small four-component contrast as a check on the signs
and Bonferroni allocation in `ppb_difference_interval`.
"""
function check_inference(grid=coverage_grid())
    alpha = 0.05
    component = clopper_pearson_interval(0, 4; alpha=alpha / 2)
    @assert component.lower == 0
    @assert isapprox(component.upper, 1 - (alpha / 4)^(1 / 4); atol=1e-13)
    empty = ppb_interval(0, 4, 0, 4)
    full = ppb_interval(4, 4, 4, 4)
    @assert empty.lower == 0 < empty.upper < 2
    @assert 0 < full.lower < full.upper == 2
    @assert isapprox(empty.upper, 2 - full.lower; atol=1e-13)
    @assert wald_interval(4, 4, 0, 4) == (lower=1.0, upper=1.0)

    # The adjusted interval is defined at every boundary, stays in [0,2], is never
    # a point interval, contains the raw estimate for every outcome at the grid's
    # trial counts, and has the same exchange and complement symmetries.
    for ns in (4, 10, 20), nn in (4, 10, 20), x in 0:ns, y in 0:nn
        a = adjusted_ppb_interval(x, ns, y, nn)
        @assert 0 <= a.lower < a.upper <= 2
        @assert a.lower - 1e-12 <= x / ns + y / nn <= a.upper + 1e-12
    end
    ai = adjusted_ppb_interval(3, 4, 2, 10)
    @assert ai == adjusted_ppb_interval(2, 10, 3, 4)
    ac = adjusted_ppb_interval(1, 4, 8, 10)
    @assert isapprox(ai.lower, 2 - ac.upper; atol=1e-13)
    @assert isapprox(ai.upper, 2 - ac.lower; atol=1e-13)
    for args in ((-1, 4, 0, 4), (5, 4, 0, 4), (0, 0, 0, 4))
        caught = false
        try
            adjusted_ppb_interval(args...)
        catch err
            caught = err isa ArgumentError
        end
        @assert caught
    end

    # Exchanging classes leaves the sum interval unchanged. Complementing
    # every response reflects it about one; unequal denominators are allowed.
    ci = ppb_interval(3, 4, 2, 10)
    swapped = ppb_interval(2, 10, 3, 4)
    complemented = ppb_interval(1, 4, 8, 10)
    @assert ci == swapped
    @assert isapprox(ci.lower, 2 - complemented.upper; atol=1e-13)
    @assert isapprox(ci.upper, 2 - complemented.lower; atol=1e-13)
    for args in ((-1, 4, 0, 4), (5, 4, 0, 4), (0, 0, 0, 4))
        caught = false
        try
            ppb_interval(args...)
        catch err
            caught = err isa ArgumentError
        end
        @assert caught
    end

    # These independently derived benchmarks were recorded in the review.
    @assert isapprox(interval_coverage(0.80, 0.05, 10, 10).wald_coverage,
                     0.869605; atol=5e-7)
    @assert isapprox(interval_coverage(0.95, 0.10, 20, 20).wald_coverage,
                     0.857643; atol=5e-7)
    @assert isapprox(interval_coverage(0.50, 0.50, 4, 4).wald_coverage,
                     0.8359375; atol=1e-13)
    @assert !isempty(grid)
    @assert all(r -> isapprox(r.mass, 1; atol=1e-12), grid)
    @assert all(r -> 1 - alpha - 1e-12 <= r.cp_coverage <= 1 + 1e-12, grid)
    @assert all(r -> 0 < r.cp_width <= 2 && 0 <= r.wald_width && 0 < r.adj_width <= 2, grid)
    @assert all(r -> 0 <= r.adj_coverage <= 1 + 1e-12, grid)
    for H in (0.0, 1.0), F in (0.0, 1.0)
        r = interval_coverage(H, F, 4, 10)
        @assert isapprox(r.cp_coverage, 1; atol=1e-13)
        @assert isapprox(r.wald_coverage, 1; atol=1e-13)
        @assert isapprox(r.adj_coverage, 1; atol=1e-13)
        @assert isapprox(r.mass, 1; atol=1e-13)
        @assert isapprox(r.zero_variance, 1; atol=1e-13)
    end

    ns1, nn1, ns2, nn2 = 2, 3, 2, 3
    H1, F1, H2, F2 = 0.8, 0.1, 0.5, 0.2
    target = H1 + F1 - H2 - F2
    contrast_coverage = 0.0
    distributions = (Binomial(ns1, H1), Binomial(nn1, F1),
                     Binomial(ns2, H2), Binomial(nn2, F2))
    for x1 in 0:ns1, y1 in 0:nn1, x2 in 0:ns2, y2 in 0:nn2
        d = ppb_difference_interval(x1, ns1, y1, nn1, x2, ns2, y2, nn2)
        reversed = ppb_difference_interval(x2, ns2, y2, nn2, x1, ns1, y1, nn1)
        @assert -2 <= d.lower < d.upper <= 2
        @assert isapprox(d.lower, -reversed.upper; atol=1e-13)
        @assert isapprox(d.upper, -reversed.lower; atol=1e-13)
        probability = prod(pdf(dist, x) for (dist, x) in
                           zip(distributions, (x1, y1, x2, y2)))
        contrast_coverage += probability * _interval_contains(d, target)
    end
    @assert contrast_coverage >= 1 - alpha - 1e-12
    return true
end

function inference_main()
    grid = coverage_grid()
    check_inference(grid)
    println("All inference checks passed; $(length(grid)) grid configurations enumerated.")
    for field in (:cp_coverage, :adj_coverage, :wald_coverage)
        row = grid[argmin([getproperty(r, field) for r in grid])]
        @printf("Minimum %s = %.9f at H=%.2f, F=%.2f, ns=%d, nn=%d\n",
                string(field), getproperty(row, field), row.H, row.F, row.ns, row.nn)
    end
    path = joinpath(@__DIR__, "..", "tables", "tbl_interval_coverage.md")
    mkpath(dirname(path))
    write(path, interval_coverage_table())
    println("Wrote: $path")
    print(interval_coverage_table())
end

if abspath(PROGRAM_FILE) == @__FILE__
    inference_main()
end
