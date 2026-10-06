# simulation.jl
# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Simulation study comparing PPB, c, and B'' under varying conditions
#
# Verifies analytical predictions from Phase 1 (boundary stability,
# aggregation, regression decomposability) and maps the parameter space
# where PPB outperforms c.
#
# Requirements (already in Project.toml):
#   ] add CairoMakie Distributions ColorSchemes
# ─────────────────────────────────────────────────────────────────────────────

using Distributions
using CairoMakie
using ColorSchemes
using Random
using Statistics

# ─── Core functions ──────────────────────────────────────────────────────────

const ϕ_dist = Normal()
Φinv(p) = quantile(ϕ_dist, p)
Φ(x) = cdf(ϕ_dist, x)
ϕpdf(x) = pdf(ϕ_dist, x)

"""Compute PPB from hit rate and false alarm rate."""
ppb(H, F) = H + F

"""Compute Youden's J from hit rate and false alarm rate."""
youden_j(H, F) = H - F

"""Compute criterion c from hit rate and false alarm rate (probit).
Returns NaN if H or F is 0 or 1."""
function criterion_c(H, F)
    (H <= 0 || H >= 1 || F <= 0 || F >= 1) && return NaN
    return -0.5 * (Φinv(H) + Φinv(F))
end

"""Compute logistic criterion c_L from hit rate and false alarm rate."""
function criterion_c_logistic(H, F)
    (H <= 0 || H >= 1 || F <= 0 || F >= 1) && return NaN
    return -0.5 * (log(H / (1 - H)) + log(F / (1 - F)))
end

"""Compute B'' (Grier 1971) from hit rate and false alarm rate."""
function b_double_prime(H, F)
    (H == F) && return 0.0
    return sign(H - F) * (H * (1 - H) - F * (1 - F)) / (H * (1 - H) + F * (1 - F))
end

"""Compute B_r (Snodgrass & Corwin 1988, Eq. 8) from hit rate and false alarm rate.
B_r = F / (1 - P_r) = F / (1 - H + F), the probability of saying yes from the
uncertain state. Undefined (0/0) exactly at H = 1, F = 0."""
function b_r(H, F)
    denom = 1 - H + F
    denom == 0 && return NaN
    return F / denom
end

"""Apply log-linear correction (Hautus 1995).
Adds 0.5 to hits/misses and FA/CR counts."""
function loglinear_correct(hits, misses, fa, cr)
    H = (hits + 0.5) / (hits + misses + 1)
    F = (fa + 0.5) / (fa + cr + 1)
    return (H, F)
end

"""Apply 1/2N correction (Macmillan & Kaplan 1985).
Only adjusts extreme rates."""
function half_n_correct(hits, n_signal, fa, n_noise)
    H = hits / n_signal
    F = fa / n_noise
    if H == 0.0
        H = 1 / (2 * n_signal)
    elseif H == 1.0
        H = 1 - 1 / (2 * n_signal)
    end
    if F == 0.0
        F = 1 / (2 * n_noise)
    elseif F == 1.0
        F = 1 - 1 / (2 * n_noise)
    end
    return (H, F)
end

"""Generate observed (hits, fa) for one subject given true rates and trial counts."""
function simulate_subject(H_true, F_true, n_signal, n_noise; rng=Random.default_rng())
    hits = rand(rng, Binomial(n_signal, H_true))
    fa = rand(rng, Binomial(n_noise, F_true))
    return (hits, n_signal - hits, fa, n_noise - fa)  # hits, misses, fa, cr
end


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION 1: Boundary stability
# Verifies Phase 1.1 predictions
# ═══════════════════════════════════════════════════════════════════════════════

function sim_boundary_stability(;
    H_true_vals = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
    F_true_vals = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50],
    n_trials_vals = [10, 20, 40, 80, 160],
    n_subjects = 5000,
    seed = 42
)
    rng = MersenneTwister(seed)

    results = Dict{String, Vector{Float64}}()
    results["H_true"] = Float64[]
    results["F_true"] = Float64[]
    results["n_trials"] = Float64[]
    results["pct_boundary"] = Float64[]          # % subjects hitting boundary (Monte Carlo)
    results["pct_boundary_exact"] = Float64[]    # % hitting boundary (closed form; no MC error)
    results["ppb_bias"] = Float64[]              # E[PPB_obs] - PPB_true
    results["ppb_rmse"] = Float64[]
    results["c_raw_pct_nan"] = Float64[]         # % subjects where c is NaN
    results["c_loglin_bias"] = Float64[]         # E[c_corrected] - c_true
    results["c_loglin_rmse"] = Float64[]
    results["c_halfn_bias"] = Float64[]
    results["c_halfn_rmse"] = Float64[]

    for H_true in H_true_vals, F_true in F_true_vals, n_trials in n_trials_vals
        n_signal = n_trials ÷ 2
        n_noise = n_trials - n_signal

        ppb_true = ppb(H_true, F_true)
        c_true = criterion_c(H_true, F_true)

        ppb_obs = Float64[]
        c_raw = Float64[]
        c_loglin = Float64[]
        c_halfn = Float64[]
        n_boundary = 0

        for _ in 1:n_subjects
            hits, misses, fa, cr = simulate_subject(H_true, F_true, n_signal, n_noise; rng)
            H_obs = hits / n_signal
            F_obs = fa / n_noise

            # Track boundary hits
            if H_obs == 0.0 || H_obs == 1.0 || F_obs == 0.0 || F_obs == 1.0
                n_boundary += 1
            end

            # PPB: always defined
            push!(ppb_obs, ppb(H_obs, F_obs))

            # c raw: may be NaN
            push!(c_raw, criterion_c(H_obs, F_obs))

            # c with log-linear correction
            H_ll, F_ll = loglinear_correct(hits, misses, fa, cr)
            push!(c_loglin, criterion_c(H_ll, F_ll))

            # c with 1/2N correction
            H_hn, F_hn = half_n_correct(hits, n_signal, fa, n_noise)
            push!(c_halfn, criterion_c(H_hn, F_hn))
        end

        push!(results["H_true"], H_true)
        push!(results["F_true"], F_true)
        push!(results["n_trials"], Float64(n_trials))
        push!(results["pct_boundary"], 100 * n_boundary / n_subjects)

        # Exact boundary probability. P(no boundary) = P(0 < hits < n_signal) * P(0 < fa < n_noise).
        # Reported in preference to the Monte Carlo estimate, which jitters by ~1 percentage
        # point with the RNG position and has caused figure/text drift.
        p_int_H = 1 - pdf(Binomial(n_signal, H_true), n_signal) - pdf(Binomial(n_signal, H_true), 0)
        p_int_F = 1 - pdf(Binomial(n_noise, F_true), n_noise) - pdf(Binomial(n_noise, F_true), 0)
        push!(results["pct_boundary_exact"], 100 * (1 - p_int_H * p_int_F))

        push!(results["ppb_bias"], mean(ppb_obs) - ppb_true)
        push!(results["ppb_rmse"], sqrt(mean((ppb_obs .- ppb_true) .^ 2)))

        push!(results["c_raw_pct_nan"], 100 * count(isnan, c_raw) / n_subjects)

        # For corrected c, compute bias and RMSE (should have no NaNs)
        push!(results["c_loglin_bias"], mean(c_loglin) - c_true)
        push!(results["c_loglin_rmse"], sqrt(mean((c_loglin .- c_true) .^ 2)))

        push!(results["c_halfn_bias"], mean(c_halfn) - c_true)
        push!(results["c_halfn_rmse"], sqrt(mean((c_halfn .- c_true) .^ 2)))
    end

    return results
end


# ─── Canonical Panel A configuration set ─────────────────────────────────────
# figures.jl (Panel A) and verify_numbers.jl both go through boundary_panel()
# so the figure and the paper's reported numbers cannot drift apart.
#
# Panel A reports exact expected values, computed by enumerating every binomial
# outcome, not a Monte Carlo draw: a single draw of 5,000 subjects put the 1/2N
# bias at 0.127 against an exact 0.123, and the text had reported the draw.

const PANEL_A_CONFIGS = [(0.80, 0.05, 20), (0.95, 0.20, 20), (0.95, 0.05, 20), (0.95, 0.10, 40)]

"""Exact expected bias of PPB and of corrected c for one observer with true rates
(H, F) and n total trials split equally between signal and noise, by enumerating
every (hits, false alarms) outcome. Also returns the exact probability of a
boundary estimate and the bias of uncorrected c among interior outcomes only
(the estimand of excluding boundary subjects before averaging)."""
function exact_boundary_bias(H, F, n)
    ns = n ÷ 2
    nn = n - ns
    c_true = criterion_c(H, F)
    e_ppb = e_ll = e_hn = 0.0
    p_int = e_int = 0.0
    for h in 0:ns, f in 0:nn
        p = pdf(Binomial(ns, H), h) * pdf(Binomial(nn, F), f)
        e_ppb += p * ppb(h / ns, f / nn)
        H_ll, F_ll = loglinear_correct(h, ns - h, f, nn - f)
        e_ll += p * criterion_c(H_ll, F_ll)
        H_hn, F_hn = half_n_correct(h, ns, f, nn)
        e_hn += p * criterion_c(H_hn, F_hn)
        if 0 < h < ns && 0 < f < nn
            p_int += p
            e_int += p * criterion_c(h / ns, f / nn)
        end
    end
    return (; H, F, n, c_true,
            ppb_bias = e_ppb - ppb(H, F),
            c_ll_bias = e_ll - c_true,
            c_hn_bias = e_hn - c_true,
            c_excl_bias = e_int / p_int - c_true,
            pct_boundary = 100 * (1 - p_int))
end

"""Exact Panel A results, one row per configuration in PANEL_A_CONFIGS order."""
boundary_panel() = [exact_boundary_bias(H, F, n) for (H, F, n) in PANEL_A_CONFIGS]


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION 2: Aggregation discrepancy
# Verifies Phase 1.2 predictions
# ═══════════════════════════════════════════════════════════════════════════════

function sim_aggregation(;
    # Each scenario: (mean_H, mean_F, sd_H, sd_F, label)
    scenarios = [
        (0.70, 0.30, 0.08, 0.08, "Moderate, symmetric"),
        (0.90, 0.15, 0.08, 0.06, "High H, low F (typical applied)"),
        (0.95, 0.05, 0.04, 0.03, "Very extreme (Honduras-like)"),
        (0.80, 0.20, 0.15, 0.10, "High variance"),
        (0.60, 0.40, 0.05, 0.05, "Near chance, low variance"),
    ],
    K_vals = [20, 50, 100, 500],  # number of subjects to average
    n_replications = 2000,
    seed = 123
)
    rng = MersenneTwister(seed)

    results = Dict{String, Vector}()
    results["scenario"] = String[]
    results["K"] = Int[]
    results["mean_H"] = Float64[]
    results["mean_F"] = Float64[]
    results["ppb_discrepancy"] = Float64[]       # should be ~0
    results["c_discrepancy_mean"] = Float64[]     # mean of (mean(c_i) - c(mean_H, mean_F))
    results["c_discrepancy_sd"] = Float64[]
    results["br_discrepancy_mean"] = Float64[]    # mean of (mean(Br_i) - Br(mean_H, mean_F))
    results["br_discrepancy_sd"] = Float64[]
    results["br_undefined"] = Int[]               # count of individual Br that were NaN
    results["c_discrepancy_can_flip"] = Bool[]    # does the discrepancy ever flip sign?
    results["analytical_prediction"] = Float64[]  # second-order approx from Phase 1.2

    for (μH, μF, σH, σF, label) in scenarios
        # Beta distribution parameters from mean and sd
        # Beta(α, β) has mean α/(α+β) and var αβ/((α+β)²(α+β+1))
        αH = μH * (μH * (1 - μH) / σH^2 - 1)
        βH = (1 - μH) * (μH * (1 - μH) / σH^2 - 1)
        αF = μF * (μF * (1 - μF) / σF^2 - 1)
        βF = (1 - μF) * (μF * (1 - μF) / σF^2 - 1)

        # Check valid Beta parameters
        if αH <= 0 || βH <= 0 || αF <= 0 || βF <= 0
            @warn "Invalid Beta parameters for scenario: $label, skipping"
            continue
        end

        dist_H = Beta(αH, βH)
        dist_F = Beta(αF, βF)

        # Analytical prediction (second-order approximation)
        z2H = isfinite(Φinv(μH)) ? Φinv(μH) / ϕpdf(Φinv(μH))^2 : NaN
        z2F = isfinite(Φinv(μF)) ? Φinv(μF) / ϕpdf(Φinv(μF))^2 : NaN
        analytical = -0.25 * (z2H * σH^2 + z2F * σF^2)

        for K in K_vals
            ppb_discreps = Float64[]
            c_discreps = Float64[]
            br_discreps = Float64[]
            n_br_undefined = 0

            for _ in 1:n_replications
                # Draw K subjects
                Hs = rand(rng, dist_H, K)
                Fs = rand(rng, dist_F, K)

                # Clamp to avoid exact 0/1 for c computation
                Hs = clamp.(Hs, 1e-6, 1 - 1e-6)
                Fs = clamp.(Fs, 1e-6, 1 - 1e-6)

                # PPB discrepancy
                ppb_avg = mean(ppb.(Hs, Fs))
                ppb_of_avg = ppb(mean(Hs), mean(Fs))
                push!(ppb_discreps, ppb_avg - ppb_of_avg)

                # c discrepancy
                c_avg = mean(criterion_c.(Hs, Fs))
                c_of_avg = criterion_c(mean(Hs), mean(Fs))
                push!(c_discreps, c_avg - c_of_avg)

                # Br discrepancy (same clamped rates used for c; see note in report)
                br_ind = b_r.(Hs, Fs)
                n_br_undefined += count(isnan, br_ind)
                br_avg = mean(br_ind)
                br_of_avg = b_r(mean(Hs), mean(Fs))
                push!(br_discreps, br_avg - br_of_avg)
            end

            push!(results["scenario"], label)
            push!(results["K"], K)
            push!(results["mean_H"], μH)
            push!(results["mean_F"], μF)
            push!(results["ppb_discrepancy"], mean(ppb_discreps))
            push!(results["c_discrepancy_mean"], mean(c_discreps))
            push!(results["c_discrepancy_sd"], std(c_discreps))
            push!(results["br_discrepancy_mean"], mean(br_discreps))
            push!(results["br_discrepancy_sd"], std(br_discreps))
            push!(results["br_undefined"], n_br_undefined)
            push!(results["analytical_prediction"], analytical)

            # Check if discrepancy could flip a group comparison
            # (sign of discrepancy vs sign of c itself)
            c_true = criterion_c(μH, μF)
            push!(results["c_discrepancy_can_flip"],
                  abs(mean(c_discreps)) > 0.5 * abs(c_true))
        end
    end

    return results
end


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION 3: Group comparison sign-flip
# Can aggregation distortion actually reverse which group appears more biased?
# ═══════════════════════════════════════════════════════════════════════════════

function sim_group_comparison(;
    n_replications = 5000,
    K = 100,  # subjects per group
    seed = 456
)
    rng = MersenneTwister(seed)

    # Two groups with similar TRUE bias (PPB) but different baseline accuracy
    # Group A: high accuracy, extreme rates → c distortion is large
    # Group B: moderate accuracy, central rates → c distortion is small

    scenarios = [
        # (label, μH_A, μF_A, σH_A, σF_A, μH_B, μF_B, σH_B, σF_B)
        ("Subtle bias diff, accuracy diff",
         0.92, 0.13, 0.06, 0.05,   # Group A: high accuracy, PPB=1.05
         0.72, 0.32, 0.08, 0.08),  # Group B: moderate accuracy, PPB=1.04
        ("Equal true PPB, accuracy diff",
         0.90, 0.15, 0.07, 0.06,   # Group A: PPB=1.05
         0.70, 0.35, 0.08, 0.08),  # Group B: PPB=1.05
        ("Small PPB diff, large accuracy diff",
         0.95, 0.08, 0.04, 0.04,   # Group A: PPB=1.03
         0.65, 0.40, 0.10, 0.10),  # Group B: PPB=1.05
    ]

    make_beta(μ, σ) = begin
        α = μ * (μ * (1 - μ) / σ^2 - 1)
        β = (1 - μ) * (μ * (1 - μ) / σ^2 - 1)
        (α > 0 && β > 0) || error("Invalid Beta params: μ=$μ, σ=$σ")
        Beta(α, β)
    end

    results = []

    for (label, μH_A, μF_A, σH_A, σF_A, μH_B, μF_B, σH_B, σF_B) in scenarios
        ppb_A_true = ppb(μH_A, μF_A)
        ppb_B_true = ppb(μH_B, μF_B)
        c_A_true = criterion_c(μH_A, μF_A)
        c_B_true = criterion_c(μH_B, μF_B)

        dHA = make_beta(μH_A, σH_A)
        dFA = make_beta(μF_A, σF_A)
        dHB = make_beta(μH_B, σH_B)
        dFB = make_beta(μF_B, σF_B)

        n_ppb_flip = 0
        n_c_flip = 0
        n_c_disagree_ppb = 0

        # Store per-replication diffs for scatter plot (Figure 3A)
        ppb_diffs = Float64[]
        c_diffs = Float64[]

        for _ in 1:n_replications
            HA = clamp.(rand(rng, dHA, K), 1e-6, 1-1e-6)
            FA = clamp.(rand(rng, dFA, K), 1e-6, 1-1e-6)
            HB = clamp.(rand(rng, dHB, K), 1e-6, 1-1e-6)
            FB = clamp.(rand(rng, dFB, K), 1e-6, 1-1e-6)

            ppb_A = mean(ppb.(HA, FA))
            ppb_B = mean(ppb.(HB, FB))
            c_A = mean(criterion_c.(HA, FA))
            c_B = mean(criterion_c.(HB, FB))

            push!(ppb_diffs, ppb_A - ppb_B)
            push!(c_diffs, c_A - c_B)

            # PPB sign flip (observed disagrees with true ordering)
            true_ppb_sign = sign(ppb_A_true - ppb_B_true)
            obs_ppb_sign = sign(ppb_A - ppb_B)
            if true_ppb_sign != 0 && obs_ppb_sign != true_ppb_sign
                n_ppb_flip += 1
            end

            # c sign flip (observed disagrees with true ordering)
            true_c_sign = sign(c_A_true - c_B_true)
            obs_c_sign = sign(c_A - c_B)
            if true_c_sign != 0 && obs_c_sign != true_c_sign
                n_c_flip += 1
            end

            # PPB and c disagree on which group is more liberal
            # (more liberal = higher PPB = more negative c)
            if sign(ppb_A - ppb_B) != -sign(c_A - c_B) && sign(ppb_A - ppb_B) != 0
                n_c_disagree_ppb += 1
            end
        end

        push!(results, (;
            label, K, n_replications,
            ppb_A_true, ppb_B_true, c_A_true, c_B_true,
            ppb_flip_rate = n_ppb_flip / n_replications,
            c_flip_rate = n_c_flip / n_replications,
            disagree_rate = n_c_disagree_ppb / n_replications,
            ppb_diffs, c_diffs,
        ))
    end

    return results
end

function report_sim3(results)
    for r in results
        println("\n── $(r.label) ──")
        println("  True PPB:  A=$(round(r.ppb_A_true, digits=3)), B=$(round(r.ppb_B_true, digits=3)), diff=$(round(r.ppb_A_true - r.ppb_B_true, digits=3))")
        println("  True c:    A=$(round(r.c_A_true, digits=3)), B=$(round(r.c_B_true, digits=3)), diff=$(round(r.c_A_true - r.c_B_true, digits=3))")
        println("  PPB sign-flip rate:  $(round(100 * r.ppb_flip_rate, digits=1))%")
        println("  c sign-flip rate:    $(round(100 * r.c_flip_rate, digits=1))%")
        println("  PPB vs c disagree:   $(round(100 * r.disagree_rate, digits=1))%")
    end
end


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION 4: Regression — covariate effects on PPB vs c
# Verifies Phase 1.3 predictions about interaction artifacts
# ═══════════════════════════════════════════════════════════════════════════════

function sim_regression_artifact(;
    n_subjects = 2000,
    n_trials = 40,
    seed = 789
)
    rng = MersenneTwister(seed)
    n_signal = n_trials ÷ 2
    n_noise = n_trials - n_signal

    # Two groups differing in baseline accuracy
    # Same covariate with same TRUE effect on bias (logit scale)
    #
    # Group A: high baseline accuracy (η₁ = 2.5, η₀ = -2.5 → TPR≈0.92, FPR≈0.08)
    # Group B: moderate accuracy (η₁ = 1.0, η₀ = -1.0 → TPR≈0.73, FPR≈0.27)

    logistic(x) = 1 / (1 + exp(-x))

    groups = [
        ("High accuracy",  2.5, -2.5),
        ("Moderate accuracy", 1.0, -1.0),
    ]

    β_bias = 0.3

    results = []

    for (label, η₁_base, η₀_base) in groups
        H_base = logistic(η₁_base)
        F_base = logistic(η₀_base)

        x = randn(rng, n_subjects)

        ppb_vals = Float64[]
        c_vals = Float64[]
        c_loglin_vals = Float64[]

        for xi in x
            η₁ = η₁_base + β_bias * xi
            η₀ = η₀_base + β_bias * xi
            H_true = logistic(η₁)
            F_true = logistic(η₀)

            hits, misses, fa, cr = simulate_subject(H_true, F_true, n_signal, n_noise; rng)
            H_obs = hits / n_signal
            F_obs = fa / n_noise

            push!(ppb_vals, ppb(H_obs, F_obs))

            H_ll, F_ll = loglinear_correct(hits, misses, fa, cr)
            push!(c_loglin_vals, criterion_c(H_ll, F_ll))

            push!(c_vals, criterion_c(H_obs, F_obs))
        end

        valid_c = .!isnan.(c_vals)
        valid_cll = .!isnan.(c_loglin_vals)

        slope_ppb = cov(ppb_vals, x) / var(x)
        slope_c_raw = cov(c_vals[valid_c], x[valid_c]) / var(x[valid_c])
        slope_c_loglin = cov(c_loglin_vals[valid_cll], x[valid_cll]) / var(x[valid_cll])

        # Analytical predictions for marginal effect at baseline
        σ_prime_1 = H_base * (1 - H_base)
        σ_prime_0 = F_base * (1 - F_base)
        predicted_ppb_slope = (σ_prime_1 + σ_prime_0) * β_bias

        ϕ_z_H = ϕpdf(Φinv(H_base))
        ϕ_z_F = ϕpdf(Φinv(F_base))
        predicted_c_slope = -0.5 * (σ_prime_1 / ϕ_z_H + σ_prime_0 / ϕ_z_F) * β_bias

        push!(results, (;
            label, H_base, F_base, β_bias, n_trials,
            ppb_baseline = ppb(H_base, F_base),
            c_baseline = criterion_c(H_base, F_base),
            slope_ppb, slope_c_raw, slope_c_loglin,
            predicted_ppb_slope, predicted_c_slope,
            n_nan = count(.!valid_c),
            n_subjects,
            # Raw data for potential figure use
            x, ppb_vals, c_loglin_vals,
        ))
    end

    return results
end

function report_sim4(results)
    r1 = results[1]
    println("\n═══ Regression artifact simulation ═══")
    println("True bias-shift coefficient (logit scale): β = $(r1.β_bias) on both η₁ and η₀")
    println("n_trials = $(r1.n_trials) per subject ($(r1.n_trials ÷ 2) signal, $(r1.n_trials - r1.n_trials ÷ 2) noise)\n")

    for r in results
        println("── $(r.label) ──")
        println("  Baseline: TPR=$(round(r.H_base, digits=3)), FPR=$(round(r.F_base, digits=3))")
        println("  Baseline PPB=$(round(r.ppb_baseline, digits=3))")
        println("  Baseline c=$(round(r.c_baseline, digits=3))")
        println("  Observed slope (PPB ~ x): $(round(r.slope_ppb, digits=4))  (predicted: $(round(r.predicted_ppb_slope, digits=4)))")
        println("  Observed slope (c_raw ~ x): $(round(r.slope_c_raw, digits=4))  ($(r.n_nan) NaN subjects dropped)")
        println("  Observed slope (c_loglin ~ x): $(round(r.slope_c_loglin, digits=4))  (predicted: $(round(r.predicted_c_slope, digits=4)))")
        println()
    end

    if length(results) >= 2
        ppb_ratio = abs(results[1].slope_ppb / results[2].slope_ppb)
        c_ratio = abs(results[1].slope_c_loglin / results[2].slope_c_loglin)
        println("  PPB slope ratio ($(results[1].label) / $(results[2].label)): $(round(ppb_ratio, digits=2))x")
        println("  c slope ratio: $(round(c_ratio, digits=2))x")
    end
end


# ─── Report functions for Sims 1 & 2 ──────────────────────────────────────────

function report_sim1(res; min_boundary_pct=1.0)
    println("\nSelected results (H_true, F_true, n_trials → %boundary, %NaN_c, PPB_bias, c_loglin_bias):")
    for i in eachindex(res["H_true"])
        pct_bnd = res["pct_boundary"][i]
        pct_bnd < min_boundary_pct && continue
        println("  H=$(res["H_true"][i]), F=$(res["F_true"][i]), n=$(Int(res["n_trials"][i])): " *
                "boundary=$(round(pct_bnd, digits=1))%, " *
                "NaN_c=$(round(res["c_raw_pct_nan"][i], digits=1))%, " *
                "PPB_bias=$(round(res["ppb_bias"][i], digits=4)), " *
                "c_loglin_bias=$(round(res["c_loglin_bias"][i], digits=4)), " *
                "c_halfn_bias=$(round(res["c_halfn_bias"][i], digits=4))")
    end
end

function report_sim2(res)
    println("\nResults (scenario, K → PPB_discrep, c_discrep, analytical_pred, can_flip?):")
    for i in eachindex(res["scenario"])
        println("  $(res["scenario"][i]), K=$(res["K"][i]): " *
                "PPB=$(round(res["ppb_discrepancy"][i], digits=5)), " *
                "c=$(round(res["c_discrepancy_mean"][i], digits=4)) ± $(round(res["c_discrepancy_sd"][i], digits=4)), " *
                "Br=$(round(res["br_discrepancy_mean"][i], digits=4)) ± $(round(res["br_discrepancy_sd"][i], digits=4)) [undef $(res["br_undefined"][i])], " *
                "analytical=$(round(res["analytical_prediction"][i], digits=4)), " *
                "flip_risk=$(res["c_discrepancy_can_flip"][i])")
    end
end


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL SIMULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

function run_all()
    println("=" ^ 70)
    println("SIMULATION 1: Boundary stability")
    println("=" ^ 70)

    res1 = sim_boundary_stability(
        H_true_vals = [0.50, 0.80, 0.90, 0.95],
        F_true_vals = [0.05, 0.10, 0.20, 0.50],
        n_trials_vals = [20, 40, 80],
        n_subjects = 5000
    )
    report_sim1(res1)

    println("\n" * "=" ^ 70)
    println("SIMULATION 2: Aggregation discrepancy")
    println("=" ^ 70)

    res2 = sim_aggregation()
    report_sim2(res2)

    println("\n" * "=" ^ 70)
    println("SIMULATION 3: Group comparison sign-flip")
    println("=" ^ 70)

    res3 = sim_group_comparison()
    report_sim3(res3)

    println("\n" * "=" ^ 70)
    println("SIMULATION 4: Regression artifact")
    println("=" ^ 70)

    res4 = sim_regression_artifact()
    report_sim4(res4)

    println("\n" * "=" ^ 70)
    println("DONE")
    println("=" ^ 70)

    return (; sim1=res1, sim2=res2, sim3=res3, sim4=res4)
end

# Run when executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all()
end
