# simulation_gaps.jl
# ─────────────────────────────────────────────────────────────────────────────
# Fills three paper-critical gaps identified in the coverage check (2026-03-25):
#   Gap A: Non-Gaussian latent distributions (Sim 6)
#   Gap B: B''/B''D added to existing simulation paradigms (Sims 1b–4b)
#   Gap C: Boundary × aggregation interaction (Sim 5)
#
# Reuses core functions from simulation.jl.
# ─────────────────────────────────────────────────────────────────────────────

using Distributions
using Random
using Statistics
using Roots  # for finding criterion that yields target PPB

# ─── Core functions (shared with simulation.jl) ─────────────────────────────

const ϕ_dist = Normal()
Φinv(p) = quantile(ϕ_dist, p)
Φ(x) = cdf(ϕ_dist, x)
ϕpdf(x) = pdf(ϕ_dist, x)

ppb(H, F) = H + F
youden_j(H, F) = H - F

function criterion_c(H, F)
    (H <= 0 || H >= 1 || F <= 0 || F >= 1) && return NaN
    return -0.5 * (Φinv(H) + Φinv(F))
end

function b_double_prime(H, F)
    (H == F) && return 0.0
    denom = H * (1 - H) + F * (1 - F)
    denom == 0 && return 0.0
    return sign(H - F) * (H * (1 - H) - F * (1 - F)) / denom
end

"""B''D (Donaldson 1992)."""
function b_double_prime_d(H, F)
    num = (1 - H) * (1 - F) - H * F
    denom = (1 - H) * (1 - F) + H * F
    denom == 0 && return 0.0
    return num / denom
end

"""B_r (Snodgrass & Corwin 1988, Eq. 8): F / (1 - P_r) = F / (1 - H + F).
The probability of saying yes from the uncertain state. Undefined at H = 1, F = 0."""
function b_r(H, F)
    denom = 1 - H + F
    denom == 0 && return NaN
    return F / denom
end

function loglinear_correct(hits, misses, fa, cr)
    H = (hits + 0.5) / (hits + misses + 1)
    F = (fa + 0.5) / (fa + cr + 1)
    return (H, F)
end

function half_n_correct(hits, n_signal, fa, n_noise)
    H = hits / n_signal
    F = fa / n_noise
    if H == 0.0; H = 1 / (2 * n_signal); elseif H == 1.0; H = 1 - 1 / (2 * n_signal); end
    if F == 0.0; F = 1 / (2 * n_noise); elseif F == 1.0; F = 1 - 1 / (2 * n_noise); end
    return (H, F)
end

function simulate_subject(H_true, F_true, n_signal, n_noise; rng=Random.default_rng())
    hits = rand(rng, Binomial(n_signal, H_true))
    fa = rand(rng, Binomial(n_noise, F_true))
    return (hits, n_signal - hits, fa, n_noise - fa)
end

"""Two-sided Welch's t-test. Returns p-value."""
function welch_ttest_p(x, y)
    nx, ny = length(x), length(y)
    mx, my = mean(x), mean(y)
    vx, vy = var(x), var(y)
    t = (mx - my) / sqrt(vx/nx + vy/ny)
    df = (vx/nx + vy/ny)^2 / ((vx/nx)^2/(nx-1) + (vy/ny)^2/(ny-1))
    return 2 * ccdf(TDist(df), abs(t))
end


# ═══════════════════════════════════════════════════════════════════════════════
# GAP B: Add B''/B''D to existing simulation paradigms
# Run abbreviated versions of Sims 1–4 that include B'' and B''D
# ═══════════════════════════════════════════════════════════════════════════════

function gap_b_boundary_stability(;
    H_true_vals = [0.50, 0.80, 0.90, 0.95],
    F_true_vals = [0.05, 0.10, 0.20, 0.50],
    n_trials_vals = [20, 40, 80],
    n_subjects = 5000,
    seed = 42
)
    rng = MersenneTwister(seed)

    results = Dict{String, Vector{Float64}}()
    for k in ["H_true", "F_true", "n_trials", "pct_boundary",
              "ppb_bias", "c_loglin_bias", "bpp_bias", "bppd_bias"]
        results[k] = Float64[]
    end

    for H_true in H_true_vals, F_true in F_true_vals, n_trials in n_trials_vals
        n_signal = n_trials ÷ 2
        n_noise = n_trials - n_signal

        ppb_true = ppb(H_true, F_true)
        bpp_true = b_double_prime(H_true, F_true)
        bppd_true = b_double_prime_d(H_true, F_true)
        c_true = criterion_c(H_true, F_true)

        ppb_obs = Float64[]
        bpp_obs = Float64[]
        bppd_obs = Float64[]
        c_loglin = Float64[]
        n_boundary = 0

        for _ in 1:n_subjects
            hits, misses, fa, cr = simulate_subject(H_true, F_true, n_signal, n_noise; rng)
            H_obs = hits / n_signal
            F_obs = fa / n_noise

            if H_obs == 0.0 || H_obs == 1.0 || F_obs == 0.0 || F_obs == 1.0
                n_boundary += 1
            end

            push!(ppb_obs, ppb(H_obs, F_obs))
            push!(bpp_obs, b_double_prime(H_obs, F_obs))
            push!(bppd_obs, b_double_prime_d(H_obs, F_obs))

            H_ll, F_ll = loglinear_correct(hits, misses, fa, cr)
            push!(c_loglin, criterion_c(H_ll, F_ll))
        end

        push!(results["H_true"], H_true)
        push!(results["F_true"], F_true)
        push!(results["n_trials"], Float64(n_trials))
        push!(results["pct_boundary"], 100 * n_boundary / n_subjects)
        push!(results["ppb_bias"], mean(ppb_obs) - ppb_true)
        push!(results["c_loglin_bias"], mean(c_loglin) - c_true)
        push!(results["bpp_bias"], mean(bpp_obs) - bpp_true)
        push!(results["bppd_bias"], mean(bppd_obs) - bppd_true)
    end

    return results
end

function report_gap_b_boundary(results; min_boundary_pct=5.0)
    println("\n═══ Gap B: Boundary stability with B'' and B''D ═══\n")
    println("H_true | F_true | n  | %bnd  | PPB_bias  | c_LL_bias | B''_bias  | B''D_bias")
    println("─" ^ 85)

    for i in eachindex(results["H_true"])
        pct_bnd = results["pct_boundary"][i]
        pct_bnd < min_boundary_pct && continue

        println("$(lpad(results["H_true"][i], 6)) | $(lpad(results["F_true"][i], 6)) | $(lpad(Int(results["n_trials"][i]), 2)) | " *
                "$(lpad(round(pct_bnd, digits=1), 5))% | " *
                "$(lpad(round(results["ppb_bias"][i], digits=4), 9)) | " *
                "$(lpad(round(results["c_loglin_bias"][i], digits=4), 9)) | " *
                "$(lpad(round(results["bpp_bias"][i], digits=4), 9)) | " *
                "$(lpad(round(results["bppd_bias"][i], digits=4), 9))")
    end
end


# Figure 2 Panel B and verify_numbers.jl both call panel_b_aggregation(), so the
# figure and the checked text come from one simulation run.
const PANEL_B_SCENARIOS = [
    (0.90, 0.15, 0.08, 0.06, "Typical applied"),
    (0.70, 0.30, 0.08, 0.08, "Symmetric"),
    (0.95, 0.05, 0.04, 0.03, "Extreme"),
]
panel_b_aggregation() = gap_b_aggregation(scenarios = PANEL_B_SCENARIOS, K = 100, n_replications = 2000)

function gap_b_aggregation(;
    scenarios = [
        (0.90, 0.15, 0.08, 0.06, "High H, low F (typical applied)"),
        (0.95, 0.05, 0.04, 0.03, "Very extreme (Honduras-like)"),
        (0.70, 0.30, 0.08, 0.08, "Moderate, symmetric"),
    ],
    K = 100,
    n_replications = 2000,
    seed = 123
)
    rng = MersenneTwister(seed)

    results = []

    for (μH, μF, σH, σF, label) in scenarios
        αH = μH * (μH * (1 - μH) / σH^2 - 1)
        βH = (1 - μH) * (μH * (1 - μH) / σH^2 - 1)
        αF = μF * (μF * (1 - μF) / σF^2 - 1)
        βF = (1 - μF) * (μF * (1 - μF) / σF^2 - 1)

        (αH <= 0 || βH <= 0 || αF <= 0 || βF <= 0) && continue

        dist_H = Beta(αH, βH)
        dist_F = Beta(αF, βF)

        ppb_discreps = Float64[]
        c_discreps = Float64[]
        bpp_discreps = Float64[]
        bppd_discreps = Float64[]
        br_discreps = Float64[]
        n_br_undefined = 0

        for _ in 1:n_replications
            Hs = clamp.(rand(rng, dist_H, K), 1e-6, 1 - 1e-6)
            Fs = clamp.(rand(rng, dist_F, K), 1e-6, 1 - 1e-6)

            push!(ppb_discreps, mean(ppb.(Hs, Fs)) - ppb(mean(Hs), mean(Fs)))
            push!(c_discreps, mean(criterion_c.(Hs, Fs)) - criterion_c(mean(Hs), mean(Fs)))
            push!(bpp_discreps, mean(b_double_prime.(Hs, Fs)) - b_double_prime(mean(Hs), mean(Fs)))
            push!(bppd_discreps, mean(b_double_prime_d.(Hs, Fs)) - b_double_prime_d(mean(Hs), mean(Fs)))

            br_ind = b_r.(Hs, Fs)
            n_br_undefined += count(isnan, br_ind)
            push!(br_discreps, mean(br_ind) - b_r(mean(Hs), mean(Fs)))
        end

        push!(results, (;
            label, μH, μF,
            ppb_discrepancy = mean(ppb_discreps),
            c_discrepancy_mean = mean(c_discreps),
            c_discrepancy_sd = std(c_discreps),
            bpp_discrepancy_mean = mean(bpp_discreps),
            bpp_discrepancy_sd = std(bpp_discreps),
            bppd_discrepancy_mean = mean(bppd_discreps),
            bppd_discrepancy_sd = std(bppd_discreps),
            br_discrepancy_mean = mean(br_discreps),
            br_discrepancy_sd = std(br_discreps),
            br_undefined = n_br_undefined,
            n_br_total = n_replications * K,
        ))
    end

    return results
end

function report_gap_b_aggregation(results)
    println("\n═══ Gap B: Aggregation discrepancy with B'' and B''D ═══\n")
    for r in results
        println("── $(r.label) ──")
        println("  PPB  discrepancy: $(round(r.ppb_discrepancy, digits=5))")
        println("  c    discrepancy: $(round(r.c_discrepancy_mean, digits=4)) ± $(round(r.c_discrepancy_sd, digits=4))")
        println("  B''  discrepancy: $(round(r.bpp_discrepancy_mean, digits=4)) ± $(round(r.bpp_discrepancy_sd, digits=4))")
        println("  B''D discrepancy: $(round(r.bppd_discrepancy_mean, digits=4)) ± $(round(r.bppd_discrepancy_sd, digits=4))")
        println("  Br   discrepancy: $(round(r.br_discrepancy_mean, digits=4)) ± $(round(r.br_discrepancy_sd, digits=4))")
        println("  Br   undefined:   $(r.br_undefined) of $(r.n_br_total) individual values")
        println()
    end
end


function gap_b_group_comparison(;
    n_replications = 5000,
    K = 100,
    seed = 456
)
    rng = MersenneTwister(seed)

    scenarios = [
        ("Small PPB diff, large accuracy diff",
         0.95, 0.08, 0.04, 0.04,
         0.65, 0.40, 0.10, 0.10),
        ("Equal true PPB, accuracy diff",
         0.90, 0.15, 0.07, 0.06,
         0.70, 0.35, 0.08, 0.08),
    ]

    make_beta(μ, σ) = begin
        α = μ * (μ * (1 - μ) / σ^2 - 1)
        β = (1 - μ) * (μ * (1 - μ) / σ^2 - 1)
        (α > 0 && β > 0) || error("Invalid Beta params: μ=$μ, σ=$σ")
        Beta(α, β)
    end

    results = []

    for (label, μH_A, μF_A, σH_A, σF_A, μH_B, μF_B, σH_B, σF_B) in scenarios
        dHA, dFA = make_beta(μH_A, σH_A), make_beta(μF_A, σF_A)
        dHB, dFB = make_beta(μH_B, σH_B), make_beta(μF_B, σF_B)

        ppb_A_true = ppb(μH_A, μF_A)
        ppb_B_true = ppb(μH_B, μF_B)

        n_ppb_disagree_bpp = 0
        n_c_disagree_bpp = 0
        n_c_disagree_ppb = 0

        for _ in 1:n_replications
            HA = clamp.(rand(rng, dHA, K), 1e-6, 1-1e-6)
            FA = clamp.(rand(rng, dFA, K), 1e-6, 1-1e-6)
            HB = clamp.(rand(rng, dHB, K), 1e-6, 1-1e-6)
            FB = clamp.(rand(rng, dFB, K), 1e-6, 1-1e-6)

            ppb_A = mean(ppb.(HA, FA))
            ppb_B = mean(ppb.(HB, FB))
            c_A = mean(criterion_c.(HA, FA))
            c_B = mean(criterion_c.(HB, FB))
            bpp_A = mean(b_double_prime.(HA, FA))
            bpp_B = mean(b_double_prime.(HB, FB))

            ppb_sign = sign(ppb_A - ppb_B)
            c_sign = sign(c_A - c_B)
            bpp_sign = sign(bpp_A - bpp_B)

            ppb_sign != 0 && ppb_sign != -c_sign && (n_c_disagree_ppb += 1)
            ppb_sign != 0 && ppb_sign != bpp_sign && (n_ppb_disagree_bpp += 1)
            c_sign != 0 && c_sign != bpp_sign && (n_c_disagree_bpp += 1)
        end

        push!(results, (;
            label, ppb_A_true, ppb_B_true,
            ppb_vs_c_disagree = n_c_disagree_ppb / n_replications,
            ppb_vs_bpp_disagree = n_ppb_disagree_bpp / n_replications,
            c_vs_bpp_disagree = n_c_disagree_bpp / n_replications,
        ))
    end

    return results
end

function report_gap_b_group(results)
    println("\n═══ Gap B: Group comparison with B'' and B''D ═══\n")
    for r in results
        println("── $(r.label) ──")
        println("  True PPB: A=$(round(r.ppb_A_true, digits=3)), B=$(round(r.ppb_B_true, digits=3))")
        println("  PPB vs c disagree:   $(round(100 * r.ppb_vs_c_disagree, digits=1))%")
        println("  PPB vs B'' disagree: $(round(100 * r.ppb_vs_bpp_disagree, digits=1))%")
        println("  c vs B'' disagree:   $(round(100 * r.c_vs_bpp_disagree, digits=1))%")
        println()
    end
end


# ═══════════════════════════════════════════════════════════════════════════════
# GAP C: Boundary × aggregation interaction (Sim 5)
# ═══════════════════════════════════════════════════════════════════════════════

function sim5_boundary_aggregation(;
    scenarios = [
        (0.95, 0.15, 20, "Asymmetric: H=0.95, F=0.15, n=20"),
        (0.95, 0.20, 20, "Asymmetric: H=0.95, F=0.20, n=20"),
        (0.90, 0.05, 20, "Asymmetric: H=0.90, F=0.05, n=20"),
        (0.95, 0.50, 20, "Liberal bias: H=0.95, F=0.50, n=20"),
        (0.95, 0.05, 20, "Symmetric: H=0.95, F=0.05, n=20"),
        (0.80, 0.20, 20, "Symmetric: H=0.80, F=0.20, n=20"),
        (0.95, 0.15, 80, "Asymmetric control: H=0.95, F=0.15, n=80"),
    ],
    K_vals = [20, 50, 100, 500],
    n_replications = 3000,
    seed = 999
)
    rng = MersenneTwister(seed)

    results = []

    for (H_true, F_true, n_trials, label) in scenarios
        n_signal = n_trials ÷ 2
        n_noise = n_trials - n_signal

        ppb_true = ppb(H_true, F_true)
        c_true = criterion_c(H_true, F_true)
        bpp_true = b_double_prime(H_true, F_true)

        scenario_results = []

        for K in K_vals
            ppb_agg = Float64[]
            c_loglin_agg = Float64[]
            c_dropnan_agg = Float64[]
            bpp_agg = Float64[]
            pct_bnd_all = Float64[]

            for _ in 1:n_replications
                ppb_subj = Float64[]
                c_loglin_subj = Float64[]
                c_raw_subj = Float64[]
                bpp_subj = Float64[]
                n_bnd = 0

                for _ in 1:K
                    hits, misses, fa, cr = simulate_subject(H_true, F_true, n_signal, n_noise; rng)
                    H_obs = hits / n_signal
                    F_obs = fa / n_noise

                    if H_obs == 0.0 || H_obs == 1.0 || F_obs == 0.0 || F_obs == 1.0
                        n_bnd += 1
                    end

                    push!(ppb_subj, ppb(H_obs, F_obs))
                    push!(bpp_subj, b_double_prime(H_obs, F_obs))
                    push!(c_raw_subj, criterion_c(H_obs, F_obs))

                    H_ll, F_ll = loglinear_correct(hits, misses, fa, cr)
                    push!(c_loglin_subj, criterion_c(H_ll, F_ll))
                end

                push!(pct_bnd_all, 100 * n_bnd / K)
                push!(ppb_agg, mean(ppb_subj))
                push!(c_loglin_agg, mean(c_loglin_subj))
                push!(bpp_agg, mean(bpp_subj))

                valid = .!isnan.(c_raw_subj)
                push!(c_dropnan_agg, count(valid) > 0 ? mean(c_raw_subj[valid]) : NaN)
            end

            push!(scenario_results, (;
                K,
                pct_boundary = mean(pct_bnd_all),
                ppb_bias = mean(ppb_agg) - ppb_true,
                c_ll_bias = mean(c_loglin_agg) - c_true,
                c_dropnan_bias = mean(filter(!isnan, c_dropnan_agg)) - c_true,
                bpp_bias = mean(bpp_agg) - bpp_true,
            ))
        end

        push!(results, (;
            label, H_true, F_true, n_trials, ppb_true, c_true,
            by_K = scenario_results,
        ))
    end

    return results
end

function report_sim5(results)
    println("\n═══ Sim 5: Boundary × aggregation interaction ═══\n")
    for r in results
        println("── $(r.label) ──")
        println("  True PPB=$(round(r.ppb_true, digits=3)), True c=$(round(r.c_true, digits=3))")
        println("  K  | %bnd  | PPB agg bias | c_LL agg bias | c_LL drop-NaN bias | B'' agg bias")
        println("  " * "─" ^ 80)

        for kr in r.by_K
            println("  $(lpad(kr.K, 3)) | $(lpad(round(kr.pct_boundary, digits=1), 5))% | " *
                    "$(lpad(round(kr.ppb_bias, digits=4), 12)) | " *
                    "$(lpad(round(kr.c_ll_bias, digits=4), 13)) | " *
                    "$(lpad(round(kr.c_dropnan_bias, digits=4), 18)) | " *
                    "$(lpad(round(kr.bpp_bias, digits=4), 12))")
        end
        println()
    end
end


# ═══════════════════════════════════════════════════════════════════════════════
# GAP A: Non-Gaussian latent distributions (Sim 6)
# ═══════════════════════════════════════════════════════════════════════════════

"""
For a given signal and noise distribution and criterion k,
compute H = P(signal > k) and F = P(noise > k).
"""
function rates_from_criterion(signal_dist, noise_dist, k)
    H = 1 - cdf(signal_dist, k)
    F = 1 - cdf(noise_dist, k)
    return (H, F)
end

"""
Find criterion k such that PPB = H + F = target_ppb.
"""
function find_criterion_for_ppb(signal_dist, noise_dist, target_ppb;
                                 bracket=(-10.0, 10.0))
    f(k) = begin
        H, F = rates_from_criterion(signal_dist, noise_dist, k)
        ppb(H, F) - target_ppb
    end
    return find_zero(f, bracket, Bisection())
end

function sim6_nongaussian(;
    target_ppbs = [0.90, 1.00, 1.05, 1.10, 1.20],
    d_prime = 1.5,
    n_trials = 40,
    n_subjects = 2000,
    n_replications = 1000,
    seed = 2025
)
    rng = MersenneTwister(seed)

    n_signal = n_trials ÷ 2
    n_noise = n_trials - n_signal

    dist_pairs = [
        ("EVSDT (Gaussian)",
         Normal(d_prime, 1.0), Normal(0.0, 1.0)),
        ("Unequal variance (σ=1.5)",
         Normal(d_prime, 1.5), Normal(0.0, 1.0)),
        ("Unequal variance (σ=2.0)",
         Normal(d_prime, 2.0), Normal(0.0, 1.0)),
        ("Logistic",
         Logistic(d_prime, 1.0), Logistic(0.0, 1.0)),
        ("Heavy-tailed (t, df=5)",
         TDist(5) + d_prime, TDist(5)),
        ("Heavy-tailed (t, df=3)",
         TDist(3) + d_prime, TDist(3)),
    ]

    # ── Part 1: Population-level rates ──
    part1 = []
    for target_ppb in target_ppbs
        for (label, sig, noi) in dist_pairs
            try
                k = find_criterion_for_ppb(sig, noi, target_ppb)
                H, F = rates_from_criterion(sig, noi, k)
                push!(part1, (;
                    target_ppb, label, H, F,
                    ppb_val = ppb(H, F),
                    c_val = criterion_c(H, F),
                    bpp_val = b_double_prime(H, F),
                    bppd_val = b_double_prime_d(H, F),
                ))
            catch e
                push!(part1, (;
                    target_ppb, label, H=NaN, F=NaN,
                    ppb_val=NaN, c_val=NaN, bpp_val=NaN, bppd_val=NaN,
                    error=string(e),
                ))
            end
        end
    end

    # ── Part 2: Sampling variability ──
    target_ppb_2 = 1.05
    part2 = []
    for (label, sig, noi) in dist_pairs
        try
            k = find_criterion_for_ppb(sig, noi, target_ppb_2)
            H_pop, F_pop = rates_from_criterion(sig, noi, k)
            ppb_pop = ppb(H_pop, F_pop)
            c_pop = criterion_c(H_pop, F_pop)
            bpp_pop = b_double_prime(H_pop, F_pop)

            ppb_obs = Float64[]
            c_obs = Float64[]
            bpp_obs = Float64[]

            for _ in 1:n_subjects
                hits, misses, fa, cr = simulate_subject(H_pop, F_pop, n_signal, n_noise; rng)
                H_ll, F_ll = loglinear_correct(hits, misses, fa, cr)
                H_raw = hits / n_signal
                F_raw = fa / n_noise

                push!(ppb_obs, ppb(H_raw, F_raw))
                push!(c_obs, criterion_c(H_ll, F_ll))
                push!(bpp_obs, b_double_prime(H_raw, F_raw))
            end

            push!(part2, (;
                label,
                ppb_bias = mean(ppb_obs) - ppb_pop,
                ppb_rmse = sqrt(mean((ppb_obs .- ppb_pop).^2)),
                c_bias = mean(c_obs) - c_pop,
                c_rmse = sqrt(mean((c_obs .- c_pop).^2)),
                bpp_bias = mean(bpp_obs) - bpp_pop,
                bpp_rmse = sqrt(mean((bpp_obs .- bpp_pop).^2)),
            ))
        catch e
            push!(part2, (; label, error=string(e)))
        end
    end

    # ── Part 3: Group comparison across distribution shapes ──
    target_ppb_3 = 1.20
    K_group = 200
    n_reps_3 = 2000

    sig_A, noi_A = Normal(d_prime, 1.0), Normal(0.0, 1.0)
    k_A = find_criterion_for_ppb(sig_A, noi_A, target_ppb_3)
    H_A, F_A = rates_from_criterion(sig_A, noi_A, k_A)
    c_A_pop = criterion_c(H_A, F_A)
    bpp_A_pop = b_double_prime(H_A, F_A)
    br_A_pop = b_r(H_A, F_A)

    part3 = []
    for (label, sig_B, noi_B) in dist_pairs[2:end]
        try
            k_B = find_criterion_for_ppb(sig_B, noi_B, target_ppb_3)
            H_B, F_B = rates_from_criterion(sig_B, noi_B, k_B)
            c_B_pop = criterion_c(H_B, F_B)
            bpp_B_pop = b_double_prime(H_B, F_B)
            br_B_pop = b_r(H_B, F_B)

            n_ppb_says_diff = 0
            n_c_says_diff = 0
            n_bpp_says_diff = 0
            n_br_says_diff = 0
            n_br_undefined_raw = 0

            for _ in 1:n_reps_3
                ppb_sA = Float64[]
                c_sA = Float64[]
                bpp_sA = Float64[]
                br_sA = Float64[]
                for _ in 1:K_group
                    hits, misses, fa, cr = simulate_subject(H_A, F_A, n_signal, n_noise; rng)
                    H_ll, F_ll = loglinear_correct(hits, misses, fa, cr)
                    push!(ppb_sA, ppb(hits/n_signal, fa/n_noise))
                    push!(c_sA, criterion_c(H_ll, F_ll))
                    push!(bpp_sA, b_double_prime(H_ll, F_ll))
                    push!(br_sA, b_r(H_ll, F_ll))
                    isnan(b_r(hits/n_signal, fa/n_noise)) && (n_br_undefined_raw += 1)
                end

                ppb_sB = Float64[]
                c_sB = Float64[]
                bpp_sB = Float64[]
                br_sB = Float64[]
                for _ in 1:K_group
                    hits, misses, fa, cr = simulate_subject(H_B, F_B, n_signal, n_noise; rng)
                    H_ll, F_ll = loglinear_correct(hits, misses, fa, cr)
                    push!(ppb_sB, ppb(hits/n_signal, fa/n_noise))
                    push!(c_sB, criterion_c(H_ll, F_ll))
                    push!(bpp_sB, b_double_prime(H_ll, F_ll))
                    push!(br_sB, b_r(H_ll, F_ll))
                    isnan(b_r(hits/n_signal, fa/n_noise)) && (n_br_undefined_raw += 1)
                end

                welch_ttest_p(ppb_sA, ppb_sB) < 0.05 && (n_ppb_says_diff += 1)
                welch_ttest_p(c_sA, c_sB) < 0.05 && (n_c_says_diff += 1)
                welch_ttest_p(bpp_sA, bpp_sB) < 0.05 && (n_bpp_says_diff += 1)
                welch_ttest_p(br_sA, br_sB) < 0.05 && (n_br_says_diff += 1)
            end

            push!(part3, (;
                label,
                c_A_pop, c_B_pop,
                c_pop_diff = c_A_pop - c_B_pop,
                bpp_pop_diff = bpp_A_pop - bpp_B_pop,
                br_A_pop, br_B_pop,
                br_pop_diff = br_A_pop - br_B_pop,
                ppb_detection_rate = n_ppb_says_diff / n_reps_3,
                c_detection_rate = n_c_says_diff / n_reps_3,
                bpp_detection_rate = n_bpp_says_diff / n_reps_3,
                br_detection_rate = n_br_says_diff / n_reps_3,
                br_undefined_raw = n_br_undefined_raw,
                n_subject_draws = 2 * K_group * n_reps_3,
            ))
        catch e
            push!(part3, (; label, error=string(e)))
        end
    end

    return (; part1, part2, part3,
              d_prime, n_trials, target_ppb_3, K_group, n_reps_3)
end

function report_sim6(results)
    println("\n═══ Sim 6: Non-Gaussian latent distributions ═══\n")
    println("d' (or location shift) = $(results.d_prime) for all distributions")
    println("n_trials = $(results.n_trials) per subject\n")

    # Part 1
    println("Part 1: Same PPB, different c values across distributions\n")
    current_ppb = NaN
    for r in results.part1
        if r.target_ppb != current_ppb
            current_ppb = r.target_ppb
            println("── Target PPB = $current_ppb ──")
            println("  Distribution          |    H    |    F    |   PPB   |     c     |    B''   |   B''D")
            println("  " * "─" ^ 85)
        end
        if hasproperty(r, :error)
            println("  $(rpad(r.label, 23)) | [could not find criterion: $(r.error)]")
        else
            println("  $(rpad(r.label, 23)) | $(lpad(round(r.H, digits=4), 7)) | " *
                    "$(lpad(round(r.F, digits=4), 7)) | " *
                    "$(lpad(round(r.ppb_val, digits=4), 7)) | " *
                    "$(lpad(round(r.c_val, digits=4), 9)) | " *
                    "$(lpad(round(r.bpp_val, digits=4), 8)) | " *
                    "$(lpad(round(r.bppd_val, digits=4), 7))")
        end
    end

    # Part 2
    println("\n\nPart 2: Finite-sample estimation (target PPB = 1.05)\n")
    println("  Distribution          | PPB bias | PPB RMSE | c bias   | c RMSE   | B'' bias | B'' RMSE")
    println("  " * "─" ^ 95)
    for r in results.part2
        if hasproperty(r, :error)
            println("  $(rpad(r.label, 23)) | [error: $(r.error)]")
        else
            println("  $(rpad(r.label, 23)) | " *
                    "$(lpad(round(r.ppb_bias, digits=4), 8)) | " *
                    "$(lpad(round(r.ppb_rmse, digits=4), 8)) | " *
                    "$(lpad(round(r.c_bias, digits=4), 8)) | " *
                    "$(lpad(round(r.c_rmse, digits=4), 8)) | " *
                    "$(lpad(round(r.bpp_bias, digits=4), 8)) | " *
                    "$(lpad(round(r.bpp_rmse, digits=4), 8))")
        end
    end

    # Part 3
    println("\n\nPart 3: Two groups, same PPB, different distributions\n")
    println("Both groups have PPB = $(results.target_ppb_3). Group A = EVSDT Gaussian. Group B varies.")
    println("K=$(results.K_group) subjects per group, $(results.n_reps_3) replications.\n")
    for r in results.part3
        if hasproperty(r, :error)
            println("  Group B: $(r.label) — [error: $(r.error)]")
        else
            println("  Group B: $(r.label)")
            println("    Population c: A=$(round(r.c_A_pop, digits=4)), B=$(round(r.c_B_pop, digits=4)), diff=$(round(r.c_pop_diff, digits=4))")
            println("    Population Br: A=$(round(r.br_A_pop, digits=4)), B=$(round(r.br_B_pop, digits=4)), diff=$(round(r.br_pop_diff, digits=4))")
            println("    PPB detects group diff: $(round(100 * r.ppb_detection_rate, digits=1))% of replications (should be ~5%)")
            println("    c detects group diff:   $(round(100 * r.c_detection_rate, digits=1))% of replications (inflated = spurious)")
            println("    B'' detects group diff: $(round(100 * r.bpp_detection_rate, digits=1))% of replications")
            println("    Br detects group diff:  $(round(100 * r.br_detection_rate, digits=1))% of replications")
            println("    Br undefined on raw rates: $(r.br_undefined_raw) of $(r.n_subject_draws) subject draws")
            println()
        end
    end
end


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════════════════════

function run_all_gaps()
    println("=" ^ 70)
    println("SIMULATION GAPS — Filling coverage gaps A, B, C")
    println("=" ^ 70)

    # Gap B: B''/B''D in existing paradigms
    res_b_boundary = gap_b_boundary_stability()
    report_gap_b_boundary(res_b_boundary)

    res_b_agg = gap_b_aggregation()
    report_gap_b_aggregation(res_b_agg)

    res_b_group = gap_b_group_comparison()
    report_gap_b_group(res_b_group)

    # Gap C: Boundary × aggregation
    res_sim5 = sim5_boundary_aggregation()
    report_sim5(res_sim5)

    # Gap A: Non-Gaussian distributions
    res_sim6 = sim6_nongaussian()
    report_sim6(res_sim6)

    println("\n" * "=" ^ 70)
    println("ALL GAP SIMULATIONS COMPLETE")
    println("=" ^ 70)

    return (; gap_b_boundary=res_b_boundary, gap_b_agg=res_b_agg,
              gap_b_group=res_b_group, sim5=res_sim5, sim6=res_sim6)
end

# Run when executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_gaps()
end
