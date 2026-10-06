# figures.jl
# ─────────────────────────────────────────────────────────────────────────────
# Generate Figures 2 and 3 for the BRM paper.
# Figure 1 (ROC geometry) is handled by figure_warp.jl.
#
# Loads simulation functions via module isolation to avoid const conflicts.
#
# Requirements: ] add CairoMakie Distributions ColorSchemes Roots
# ─────────────────────────────────────────────────────────────────────────────

using CairoMakie
using ColorSchemes
using Statistics

# Load simulation code in isolated modules
module Sim
    include("simulation.jl")
end

module SimGaps
    include("simulation_gaps.jl")
end

# ─── Color palette ────────────────────────────────────────────────────────────

const COL_PPB = ColorSchemes.batlowS[1]       # PPB color
const COL_C_LL = ColorSchemes.batlowS[5]      # c log-linear color
const COL_C_HN = ColorSchemes.batlowS[7]      # c half-N color
const COL_BPP = ColorSchemes.batlowS[9]       # B'' color
const COL_BR = ColorSchemes.batlowS[8]        # Br color
const COL_BPPD = ColorSchemes.batlowS[11]     # B''D color

# ─── Safe save ────────────────────────────────────────────────────────────────
# Writing straight into the iCloud-synced figures/ directory can fail with
# "SystemError: close: Operation timed out", and the failed write leaves the
# target *deleted*. Render to a local temp file first, then copy it in.

function safe_save(dest, fig; kwargs...)
    tmpdir = mktempdir()
    try
        tmp = joinpath(tmpdir, basename(dest))
        save(tmp, fig; kwargs...)
        mkpath(dirname(dest))
        cp(tmp, dest; force = true)
        println("Saved $dest")
    finally
        rm(tmpdir; recursive = true, force = true)
    end
    return dest
end


# ─── Figure 2: Three practical advantages ─────────────────────────────────────

function figure2()
    println("Computing Figure 2 data...")

    # ── Panel A: Boundary correction bias (exact enumeration) ──
    # Shared code path with verify_numbers.jl so labels and paper text cannot drift.
    panel = Sim.boundary_panel()

    config_labels = [
        "H=.80\nF=.05\nn=20",
        "H=.95\nF=.20\nn=20",
        "H=.95\nF=.05\nn=20",
        "H=.95\nF=.10\nn=40",
    ]
    boundary_pcts = [r.pct_boundary for r in panel]
    ppb_biases = [r.ppb_bias for r in panel]
    c_ll_biases = [r.c_ll_bias for r in panel]
    c_hn_biases = [r.c_hn_bias for r in panel]

    # ── Panel B: Aggregation discrepancy ──
    # One simulation for every series, the same call verify_numbers.jl checks
    # the text against (it previously took PPB and c from a second simulation).
    res2b = SimGaps.panel_b_aggregation()

    agg_labels = [replace(r.label, "Typical applied" => "Typical\napplied") for r in res2b]
    ppb_agg = [r.ppb_discrepancy for r in res2b]
    c_agg = [r.c_discrepancy_mean for r in res2b]
    bpp_agg = [r.bpp_discrepancy_mean for r in res2b]
    bppd_agg = [r.bppd_discrepancy_mean for r in res2b]
    br_agg = [r.br_discrepancy_mean for r in res2b]

    # ── Panel C: Regression compression (Sim 4) ──
    res4 = Sim.sim_regression_artifact()

    # ── Build figure ──
    fig = Figure(size = (1200, 500), fontsize = 21)
    l1 = GridLayout(fig[1, 1])
    l2 = GridLayout(fig[1, 2])
    l3 = GridLayout(fig[1, 3])

    # Panel A
    ax_a = Axis(l1[1, 1],
        ylabel = "Bias (estimated − true)",
        title = "Boundary correction bias",
        xticks = (1:4, config_labels),
        xticklabelsize = 18,
    )
    ylims!(ax_a, -0.17, 0.18)
    hlines!(ax_a, [0], color = (:gray, 0.5), linestyle = :dash)

    dodge_width = 0.25
    for (j, (vals, col, lab)) in enumerate([
        (ppb_biases, COL_PPB, "PPB"),
        (c_ll_biases, COL_C_LL, "c (log-linear)"),
        (c_hn_biases, COL_C_HN, "c (1/2N)"),
    ])
        xs = (1:4) .+ (j - 2) * dodge_width
        scatter!(ax_a, xs, vals, color = col, markersize = 12, label = lab)
    end

    # Annotate boundary % above the most extreme dot per config
    for (i, pct) in enumerate(boundary_pcts)
        max_abs = maximum(abs.([ppb_biases[i], c_ll_biases[i], c_hn_biases[i]]))
        # Place above the highest magnitude, with sign matching the dominant direction
        y_extreme = c_ll_biases[i] > 0 ? max_abs : -max_abs
        y_pos = y_extreme + sign(y_extreme) * 0.018
        text!(ax_a, i, y_pos;
            text = "$(round(Int, pct))%", align = (:center, y_extreme > 0 ? :bottom : :top),
            fontsize = 16, color = :gray40)
    end

    # Panel B
    ax_b = Axis(l2[1, 1],
        ylabel = "Aggregation discrepancy",
        title = "Aggregation discrepancy",
        xticks = (1:3, agg_labels),
        xticklabelsize = 18,
    )
    hlines!(ax_b, [0], color = (:gray, 0.5), linestyle = :dash)

    b_series = [
        (ppb_agg, COL_PPB, "PPB"),
        (c_agg, COL_C_LL, "c"),
        (bpp_agg, COL_BPP, "B''"),
        (bppd_agg, COL_BPPD, "B''D"),
        (br_agg, COL_BR, "Br"),
    ]
    for (j, (vals, col, lab)) in enumerate(b_series)
        xs = (1:3) .+ (j - (length(b_series) + 1) / 2) * 0.18
        scatter!(ax_b, xs, vals, color = col, markersize = 12, label = lab)
    end

    # Panel-local legend: the shared legend below the figure carries Panel A only
    Legend(l2[1, 1],
        [MarkerElement(color = col, marker = :circle, markersize = 10)
         for (_, col, _) in b_series],
        [lab for (_, _, lab) in b_series],
        tellwidth = false, tellheight = false,
        halign = :center, valign = :bottom,
        framevisible = false, labelsize = 17, nbanks = 2,
        margin = (0, 0, 10, 0))

    # Panel C
    ax_c = Axis(l3[1, 1],
        ylabel = "Observed slope (bias ~ covariate)",
        title = "Covariate effects",
        xticks = (1:2, [replace(r.label, " accuracy" => "\naccuracy") for r in res4]),
        xticklabelsize = 18,
    )

    ppb_slopes = [abs(r.slope_ppb) for r in res4]
    c_slopes = [abs(r.slope_c_loglin) for r in res4]

    c_series = [
        (ppb_slopes, COL_PPB, "PPB"),
        (c_slopes, COL_C_LL, "c (log-linear)"),
    ]
    for (j, (vals, col, lab)) in enumerate(c_series)
        xs = (1:2) .+ (j - 1.5) * dodge_width
        scatter!(ax_c, xs, vals, color = col, markersize = 12, label = lab)
    end

    Legend(l3[1, 1],
        [MarkerElement(color = col, marker = :circle, markersize = 10)
         for (_, col, _) in c_series],
        [lab for (_, _, lab) in c_series],
        tellwidth = false, tellheight = false,
        halign = :right, valign = :bottom,
        framevisible = false, labelsize = 17,
        margin = (0, 10, 10, 0))

    # Connect dots and annotate ratios
    # Moderate-accuracy : high-accuracy, the direction the text reports.
    ppb_ratio = ppb_slopes[2] / ppb_slopes[1]
    c_ratio = c_slopes[2] / c_slopes[1]

    ppb_xs = (1:2) .- 0.5 * dodge_width
    c_xs = (1:2) .+ 0.5 * dodge_width
    lines!(ax_c, ppb_xs, ppb_slopes, color = (COL_PPB, 0.5), linewidth = 1.5)
    lines!(ax_c, c_xs, c_slopes, color = (COL_C_LL, 0.5), linewidth = 1.5)

    # Ratio annotations
    text!(ax_c, mean(ppb_xs), mean(ppb_slopes) + 0.005;
        text = "$(round(ppb_ratio, digits=1))× (mod./high)",
        align = (:center, :bottom), fontsize = 17, color = COL_PPB)
    text!(ax_c, mean(c_xs), mean(c_slopes) - 0.005;
        text = "$(round(c_ratio, digits=1))× (mod./high)",
        align = (:center, :top), fontsize = 17, color = COL_C_LL)

    # Legend for Panel A only; Panels B and C carry their own (the series differ).
    Legend(l1[2, 1], ax_a, orientation = :horizontal, framevisible = false,
           nbanks = 2, labelsize = 17, tellheight = true)

    # Panel labels
    for (label, layout) in zip(["A", "B", "C"], [l1, l2, l3])
        Label(layout[1, 1, TopLeft()], label,
            fontsize = 26, font = :bold,
            padding = (0, 5, 5, 0), halign = :right)
    end

    resize_to_layout!(fig)
    safe_save("figures/fg_advantages.png", fig, px_per_unit = 2)

    return fig
end


# ─── Supplement figure (fig-conceptual): PPB and c measure different things ─────────────────────────────

function figure3()
    println("Computing Figure 3 data...")

    # ── Panel A: Group comparison disagreement (Sim 3) ──
    res3 = Sim.sim_group_comparison()
    # Use the "Small PPB diff, large accuracy diff" scenario (91.5% disagreement)
    scenario = res3[3]

    # ── Panel B: Spurious detection (Sim 6, Part 3) ──
    res6 = SimGaps.sim6_nongaussian()

    # ── Build figure ──
    fig = Figure(size = (900, 400))
    l1 = GridLayout(fig[1, 1])
    l2 = GridLayout(fig[1, 2])

    # Panel A: scatter of per-replication PPB diff vs c diff
    ax_a = Axis(l1[1, 1],
        xlabel = "PPB difference (A − B)",
        ylabel = "c difference (A − B)",
        title = "Group comparison disagreement",
        aspect = 1,
    )

    hlines!(ax_a, [0], color = (:gray, 0.4), linestyle = :dash)
    vlines!(ax_a, [0], color = (:gray, 0.4), linestyle = :dash)

    # Subsample for visual clarity
    n_show = 500
    step = max(1, length(scenario.ppb_diffs) ÷ n_show)
    ppb_d = scenario.ppb_diffs[1:step:end]
    c_d = scenario.c_diffs[1:step:end]

    scatter!(ax_a, ppb_d, c_d,
        color = (ColorSchemes.batlowS[3], 0.3), markersize = 5)

    # Shade disagreement quadrant and annotate
    disagree_pct = round(100 * scenario.disagree_rate, digits=1)

    # Count points in each quadrant for this subsample
    # PPB says B more liberal (ppb_diff < 0) but c says A more liberal (c_diff < 0)
    # needs sign convention check: more liberal = higher PPB = more negative c
    text!(ax_a, minimum(ppb_d) * 0.5, minimum(c_d) * 0.5;
        text = "$(disagree_pct)%\ndisagree",
        align = (:center, :center), fontsize = 14,
        color = :black, font = :bold)

    # Panel B: bar chart of detection rates
    labels_short = [
        "Uneq. var.\nσ=1.5",
        "Uneq. var.\nσ=2.0",
        "Logistic",
        "Heavy-tail\ndf=5",
        "Heavy-tail\ndf=3",
    ]

    b_rates = [
        ([100 * r.ppb_detection_rate for r in res6.part3], COL_PPB, "PPB"),
        ([100 * r.c_detection_rate   for r in res6.part3], COL_C_LL, "c"),
        ([100 * r.bpp_detection_rate for r in res6.part3], COL_BPP, "B''"),
        ([100 * r.br_detection_rate  for r in res6.part3], COL_BR, "Br"),
    ]

    # Order families by how far their J sits from Group A's, so the panel reads as
    # the J spread it is. Group A is the equal-variance Gaussian (dist_pairs[1]).
    J_A = let r = first(filter(x -> x.target_ppb == res6.target_ppb_3 &&
                                    x.label == "EVSDT (Gaussian)", res6.part1))
        r.H - r.F
    end
    J_B = [let lab = r.label
               q = first(filter(x -> x.target_ppb == res6.target_ppb_3 &&
                                     x.label == lab, res6.part1))
               q.H - q.F
           end for r in res6.part3]
    ord = sortperm(abs.(J_B .- J_A))

    ax_b = Axis(l2[1, 1],
        xlabel = "Group B distribution (ordered by |ΔJ| from Group A)",
        ylabel = "Detection rate (%)",
        title = "Group differences at fixed PPB",
        xticks = (1:5, labels_short[ord]),
        xticklabelsize = 9,
    )

    hlines!(ax_b, [5], color = (:gray, 0.5), linestyle = :dash)

    nser = length(b_rates)
    barw = 0.8 / nser
    for (j, (vals, col, _)) in enumerate(b_rates)
        xs = (1:5) .+ (j - (nser + 1) / 2) * barw
        barplot!(ax_b, xs, vals[ord], color = col, width = barw * 0.9)
    end

    # ΔJ annotation under each family
    for (i, k) in enumerate(ord)
        text!(ax_b, i, -6.0;
            text = "ΔJ=$(round(J_B[k] - J_A, digits = 2))",
            align = (:center, :top), fontsize = 8, color = :gray40)
    end

    Legend(l2[1, 1],
        vcat([PolyElement(color = col) for (_, col, _) in b_rates],
             [LineElement(color = :gray, linestyle = :dash)]),
        vcat([lab for (_, _, lab) in b_rates], ["5% nominal"]),
        tellwidth = false, tellheight = false,
        halign = :left, valign = :top,
        framevisible = false, labelsize = 11, nbanks = 2,
        margin = (10, 0, 0, 10))

    # Panel labels
    for (label, layout) in zip(["A", "B"], [l1, l2])
        Label(layout[1, 1, TopLeft()], label,
            fontsize = 26, font = :bold,
            padding = (0, 5, 5, 0), halign = :right)
    end

    resize_to_layout!(fig)
    safe_save("figures/fg_conceptual.png", fig, px_per_unit = 2)

    return fig
end


# ─── Main ─────────────────────────────────────────────────────────────────────

function generate_all()
    fig2 = figure2()
    fig3 = figure3()
    println("\nAll figures generated.")
    return (; fig2, fig3)
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_all()
end
