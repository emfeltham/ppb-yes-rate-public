# figure_wrap.jl
# ─────────────────────────────────────────────────────────────────────────────
# ROC illustrations with CairoMakie (Julia)
# Plots:
#   (1) J / PPB orthogonal lines through an observed (F, H)
#   (2) iso-d′ curves showing SDT warping
#
# Requirements:
#   ] add CairoMakie Distributions
# ─────────────────────────────────────────────────────────────────────────────

using Distributions
using CairoMakie
using ColorSchemes

# Convenience: standard normal functions
const ϕ = Normal()
Φinv(p) = quantile(ϕ, p)
Φ(x) = cdf(ϕ, x)

# Example operating point (False Positive Rate F, True Positive Rate H)
F_ex = 0.30
H_ex = 0.80

# Derived metrics
J = H_ex - F_ex
PPB = H_ex + F_ex
d_ex = Φinv(H_ex) - Φinv(F_ex)
c_ex = -0.5 * (Φinv(H_ex) + Φinv(F_ex))

# Helper: clip lines to [0,1]^2 box
function clipped_line(xs, ys)
    # Return only points that fall within ROC square
    mask = (xs .>= 0) .& (xs .<= 1) .& (ys .>= 0) .& (ys .<= 1)
    return (xs[mask], ys[mask])
end

# Helper: transform (F, H) coordinates to (J, PPB) coordinates
function transform_to_j_ppb(F_vals, H_vals)
    J_vals = H_vals .- F_vals
    PPB_vals = H_vals .+ F_vals
    return (J_vals, PPB_vals)
end

##
fg = Figure();
l1 = GridLayout(fg[1, 1])
l2 = GridLayout(fg[1, 2])
l3 = GridLayout(fg[1, 3])

# ── (1) ROC with J, PPB as orthogonal axes ───────────────────────────────────
ax_1 = let l = l1
    ax = Axis(l[1, 1],
        xlabel = "False Positive Rate (FPR)",
        ylabel = "True Positive Rate (TPR)",
        title  = "ROC space",
        aspect = 1
    )

    # Chance line y = x
    lines!(ax, [0, 1], [0, 1], color = (:gray, 0.6), linestyle = :dot, label = "Chance")

    # The observed point
    scatter!(
        ax, [F_ex], [H_ex], markersize = 14, color = :black,
        label = "Observed point"
    )

    # Constant-J line: H = F + J
    xs = range(0, 1, length = 400)
    ys_J = xs .+ J
    xsJ, ysJ = clipped_line(xs, ys_J)
    lines!(ax, xsJ, ysJ, linestyle = :dash, color = ColorSchemes.batlowS[1], label = "Constant J = $(round(J, digits=2))")

    # Constant-PPB line: H = -F + PPB
    ys_PPB = .-xs .+ PPB
    xsP, ysP = clipped_line(xs, ys_PPB)
    lines!(ax, xsP, ysP, linestyle = :dash, color = ColorSchemes.batlowS[2], label = "Constant PPB = $(round(PPB, digits=2))")

    # PPB = 1 reference line (unbiased locus: H + F = 1, i.e., H = 1 - F)
    ys_ppb1 = 1.0 .- xs
    lines!(ax, collect(xs), ys_ppb1, color = (:gray, 0.3), linestyle = :dashdot, label = "PPB = 1 (unbiased)")

    Legend(l[2, 1], ax, framevisible = false)
    xlims!(ax, 0, 1)
    ylims!(ax, 0, 1)

end

# ── (2) ROC with iso-d′ curves (EVSDT warping) ───────────────────────────────
fg_2 = let l = l2
    ax = Axis(
        l[1, 1],
        xlabel = "False Positive Rate (FPR)",
        ylabel = "True Positive Rate (TPR)",
        title  = "SDT parameterization",
        aspect = 1
    )

    # Chance line
    lines!(ax, [0, 1], [0, 1], color = (:gray, 0.6), linestyle = :dot) #, label = "Chance")

    # A set of d′ values to show
    dvals = [0.5, 1.0, 1.5, 2.0]
    # A set of c values to show
    cvals = [-1.0, -0.5, 0.0, 0.5, 1.0]

    # For EVSDT with equal variance, iso-d′ curves satisfy:
    #   H = Φ( d′ + Φ^{-1}(F) )
    Fgrid = range(1e-3, 1 - 1e-3, length = 500)
    for (i, d′) in enumerate(dvals)
        Hcurve = [Φ(d′ + Φinv(F)) for F in Fgrid]
        # Use Batlow colormap with different positions for each curve
        color_pos = (i - 1) / (length(dvals) - 1)
        lines!(
            ax, Fgrid, Hcurve, color = ColorSchemes.batlow[color_pos],
            label = "d′ = $(round(d′, digits=1))"
        )
    end

    # Iso-c curves: H = Φ(-2c - Φ^{-1}(F))
    for (i, c) in enumerate(cvals)
        Hcurve = [Φ(-2c - Φinv(F)) for F in Fgrid]
        # Use BatlowS colormap for c curves to distinguish from d′ curves
        color_pos = (i - 1) / (length(cvals) - 1)
        lines!(
            ax, Fgrid, Hcurve, color = ColorSchemes.batlowS[color_pos],
            linestyle = :dash, label = "c = $(round(c, digits=1))"
        )
    end

    # Mark the example point
    scatter!(
        ax, [F_ex], [H_ex], markersize = 14, color = :black,
        label = "Observed point"
    )

    Legend(l[2, 1], ax, framevisible = false, nbanks = 2)
    xlims!(ax, 0, 1)
    ylims!(ax, 0, 1)

end

# ── (3) J, PPB space with rotated iso-d′ and iso-c curves ───────────────────
fg_3 = let l = l3
    ax = Axis(
        l[1, 1],
        xlabel = "PPB",
        ylabel = "J (Youden's index)",
        title  = "PPB, J coordinates",
        aspect = 1
    )

    # Grids and values (same as panel 2)
    Fgrid = range(1e-3, 1 - 1e-3, length = 500)
    dvals = [0.5, 1.0, 1.5, 2.0]
    cvals = [-1.0, -0.5, 0.0, 0.5, 1.0]

    # Transform iso-d′ curves to J,PPB space
    for (i, d′) in enumerate(dvals)
        Hcurve = [Φ(d′ + Φinv(F)) for F in Fgrid]
        J_vals, PPB_vals = transform_to_j_ppb(Fgrid, Hcurve)

        # Filter to reasonable bounds for PPB,J space
        valid_mask = (J_vals .>= -1) .& (J_vals .<= 1) .& (PPB_vals .>= 0) .& (PPB_vals .<= 2)
        J_filtered = J_vals[valid_mask]
        PPB_filtered = PPB_vals[valid_mask]

        if length(J_filtered) > 0
            color_pos = (i - 1) / (length(dvals) - 1)
            lines!(
                ax, PPB_filtered, J_filtered,
                color = ColorSchemes.batlow[color_pos],
                label = "d′ = $(round(d′, digits=1))"
            )
        end
    end

    # Transform iso-c curves to J,PPB space
    for (i, c) in enumerate(cvals)
        Hcurve = [Φ(-2c - Φinv(F)) for F in Fgrid]
        J_vals, PPB_vals = transform_to_j_ppb(Fgrid, Hcurve)

        # Filter to reasonable bounds for PPB,J space
        valid_mask = (J_vals .>= -1) .& (J_vals .<= 1) .& (PPB_vals .>= 0) .& (PPB_vals .<= 2)
        J_filtered = J_vals[valid_mask]
        PPB_filtered = PPB_vals[valid_mask]

        if length(J_filtered) > 0
            color_pos = (i - 1) / (length(cvals) - 1)
            lines!(
                ax, PPB_filtered, J_filtered,
                color = ColorSchemes.batlowS[color_pos],
                linestyle = :dash, label = "c = $(round(c, digits=1))"
            )
        end
    end

    # Transform and mark the example point
    J_ex, PPB_ex = transform_to_j_ppb([F_ex], [H_ex])
    scatter!(
        ax, PPB_ex, J_ex, markersize = 14, color = :black,
        label = "Observed point"
    )


    Legend(l[2, 1], ax, framevisible = false, nbanks = 2)
    xlims!(ax, 0, 2)
    ylims!(ax, -1, 1)

end

# Add panel labels
for (label, layout) in zip(["A", "B", "C"], [l1, l2, l3])
    Label(layout[1, 1, TopLeft()], label,
        fontsize = 26,
        font = :bold,
        padding = (0, 5, 5, 0),
        halign = :right)
end

resize_to_layout!(fg)
# Render to a temp file and copy in: a failed direct write into the iCloud-synced
# figures/ directory deletes the target (see code/README.md).
let tmp = joinpath(mktempdir(), "fg_warp.pdf")
    save(tmp, fg)
    cp(tmp, "figures/fg_warp.pdf"; force = true)
end

fg
