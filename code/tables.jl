# tables.jl
# ─────────────────────────────────────────────────────────────────────────────
# Generate paper tables programmatically from empirical data.
# Writes Markdown pipe tables to tables/ for inclusion in ppb_paper.qmd.
#
# verify_numbers.jl calls all_tables() and fails if
# the files in tables/ differ from what this code produces now.
#
# Run: julia +1.12 --project=code code/tables.jl
# ─────────────────────────────────────────────────────────────────────────────

using Printf

include("colloff.jl")
include("colloff_wixted.jl")

const TABLE_DIR = joinpath(@__DIR__, "..", "tables")
const TABLE_FILES = ("tbl_empirical.md", "tbl_quartiles.md", "tbl_lineup_responses.md",
                     "tbl_lineup_sensitivity.md", "tbl_lineup_paired.md",
                     "tbl_cw_conditions.md", "tbl_cw_contrast.md")

# ─── Table 1: Boundary and aggregation summary ────────────────────────────
# Each function returns a Markdown pipe table, with no caption; the paper
# includes the file and supplies the caption and label, so the tables render
# in any output format.

"""Number with a typographic minus (U+2212) in place of the hyphen."""
signed(x::AbstractString) = replace(x, "-" => "−")

"""Approximate printed width of a cell: math delimiters and markup dropped."""
visible_width(s) = length(replace(s, r"\\text\{([^}]*)\}" => s"\1", r"[\$_{}]" => ""))

"""Pipe table with padded columns. Pandoc sets relative column widths from the
dashes in the separator row, so each column gets dashes in proportion to its
widest printed cell; headers then wrap only where the page forces it.
`align` holds one of 'l' or 'r' per column."""
function pipe_table(header, rows, align)
    cols = [[header[j]; [r[j] for r in rows]] for j in eachindex(header)]
    w = [maximum(length, c) for c in cols]
    dashes = [maximum(visible_width, c) + 2 for c in cols]   # + 2 for the cell inset
    line(cells) = "| " * join((rpad(c, w[j]) for (j, c) in enumerate(cells)), " | ") * " |\n"
    rule = "|" * join((a == 'l' ? ":" * "-"^d : "-"^d * ":" for (a, d) in zip(align, dashes)), "|") * "|\n"
    return line(header) * rule * join(line.(rows))
end

function table_empirical_md(subj, cond)
    n_subj = nrow(subj)
    bnd_subj = round(100 * mean(subj.at_boundary), digits=1)

    ppb_disc = round(mean(subj.ppb_val) - ppb(mean(subj.H), mean(subj.F)), digits=4)
    c_mean_ind = mean(subj.c_ll)
    c_of_means = criterion_c(mean(subj.H_ll), mean(subj.F_ll))
    c_disc = round(c_mean_ind - c_of_means, digits=4)
    c_disc_pct = round(100 * (c_mean_ind - c_of_means) / c_of_means, digits=1)

    n_cond = nrow(cond)
    bnd_cond = round(100 * mean(cond.at_boundary), digits=1)

    header = ["Level", "\$N\$", "Boundary %", "PPB agg. disc.", "\$c\$ agg. disc.", "\$c\$ disc. as % of \$c\$"]
    rows = [["Subject (4+4 trials)", "$n_subj", "$(bnd_subj)%", ppb_disc == 0 ? "0 (exact)" : string(ppb_disc),
             signed(string(c_disc)), "$(c_disc_pct)%"],
            ["Condition (2+2 trials)", "$n_cond", "$(bnd_cond)%", "---", "---", "---"]]
    return pipe_table(header, rows, "lrrrrr")
end

# ─── Table 2: Choosing-rate separation quartiles ───────────────────────────
# Ties in J at the cutpoints are split proportionally (colloff.jl), so the
# table does not depend on the row order of the data file.

function table_quartiles_md(subj)
    labels = ("Q1 (lowest)", "Q2", "Q3", "Q4 (highest)")

    header = ["Quartile", "\$n\$", "Mean \$H\$", "Mean \$F\$", "Mean \$J\$", "Boundary %", "Mean PPB",
              "Mean \$c_{\\text{LL}}\$"]
    rows = map(zip(labels, separation_quartiles(subj))) do (label, q)
        mc = @sprintf("%.3f", q.c_ll)
        [label, string(round(Int, q.n)), @sprintf("%.3f", q.H), @sprintf("%.3f", q.F),
         signed(@sprintf("%.2f", q.J)), @sprintf("%.0f%%", q.boundary_pct), @sprintf("%.3f", q.ppb),
         startswith(mc, "-") ? signed(mc) : "+" * mc]
    end
    return pipe_table(header, rows, "lrrrrrrr")
end

"""Observed response categories and the two numerator definitions."""
function table_lineup_responses_md(raw)
    header = ["Target", "Response", "Own-race", "Other-race", "Total", "Choose coding", "Perpetrator-only"]
    rows = Vector{String}[]
    for (target, response) in (("yes", "perpetrator"), ("yes", "foil"), ("yes", "reject"),
                               ("no", "foil"), ("no", "reject"))
        d = filter(r -> r.TargetPresent == target && r.IDResponse == response, raw)
        positive = response != "reject"
        legacy_positive = target == "yes" ? response == "perpetrator" : response == "foil"
        numerator = target == "yes" ? "H numerator" : "F numerator"
        push!(rows, [target == "yes" ? "Present" : "Absent", response,
                     string(count(==("ownRace"), d.OwnRace)),
                     string(count(==("otherRace"), d.OwnRace)), string(nrow(d)),
                     positive ? numerator : "Negative", legacy_positive ? numerator : "Negative"])
    end
    return pipe_table(header, rows, "llrrrll")
end

"""Primary choosing analysis and legacy identification sensitivity summary."""
function table_lineup_sensitivity_md(primary, legacy)
    header = ["Coding", "Level / race", "Mean H", "Mean F", "Mean H+F", "Mean J", "Mean c (LL)", "Boundary %"]
    rows = Vector{String}[]
    for (label, (subj, cond)) in (("Choose", primary), ("Perpetrator-only", legacy))
        own = filter(:OwnRace => ==("ownRace"), cond)
        other = filter(:OwnRace => ==("otherRace"), cond)
        for (scope, d) in (("Subject / both", subj), ("Condition / own", own), ("Condition / other", other))
            push!(rows, [label, scope, @sprintf("%.3f", mean(d.H)), @sprintf("%.3f", mean(d.F)),
                         @sprintf("%.3f", mean(d.ppb_val)), signed(@sprintf("%.3f", mean(d.J))),
                         signed(@sprintf("%.3f", mean(d.c_ll))), @sprintf("%.1f%%", 100 * mean(d.at_boundary))])
        end
    end
    return pipe_table(header, rows, "llrrrrrr")
end

"""Own-minus-other participant-paired differences and 95% t intervals."""
function table_lineup_paired_md(primary_cond, legacy_cond)
    header = ["Coding", "Quantity", "Mean difference", "95% CI"]
    rows = Vector{String}[]
    for (label, cond) in (("Choose", primary_cond), ("Perpetrator-only", legacy_cond))
        for (quantity, scale) in (("H+F", 1.0), ("(H+F)/2", 0.5))
            p = paired_ownrace_summary(cond; scale)
            push!(rows, [label, quantity, signed(@sprintf("%.4f", p.mean)),
                         signed(@sprintf("[%.4f, %.4f]", p.lo, p.hi))])
        end
    end
    return pipe_table(header, rows, "llrr")
end

const CW_LABELS = Dict("showup" => "Showup", "simultaneousShowup" => "Simultaneous showup", "control" => "Standard lineup")

"""Colloff and Wixted (2020): each condition's participants, mean rates, PPB and J."""
function table_cw_conditions_md()
    header = ["Experiment", "Condition", "Participants", "Mean H", "Mean F", "Mean H+F", "Mean J"]
    rows = Vector{String}[]
    for e in 1:3, r in condition_summary(e)
        push!(rows, [string(e), CW_LABELS[r.condition], @sprintf("%d", r.n), @sprintf("%.3f", r.H),
                     @sprintf("%.3f", r.F), @sprintf("%.3f", r.ppb), signed(@sprintf("%.3f", r.J))])
    end
    return pipe_table(header, rows, "llrrrrr")
end

"""Showup minus simultaneous showup, with Welch 95% intervals on participant-level values."""
function table_cw_contrast_md()
    header = ["Experiment", "Participants", "Quantity", "Difference", "95% CI", "p"]
    fmtp(p) = p < 0.001 ? "<.001" : replace(@sprintf("%.3f", p), r"^0" => "")
    rows = Vector{String}[]
    for e in 1:3
        r = showup_contrast(e)
        for (q, t) in (("H+F", r.ppb), ("J", r.J), ("c (log-linear)", r.c_ll))
            push!(rows, [string(e), @sprintf("%d", r.n), q, signed(@sprintf("%.3f", t.diff)),
                         signed(@sprintf("[%.3f, %.3f]", t.lo, t.hi)), fmtp(t.p)])
        end
    end
    return pipe_table(header, rows, "lrlrrr")
end

"""Current contents for each file in TABLE_FILES."""
function all_tables()
    primary = load_colloff()
    legacy = load_colloff(; response = :identification)
    subj, cond = primary
    return Dict("tbl_empirical.md" => table_empirical_md(subj, cond),
                "tbl_quartiles.md" => table_quartiles_md(subj),
                "tbl_lineup_responses.md" => table_lineup_responses_md(load_colloff_raw()),
                "tbl_lineup_sensitivity.md" => table_lineup_sensitivity_md(primary, legacy),
                "tbl_lineup_paired.md" => table_lineup_paired_md(cond, legacy[2]),
                "tbl_cw_conditions.md" => table_cw_conditions_md(),
                "tbl_cw_contrast.md" => table_cw_contrast_md())
end

# ─── Main ──────────────────────────────────────────────────────────────────

function main()
    println("Generating tables from Colloff et al. (2022) data...")
    mkpath(TABLE_DIR)
    for (name, md) in all_tables()
        path = joinpath(TABLE_DIR, name)
        write(path, md)
        println("  Wrote: $path")
    end
    println("Done.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
