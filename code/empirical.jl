# empirical.jl
# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Empirical demonstration using Colloff, Flowe, Smith et al. (2022)
# Experiment 1 — eyewitness lineup identification with own-race/other-race design
#
# Data: 220 participants × 8 trials (4 target-present, 4 target-absent)
# Source: pyWitness published data (GitHub)
#
# Demonstrates PPB advantages over c:
#   1. Boundary stability — c undefined at extreme choosing rates
#   2. Aggregation — mean(c) ≠ c(mean rates)
#   3. Group comparison — own-race vs. other-race bias
#   4. Sensitivity to the response event and paired own-minus-other uncertainty
#
# Requirements: CSV, DataFrames, Distributions, Statistics (code/Project.toml)
# ─────────────────────────────────────────────────────────────────────────────

# All coding, corrections and measures come from the table/verifier loader.
include("colloff.jl")

load_data(; response = :choose) = load_colloff_raw(; response)
aggregate_trials(raw, groupcols) = aggregate_colloff(raw, groupcols)

# ─── Analysis 1: Boundary stability ───────────────────────────────────────

function analysis_boundary(subj)
    println("=" ^ 70)
    println("ANALYSIS 1: BOUNDARY STABILITY")
    println("=" ^ 70)
    println("(Per subject, collapsing across race conditions: 4 TP + 4 TA trials)")
    println()

    n = nrow(subj)
    n_boundary = sum(subj.at_boundary)
    n_nan_c = sum(isnan.(subj.c_raw))

    println("Total subjects: $n")
    println("At boundary (H or F ∈ {0,1}): $n_boundary ($(round(100*n_boundary/n, digits=1))%)")
    println("  H=0: $(sum(subj.H .== 0))  H=1: $(sum(subj.H .== 1))")
    println("  F=0: $(sum(subj.F .== 0))  F=1: $(sum(subj.F .== 1))")
    println("c undefined (NaN): $n_nan_c ($(round(100*n_nan_c/n, digits=1))%)")
    n_nan_br = sum(subj.br_undefined)
    println("Br undefined (H=1 and F=0): $n_nan_br ($(round(100*n_nan_br/n, digits=1))%)")
    println()

    bsub = filter(:at_boundary => identity, subj)
    interior = filter(:at_boundary => !, subj)

    if nrow(interior) > 0
        println("Interior subjects (n=$(nrow(interior))):")
        valid_c = filter(x -> !isnan(x), interior.c_raw)
        println("  c range: $(round(minimum(valid_c), digits=3)) to $(round(maximum(valid_c), digits=3))")
        println("  PPB range: $(round(minimum(interior.ppb_val), digits=3)) to $(round(maximum(interior.ppb_val), digits=3))")
        println()
    end

    if nrow(bsub) > 0
        println("Boundary subjects (n=$(nrow(bsub))):")
        println("  PPB range: $(round(minimum(bsub.ppb_val), digits=3)) to $(round(maximum(bsub.ppb_val), digits=3))")
        println("  c is NaN for all — must apply correction")
        println()

        println("Correction method disagreement (boundary subjects only):")
        println("  Mean |c_ll - c_hn|: $(round(mean(abs.(bsub.c_ll .- bsub.c_hn)), digits=4))")
        println("  Max  |c_ll - c_hn|: $(round(maximum(abs.(bsub.c_ll .- bsub.c_hn)), digits=4))")
        println("  PPB needs no boundary rate correction")
        println()
    end
end

# ─── Analysis 2: Aggregation ──────────────────────────────────────────────

function analysis_aggregation(subj)
    println("=" ^ 70)
    println("ANALYSIS 2: AGGREGATION DISCREPANCY")
    println("=" ^ 70)
    println()

    # PPB: mean of individual PPBs vs. PPB of mean rates
    mean_ppb = mean(subj.ppb_val)
    ppb_of_means = ppb(mean(subj.H), mean(subj.F))
    ppb_disc = mean_ppb - ppb_of_means

    println("PPB:")
    println("  mean(PPB_i):  $(round(mean_ppb, digits=4))")
    println("  PPB(mean H, mean F): $(round(ppb_of_means, digits=4))")
    println("  Discrepancy:  $(round(ppb_disc, digits=6)) (expected: exactly 0)")
    println()

    # c with log-linear correction (all subjects computable)
    mean_c_ll = mean(subj.c_ll)
    c_of_means_ll = criterion_c(mean(subj.H_ll), mean(subj.F_ll))
    c_disc_ll = mean_c_ll - c_of_means_ll

    println("c (log-linear, all $(nrow(subj)) subjects):")
    println("  mean(c_i):    $(round(mean_c_ll, digits=4))")
    println("  c(mean H, mean F): $(round(c_of_means_ll, digits=4))")
    println("  Discrepancy:  $(round(c_disc_ll, digits=4))")
    if c_of_means_ll != 0
        println("  As % of c(means): $(round(100*c_disc_ll/c_of_means_ll, digits=1))%")
    end
    println()

    # B'' with log-linear correction
    valid_bpp = filter(x -> !isnan(x), subj.bpp_ll)
    mean_bpp = mean(valid_bpp)
    bpp_of_means = b_double_prime(mean(subj.H_ll), mean(subj.F_ll))
    bpp_disc = mean_bpp - bpp_of_means

    println("B'' (log-linear, n=$(length(valid_bpp)) with valid values):")
    println("  mean(B''_i):  $(round(mean_bpp, digits=4))")
    println("  B''(mean H, mean F): $(round(bpp_of_means, digits=4))")
    println("  Discrepancy:  $(round(bpp_disc, digits=4))")
    println()

    # Br with log-linear correction (same corrected rates used for c and B'')
    valid_br = filter(x -> !isnan(x), subj.br_ll)
    mean_br = mean(valid_br)
    br_of_means = b_r(mean(subj.H_ll), mean(subj.F_ll))
    br_disc = mean_br - br_of_means

    println("Br (log-linear, n=$(length(valid_br)) with valid values):")
    println("  mean(Br_i):   $(round(mean_br, digits=4))")
    println("  Br(mean H, mean F): $(round(br_of_means, digits=4))")
    println("  Discrepancy:  $(round(br_disc, digits=4))")
    if br_of_means != 0
        println("  As % of Br(means): $(round(100*br_disc/br_of_means, digits=1))%")
    end
    println("  Br undefined on raw rates (H=1, F=0): $(sum(subj.br_undefined)) of $(nrow(subj))")
    println()
end

# ─── Analysis 3: Group comparisons (own-race vs. other-race) ──────────────

function analysis_group_comparison(subj_by_cond)
    println("=" ^ 70)
    println("ANALYSIS 3: GROUP COMPARISON (OWN-RACE vs. OTHER-RACE)")
    println("=" ^ 70)
    println("(Per subject × race condition: 2 TP + 2 TA trials)")
    println()

    own = filter(:OwnRace => ==("ownRace"), subj_by_cond)
    other = filter(:OwnRace => ==("otherRace"), subj_by_cond)

    for (label, grp) in [("Own-race lineups", own), ("Other-race lineups", other)]
        n_bnd = sum(grp.at_boundary)
        valid_bpp = filter(x -> !isnan(x), grp.bpp)
        println("$label (n=$(nrow(grp))):")
        println("  Mean H: $(round(mean(grp.H), digits=3))  Mean F: $(round(mean(grp.F), digits=3))")
        println("  Mean PPB:       $(round(mean(grp.ppb_val), digits=4))")
        println("  Mean J:         $(round(mean(grp.J), digits=4)) (TP minus TA response rate)")
        println("  Mean c_ll:      $(round(mean(grp.c_ll), digits=4))")
        println("  Mean B''_ll:    $(round(mean(grp.bpp_ll), digits=4))")
        println("  At boundary:    $n_bnd/$(nrow(grp)) ($(round(100*n_bnd/nrow(grp), digits=1))%)")
        println()
    end

    # Direction of group difference
    ppb_diff = mean(own.ppb_val) - mean(other.ppb_val)
    c_diff = mean(own.c_ll) - mean(other.c_ll)
    bpp_diff = mean(own.bpp_ll) - mean(other.bpp_ll)

    dir(d) = d > 0 ? "own > other" : d < 0 ? "other > own" : "equal"

    println("Group differences (own - other):")
    println("  PPB:  $(round(ppb_diff, digits=4))  ($(dir(ppb_diff)))")
    println("  c_ll: $(round(c_diff, digits=4))  (criterion value: $(dir(c_diff)); lower c indicates more positive responding)")
    println("  B''_ll: $(round(bpp_diff, digits=4))  (index value: $(dir(bpp_diff)))")
    println()

    agree_ppb_c = sign(ppb_diff) == -sign(c_diff)
    println("  PPB and c agree on response tendency (opposite signs): $agree_ppb_c")
    println("  Higher PPB and lower c each indicate more positive responding.")
    println()

    for (label, scale) in (("PPB", 1.0), ("PPB/2", 0.5))
        p = paired_ownrace_summary(subj_by_cond; scale)
        println("  Paired $label own-minus-other: $(round(p.mean, digits=4)); 95% t CI [$(round(p.lo, digits=4)), $(round(p.hi, digits=4))], n=$(p.n), df=$(p.df)")
    end
    println()

    # Break down by subject race
    println("─── By subject race ───")
    for race in ["caucasian", "south_asian"]
        rsub = filter(:SubjectRace => ==(race), subj_by_cond)
        rown = filter(:OwnRace => ==("ownRace"), rsub)
        rother = filter(:OwnRace => ==("otherRace"), rsub)
        d_ppb = mean(rown.ppb_val) - mean(rother.ppb_val)
        d_c = mean(rown.c_ll) - mean(rother.c_ll)
        println("\n  $race subjects (n=$(nrow(rown)) per condition):")
        println("    Own-race:   H=$(round(mean(rown.H), digits=3))  F=$(round(mean(rown.F), digits=3))  PPB=$(round(mean(rown.ppb_val), digits=3))  c_ll=$(round(mean(rown.c_ll), digits=3))")
        println("    Other-race: H=$(round(mean(rother.H), digits=3))  F=$(round(mean(rother.F), digits=3))  PPB=$(round(mean(rother.ppb_val), digits=3))  c_ll=$(round(mean(rother.c_ll), digits=3))")
        println("    Δ PPB: $(round(d_ppb, digits=4))  Δ c_ll: $(round(d_c, digits=4))  Same response tendency: $(sign(d_ppb) == -sign(d_c))")
    end
    println()
end

"""Show exactly which response categories drive the coding sensitivity."""
function analysis_response_sensitivity()
    println("=" ^ 70)
    println("RESPONSE CODING SENSITIVITY")
    println("=" ^ 70)
    raw = load_colloff_raw()
    counts = combine(groupby(raw, [:OwnRace, :TargetPresent, :IDResponse]), nrow => :count)
    show(stdout, MIME("text/plain"), sort(counts, [:OwnRace, :TargetPresent, :IDResponse]); allrows = true)
    println("\n")
    for response in (:choose, :identification)
        subj, cond = load_colloff(; response)
        println("Coding: $response")
        for (label, d) in (("Subject / both", subj),
                           ("Condition / own", filter(:OwnRace => ==("ownRace"), cond)),
                           ("Condition / other", filter(:OwnRace => ==("otherRace"), cond)))
            println("  $label: H=$(round(mean(d.H), digits=4)), F=$(round(mean(d.F), digits=4)), H+F=$(round(mean(d.ppb_val), digits=4)), J=$(round(mean(d.J), digits=4)), c_LL=$(round(mean(d.c_ll), digits=4)), boundary=$(sum(d.at_boundary))/$(nrow(d))")
        end
        for (quantity, scale) in (("H+F", 1.0), ("(H+F)/2", 0.5))
            p = paired_ownrace_summary(cond; scale)
            println("  Paired $quantity own-minus-other: $(round(p.mean, digits=4)); 95% t CI [$(round(p.lo, digits=4)), $(round(p.hi, digits=4))], n=$(p.n), df=$(p.df)")
        end
    end
    println("Primary choosing PPB describes identification propensity, not culprit identification accuracy.")
    println("The legacy H+F combines TP culprit IDs with TA foil IDs, so it is not a single consistently defined binary response propensity.")
    println()
end

# ─── Analysis 4: Aggregation within conditions ────────────────────────────

function analysis_conditional_aggregation(subj_by_cond)
    println("=" ^ 70)
    println("ANALYSIS 4: AGGREGATION DISCREPANCY BY CONDITION")
    println("=" ^ 70)
    println()

    for cond in ["ownRace", "otherRace"]
        csub = filter(:OwnRace => ==(cond), subj_by_cond)
        println("$cond (n=$(nrow(csub)), 2 TP + 2 TA per subject):")

        # PPB
        mean_ppb = mean(csub.ppb_val)
        ppb_of_means = ppb(mean(csub.H), mean(csub.F))
        println("  PPB discrepancy: $(round(mean_ppb - ppb_of_means, digits=6))")

        # c (log-linear)
        mean_c = mean(csub.c_ll)
        c_of_means = criterion_c(mean(csub.H_ll), mean(csub.F_ll))
        disc = mean_c - c_of_means
        println("  c_ll discrepancy: $(round(disc, digits=4))")
        if c_of_means != 0
            println("  As % of c(means): $(round(100*disc/c_of_means, digits=1))%")
        end
        println()
    end
end

# ─── Summary ───────────────────────────────────────────────────────────────

function print_summary(subj, subj_by_cond)
    println("=" ^ 70)
    println("SUMMARY: COLLOFF ET AL. (2022) EXP 1 — PPB vs. c vs. B''")
    println("=" ^ 70)
    println()

    n = nrow(subj)
    n_bnd = sum(subj.at_boundary)
    n_cond = nrow(subj_by_cond)
    n_bnd_cond = sum(subj_by_cond.at_boundary)

    println("Subject-level (4 TP + 4 TA, N=$n):")
    println("  Boundary subjects: $n_bnd/$n ($(round(100*n_bnd/n, digits=1))%)")
    ppb_disc = mean(subj.ppb_val) - ppb(mean(subj.H), mean(subj.F))
    c_disc = mean(subj.c_ll) - criterion_c(mean(subj.H_ll), mean(subj.F_ll))
    br_disc = mean(filter(!isnan, subj.br_ll)) - b_r(mean(subj.H_ll), mean(subj.F_ll))
    println("  Aggregation discrepancy — PPB: $(round(ppb_disc, digits=6))  c_ll: $(round(c_disc, digits=4))  Br_ll: $(round(br_disc, digits=4))")
    n_br_undef = sum(subj.br_undefined)
    println("  Br undefined (H=1 & F=0): $n_br_undef/$n ($(round(100*n_br_undef/n, digits=1))%)")
    println()

    println("Condition-level (2 TP + 2 TA, N=$n_cond):")
    println("  Boundary observations: $n_bnd_cond/$n_cond ($(round(100*n_bnd_cond/n_cond, digits=1))%)")
    n_br_undef_cond = sum(subj_by_cond.br_undefined)
    println("  Br undefined (H=1 & F=0): $n_br_undef_cond/$n_cond ($(round(100*n_br_undef_cond/n_cond, digits=1))%)")
    ppb_disc_cond = mean(subj_by_cond.ppb_val) - ppb(mean(subj_by_cond.H), mean(subj_by_cond.F))
    c_disc_cond = mean(subj_by_cond.c_ll) - criterion_c(mean(subj_by_cond.H_ll), mean(subj_by_cond.F_ll))
    br_disc_cond = mean(filter(!isnan, subj_by_cond.br_ll)) - b_r(mean(subj_by_cond.H_ll), mean(subj_by_cond.F_ll))
    println("  Aggregation discrepancy — PPB: $(round(ppb_disc_cond, digits=6))  c_ll: $(round(c_disc_cond, digits=4))  Br_ll: $(round(br_disc_cond, digits=4))")

    own = filter(:OwnRace => ==("ownRace"), subj_by_cond)
    other = filter(:OwnRace => ==("otherRace"), subj_by_cond)
    ppb_diff = mean(own.ppb_val) - mean(other.ppb_val)
    c_diff = mean(own.c_ll) - mean(other.c_ll)
    println("  Own vs. other-race — PPB: $(round(ppb_diff, digits=4))  c_ll: $(round(c_diff, digits=4))  Agree: $(sign(ppb_diff) == -sign(c_diff))")
    println()

    println("Key findings:")
    println("  1. c is undefined for $(round(100*n_bnd/n, digits=0))% of subjects — PPB defined for all")
    println("  2. Correction method choice shifts c by up to $(round(maximum(abs.(subj[subj.at_boundary, :].c_ll .- subj[subj.at_boundary, :].c_hn)), digits=2)) for boundary subjects")
    println("  3. PPB aggregation discrepancy is exactly 0; c discrepancy is nonzero")
    println("  4. PPB and c agree on response tendency: $(sign(ppb_diff) == -sign(c_diff)); $(round(100*n_bnd_cond/n_cond, digits=1))% of condition-level observations have boundary rates")
    println()
end

# ─── Main ──────────────────────────────────────────────────────────────────

function main(; response = :choose)
    println("Loading Colloff, Flowe, Smith et al. (2022) Experiment 1...")
    println("Response coding: $response")
    println(response == :choose ?
        "Positive response = any lineup identification; J separates TP and TA choosing rates, not culprit identification accuracy." :
        "Legacy sensitivity: TP perpetrator IDs plus TA foil IDs; H+F sums different response events.")
    raw = load_data(; response)
    println("  $(nrow(raw)) trials from $(length(unique(raw.ParticipantId))) participants\n")

    # Subject-level: collapse across race conditions (4 TP + 4 TA)
    subj = aggregate_trials(raw, [:ParticipantId, :SubjectRace])
    println("Subject-level: $(nrow(subj)) subjects (4 TP + 4 TA each)\n")

    # Condition-level: per subject × own/other race (2 TP + 2 TA)
    subj_by_cond = aggregate_trials(raw, [:ParticipantId, :SubjectRace, :PerpRace, :OwnRace])
    println("Condition-level: $(nrow(subj_by_cond)) observations (2 TP + 2 TA each)\n")

    analysis_boundary(subj)
    analysis_aggregation(subj)
    analysis_group_comparison(subj_by_cond)
    analysis_conditional_aggregation(subj_by_cond)
    print_summary(subj, subj_by_cond)
    analysis_response_sensitivity()

    return (raw=raw, subj=subj, subj_by_cond=subj_by_cond)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
