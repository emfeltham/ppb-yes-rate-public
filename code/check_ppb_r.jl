# check_ppb_r.jl
# Parity check: code/ppb.R against inference.jl on a grid including boundary
# counts. Requires Rscript; not part of make.jl.
# Run: julia +1.12 --project=code code/check_ppb_r.jl

include(joinpath(@__DIR__, "inference.jl"))

cases = Tuple{Int,Int,Int,Int}[]
for (ns, nn) in ((10, 10), (20, 15), (50, 50)),
    h in (0, 1, ns ÷ 2, ns - 1, ns), f in (0, 1, nn ÷ 2, nn - 1, nn)
    push!(cases, (h, ns, f, nn))
end
unique!(cases)

# Julia reference values, then R values for the same cases, compared in R.
ref = joinpath(mktempdir(), "ref.txt")
open(ref, "w") do io
    for (h, ns, f, nn) in cases
        vals = Float64[]
        for ci in (ppb_interval(h, ns, f, nn), wald_interval(h, ns, f, nn),
                   adjusted_ppb_interval(h, ns, f, nn),
                   ppb_difference_interval(h, ns, f, nn, nn ÷ 2, ns, ns ÷ 2, nn))
            push!(vals, ci.lower, ci.upper)
        end
        println(io, join([h, ns, f, nn], " "), " ", join(vals, " "))
    end
end

rcode = """
source('$(joinpath(@__DIR__, "ppb.R"))')
d <- as.matrix(read.table('$ref'))
worst <- 0
for (i in seq_len(nrow(d))) {
  h <- d[i,1]; ns <- d[i,2]; f <- d[i,3]; nn <- d[i,4]
  r <- c(ppb_interval(h,ns,f,nn), wald_interval(h,ns,f,nn),
         adjusted_ppb_interval(h,ns,f,nn),
         ppb_difference_interval(h,ns,f,nn, nn %/% 2, ns, ns %/% 2, nn))
  worst <- max(worst, abs(unname(r) - d[i,5:12]))
}
cat('cases:', nrow(d), ' max abs difference:', worst, '\\n')
quit(status = if (worst < 1e-10) 0 else 1)
"""
run(`Rscript -e $rcode`)
println("R port matches inference.jl")
