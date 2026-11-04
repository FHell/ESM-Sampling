# Summary of the replica runs (ledger E23, E24): carrier totals and penalty-free cost against
# step, probe gaps and equilibrated scores per carrier, weather features of the replicas
# against the prior draws, cost per step. Needs `featnames` (replicas.jl) and P in the session,
# and results/replicas/prior_test.jls for the prior reference.
using Serialization, Statistics, Printf

function rep_summary(file; names_k=["onwind", "solar", "offwind", "gas", "battery", "H2"], win=50, io=stdout)
    D = deserialize(file)
    tr, pb = D.trace, D.probes
    n = length(tr)
    @printf(io, "%s: %s start, λ = %g, M = %d, %d steps, %.0f s (%.2f s per step incl. probes)\n", file, D.kind, D.λ, D.M,
            n, D.t, D.t / n)
    println(io, "  step   totals GW [", join(names_k, ", "), "]   capex   pf cost (mean over replicas)   shed MWh")
    for i in unique(vcat(1, 10:10:n ÷ 2, (n ÷ 2):25:n, n))
        x = tr[i]
        @printf(io, "  %4d  [%s]  %.4e  %.4e  %.3g\n", x.it, join([@sprintf("%6.1f", t / 1e3) for t in x.tot], ", "),
                x.capex, mean(x.pfcost), mean(x.shed))
    end
    lw = max(1, n - win + 1):n
    tot = reduce(hcat, [x.tot for x in tr[lw]]) ./ 1e3
    @printf(io, "  last %d steps: totals [%s] ± [%s] GW, pf cost %.4e, capex %.4e, op %.4e, shed %.3g MWh\n", length(lw),
            join([@sprintf("%.1f", x) for x in mean(tot, dims=2)], ", "), join([@sprintf("%.1f", x) for x in std(tot, dims=2)], ", "),
            mean(mean(x.pfcost) for x in tr[lw]), mean(x.capex for x in tr[lw]), mean(mean(x.op) for x in tr[lw]),
            mean(mean(x.shed) for x in tr[lw]))
    if !isempty(pb)
        G = reduce(hcat, [p.gap for p in pb]); Req = reduce(hcat, [p.rs_eq for p in pb]); Fk = reduce(hcat, [p.fk for p in pb])
        println(io, "  probes (it: gap/fk per carrier, ηw after)")
        for p in pb
            @printf(io, "   %4d rep %d  [%s]  ηw [%s]\n", p.it, p.replica, join([@sprintf("%+.2f", x) for x in p.gap], ", "),
                    join([@sprintf("%.1e", x) for x in p.η1], ", "))
        end
        h = max(1, length(pb) ÷ 2 + 1):length(pb)
        @printf(io, "  second half of the probes (%d): mean gap/fk [%s]\n", length(h), join([@sprintf("%+.2f", x) for x in mean(G[:, h], dims=2)], ", "))
        @printf(io, "     ⟨eq r ŝ_k⟩ [%s]\n     ⟨β_k ∂K⟩  [%s]\n     ratio [%s]\n",
                join([@sprintf("%.2e", x) for x in mean(Req[:, h], dims=2)], ", "), join([@sprintf("%.2e", x) for x in mean(Fk[:, h], dims=2)], ", "),
                join([@sprintf("%.2f", x) for x in mean(Req[:, h], dims=2) ./ mean(Fk[:, h], dims=2)], ", "))
    end
    return D
end

"Weather features of all replicas over the run against the prior draws (mean, sd, z of the replica mean)."
function rep_weather(Ds, exact, fnames; io=stdout)
    X = reduce(hcat, [f for D in Ds for x in D.trace for f in x.feats])
    nf = size(X, 1)
    for j in 1:nf
        @printf(io, "  %-16s prior %9.3f ± %7.3f   replicas %9.3f ± %7.3f  (min %.3f max %.3f)\n", fnames[j], mean(exact[j, :]),
                std(exact[j, :]), mean(X[j, :]), std(X[j, :]), minimum(X[j, :]), maximum(X[j, :]))
    end
end
