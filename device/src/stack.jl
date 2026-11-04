# Term batches built directly in the stacked form (docs/system/device.md §3, term axis): every field of
# a term type is one gather of a (B, n) index table, (C, B, 1, n), instead of n gathers of (C, B, 1)
# pieces concatenated along the term axis. The program then has one gather per field whatever the
# number of terms (no concatenations of term pieces, a trace that does not grow with the terms), and
# the commits are one scatter-add per state array.

"X (C, K) at the index table idx (B, n): (C, B, 1, n)."
G4(X, idx) = (C = size(X, 1); (B, n) = size(idx); rs(X[:, vec(idx)], C, B, 1, n))
"Static per-item vector v at the index table idx (B, n): (1, B, 1, n)."
S4(v, idx) = ((B, n) = size(idx); rs(v[vec(idx)], 1, B, 1, n))
"A (B, n) table as (1, B, 1, n)."
M4(m) = rs(m, 1, size(m, 1), 1, size(m, 2))
"(C, B, 1) as (C, B, 1, 1)."
c4(x) = rs(x, size(x, 1), size(x, 2), 1, 1)
"Columns of two (B, n) tables side by side (index tables of a round)."
hcat2(a, b) = cat(a, b; dims=2)
"X[:, idx] += V (C, B, 1, n) for the index table idx (B, n): one scatter-add."
scadd4(X, idx, V) = scadd(X, vec(idx), rs(V, size(V, 1), size(V, 2) * size(V, 4)))

"""
Stacked generator terms: fields (C|1, B, 1, n) or `nothing` (static zero), static per column: the
balance slot `slot`, the kink `kid` (0: linear), whether the linear slope moves (`mva`) and whether
the kinked part moves (`mvk`).
"""
function sgen(z4, n; y0, a=nothing, κ, w, mc, clo, chi, plo, hv, yk, ak=nothing, kid=zeros(Int, n), slot,
              mva=fill(a !== nothing, n), mvk=(kid .> 0) .& (ak !== nothing))
    F(x) = x === nothing ? nothing : x .+ z4
    return (y0=F(y0), a=F(a), κ=F(κ), w=F(w), mc=F(mc), clo=F(clo), chi=F(chi), plo=F(plo), hv=F(hv), yk=F(yk), ak=F(ak),
            kid=copy(kid), slot=copy(slot), mva=collect(mva), mvk=collect(mvk), cnt=Int.(mva) .+ Int.(mvk))
end

"Stacked commitment terms (arguments y, u, u1 each x0 + a t + ak φ_kid; per-column moving counts cy, cu, cu1)."
function suc(z4, n; y0, ay=nothing, u0, au=nothing, u10, au1=nothing, κ, w, gmin, su, nl, on,
             aky=nothing, kidy=zeros(Int, n), aku=nothing, kidu=zeros(Int, n), aku1=nothing, kidu1=zeros(Int, n),
             cy, cu, cu1)
    F(x) = x === nothing ? nothing : x .+ z4
    return (y0=F(y0), ay=F(ay), u0=F(u0), au=F(au), u10=F(u10), au1=F(au1), κ=F(κ), w=F(w), gmin=F(gmin), su=F(su), nl=F(nl),
            on=F(on), aky=F(aky), aku=F(aku), aku1=F(aku1), kidy=copy(kidy), kidu=copy(kidu), kidu1=copy(kidu1),
            cy=copy(cy), cu=copy(cu), cu1=copy(cu1))
end

"Stacked line terms: flows f = f0 + df t (oriented near → far), static slots `near`, `far` (0: none)."
sline(z4; f0, df, s, α, near, far, mv=fill(df !== nothing, length(near))) =
    (f0=f0 .+ z4, df=df === nothing ? nothing : df .+ z4, s=s .+ z4, α=α .+ z4, near=copy(near), far=copy(far), mv=collect(mv))

"Stacked SoC terms: (e, e₁) each linear with slopes ae, ae1 (`nothing`: static), static slots."
function ssoc(z4, n; e0, ae=nothing, e10, ae1=nothing, k, w, h, ηin, ηout, qt, mc, slot,
              mve=fill(ae !== nothing, n), mve1=fill(ae1 !== nothing, n))
    F(x) = x === nothing ? nothing : x .+ z4
    return (e0=F(e0), ae=F(ae), e10=F(e10), ae1=F(ae1), k=F(k), w=F(w), h=F(h), ηin=F(ηin), ηout=F(ηout), qt=F(qt), mc=F(mc),
            slot=copy(slot), mve=collect(mve), mve1=collect(mve1), cnt=Int.(mve) .+ Int.(mve1))
end

"Stacked kinks (C, B, 1, nk): requirement Q, change g, Jacobian data P, e, nm, active flag."
skinks(z4; Q, g, P, e, nm, on) = (Q=Q .+ z4, g=g .+ z4, P=P .+ z4, e=e .+ z4, nm=nm .+ z4, on=on .+ z4, n=size(Q, 4))

"""
A term batch from stacked parts (`sgen`, `suc`, `sline`, `ssoc`, `skinks`; `nothing` where absent) and
the balance-slot residuals r0 (C, B, 1, R) (`nothing`: no slots). z: (C, B, 1) zeros.
"""
function TermBatchS(z; gen=nothing, uc=nothing, link=nothing, line=nothing, soc=nothing, kk=nothing, r0=nothing)
    st = (z=z, gen=gen, uc=uc, link=link, line=line, soc=soc, kk=kk, r0=r0, nr=r0 === nothing ? 0 : size(r0, 4))
    return TermBatch(nothing, nothing, nothing, nothing, nothing, Any[], kk === nothing ? nothing : :stacked, st)
end

"φ_k(t) of every kink (C, B, 1, nk) at t (C, B, 1); nothing without kinks."
function kinkφ4(tb::TermBatch, t)
    kk = tb.st.kk
    kk === nothing && return nothing
    t4 = c4(t)
    return max.(kk.Q .+ kk.g .* t4, 0.0) .- max.(kk.Q, 0.0)
end

"""
Pro-rata data of the units u (C, B, 1, G) at the node slots of a site (columns grouped by slot: G
units per slot, nslot slots; validity `v`, outputs ω): P, Q, e, nm per slot (C, B, 1, nslot) and the
activity flag, for a requirement change g (C, B, 1, nslot) per unit t; Q = r (C, B, 1, nslot) + Σ κ clamp(ω).
"""
function prorata4(κ, ω, v, r, g, nslot::Int)
    C, B = size(r, 1), size(r, 2)
    G = size(κ, 4) ÷ nslot
    sum_slot(x) = rs(sum(rs(x .+ 0.0 .* κ, C, B, G, nslot); dims=3), C, B, 1, nslot)
    P = sum_slot(ifelse.(v, κ .* ω, 0.0))
    Qs = sum_slot(ifelse.(v, κ .* clamp01.(ω), 0.0))
    nu = sum_slot(ifelse.(v, 1.0, 0.0))
    Q = r .+ Qs
    act = (g .!= 0.0) .& (nu .> 0.0) .& (absd.(P) .>= 1e-6)
    return (P=P, Q=Q, e=P .- max.(Q, 0.0), nm=nu .- 1.0, act=act)
end
