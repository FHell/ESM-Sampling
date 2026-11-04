# Dense angle directions with compensation (CPU `prep_tdir!`, `prep_tline!`, `prep_amode!`): the
# transport modes `tmode`, the binding-line normals `tline` and the learned modes `amode`, and their
# uncompensated (…0) and learned-compensation (…L) variants, docs/system/moves/transport.md, learned.md.
#
#   :tdir   T2 (share-preserving, one kink per node, Jacobian step): tmode, tline, amode
#   :tdir0  none (the residuals absorb the flow change): tmode0, tline0, amode0
#   :tdirL  LINEAR (learned coefficients): tmodeL, tlineL, amodeL
#
# A site is one step τ of a time class with, per chain, a node direction d (N) and the requirement
# change g (N) per unit t of the nodes that compensate (g = 0 elsewhere), host data in the index table
# (`dv`, `gv`, C·N per site): the directions depend on the chain's (k, x) (frames) and on the learned
# modes only, so they are drawn on the host (`TDIR_SOURCE`), one per site and chain, and refreshed
# between device calls with the same table shapes (`tdir_refresh`). Terms of a site: every line (static bus slots), every node's balance slot,
# G unit columns per node (and their commitment terms), one kink per node (T2).

const TDIR_FAMILIES = (:tdir, :tdir0, :tdirL)
tdir_rule(f::Symbol) = f == :tdir ? :T2 : f == :tdirL ? :L : :none

"Topology of the plan being built (static line slots of the dense angle families)."
const TDIR_TOPO = Ref{Any}(nothing)

"""
Host source of the directions: (fam, τ, rng, c) → (d, g) of chain c (Vectors of length N) or
`nothing` (the site stays: d = 0). Set by the driver (`tdir_source`) or a test before `plan`, with
the chain count in `TDIR_CHAINS`; `nothing`: the families have no sites.
"""
const TDIR_SOURCE = Ref{Any}(nothing)
const TDIR_CHAINS = Ref(1)

function tdir_rounds(fam::Symbol, tp::Topo, M::DeviceModel, cls, rng)
    TDIR_TOPO[] = tp
    src = TDIR_SOURCE[]
    src === nothing && return Vector{Vector{Tuple}}()
    (fam == :tdir && tracking(M)) && error(":tdir needs a model without line losses and lossy links (use :tdirL or :tdir0)")
    rounds = Vector{Vector{Tuple}}()
    for c in cls
        push!(rounds, Tuple[tdir_site(src, fam, τ, tp.N, TDIR_CHAINS[], rng) for τ in c])
    end
    return rounds
end

"A site (τ, d, g) from the source, d and g (C, N); a zero direction for a chain without one."
function tdir_site(src, fam, τ, N, C, rng)
    d, g = zeros(C, N), zeros(C, N)
    for c in 1:C
        dg = src(fam, τ, rng, c)
        dg === nothing && continue
        d[c, :] .= dg[1]; g[c, :] .= dg[2]
    end
    return (τ, d, g)
end

function tdir_cols(tp::Topo, site)
    T = tp.T
    G = gmax(tp)
    τ, d, g = site
    cols = [gencols(tp, n, τ, G) for n in 1:tp.N]
    return (cθ=[colτ(T, τ, n) for n in 1:tp.N], dv=vec(d), gv=vec(g),
            gq=reduce(vcat, [c[1] for c in cols]), gcm=reduce(vcat, [c[2] for c in cols]), gc=reduce(vcat, [c[3] for c in cols]),
            gc1=reduce(vcat, [c[4] for c in cols]))
end
for f in TDIR_FAMILIES
    @eval family_rounds(::Val{$(QuoteNode(f))}, tp::Topo, M::DeviceModel, cls, rng) = tdir_rounds($(QuoteNode(f)), tp, M, cls, rng)
    @eval site_cols(::Val{$(QuoteNode(f))}, tp::Topo, site, D::Int) = tdir_cols(tp, site)
    @eval device_extra(::Val{$(QuoteNode(f))}, ht) = (gon=Float64.(ht.gq .> 0),)
    FOOT[f] = [(:cθ, :θ, true), (:cθ, :r, true), (:gc, :ω, true), (:gc, :uc, true), (:gcm, :ω, false), (:gcm, :uc, false),
               (:gc1, :uc, false)]
end

function build_tdir(st, cp, sp, tab, r, k, ::Val{rule}, fam::Symbol) where {rule}
    tp = TDIR_TOPO[]
    N = tp.N
    e0, e1 = tp.ebus0, tp.ebus1
    nl = length(e0)
    on = m3(tab.on[:, r])
    cθ = tab.cθ[:, :, r]
    z = 0.0 .* g3(st.θ, cθ[:, 1])
    z4 = c4(z)
    a4 = c4(on .+ z)
    θ = G4(st.θ, cθ)
    C = size(z, 1)
    perchain(X) = (B = size(X, 1); rs(permutedims(rs(X, B, C, N), (2, 1, 3)), C, B, 1, N))
    d = perchain(tab.dv[:, :, r]) .* a4                         # direction (C, B, 1, N); 0 at padded sites
    lv(v) = rs(v[1:nl], 1, 1, 1, nl)
    bl = lv(sp.lb)
    line = sline(z4; f0=bl .* (θ[:, :, :, e0] .- θ[:, :, :, e1]), df=bl .* (d[:, :, :, e0] .- d[:, :, :, e1]),
                 s=lv(sp.ls), α=lv(sp.lα), near=copy(e0), far=copy(e1))
    r0 = G4(st.r, cθ)
    gen, uc, kk, x, ucq, uj, G = nothing, nothing, nothing, nothing, nothing, Int[], 0
    if rule !== :none
        gm = perchain(tab.gv[:, :, r]) .* a4                    # requirement change per node (C, B, 1, N)
        q = tab.gq[:, :, r]
        G = size(q, 2) ÷ N
        gc, gcm, gc1 = tab.gc[:, :, r], tab.gcm[:, :, r], tab.gc1[:, :, r]
        gon = M4(tab.gon[:, :, r]) .> 0.0
        κ = G4(cp.κg, q)
        sl = slotmap(G, N)
        if rule === :T2
            ω = G4(st.ω, gc)
            v = gon .& (absd.(κ) .>= k.κmin)
            dd = prorata4(κ, ω, v, r0, gm, N)
            kk = skinks(z4; Q=dd.Q, g=ifelse.(dd.act, gm, 0.0), P=dd.P, e=dd.e, nm=dd.nm, on=ifelse.(dd.act, 1.0, 0.0))
            x = ifelse.(dd.act[:, :, :, sl] .& v, ω ./ ifelse.(dd.act[:, :, :, sl], dd.P[:, :, :, sl], 1.0), 0.0)
            gen = gen4(st, cp, sp, z4, q, gc, sl; ak=x, kid=sl)
        else
            β = βof4(cp, sp, q, G4(cp.cls, cθ)[:, :, :, sl])
            gmc = gm[:, :, :, sl]
            v = (a4 .> 0.0) .& gon .& (absd.(κ) .>= k.κmin) .& (β .!= 0.0) .& (gmc .!= 0.0)
            x = ifelse.(v, β .* gmc ./ ifelse.(v, κ, 1.0), 0.0)
            gen = gen4(st, cp, sp, z4, q, gc, sl; a=x)
        end
        uj = [j for j in 1:N*G if ucon(fam, j)]
        if !isempty(uj)
            ucq = ucflags(sp, q, uj, gon)
            xu = x[:, :, :, uj]
            ku = sl[uj]
            sets = [(cy=gc[:, uj], y=xu, ky=ku, cu=gc[:, uj], u=xu .* ucq, ku=ku, cu1=gc1[:, uj], u1=nothing, ku1=nothing),
                    (cy=gcm[:, uj], y=nothing, ky=nothing, cu=gcm[:, uj], u=nothing, ku=nothing, cu1=gc[:, uj], u1=xu .* ucq, ku1=ku)]
            uc = uc4s(st, cp, sp, z4, q[:, uj], ucq, sets, rule === :T2)
        end
    end
    tb = TermBatchS(z; gen=gen, uc=uc, line=line, kk=kk, r0=r0)
    return tb, (a4, d, x, ucq, uj, G, N, rule)
end

for f in TDIR_FAMILIES
    @eval build(::Val{$(QuoteNode(f))}, st::FlatState, cp::ChainParams, sp::StaticParams, tab, r, k::Consts) =
        build_tdir(st, cp, sp, tab, r, k, Val($(QuoteNode(tdir_rule(f)))), $(QuoteNode(f)))
    @eval commit_tb(::Val{$(QuoteNode(f))}, st, tab, r, t, dr, aux, tb) = commit_tdir(st, tab, r, t, dr, aux, tb)
    # every line moves: with line losses a balance residual crosses 0 up to 5 times (test_comp G); 6 kept
    @eval nroots(::Val{$(QuoteNode(f))}) = Val(6)
end

function commit_tdir(st, tab, r, t, dr, aux, tb)
    a4, d, x, ucq, uj, G, N, rule = aux
    cθ = tab.cθ[:, :, r]
    θ = scadd4(st.θ, cθ, d .* c4(t))
    rr = scadd4(st.r, cθ, msk(dr4(dr), a4 .+ 0.0 .* d))
    ω, uc = st.ω, st.uc
    if x !== nothing
        gc = tab.gc[:, :, r]
        Δ = unitΔ(x, tb, t, slotmap(G, N), rule === :T2)
        ω = scadd4(st.ω, gc, Δ)
        ucq === nothing || (uc = scadd4(st.uc, gc[:, uj], Δ[:, :, :, uj] .* ucq))
    end
    return FlatState(θ, ω, uc, st.e, st.u, rr)
end

# ---------------------------------------------------------------------------
# host: directions and requirement changes, as the CPU preps
# ---------------------------------------------------------------------------

"""
Requirement change per node of the angle direction d (the loop of `prep_tdir!`): g = L_b d over the
lines in order, kept where |g_n| > 1e-9 max |g| (0 elsewhere); d itself with |d_i| ≤ 1e-14 zeroed
(the entries `prep_tdir!` leaves out).
"""
function tdir_g(sys, d::AbstractVector)
    d = [abs(x) > 1e-14 ? x : 0.0 for x in d]
    g = zeros(sys.N)
    for e in eachindex(sys.line_s)
        a, b = sys.line_bus0[e], sys.line_bus1[e]
        df = sys.line_b[e] * (d[a] - d[b])
        g[a] += df; g[b] -= df
    end
    gmax = maximum(abs, g; init=0.0)
    return d, [gmax != 0.0 && abs(x) > 1e-9 * gmax ? x : 0.0 for x in g]
end

"""
Direction of a binding-line normal (`prep_tline!`): the injection pair +1 at bus0, −1 at bus1 of
line e re-routed over its component (Δθ = L⁺(e_a − e_b), entries |Δθ| ≤ 1e-14 left out), compensated
at both ends (g = +1 at a, −1 at b).
"""
function tline_dir(sys, LM, e::Int)
    a, b = sys.line_bus0[e], sys.line_bus1[e]
    d = zeros(sys.N)
    ML.reroute!(d, LM.GS, a, b)
    dd = zeros(sys.N)
    for i in LM.GS.nodes[LM.comp[a]]
        abs(d[i]) > 1e-14 && (dd[i] = d[i])
    end
    g = zeros(sys.N); g[a] += 1.0; g[b] -= 1.0
    return dd, g
end

"""
    tdir_source(P, LM; FFs, AM, rates) -> (fam, τ, rng, c) → (d, g) or nothing

Direction source of the dense angle families: per site one of the CPU families of its rule, with the
CPU's relative rates (`rates`: family → weight; default the adaptive kernel's), among those available:
`tmode` needs frames with transport modes (`FFs[c]` of chain c, flow basis), `tline` a predicted
binding line at τ, `amode` learned modes (`AM` with an update; projected onto the chain's face). A
function of the chain's (k, x) and the learned tables only.
"""
function tdir_source(P::ML.Problem, LM; FFs=nothing, AM=nothing,
                     rates=Dict(:tmode => 0.05, :tline => 0.05, :amode => 0.03, :tmode0 => 0.05, :tline0 => 0.05,
                                :amode0 => 0.03, :tmodeL => 0.03, :tlineL => 0.05, :amodeL => 0.02))
    sys = P.E.sys
    members = Dict(:tdir => (:tmode, :tline, :amode), :tdir0 => (:tmode0, :tline0, :amode0), :tdirL => (:tmodeL, :tlineL, :amodeL))
    function avail(m, τ, FF)
        s = string(m)
        startswith(s, "tmode") && return FF !== nothing && size(FF.D, 2) > 0
        startswith(s, "tline") && return FF !== nothing && !isempty(FF.binding[τ])
        return AM !== nothing && size(AM.Dθ, 2) > 0
    end
    return function (fam, τ, rng, c)
        FF = FFs === nothing ? nothing : FFs[c]
        ms = [m for m in members[fam] if avail(m, τ, FF)]
        isempty(ms) && return nothing
        w = cumsum([rates[m] for m in ms]); u = rand(rng) * w[end]
        m = ms[findfirst(>=(u), w)]
        s = string(m)
        if startswith(s, "tmode")
            return tdir_g(sys, FF.D[:, rand(rng, 1:size(FF.D, 2)), τ])
        elseif startswith(s, "tline")
            return tline_dir(sys, LM, rand(rng, FF.binding[τ]))
        end
        d = AM.Dθ[:, rand(rng, 1:size(AM.Dθ, 2))]
        return tdir_g(sys, FF === nothing ? d : ML.project_face(d, FF, sys, τ))
    end
end

"""
New directions for the dense angle families of a plan (same sites per round, same table shapes: an
upload without recompilation): the device tables of `pl.fams` ∩ `TDIR_FAMILIES` redrawn from the
current `TDIR_SOURCE`. Returns a NamedTuple family → device table.
"""
function tdir_refresh(pl, rng)
    fs = Tuple(f for f in pl.fams if f in TDIR_FAMILIES)
    src = TDIR_SOURCE[]
    out = Pair{Symbol,Any}[]
    for f in fs
        rounds = [[tdir_site(src, f, site[1], pl.tp.N, TDIR_CHAINS[], rng) for site in r] for r in pl.sched[f]]
        push!(out, f => device_table(f, host_table(f, pl.tp, rounds)))
    end
    return (; out...)
end
