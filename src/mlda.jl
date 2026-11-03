# Multilevel delayed acceptance (§5).
#
# Level 0: random scan over π₀-reversible moves, hence itself π₀-reversible:
#   * FFBS: exact resampling of the regime path given z0 (prob. p_ffbs);
#   * carrier scaling (prob. p_scale): for one investable carrier multiply its
#     capacity coefficients by c and all its utilization coefficients by 1/c.
#     Nodal dispatch is unchanged, so this follows the ridge k·ω = const along
#     which investments move; MH with Jacobian c^(m_k − m_ω T);
#   * reflective HMC on (k, v, z0) | m otherwise.
#
# Level ℓ ≥ 1: a subchain of random length n ~ U{1..J_ℓ} at level ℓ−1 from the
# coarse part of the current state, then new fine modes proposed independently
# of the coarse ones: weather innovations by pCN, k and v details by a
# symmetric random walk on a random subset of coarse time blocks. Accept with
# exp(Δ(s) − Δ(ψ)), Δ = L_{ℓ−1}(coarse part) − L_ℓ.

mutable struct LevelStats
    nprop::Int
    nacc::Int
    ninfeasible::Int
    violated::Dict{Symbol,Int}     # infeasible proposals by the first violated field
end
LevelStats() = LevelStats(0, 0, 0, Dict{Symbol,Int}())

"Fields whose hard constraints a level-ℓ state violates."
function violated_fields(P::Problem, ℓ::Int, k, v; tol=1e-9)
    C = P.cons[ℓ+1]
    lk, lv = P.lk[ℓ+1], P.lv[ℓ+1]
    q = vcat(k, v)
    field(i) = i <= lk.n ? P.E.kspecs[findlast(o -> o < i, lk.off)].name :
                           P.E.vspecs[findlast(o -> o < i - lk.n, lv.off)].name
    out = Set{Symbol}()
    for i in 1:C.n
        (q[i] < C.lo[i] - tol * (1 + abs(C.lo[i])) || q[i] > C.hi[i] + tol * (1 + abs(C.hi[i]))) && push!(out, field(i))
    end
    Aq = C.A * q
    for j in eachindex(C.b)
        Aq[j] > C.b[j] + tol * (1 + abs(C.b[j])) && push!(out, field(C.At.rowval[C.At.colptr[j]]))
    end
    return out
end

mutable struct MLDASampler
    J::Vector{Int}                   # subchain lengths J_ℓ, ℓ = 1..L
    hmc::HMCState
    ws::BilliardWS
    p_ffbs::Float64
    p_scale::Float64
    σscale::Vector{Float64}          # log-scale step per k field
    nscale::Vector{Int}
    accscale::Vector{Int}
    ϱ::Vector{Float64}               # pCN parameters per level
    σk::Vector{Vector{Float64}}      # RW scales per k field, per level
    σv::Vector{Vector{Float64}}      # RW scales per v field, per level
    logscale::Vector{Float64}        # adapted multiplier per level
    frac::Float64                    # fraction of coarse blocks perturbed per proposal
    adapt::Bool
    nadapt::Vector{Int}
    stats::Vector{LevelStats}
    record::Bool
    pairs::Vector{Vector{Tuple{Vector{Float64},Vector{Float64}}}}   # per level ℓ ≥ 1: (Q_ℓ(s), Q_{ℓ-1}(ψ_end))
    hmc_acc::Vector{Float64}
    kernel::Symbol                   # :hmc or :gibbs at level 0
    gibbs::Any                       # GibbsState (gibbs.jl) or nothing
    p_fine::Float64                  # levels ≥ 1: probability of a fixed-level Gibbs sweep
    nfine::Vector{Int}
    loc::Any                         # kernel = :local: per level (LocalModel, move workspaces), lazily built
end

"Quantities of interest Q_ℓ(s): capacity totals, costs, shed, regime frequencies, weather means."
function quantities(P::Problem, s::State)
    E = P.E
    lk = P.lk[s.ℓ+1]
    caps = capacity_totals(E, lk, s.k)
    p = s.parts
    K = P.W.K
    rf = [count(==(k), s.m) / length(s.m) for k in 1:K]
    lay = P.W.lays[1]
    wm = [mean(fieldview(s.z0, lay, P.W.specs, f)[1, :]) for f in eachindex(P.W.specs)]
    return vcat(caps, [p.capex, p.op, p.capex + p.op, p.shed, p.penalty], rf, wm, [p.co2])
end

function quantity_names(P::Problem)
    vcat([String(s.name) for s in P.E.kspecs], ["capex", "opex", "total", "shed_MWh", "penalty"],
         ["regime$k" for k in 1:P.W.K], ["z0mean_" * String(s.name) for s in P.W.specs], ["co2_t"])
end

"v fields scaled together with k field f (utilizations of that carrier)."
function scaled_vfields(E::EnergyModel, f::Int)
    for g in E.gens
        g.kf == f && return g.vf > 0 ? (g.ucf > 0 ? [g.vf, g.ucf] : [g.vf]) : Int[]
    end
    for st in E.stors
        st.kf == f && return st.fp > 0 ? [st.fp, st.fm, st.fe] : [st.fe]
    end
    return Int[]
end

function scale_step(rng, P::Problem, s::State, S::MLDASampler)
    E = P.E
    lk, lv = P.lk[s.ℓ+1], P.lv[s.ℓ+1]
    f = rand(rng, 1:length(E.kspecs))
    logc = S.σscale[f] * randn(rng)
    c = exp(logc)
    sn = copy(s)
    fieldview(sn.k, lk, E.kspecs, f) .*= c
    nlogJ = lk.m[f]
    for vf in scaled_vfields(E, f)
        fieldview(sn.v, lv, E.vspecs, vf) ./= c
        nlogJ -= lv.m[vf] * ntime(lv, E.vspecs[vf])
    end
    evaluate!(P, sn)
    logα = sn.L - s.L + nlogJ * logc
    acc = isfinite(logα) ? min(1.0, exp(logα)) : 0.0
    S.nscale[f] += 1
    if S.adapt
        S.σscale[f] *= exp((acc - 0.3) / sqrt(S.nscale[f]))
    end
    if rand(rng) < acc
        S.accscale[f] += 1
        return sn
    end
    return s
end

function base_step(rng, P::Problem, s::State, S::MLDASampler)
    S.kernel == :gibbs && return gibbs_step(rng, P, s, S, S.gibbs)
    S.kernel == :local && return local_base_step(rng, P, s, S)
    s = copy(s)
    u = rand(rng)
    if S.p_ffbs <= u < S.p_ffbs + S.p_scale
        return scale_step(rng, P, s, S)
    end
    if u < S.p_ffbs
        ll = regime_loglik(P.W, s.z0)
        s.m = ffbs(rng, ll, P.W.Ptr, P.W.π0)
        s.xs = [s.z0]
        return s     # L_0 does not depend on m
    end
    q = vcat(s.k, s.v, s.z0)
    acc = hmc_step!(rng, q, level0_potential(P, s.m), P.cons0, S.hmc, S.ws)
    push!(S.hmc_acc, acc)
    S.adapt && adapt_stepsize!(S.hmc, acc)
    nk, nv = length(s.k), length(s.v)
    s.k = q[1:nk]
    s.v = q[nk+1:nk+nv]
    s.z0 = q[nk+nv+1:end]
    s.xs = [s.z0]
    return evaluate!(P, s)
end

function level_step(rng, P, s, S, ℓ)
    ℓ == 0 && return base_step(rng, P, s, S)
    if S.kernel in (:gibbs, :local) && rand(rng) < S.p_fine
        S.nfine[ℓ] += 1
        S.kernel == :local && return fine_local_step(rng, P, s, S)
        return fine_gibbs_step(rng, P, s, S.gibbs)
    end
    return mlda_step(rng, P, s, S)
end

"Random-walk perturbation of the detail coordinates on a random subset of coarse blocks."
function perturb_details(rng, ξ, tr::Transfer, lc::Layout, lf::Layout, specs, σ, scale, sel::BitVector)
    ξn = copy(ξ)
    for f in eachindex(specs)
        o = tr.doff[f]
        len = (f < length(specs) ? tr.doff[f+1] : tr.n) - o
        for i in o+1:o+len
            b = tr.dblock[i]
            (b == 0 ? sel[1] : sel[b]) || continue
            ξn[i] += scale * σ[f] * randn(rng)
        end
    end
    return ξn
end

function pcn(rng, w, ϱ, dblock, sel)
    wn = copy(w)
    c = sqrt(1 - ϱ^2)
    for i in eachindex(w)
        sel[dblock[i]] || continue
        wn[i] = c * w[i] + ϱ * randn(rng)
    end
    return wn
end

"""
    mlda_core(rng, s, ℓ, J, ops) -> (snew, ψend, accprob, ψ)

One MLDA transition at level ℓ ≥ 1, generic in the state type. `ops` provides
  `split(s)`              → (coarse state at ℓ-1 with its log target, fine modes)
  `step(rng, sc)`         → one π_{ℓ-1}-reversible transition of the coarse state
  `propose(rng, fine)`    → fine modes, independent of the coarse modes and
                            symmetric (or pCN-reversible w.r.t. their reference)
  `assemble(sc, fine)`    → fine state with its log target
  `logtarget(s)`          → cached log target L (−Inf if infeasible)
  `record`                → continue the subchain to length J (estimator)
Acceptance: exp(Δ(s) − Δ(ψ)) with Δ = L_{ℓ-1}(coarse part) − L_ℓ.
"""
function mlda_core(rng, s, J::Int, ops)
    sc, fine = ops.split(s)
    n = rand(rng, 1:J)
    ψc = sc
    for _ in 1:n
        ψc = ops.step(rng, ψc)
    end
    ψend = ψc
    if ops.record
        for _ in n+1:J
            ψend = ops.step(rng, ψend)
        end
    end
    ψ = ops.assemble(ψc, ops.propose(rng, fine))
    Lψ = ops.logtarget(ψ)
    acc = 0.0
    if isfinite(Lψ)
        logα = (Lψ - ops.logtarget(s)) - (ops.logtarget(ψc) - ops.logtarget(sc))
        acc = isnan(logα) ? 0.0 : min(1.0, exp(logα))
    end
    snew = rand(rng) < acc ? ψ : s
    return snew, ψend, acc, ψ
end

function mlda_step(rng, P::Problem, s::State, S::MLDASampler)
    ℓ = s.ℓ
    Tc = P.lv[ℓ].T
    function propose(rng, fine)
        w, ξk, ξv = fine
        # random subset of coarse blocks, chosen independently of the state
        sel = BitVector(rand(rng, Tc) .< S.frac)
        any(sel) || (sel[rand(rng, 1:Tc)] = true)
        scale = exp(S.logscale[ℓ])
        return (pcn(rng, w, min(1.0, scale * S.ϱ[ℓ]), P.W.trs[ℓ].dblock, sel),
                perturb_details(rng, ξk, P.trk[ℓ], P.lk[ℓ], P.lk[ℓ+1], P.E.kspecs, S.σk[ℓ], scale, sel),
                perturb_details(rng, ξv, P.trv[ℓ], P.lv[ℓ], P.lv[ℓ+1], P.E.vspecs, S.σv[ℓ], scale, sel))
    end
    ops = (split=s_ -> split_level(P, s_),
           step=(r, sc) -> level_step(r, P, sc, S, ℓ - 1),
           propose=propose,
           assemble=(sc, fine) -> assemble(P, sc, fine...),
           logtarget=s_ -> s_.L,
           record=S.record)
    snew, ψend, acc, ψ = mlda_core(rng, s, S.J[ℓ], ops)
    st = S.stats[ℓ]
    st.nprop += 1
    snew === ψ && (st.nacc += 1)
    if !isfinite(ψ.L)
        st.ninfeasible += 1
        for f in violated_fields(P, ℓ, ψ.k, ψ.v)
            st.violated[f] = get(st.violated, f, 0) + 1
        end
    end
    if S.adapt   # Robbins-Monro on the log proposal scale, target acceptance 0.25
        S.nadapt[ℓ] += 1
        S.logscale[ℓ] += 2.0 / sqrt(S.nadapt[ℓ]) * (acc - 0.25)
    end
    S.record && push!(S.pairs[ℓ], (quantities(P, snew), quantities(P, ψend)))
    return snew
end

"Per-field RW scales from the spread of the detail coordinates of a reference state."
function detail_scales(tr::Transfer, specs, ξ; rel=0.05, floor=1e-6)
    σ = zeros(length(specs))
    for f in eachindex(specs)
        o = tr.doff[f]
        len = (f < length(specs) ? tr.doff[f+1] : tr.n) - o
        len == 0 && continue
        σ[f] = rel * max(std(view(ξ, o+1:o+len); corrected=false), floor)
    end
    return σ
end
