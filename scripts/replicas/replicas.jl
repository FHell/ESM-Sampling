# Replica Langevin sampler prototype (docs/replica_sampler.md, ledger E22–E25):
#   - M weather replicas (x_m, v_m) share the capacities k;
#   - the weather of every replica follows its own prior-reversible process (pCN in the
#     non-centred innovation coordinates, FFBS of the regimes), which never reads k or v, so
#     every replica's weather is an autonomous Markov chain with stationary law ρ^x;
#   - the operation of every replica follows its weather and the capacities by operation
#     sweeps (with the compensated initialisation after a weather step);
#   - the capacities take a Langevin step on the replica-averaged sampled score (the weights
#     w_m ∝ exp(λ Φ(x_m)) of the stress tilt, 1/M at λ = 0);
#   - a lag probe freezes (k, x) on a copy of one replica, compares its equilibrated score with
#     the in-flight score per carrier and adapts the step per carrier.
# Needs the session of scripts/langevin/setup40.jl and scripts/langevin/langevin.jl
# (P, S, LM, lgv_op!, lgv_scores, lgv_units, lgv_set_disp!).

# ----------------------------------------------------------------------------------------
# weather: pCN in the innovation coordinates, regimes by FFBS
# ----------------------------------------------------------------------------------------

"Square roots of the VAR covariances (the factors of `sample_weather`): Lk[k] Lk[k]' = Σ_k, L0 L0' = Σ_0."
struct VarFactors
    Lk::Vector{LowerTriangular{Float64,Matrix{Float64}}}
    L0::LowerTriangular{Float64,Matrix{Float64}}
end
VarFactors(W::ML.WeatherModel) =
    VarFactors([cholesky(Symmetric(inv(W.Σinv[k]))).L for k in 1:W.K], cholesky(Symmetric(inv(W.Σ0inv))).L)

"VAR mean of step τ ≥ 2 given z0(τ−1), regime k."
function var_mean(W, B, zprev, τ, T0, n)
    c, cs, sn = ML.harmonics(τ, T0)
    return B[:, 1:n] * zprev .+ c .* B[:, n+1] .+ cs .* B[:, n+2] .+ sn .* B[:, n+3]
end

"Innovations e (n0 × T0) of z0 given the regime path: z0(1) = L0 e₁, z0(τ) = μ_m(τ) + L_m e_τ."
function z0_innov(W::ML.WeatherModel, F::VarFactors, m, z0)
    lay = W.lays[1]
    Z = ML.z0matrix(z0, lay, W.specs)
    n, T0 = size(Z)
    Ein = zeros(n, T0)
    Ein[:, 1] .= F.L0 \ Z[:, 1]
    for τ in 2:T0
        Ein[:, τ] .= F.Lk[m[τ]] \ (Z[:, τ] .- var_mean(W, W.Bm[m[τ]], Z[:, τ-1], τ, T0, n))
    end
    return Ein
end

"z0 from its innovations (the recursion of `sample_weather`)."
function z0_from_innov(W::ML.WeatherModel, F::VarFactors, m, Ein)
    lay = W.lays[1]
    n, T0 = size(Ein)
    Z = zeros(n, T0)
    Z[:, 1] .= F.L0 * Ein[:, 1]
    for τ in 2:T0
        Z[:, τ] .= var_mean(W, W.Bm[m[τ]], Z[:, τ-1], τ, T0, n) .+ F.Lk[m[τ]] * Ein[:, τ]
    end
    return ML.z0vector(Z, lay, W.specs)
end

"""
    pcn_weather(rng, W, F, m, z0, ws, θ) -> (z0′, ws′)

pCN step of angle θ in the innovation coordinates (e, w) at fixed regimes:
(e, w) → cos θ (e, w) + sin θ ξ, ξ ~ N(0, I). Reversible for the prior p(z0 | m) ∏ N(w; 0, I).
"""
function pcn_weather(rng, W::ML.WeatherModel, F::VarFactors, m, z0, ws, θ)
    c, sn = cos(θ), sin(θ)
    Ein = z0_innov(W, F, m, z0)
    Ein .= c .* Ein .+ sn .* randn(rng, size(Ein)...)
    z1 = z0_from_innov(W, F, m, Ein)
    ws1 = [c .* w .+ sn .* randn(rng, length(w)) for w in ws]
    return z1, ws1
end

"""
    regime_window(rng, W, m, a, b) -> m′

Exact draw of the regimes m[a..b] from the Markov-chain prior given the regimes outside the
window (π₀ at a = 1, no constraint at b = T0): backward messages β_τ(i) = (Pᵇ⁺¹⁻τ)[i, m_{b+1}],
then forward sampling.
"""
function regime_window(rng, W::ML.WeatherModel, m, a, b)
    T0, K = length(m), W.K
    Ptr = W.Ptr
    β = ones(K, b - a + 1)
    if b < T0
        β[:, end] .= Ptr[:, m[b+1]]
        for τ in b-1:-1:a
            β[:, τ-a+1] .= Ptr * β[:, τ-a+2]
        end
    end
    m1 = copy(m)
    for τ in a:b
        p = (τ == 1 ? W.π0 : Ptr[m1[τ-1], :]) .* β[:, τ-a+1]
        m1[τ] = ML.sample_log(rng, log.(p))
    end
    return m1
end

"""
    weather_step(wrng, P, F, s, θ; regime, regmode, B) -> s′

One step of the weather process of a replica, in the non-centred coordinates (m, e, w) whose
prior is p(m) ⊗ N(0, I) ⊗ N(0, I): the pCN step of (e, w) at fixed m (θ > 0), then with
`regime = true` a regime update:
- `regmode = :window` (non-centred): the regimes of a random window of `B` level-0 steps are
  redrawn from their Markov prior given the rest, at fixed innovations (e, w); z0 and the
  details follow from the recursion;
- `regmode = :ffbs` (centred): FFBS of the regimes given the anomalies of all levels
  (`fine_regime_step!`), the innovations recomputed; the anomalies do not change.
Reads neither k nor v; the dispatch is kept.
"""
function weather_step(wrng, P, F::VarFactors, s, θ; regime::Bool=false, regmode::Symbol=:window, B::Int=7)
    m, z0, ws = s.m, s.z0, s.ws
    changed = false
    if θ > 0
        z0, ws = pcn_weather(wrng, P.W, F, m, z0, ws, θ)
        changed = true
    end
    if regime && regmode == :window
        T0 = length(m)
        a = rand(wrng, 1:T0)
        b = min(T0, a + B - 1)
        Ein = z0_innov(P.W, F, m, z0)
        m = regime_window(wrng, P.W, m, a, b)
        z0 = z0_from_innov(P.W, F, m, Ein)
        changed = true
    end
    s = changed ? ML.make_state(P, s.ℓ, m, z0, ws, s.k, s.v) : copy(s)
    regime && regmode == :ffbs && ML.fine_regime_step!(wrng, P, s)
    return s
end

"""
    compensate!(P, LM, r0, s1) -> s1

Compensated initialisation after a weather change: the dispatchable (weather-independent)
generators at every node and step take up the change of the node residual from `r0` (the
residual before the change) to that of `s1`: an increase in proportion to their headroom (up
to it), a decrease in proportion to their output; the rest stays on the residual.
"""
function compensate!(P, LM, r0, s1)
    E = P.E
    lv = P.lv[s1.ℓ+1]
    LC = ML.LocalCost(P, LM, s1)
    Δ = LC.r .- r0
    for n in axes(Δ, 1), τ in axes(Δ, 2)
        need = Δ[n, τ]
        abs(need) < 1e-6 && continue
        us = [(f, i) for (f, i) in LM.genunits[n] if E.gens[LM.role[f][2]].wf == 0]
        isempty(us) && continue
        κs_ = [ML.κunit(P, LM, LC, f, i, τ) for (f, i) in us]
        ps = [κ * clamp(LC.Y[f][i, τ], 0.0, 1.0) for ((f, i), κ) in zip(us, κs_)]
        ws_ = need > 0 ? max.(κs_ .- ps, 0.0) : ps
        tot = sum(ws_)
        tot > 1e-9 || continue
        a = need > 0 ? min(need, tot) / tot : max(need, -tot) / tot
        for ((f, i), κ, wq) in zip(us, κs_, ws_)
            abs(κ) < 1e-9 && continue
            Δy = a * wq / κ
            Δy == 0.0 && continue
            m_ = lv.m[f]; o = lv.off[f] + (τ - 1) * m_
            Ut = E.vspecs[f].B.Ut
            for q in ML.nzrange(Ut, i)
                j = ML.rowvals(Ut)[q]
                j <= m_ && (s1.v[o+j] += Δy * ML.nonzeros(Ut)[q])
            end
        end
    end
    return ML.evaluate!(P, s1)
end

# ----------------------------------------------------------------------------------------
# weather features (cheap functionals of x, for the prior check and the tilt)
# ----------------------------------------------------------------------------------------

"""
    wfeatures(P, x, kref) -> NamedTuple

Features of the level-L anomalies x after the data map: annual mean system load (GW), annual
mean capacity factors of the weather-dependent carriers (unweighted node means), the residual
load D_t − Σ k_ref·a_t with the fixed reference capacities `kref` (node vectors per k field):
annual mean, peak day, and the worst 7-day mean (GW).
"""
function wfeatures(P, x, kref)
    E = P.E
    L = length(P.W.lays) - 1
    Xn = ML.weather_nodes(P.W, L, x)
    D = vec(sum(max.(Xn[1], 0.0), dims=1))
    R = zeros(length(D))
    cf = Float64[]
    for g in E.gens
        g.wf > 0 || continue
        A = clamp.(Xn[g.wf], 0.0, 1.0)
        push!(cf, mean(A))
        g.kf > 0 && (R .+= vec(kref[g.kf]' * A))
    end
    res = (D .- R) ./ 1e3
    T = length(res)
    w7 = maximum(mean(res[mod1.(t:t+6, T)]) for t in 1:T)
    return (; load=mean(D) / 1e3, cf, resmean=mean(res), respeak=maximum(res), res7=w7)
end
featvec(f) = vcat(f.load, f.cf, f.resmean, f.respeak, f.res7)
featnames(P) = vcat("load GW", ["cf " * String(P.E.wspecs[g.wf].name) for g in P.E.gens if g.wf > 0], "resload mean GW",
                    "resload peak GW", "resload 7d GW")

# ----------------------------------------------------------------------------------------
# Langevin step on the shared capacities
# ----------------------------------------------------------------------------------------

"New raw node capacities per k field from the scores (sk, fk): the Langevin step of `langevin_step!` (one noise draw)."
function lgv_knew(rng, P, units, Kraw, sk, fk; ηw, κs, κmin=1.0, clip=Inf, box::Bool=false)
    E = P.E
    r = P.βk / P.βv
    knew = deepcopy(Kraw)
    nclip = 0
    for kf in unique(u[4] for u in units)
        ηw[kf] == 0 && continue                                  # capacities of this field frozen
        k = Kraw[kf]
        g = -fk[kf] .+ r .* sk[kf]
        w = log.(k .+ κs[kf])
        dw0 = ηw[kf] .* ((k .+ κs[kf]) .* g .+ 1)
        nclip += count(abs.(dw0) .> clip)
        dw = clamp.(dw0, -clip, clip)
        w1 = w .+ dw .+ sqrt(2ηw[kf]) .* randn(rng, length(k))
        wmin = log(κmin + κs[kf])
        w1 = [x < wmin ? 2wmin - x : x for x in w1]
        if box
            wmax = log.(max.(E.kspecs[kf].hi, κmin + 1.0) .+ κs[kf])
            w1 = [x > wm && w0 <= wm ? max(2wm - x, wmin) : x for (x, wm, w0) in zip(w1, wmax, w)]
        end
        knew[kf] = exp.(w1) .- κs[kf]
    end
    return knew, nclip
end

"Move the capacities of `s` from Kraw to knew with the dispatch following (fixed output for dispatchable generators, the kop curve for renewables and storage)."
function lgv_apply!(P, LM, s, units, Kraw, knew; κmin=1.0)
    E = P.E
    LC = ML.LocalCost(P, LM, s)
    for (kind, gi, i, kf) in units
        kind == :disp || continue
        lgv_set_disp!(P, LM, LC, s, gi, i, Kraw[kf][i], knew[kf][i])
    end
    LC = ML.LocalCost(P, LM, s)
    Kr = ML.knodes(P, s.ℓ, s.k)
    for (kind, gi, i, kf) in units
        kind == :disp && continue
        k0 = Kr[kf][i]
        mv = ML.kop_prepare(P, LM, LC, s, [(kind == :ren ? 1 : 2, gi, i)]; κmin=κmin, Kraw=Kr)
        if mv === nothing
            Ut = E.kspecs[kf].B.Ut
            for q in ML.nzrange(Ut, i)
                s.k[P.lk[s.ℓ+1].off[kf]+ML.rowvals(Ut)[q]] += (knew[kf][i] - k0) * ML.nonzeros(Ut)[q]
            end
            LC = ML.LocalCost(P, LM, s); Kr = ML.knodes(P, s.ℓ, s.k)
            continue
        end
        ML.kop_commit!(P, LM, LC, s, mv, log(knew[kf][i] / k0); Kraw=Kr)
    end
    return ML.evaluate!(P, s)
end

# ----------------------------------------------------------------------------------------
# the replica sampler
# ----------------------------------------------------------------------------------------

"Settings of the replica sampler."
Base.@kwdef mutable struct RepCtl
    ηw::Vector{Float64}                  # Langevin step in w per k field
    κs::Vector{Float64}                  # offsets of w = log(k + κs) per k field (MW)
    clip::Float64 = 0.02                 # drift clip per step in w
    box::Bool = true                     # reflect at the upper capacity bound
    inner::Int = 1                       # operation sweeps per replica and step
    θ::Float64 = 0.05                    # pCN angle per step (continuous mode, burst_every = 0)
    burst_every::Int = 0                 # burst mode: a pCN step of angle φ per replica every that many steps (staggered)
    φ::Float64 = 0.3                     # burst angle
    gate::Int = 20                       # steps after a burst in which a replica's scores get weight 0
    burst_regime::Float64 = 0.5          # probability of a regime window move with a burst
    comp::Bool = true                    # compensated initialisation after a weather step
    nreg::Int = 5                        # regime update every nreg steps (0: never)
    regmode::Symbol = :window            # :window (non-centred prior window draw) or :ffbs (centred)
    regB::Int = 7                        # window length of the :window regime update (level-0 steps)
    λ::Float64 = 0.0                     # stress tilt (0: off)
    Φ::Function = x -> 0.0               # declared weather functional of the tilt
    probe_every::Int = 0                 # lag probe every that many steps (0: off)
    probe_len::Int = 30                  # operation sweeps of a probe
    probe_tail::Int = 15                 # equilibrated score: mean over the last sweeps of the probe
    probe_win::Int = 10                  # in-flight score: mean over the replica's last steps
    adapt::Bool = true                   # adapt ηw per carrier from the probes
    gap_hi::Float64 = 0.10               # |gap| / β_k∂K above which the step shrinks
    gap_lo::Float64 = 0.05               # below which it grows
    ηmin::Float64 = 2e-5
    ηmax::Float64 = 2e-3
    shrink::Float64 = 0.7
    grow::Float64 = 1.2
end

"Interior nodes per k field (capacity above 50 MW and below the box)."
interior_mask(P, Kr) = [(Kr[f] .> 50) .& (Kr[f] .< P.E.kspecs[f].hi .- 1) for f in eachindex(Kr)]

"""
Self-normalised tilt weights over the active replicas, exp(λΦ_m)/Σ_active exp(λΦ) (exactly
1/M_active at λ = 0); inactive replicas (gated after a weather burst) get 0. The gate depends on
the step count only, not on the weather, so it does not tilt the weather the scores see.
"""
function tilt_weights(λ, Φs, active=trues(length(Φs)))
    any(active) || (active = trues(length(Φs)))
    a = [act ? λ * Φ : -Inf for (act, Φ) in zip(active, Φs)]
    w = exp.(a .- maximum(a))
    return w ./ sum(w)
end

"""
    rep_run!(rng, wrngs, prng, P, S, reps, nsteps, ctl; trace, probes, it0, log_every, label) -> reps

`nsteps` steps of the replica sampler. Per step and replica: one weather step (pCN, regimes every
`nreg` steps, compensated initialisation), `inner` operation sweeps with their scores; then the
Langevin step of the shared capacities on the tilt-weighted replica mean of the scores, applied to
every replica with its dispatch following. `wrngs[m]` drives the weather of replica m only (so its
weather path does not depend on k or v), `rng` the operation sweeps and the Langevin noise, `prng`
the probes (which do not change the replicas).
"""
function rep_run!(rng, wrngs, prng, P, S, reps, nsteps, ctl::RepCtl; trace=NamedTuple[], probes=NamedTuple[], it0=0,
                  log_every=10, label="", F=VarFactors(P.W), kref=nothing, hist=nothing, since=nothing)
    K = S.loc
    LM = ML.local_model(K, P, reps[1].ℓ)
    units = lgv_units(P, LM)
    M = length(reps)
    r = P.βk / P.βv
    nk = length(P.E.kspecs)
    hist = hist === nothing ? [Vector{Vector{Float64}}[] for _ in 1:M] : hist   # per replica: recent r ŝ_k node vectors
    since = since === nothing ? fill(typemax(Int) ÷ 2, M) : since                 # per replica: steps since its last burst
    nb = ctl.burst_every
    t0 = time()
    for it in it0+1:it0+nsteps
        tw = time()
        sks = Vector{Vector{Vector{Float64}}}(undef, M)
        fk = nothing
        for m in 1:M
            s = reps[m]
            if nb == 0                                              # continuous: angle θ every step
                θm = ctl.θ
                regm = ctl.nreg > 0 && it % ctl.nreg == 0
            else                                                    # burst: angle φ every nb steps, staggered
                burst = mod(it + (m - 1) * nb ÷ M, nb) == 0
                θm = burst ? ctl.φ : 0.0
                regm = burst && rand(wrngs[m]) < ctl.burst_regime
                if burst
                    since[m] = 0
                    empty!(hist[m])
                else
                    since[m] += 1
                end
            end
            if θm > 0 || regm
                r0 = ctl.comp ? copy(ML.LocalCost(P, LM, s).r) : nothing
                s = weather_step(wrngs[m], P, F, s, θm; regime=regm, regmode=ctl.regmode, B=ctl.regB)
                r0 === nothing || compensate!(P, LM, r0, s)
            end
            acc = nothing
            for _ in 1:ctl.inner
                s = lgv_op!(rng, P, K, LM, s)
                sk, fk_ = lgv_scores(P, s)
                fk = fk_
                acc = acc === nothing ? sk ./ ctl.inner : acc .+ sk ./ ctl.inner
            end
            sks[m] = acc
            if nb == 0 || since[m] >= ctl.gate
                push!(hist[m], [r .* x for x in acc]); length(hist[m]) > ctl.probe_win && popfirst!(hist[m])
            end
            reps[m] = s
        end
        tw = time() - tw
        Φs = [ctl.Φ(s.xs[end]) for s in reps]
        active = [nb == 0 || since[m] >= ctl.gate for m in 1:M]
        wts = tilt_weights(ctl.λ, Φs, active)
        sk̄ = [sum(wts[m] .* sks[m][f] for m in 1:M) for f in 1:nk]
        Kraw = ML.knodes(P, reps[1].ℓ, reps[1].k)
        knew, nclip = lgv_knew(rng, P, units, Kraw, sk̄, fk; ηw=ctl.ηw, κs=ctl.κs, clip=ctl.clip, box=ctl.box)
        if any(>(0), ctl.ηw)
            for m in 1:M
                reps[m] = lgv_apply!(P, LM, reps[m], units, Kraw, knew)
            end
        end
        for m in 2:M
            reps[m].k == reps[1].k || error("replica capacities diverged")
        end
        mask = interior_mask(P, Kraw)
        rs̄ = [any(mask[f]) ? mean(r .* sk̄[f][mask[f]]) : NaN for f in 1:nk]
        fk̄ = [any(mask[f]) ? mean(fk[f][mask[f]]) : NaN for f in 1:nk]
        tot = [sum(k) for k in ML.knodes(P, reps[1].ℓ, reps[1].k)]
        pf = [s.parts.capex + s.parts.gen + s.parts.storage for s in reps]
        feats = kref === nothing ? nothing : [featvec(wfeatures(P, s.xs[end], kref)) for s in reps]
        push!(trace, (; it, tot, capex=reps[1].parts.capex, op=[s.parts.op for s in reps], shed=[s.parts.shed for s in reps],
                      pen=[s.parts.penalty for s in reps], pfcost=pf, rs=rs̄, fk=fk̄, wts, Φ=Φs, active, nclip, ηw=copy(ctl.ηw),
                      feats, t=time() - t0, tw))
        # lag probe
        j = nb == 0 ? mod1(it ÷ max(ctl.probe_every, 1), M) : argmax([active[m] ? since[m] : -1 for m in 1:M])
        if ctl.probe_every > 0 && it % ctl.probe_every == 0 && length(hist[j]) >= 5
            pr = lag_probe(prng, P, S, LM, reps[j], hist[j], ctl)
            gap = pr.gap
            η0 = copy(ctl.ηw)
            if ctl.adapt
                for f in 1:nk
                    isfinite(gap[f]) || continue
                    if abs(gap[f]) > ctl.gap_hi
                        ctl.ηw[f] = max(ctl.ηmin, ctl.ηw[f] * ctl.shrink)
                    elseif abs(gap[f]) < ctl.gap_lo
                        ctl.ηw[f] = min(ctl.ηmax, ctl.ηw[f] * ctl.grow)
                    end
                end
            end
            push!(probes, (; it, replica=j, pr..., η0, η1=copy(ctl.ηw)))
            @printf("%s probe %4d (rep %d): gap/fk [%s]  eq rs [%s]  fk [%s]  ηw [%s]\n", label, it, j,
                    join([@sprintf("%+.2f", x) for x in gap], ", "), join([@sprintf("%.2e", x) for x in pr.rs_eq], ", "),
                    join([@sprintf("%.2e", x) for x in pr.fk], ", "), join([@sprintf("%.1e", x) for x in ctl.ηw], ", "))
        end
        if it % log_every == 0
            @printf("%s %4d (%5.0f s): k [%s] GW, capex %.4e op %.4e pf %.4e shed %.3g MWh, clip %d\n", label, it, time() - t0,
                    join([@sprintf("%.1f", x / 1e3) for x in tot], ", "), reps[1].parts.capex, mean(s.parts.op for s in reps),
                    mean(pf), mean(s.parts.shed for s in reps), nclip)
            flush(stdout)
        end
    end
    return reps, hist, since
end

"""
    lag_probe(prng, P, S, LM, s, hist, ctl) -> NamedTuple

Freeze (k, x) on a copy of replica `s`, run `probe_len` operation sweeps, and compare the mean
score of the last `probe_tail` sweeps (equilibrated) with the replica's in-flight score (mean
of its last `probe_win` steps), per carrier on the interior nodes: gap = ⟨r ŝ_in − r ŝ_eq⟩ /
⟨β_k ∂K⟩.
"""
function lag_probe(prng, P, S, LM, s, hist, ctl)
    r = P.βk / P.βv
    c = copy(s)
    acc = nothing; fk = nothing
    for q in 1:ctl.probe_len
        c = lgv_op!(prng, P, S.loc, LM, c)
        if q > ctl.probe_len - ctl.probe_tail
            sk, fk = lgv_scores(P, c)
            acc = acc === nothing ? [r .* x ./ ctl.probe_tail for x in sk] : acc .+ [r .* x ./ ctl.probe_tail for x in sk]
        end
    end
    inflight = sum(hist) ./ length(hist)
    mask = interior_mask(P, ML.knodes(P, s.ℓ, s.k))
    nk = length(acc)
    mm(v, f) = any(mask[f]) ? mean(v[mask[f]]) : NaN
    rs_eq = [mm(acc[f], f) for f in 1:nk]
    rs_in = [mm(inflight[f], f) for f in 1:nk]
    fkm = [mm(fk[f], f) for f in 1:nk]
    gap = (rs_in .- rs_eq) ./ fkm
    return (; gap, rs_eq, rs_in, fk=fkm, op_in=s.parts.op, op_eq=c.parts.op, shed_in=s.parts.shed, shed_eq=c.parts.shed)
end
