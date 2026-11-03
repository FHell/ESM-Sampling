# Multilevel Gaussian-mixture weather model (§3) on the Laplace-mode/time
# hierarchy.
#
# Coordinates: standardized anomalies x' (physical anomaly / scale_f) in the
# level layout of the weather fields. Coarsest level: regime-switching VAR(1)
# on the level-0 steps with harmonic seasonal regime means. Level ℓ → ℓ+1:
# non-centered Gaussian details ξ = A^{m,s,p} ρ + (Q^{m,s,p})^{1/2} w with
#   time details of an existing mode i:   ρ = [1, z_i(j-1), z_i(j), z_i(j+1)]
#   steps of a newly added mode i:        ρ = [1, z_1(j)]  (field mean mode)
# for coarse block j, its regime m, meteorological season s and phase of day p.
# Detail blocks are independent given the coarse level.

struct DetailModel
    r::Int
    P::Int
    AT::Vector{Array{Float64,4}}   # (r-1, 4, m_c, G)
    LT::Vector{Array{Float64,4}}   # (r-1, r-1, m_c, G)
    AS::Vector{Array{Float64,4}}   # (r, 2, m_f - m_c, G)
    LS::Vector{Array{Float64,4}}   # (r, r, m_f - m_c, G)
end

struct WeatherModel
    specs::Vector{FieldSpec}
    lays::Vector{Layout}
    trs::Vector{Transfer}
    scale::Vector{Float64}
    xbar::Vector{Vector{Float64}}        # climatology per level (physical units, level layout)
    K::Int
    Ptr::Matrix{Float64}                 # regime transition matrix
    π0::Vector{Float64}
    Bm::Vector{Matrix{Float64}}          # [Φ_k  M_k], n0 × (n0 + 3)
    Σinv::Vector{Matrix{Float64}}
    logdetΣ::Vector{Float64}
    Σ0inv::Matrix{Float64}
    logdetΣ0::Float64
    details::Vector{DetailModel}
    data::Vector{Vector{Vector{Float64}}}   # data[ℓ+1][year]
    γ::Vector{Matrix{Float64}}              # regime responsibilities per year, K × T0
    loglik::Vector{Float64}                 # EM trace
end

n0(W::WeatherModel) = W.lays[1].n ÷ W.lays[1].T
harmonics(τ, T0) = (1.0, cos(2π * τ / T0), sin(2π * τ / T0))

"Meteorological season (DJF=1, MAM=2, JJA=3, SON=4) of the level-0 step τ0."
function season(τ0::Int, h0::Int)
    d = (τ0 - 1) * h0 ÷ 24 + 1
    d >= 335 || d <= 59 ? 1 : d <= 151 ? 2 : d <= 243 ? 3 : 4
end

ancestor(j::Int, h::Int, h0::Int) = (j - 1) * h ÷ h0 + 1
phases(h::Int) = max(1, 24 ÷ h)
phase(j::Int, h::Int) = (j - 1) % phases(h) + 1
groupindex(k, s, p, P) = ((k - 1) * 4 + (s - 1)) * P + p
ngroups(K, P) = K * 4 * P

"Level-0 coordinates as an n0 × T0 matrix (rows: fields' modes stacked)."
function z0matrix(z::AbstractVector, lay::Layout, specs)
    reduce(vcat, [fieldview(z, lay, specs, f) for f in eachindex(specs)])
end
function z0vector(Z::AbstractMatrix, lay::Layout, specs)
    z = zeros(lay.n)
    r = 0
    for f in eachindex(specs)
        m = lay.m[f]
        fieldview(z, lay, specs, f) .= view(Z, r+1:r+m, :)
        r += m
    end
    return z
end

# --------------------------------------------------------------------------
# data preparation
# --------------------------------------------------------------------------

"Hourly node data (S × 8760 × Y) → step means (S × T × Y)."
function aggregate(X::Array{Float32,3}, h::Int)
    S, H, Y = size(X)
    T = H ÷ h
    out = zeros(S, T, Y)
    for y in 1:Y, t in 1:T, s in 1:S
        acc = 0.0
        for o in 1:h
            acc += X[s, (t-1)*h+o, y]
        end
        out[s, t, y] = acc / h
    end
    return out
end

"Hour-of-year climatology, smoothed over ±`window` days for each step of the day."
function climatology(X::Array{Float64,3}, h::Int; window=7)
    S, T, Y = size(X)
    q = phases(h)
    m = dropdims(mean(X; dims=3); dims=3)
    D = T ÷ q
    out = zeros(S, T)
    for p in 1:q, d in 1:D, s in 1:S
        acc = 0.0
        for o in -window:window
            dd = mod(d - 1 + o, D)
            acc += m[s, dd*q+p]
        end
        out[s, (d-1)*q+p] = acc / (2window + 1)
    end
    return out
end

# --------------------------------------------------------------------------
# coarsest level: regime-switching VAR(1), fitted by EM
# --------------------------------------------------------------------------

function emission_loglik(Z::AbstractMatrix, Bm, Σinv, logdetΣ, Σ0inv, logdetΣ0, T0)
    n, T = size(Z)
    K = length(Bm)
    ll = zeros(K, T)
    e = zeros(n)
    for k in 1:K, τ in 1:T
        if τ == 1
            ll[k, τ] = -0.5 * (dot(Z[:, 1], Σ0inv * Z[:, 1]) + logdetΣ0 + n * log(2π))
        else
            c, cs, sn = harmonics(τ, T0)
            B = Bm[k]
            @views e .= Z[:, τ] .- B[:, 1:n] * Z[:, τ-1] .- c .* B[:, n+1] .- cs .* B[:, n+2] .- sn .* B[:, n+3]
            ll[k, τ] = -0.5 * (dot(e, Σinv[k] * e) + logdetΣ[k] + n * log(2π))
        end
    end
    return ll
end

logsumexp(v) = (m = maximum(v); m + log(sum(exp.(v .- m))))

"Forward-backward; returns (γ, ξsum, loglik)."
function forward_backward(ll::Matrix{Float64}, Ptr, π0)
    K, T = size(ll)
    la = zeros(K, T)
    lP = log.(Ptr)
    la[:, 1] .= log.(π0) .+ ll[:, 1]
    for τ in 2:T, k in 1:K
        la[k, τ] = logsumexp(la[:, τ-1] .+ lP[:, k]) + ll[k, τ]
    end
    lb = zeros(K, T)
    for τ in T-1:-1:1, k in 1:K
        lb[k, τ] = logsumexp(lP[k, :] .+ ll[:, τ+1] .+ lb[:, τ+1])
    end
    L = logsumexp(la[:, T])
    γ = exp.(la .+ lb .- L)
    ξ = zeros(K, K)
    for τ in 1:T-1, i in 1:K, j in 1:K
        ξ[i, j] += exp(la[i, τ] + lP[i, j] + ll[j, τ+1] + lb[j, τ+1] - L)
    end
    return γ, ξ, L
end

"Forward filtering, backward sampling of the regime path."
function ffbs(rng, ll::Matrix{Float64}, Ptr, π0)
    K, T = size(ll)
    la = zeros(K, T)
    lP = log.(Ptr)
    la[:, 1] .= log.(π0) .+ ll[:, 1]
    for τ in 2:T, k in 1:K
        la[k, τ] = logsumexp(la[:, τ-1] .+ lP[:, k]) + ll[k, τ]
    end
    m = zeros(Int, T)
    m[T] = sample_log(rng, la[:, T])
    for τ in T-1:-1:1
        m[τ] = sample_log(rng, la[:, τ] .+ lP[:, m[τ+1]])
    end
    return m
end

function sample_log(rng, lw)
    w = exp.(lw .- maximum(lw))
    u = rand(rng) * sum(w)
    c = 0.0
    for k in eachindex(w)
        c += w[k]
        u <= c && return k
    end
    return length(w)
end

function kmeans_labels(rng, X::Matrix{Float64}, K; iters=30)
    n, N = size(X)
    C = X[:, randperm(rng, N)[1:K]]
    lab = zeros(Int, N)
    for _ in 1:iters
        for i in 1:N
            lab[i] = argmin([sum(abs2, X[:, i] .- C[:, k]) for k in 1:K])
        end
        for k in 1:K
            sel = findall(==(k), lab)
            isempty(sel) || (C[:, k] .= vec(mean(X[:, sel]; dims=2)))
        end
    end
    return lab
end

function fit_switching_var(rng, Zs::Vector{Matrix{Float64}}, K::Int; iters=60, ridge=1e-2)
    n, T0 = size(Zs[1])
    allZ = reduce(hcat, Zs)
    Σ0 = cov(allZ') + 1e-6I
    Σ0inv, logdetΣ0 = inv(Symmetric(Σ0)), logdet(Symmetric(Σ0))
    # initialize responsibilities from k-means on 7-day smoothed coarse weather
    sm = [hcat([vec(mean(Z[:, max(1, τ - 3):min(T0, τ + 3)]; dims=2)) for τ in 1:T0]...) for Z in Zs]
    lab = kmeans_labels(rng, reduce(hcat, sm), K)
    γs = [zeros(K, T0) for _ in Zs]
    for (y, γ) in enumerate(γs), τ in 1:T0
        γ[lab[(y-1)*T0+τ], τ] = 1.0
    end
    Bm = [zeros(n, n + 3) for _ in 1:K]
    Σinv = [Matrix{Float64}(I, n, n) for _ in 1:K]
    logdetΣ = zeros(K)
    Ptr = fill(0.1 / (K - 1), K, K) + (0.9 - 0.1 / (K - 1)) * I
    π0 = fill(1 / K, K)
    trace = Float64[]
    for it in 1:iters
        # ---- M step
        for k in 1:K
            XtX = zeros(n + 3, n + 3)
            XtY = zeros(n + 3, n)
            sw = 0.0
            for (y, Z) in enumerate(Zs), τ in 2:T0
                w = γs[y][k, τ]
                w < 1e-8 && continue
                x = vcat(Z[:, τ-1], collect(harmonics(τ, T0)))
                XtX .+= w .* (x * x')
                XtY .+= w .* (x * Z[:, τ]')
                sw += w
            end
            B = ((XtX + ridge * sw * I) \ XtY)'
            S = zeros(n, n)
            for (y, Z) in enumerate(Zs), τ in 2:T0
                w = γs[y][k, τ]
                w < 1e-8 && continue
                x = vcat(Z[:, τ-1], collect(harmonics(τ, T0)))
                e = Z[:, τ] .- B * x
                S .+= w .* (e * e')
            end
            S = S ./ sw + 1e-4I
            Bm[k] = B
            Σinv[k] = inv(Symmetric(S))
            logdetΣ[k] = logdet(Symmetric(S))
        end
        if it > 1
            ξtot = zeros(K, K)
            g1 = zeros(K)
            for (y, Z) in enumerate(Zs)
                ll = emission_loglik(Z, Bm, Σinv, logdetΣ, Σ0inv, logdetΣ0, T0)
                _, ξ, _ = forward_backward(ll, Ptr, π0)
                ξtot .+= ξ
                g1 .+= γs[y][:, 1]
            end
            Ptr = (ξtot .+ 1e-3) ./ sum(ξtot .+ 1e-3; dims=2)
            π0 = (g1 .+ 1e-3) ./ sum(g1 .+ 1e-3)
        end
        # ---- E step
        L = 0.0
        for (y, Z) in enumerate(Zs)
            ll = emission_loglik(Z, Bm, Σinv, logdetΣ, Σ0inv, logdetΣ0, T0)
            γ, _, Ly = forward_backward(ll, Ptr, π0)
            γs[y] = γ
            L += Ly
        end
        push!(trace, L)
        it > 5 && abs(trace[end] - trace[end-1]) < 1e-6 * abs(trace[end]) && break
    end
    return (; Bm, Σinv, logdetΣ, Σ0inv, logdetΣ0, Ptr, π0, γs, trace)
end

# --------------------------------------------------------------------------
# details
# --------------------------------------------------------------------------

function fit_details(specs, lc::Layout, lf::Layout, tr::Transfer, datac, dataf, γs, K, h0; ridge=1e-3, κ=20.0)
    r, P = tr.r, phases(lc.h)
    G = ngroups(K, P)
    splits = [split_state(tr, lc, lf, specs, dataf[y]) for y in eachindex(dataf)]
    AT, LT, AS, LS = Array{Float64,4}[], Array{Float64,4}[], Array{Float64,4}[], Array{Float64,4}[]
    for (f, s) in enumerate(specs)
        mc, mf = lc.m[f], lf.m[f]
        push!(AT, zeros(r - 1, 4, mc, G))
        push!(LT, zeros(r - 1, r - 1, mc, G))
        push!(AS, zeros(r, 2, mf - mc, G))
        push!(LS, zeros(r, r, mf - mc, G))
        for part in (:T, :S)
            dy = part == :T ? r - 1 : r
            dx = part == :T ? 4 : 2
            nm = part == :T ? mc : mf - mc
            (dy == 0 || nm == 0) && continue
            XtX = zeros(dx, dx, nm, G)
            XtY = zeros(dx, dy, nm, G)
            YtY = zeros(dy, dy, nm, G)
            sw = zeros(nm, G)
            x = zeros(dx)
            yv = zeros(dy)
            for y in eachindex(dataf)
                zc, ξ = splits[y]
                Zc = fieldview(zc, lc, specs, f)
                ξT, ξS = detail_views(ξ, tr, lc, lf, specs, f)
                for j in 1:lc.T
                    τ0 = ancestor(j, lc.h, h0)
                    sea, ph = season(τ0, h0), phase(j, lc.h)
                    for i in 1:nm
                        if part == :T
                            x .= (1.0, Zc[i, max(j - 1, 1)], Zc[i, j], Zc[i, min(j + 1, lc.T)])
                            yv .= view(ξT, i, :, j)
                        else
                            x .= (1.0, Zc[1, j])
                            yv .= view(ξS, i, :, j)
                        end
                        xx, xy, yy = x * x', x * yv', yv * yv'
                        for k in 1:K
                            w = γs[y][k, τ0]
                            w < 1e-6 && continue
                            g = groupindex(k, sea, ph, P)
                            @views XtX[:, :, i, g] .+= w .* xx
                            @views XtY[:, :, i, g] .+= w .* xy
                            @views YtY[:, :, i, g] .+= w .* yy
                            sw[i, g] += w
                        end
                    end
                end
            end
            Aout, Lout = part == :T ? (AT[f], LT[f]) : (AS[f], LS[f])
            for i in 1:nm
                # pooled statistics act as a prior for sparsely populated groups
                pXX = dropdims(sum(XtX[:, :, i, :]; dims=3); dims=3)
                pXY = dropdims(sum(XtY[:, :, i, :]; dims=3); dims=3)
                pYY = dropdims(sum(YtY[:, :, i, :]; dims=3); dims=3)
                psw = sum(sw[i, :])
                for g in 1:G
                    w0 = κ / max(psw, 1.0)
                    XX = XtX[:, :, i, g] .+ w0 .* pXX
                    XY = XtY[:, :, i, g] .+ w0 .* pXY
                    YY = YtY[:, :, i, g] .+ w0 .* pYY
                    W = sw[i, g] + w0 * psw
                    B = (XX + ridge * W * I) \ XY
                    Q = (YY .- XY' * B .- B' * XY .+ B' * XX * B) ./ W
                    Q = Symmetric(Q + (1e-8 + 1e-6 * tr_(Q) / dy) * I)
                    Aout[:, :, i, g] .= B'
                    Lout[:, :, i, g] .= cholesky(Q).L
                end
            end
        end
    end
    return DetailModel(r, P, AT, LT, AS, LS)
end

tr_(Q) = sum(Q[i, i] for i in axes(Q, 1))

"Detail coordinates ξ from the coarse state zc, regime path m and innovations w."
function details_from_w(D::DetailModel, specs, lc::Layout, lf::Layout, tr::Transfer, zc, m, w, h0)
    ξ = zeros(tr.n)
    r, P = D.r, D.P
    for (f, s) in enumerate(specs)
        Zc = fieldview(zc, lc, specs, f)
        ξT, ξS = detail_views(ξ, tr, lc, lf, specs, f)
        wT, wS = detail_views(w, tr, lc, lf, specs, f)
        mc, mf = lc.m[f], lf.m[f]
        for j in 1:lc.T
            τ0 = ancestor(j, lc.h, h0)
            g = groupindex(m[τ0], season(τ0, h0), phase(j, lc.h), P)
            if r > 1
                for i in 1:mc
                    ρ = (1.0, Zc[i, max(j - 1, 1)], Zc[i, j], Zc[i, min(j + 1, lc.T)])
                    A = view(D.AT[f], :, :, i, g)
                    L = view(D.LT[f], :, :, i, g)
                    for a in 1:r-1
                        acc = A[a, 1] * ρ[1] + A[a, 2] * ρ[2] + A[a, 3] * ρ[3] + A[a, 4] * ρ[4]
                        for b in 1:a
                            acc += L[a, b] * wT[i, b, j]
                        end
                        ξT[i, a, j] = acc
                    end
                end
            end
            for i in 1:mf-mc
                A = view(D.AS[f], :, :, i, g)
                L = view(D.LS[f], :, :, i, g)
                for a in 1:r
                    acc = A[a, 1] + A[a, 2] * Zc[1, j]
                    for b in 1:a
                        acc += L[a, b] * wS[i, b, j]
                    end
                    ξS[i, a, j] = acc
                end
            end
        end
    end
    return ξ
end

"Standardized anomalies at level ℓ from (m, z0, w₁…w_ℓ)."
function weather_at_level(W::WeatherModel, ℓ::Int, m, z0, ws)
    x = z0
    h0 = W.lays[1].h
    for j in 1:ℓ
        ξ = details_from_w(W.details[j], W.specs, W.lays[j], W.lays[j+1], W.trs[j], x, m, ws[j], h0)
        x = merge_state(W.trs[j], W.lays[j], W.lays[j+1], W.specs, x, ξ)
    end
    return x
end

"Node-level physical weather at level ℓ from standardized anomalies x."
function weather_nodes(W::WeatherModel, ℓ::Int, x::AbstractVector)
    lay = W.lays[ℓ+1]
    out = Matrix{Float64}[]
    for f in eachindex(W.specs)
        U = cols(W.specs[f].B, lay.m[f])
        Z = fieldview(W.xbar[ℓ+1], lay, W.specs, f) .+ W.scale[f] .* fieldview(x, lay, W.specs, f)
        push!(out, U * Z)
    end
    return out
end

"Chain rule: gradient w.r.t. node weather → gradient w.r.t. standardized anomalies x."
function weather_grad_to_x(W::WeatherModel, ℓ::Int, gX)
    lay = W.lays[ℓ+1]
    g = zeros(lay.n)
    for f in eachindex(W.specs)
        U = cols(W.specs[f].B, lay.m[f])
        fieldview(g, lay, W.specs, f) .= W.scale[f] .* (U' * gX[f])
    end
    return g
end

# --------------------------------------------------------------------------
# densities and sampling
# --------------------------------------------------------------------------

regime_loglik(W::WeatherModel, z0) =
    emission_loglik(z0matrix(z0, W.lays[1], W.specs), W.Bm, W.Σinv, W.logdetΣ, W.Σ0inv, W.logdetΣ0, W.lays[1].T)

"log p(z0 | m) (up to a constant) and its gradient."
function logp_z0(W::WeatherModel, m, z0)
    lay = W.lays[1]
    Z = z0matrix(z0, lay, W.specs)
    n, T0 = size(Z)
    G = zeros(n, T0)
    lp = -0.5 * dot(Z[:, 1], W.Σ0inv * Z[:, 1])
    G[:, 1] .-= W.Σ0inv * Z[:, 1]
    for τ in 2:T0
        k = m[τ]
        B = W.Bm[k]
        c, cs, sn = harmonics(τ, T0)
        e = Z[:, τ] .- B[:, 1:n] * Z[:, τ-1] .- c .* B[:, n+1] .- cs .* B[:, n+2] .- sn .* B[:, n+3]
        Se = W.Σinv[k] * e
        lp -= 0.5 * dot(e, Se) + 0.5 * W.logdetΣ[k]
        G[:, τ] .-= Se
        G[:, τ-1] .+= B[:, 1:n]' * Se
    end
    return lp, z0vector(G, lay, W.specs)
end

"Exact draw (m, z0, w₁…w_L) from the weather model."
function sample_weather(rng, W::WeatherModel, L::Int)
    lay = W.lays[1]
    T0 = lay.T
    n = n0(W)
    m = zeros(Int, T0)
    m[1] = sample_log(rng, log.(W.π0))
    for τ in 2:T0
        m[τ] = sample_log(rng, log.(W.Ptr[m[τ-1], :]))
    end
    Z = zeros(n, T0)
    Z[:, 1] .= cholesky(Symmetric(inv(W.Σ0inv))).L * randn(rng, n)
    for τ in 2:T0
        k = m[τ]
        B = W.Bm[k]
        c, cs, sn = harmonics(τ, T0)
        μ = B[:, 1:n] * Z[:, τ-1] .+ c .* B[:, n+1] .+ cs .* B[:, n+2] .+ sn .* B[:, n+3]
        Z[:, τ] .= μ .+ cholesky(Symmetric(inv(W.Σinv[k]))).L * randn(rng, n)
    end
    ws = [randn(rng, W.trs[j].n) for j in 1:L]
    return m, z0vector(Z, lay, W.specs), ws
end

"Content fingerprint of the weather tensors of a data file (cache keys survive additive changes to the file)."
function weather_fingerprint(path::String)
    h5open(path, "r") do f
        h = hash(:weather)
        for k in sort(collect(keys(f["weather"])))
            d = read(f["weather/$k/data"])
            h = hash((k, size(d), sum(Float64, d), sum(x -> Float64(x)^2, d), read(f["weather/$k/nodes"])), h)
        end
        h
    end
end

"""
    weather_model_cached(E, levels; K, seed, cachedir) -> WeatherModel

Fit the weather model with its own random stream (seed-derived, so that cached and
fresh fits agree) or load the fitted parameters from `cachedir`, keyed by the data,
the weather bases, the levels, K and the seed.
"""
function weather_model_cached(E::EnergyModel, levels; K::Int=4, seed::Int=1, cachedir::String="results/cache", datafile::String="")
    specs = E.wspecs
    key = hash((isempty(datafile) ? 0.0 : weather_fingerprint(datafile), levels, K, seed, [hash((s.B.U, s.B.support)) for s in specs], :weather_v2))
    path = joinpath(cachedir, "weather_$(string(key; base=16)).jls")
    lays = [Layout(specs, M, h) for (M, h) in levels]
    trs = [Transfer(specs, lays[j], lays[j+1]) for j in 1:length(lays)-1]
    if isfile(path)
        f = deserialize(path)
        return WeatherModel(specs, lays, trs, f...)
    end
    W = fit_weather_model(MersenneTwister(seed + 7919), E, levels; K=K)
    mkpath(cachedir)
    serialize(path, (W.scale, W.xbar, W.K, W.Ptr, W.π0, W.Bm, W.Σinv, W.logdetΣ, W.Σ0inv, W.logdetΣ0, W.details, W.data, W.γ, W.loglik))
    return W
end

"""
    fit_weather_model(rng, E, levels; K)

`levels` is the vector of (M_ℓ, h_ℓ). Uses all data years.
"""
function fit_weather_model(rng, E::EnergyModel, levels; K::Int=4)
    sys = E.sys
    specs = E.wspecs
    lays = [Layout(specs, M, h) for (M, h) in levels]
    trs = [Transfer(specs, lays[j], lays[j+1]) for j in 1:length(lays)-1]
    hL = levels[end][2]
    Lf = lays[end]
    Y = length(sys.years)
    xbarL = zeros(Lf.n)
    scale = zeros(length(specs))
    anoms = [zeros(Lf.n) for _ in 1:Y]
    for (f, s) in enumerate(specs)
        X = aggregate(sys.weather[s.name][2], hL)
        clim = climatology(X, hL)
        A = X .- clim
        scale[f] = std(A)
        U = cols(s.B, Lf.m[f])
        fieldview(xbarL, Lf, specs, f) .= U' * clim
        for y in 1:Y
            fieldview(anoms[y], Lf, specs, f) .= U' * (A[:, :, y] ./ scale[f])
        end
    end
    nl = length(lays)
    data = [Vector{Vector{Float64}}(undef, Y) for _ in 1:nl]
    xbar = Vector{Vector{Float64}}(undef, nl)
    data[nl] = anoms
    xbar[nl] = xbarL
    for ℓ in nl-1:-1:1
        xbar[ℓ], _ = split_state(trs[ℓ], lays[ℓ], lays[ℓ+1], specs, xbar[ℓ+1])
        for y in 1:Y
            data[ℓ][y], _ = split_state(trs[ℓ], lays[ℓ], lays[ℓ+1], specs, data[ℓ+1][y])
        end
    end
    Zs = [z0matrix(data[1][y], lays[1], specs) for y in 1:Y]
    fit = fit_switching_var(rng, Zs, K)
    h0 = lays[1].h
    details = [fit_details(specs, lays[j], lays[j+1], trs[j], data[j], data[j+1], fit.γs, K, h0) for j in 1:nl-1]
    return WeatherModel(specs, lays, trs, scale, xbar, K, fit.Ptr, fit.π0, fit.Bm, fit.Σinv, fit.logdetΣ,
                        fit.Σ0inv, fit.logdetΣ0, details, data, fit.γs, fit.trace)
end
