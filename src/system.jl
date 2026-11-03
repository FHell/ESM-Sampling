# Network and technology data, read from the HDF5 file written by
# scripts/prepare_data.py (h5py writes row-major, so array axes are reversed).

struct System
    N::Int
    nodes::Vector{String}
    country::Vector{String}
    x::Vector{Float64}
    y::Vector{Float64}
    line_bus0::Vector{Int}
    line_bus1::Vector{Int}
    line_b::Vector{Float64}
    line_s::Vector{Float64}
    link_bus0::Vector{Int}
    link_bus1::Vector{Int}
    link_cap::Vector{Float64}
    gen::Dict{Symbol,Any}
    stor::Dict{Symbol,Any}
    weather::Dict{Symbol,Tuple{Vector{Int},Array{Float32,3}}}  # field => (nodes, data[node, hour, year])
    years::Vector{Int}
    line_alpha::Vector{Float64}      # quadratic loss coefficient per line (loss = α f², MW)
    link_eta::Vector{Float64}        # DC link efficiency (1: lossless)
    bfgen::Dict{Symbol,Any}          # fixed brownfield fleets (coal, lignite, ...), used with brownfield = true
    co2_data::Float64                # CO₂ price (€/t) contained in the marginal costs of the file
end

"Fill the model-completion keys of a generator table (older files): ef, k0, mc0, ef0."
function completion_keys!(gen::Dict{Symbol,Any})
    n = length(gen[:mc])
    haskey(gen, :ef) || (gen[:ef] = zeros(n))
    haskey(gen, :k0) || (gen[:k0] = zeros(n))
    haskey(gen, :mc0) || (gen[:mc0] = copy(gen[:mc]))
    haskey(gen, :ef0) || (gen[:ef0] = copy(gen[:ef]))
    return gen
end

function load_system(path::String)
    h5open(path, "r") do f
        str(x) = String.(x)
        nodes = str(read(f["network/nodes"]))
        N = length(nodes)
        idx(x) = Int.(round.(x)) .+ 1
        net = f["network"]
        gen = Dict{Symbol,Any}()
        for k in keys(f["generators"])
            v = read(f["generators/$k"])
            gen[Symbol(k)] = k == "carrier" ? str(v) : Float64.(v)
        end
        gen[:bus] = idx(gen[:bus])
        stor = Dict{Symbol,Any}()
        for k in keys(f["storage"])
            v = read(f["storage/$k"])
            stor[Symbol(k)] = k == "carrier" ? str(v) : Float64.(v)
        end
        stor[:bus] = idx(stor[:bus])
        weather = Dict{Symbol,Tuple{Vector{Int},Array{Float32,3}}}()
        for k in keys(f["weather"])
            weather[Symbol(k)] = (idx(read(f["weather/$k/nodes"])), read(f["weather/$k/data"]))
        end
        completion_keys!(gen)
        bfgen = Dict{Symbol,Any}()
        if haskey(f, "brownfield/generators")
            for k in keys(f["brownfield/generators"])
                v = read(f["brownfield/generators/$k"])
                bfgen[Symbol(k)] = k == "carrier" ? str(v) : Float64.(v)
            end
            bfgen[:bus] = idx(bfgen[:bus])
            completion_keys!(bfgen)
        end
        nl, nk = length(read(net["line_s"])), length(read(net["link_cap"]))
        attrs = HDF5.attributes(f)
        System(N, nodes, str(read(net["country"])), read(net["x"]), read(net["y"]),
               idx(read(net["line_bus0"])), idx(read(net["line_bus1"])), read(net["line_b"]), read(net["line_s"]),
               idx(read(net["link_bus0"])), idx(read(net["link_bus1"])), read(net["link_cap"]),
               gen, stor, weather, Int.(read(attrs["years"])),
               haskey(net, "line_alpha") ? read(net["line_alpha"]) : zeros(nl),
               haskey(net, "link_eta") ? read(net["link_eta"]) : ones(nk),
               bfgen, haskey(attrs, "co2_price") ? Float64(read(attrs["co2_price"])) : 0.0)
    end
end

"Edges of the transfer graph (lines and links) used for the Haar contrasts: (a, b, capacity)."
transfer_edges(sys::System) =
    vcat([(a, b, s) for (a, b, s) in zip(sys.line_bus0, sys.line_bus1, sys.line_s)],
         [(a, b, c) for (a, b, c) in zip(sys.link_bus0, sys.link_bus1, sys.link_cap)])

"""
    subsystem(sys, nodes) -> System

The sub-network on `nodes` (indices into `sys`): buses, internal lines and links,
generators, storage and weather of those nodes, indices remapped. For small exactness
tests.
"""
function subsystem(sys::System, nodes::Vector{Int})
    map_ = Dict(n => i for (i, n) in enumerate(nodes))
    keepl = [e for e in eachindex(sys.line_s) if haskey(map_, sys.line_bus0[e]) && haskey(map_, sys.line_bus1[e])]
    keepk = [l for l in eachindex(sys.link_cap) if haskey(map_, sys.link_bus0[l]) && haskey(map_, sys.link_bus1[l])]
    sub(d::Dict, key) = begin
        isempty(d) && return Dict{Symbol,Any}()
        u = findall(b -> haskey(map_, b), d[:bus])
        out = Dict{Symbol,Any}()
        for (k, v) in d
            out[k] = k == :bus ? [map_[b] for b in v[u]] : v[u]
        end
        out
    end
    weather = Dict{Symbol,Tuple{Vector{Int},Array{Float32,3}}}()
    for (k, (wn, data)) in sys.weather
        idx = findall(n -> haskey(map_, n), wn)
        weather[k] = ([map_[wn[i]] for i in idx], data[idx, :, :])
    end
    return System(length(nodes), sys.nodes[nodes], sys.country[nodes], sys.x[nodes], sys.y[nodes],
                  [map_[b] for b in sys.line_bus0[keepl]], [map_[b] for b in sys.line_bus1[keepl]], sys.line_b[keepl], sys.line_s[keepl],
                  [map_[b] for b in sys.link_bus0[keepk]], [map_[b] for b in sys.link_bus1[keepk]], sys.link_cap[keepk],
                  sub(sys.gen, :gen), sub(sys.stor, :stor), weather, sys.years,
                  sys.line_alpha[keepl], sys.link_eta[keepk], sub(sys.bfgen, :gen), sys.co2_data)
end
