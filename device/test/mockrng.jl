"An RNG that returns pre-drawn normals (randn) and uniforms (rand) in order (for comparisons with the CPU reference)."
mutable struct MockRNG <: Random.AbstractRNG
    z::Vector{Float64}
    u::Vector{Float64}
    iz::Int
    iu::Int
end
MockRNG(z, u) = MockRNG(z, u, 0, 0)
Random.randn(r::MockRNG) = (r.iz += 1; r.z[r.iz])
Base.rand(r::MockRNG) = (r.iu += 1; r.u[r.iu])
