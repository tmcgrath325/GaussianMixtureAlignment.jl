"""
    tgmm = tivgmm(gmm::IsotropicGMM, c=Inf)
    tgmm = tivgmm(mgmm::MultiGMM, c=Inf)

Returns a new `IsotropicGMM` or `MultiGMM` containing up to `c*length(gmm)` translation invariant vectors (TIVs) connecting Gaussian means in `gmm`.
TIVs are chosen to maximize length multiplied by the weights of the connected distributions. 

See [Li et. al. (2019)](https://arxiv.org/abs/1812.11307) for a description of TIV construction.
"""
function tivgmm(gmm::AbstractIsotropicGMM, c = Inf)
    t = numbertype(gmm)
    npts, ndims = size(gmm)
    n = ceil(c * npts)
    if npts^2 < n
        n = npts^2
    end
    scores = fill(zero(t), npts, npts)
    for i in 1:npts
        for j in i:npts
            scores[i, j] = scores[j, i] = norm(gmm.gaussians[i].μ - gmm.gaussians[j].μ) * √(gmm.gaussians[i].ϕ * gmm.gaussians[j].ϕ)
        end
    end

    tivgaussians = IsotropicGaussian{ndims, t}[]
    order = sortperm(vec(scores), rev = true)
    for idx in order[1:Int(n)]
        i = Int(floor((idx - 1) / npts) + 1)
        j = mod(idx - 1, npts) + 1
        x, y = gmm.gaussians[i], gmm.gaussians[j]
        push!(tivgaussians, IsotropicGaussian(x.μ - y.μ, √(x.σ * y.σ), √(x.ϕ * y.ϕ)))
    end
    return IsotropicGMM(tivgaussians)
end

"""
    tgmm = tivgmm(gmm::AbstractLabeledIsotropicGMM, c=Inf)

Build TIVs for a labeled GMM, returning a [`TIVGMM`](@ref) that keeps the widths, weights, and
labels of each TIV's two endpoint features: the TIV connecting feature `i` (head) to feature
`j` (tail) has mean `μᵢ - μⱼ`, head data from feature `i`, and tail data from feature `j`.
TIV selection is identical to the unlabeled method, except that zero-length TIVs (`i == j`)
are excluded: their overlap with every other TIV is independent of rotation, so they only add
a constant to the rotation objective. With `c = Inf` this yields `n² - n` TIVs for `n`
features.

The endpoint data let a TIV rotation search weight each TIV match by the interactions of both
endpoint pairs separately (see `tiv_pairwise_consts`).
"""
function tivgmm(gmm::AbstractLabeledIsotropicGMM{N, T, K}, c = Inf) where {N, T, K}
    npts = length(gmm)
    n = Int(min(ceil(c * npts), npts^2 - npts))
    scores = fill(zero(T), npts, npts)
    for i in 1:npts
        for j in i:npts
            scores[i, j] = scores[j, i] = norm(gmm.gaussians[i].μ - gmm.gaussians[j].μ) * √(gmm.gaussians[i].ϕ * gmm.gaussians[j].ϕ)
        end
    end

    tivgaussians = IsotropicGaussian{N, T}[]
    headσ, headϕ, headlabels = T[], T[], K[]
    tailσ, tailϕ, taillabels = T[], T[], K[]
    order = sortperm(vec(scores), rev = true)
    for idx in order
        length(tivgaussians) == n && break
        i = Int(floor((idx - 1) / npts) + 1)
        j = mod(idx - 1, npts) + 1
        i == j && continue
        x, y = gmm.gaussians[i], gmm.gaussians[j]
        push!(tivgaussians, IsotropicGaussian(x.μ - y.μ, √(x.σ * y.σ), √(x.ϕ * y.ϕ)))
        push!(headσ, x.σ)
        push!(headϕ, x.ϕ)
        push!(headlabels, gmm.labels[i])
        push!(tailσ, y.σ)
        push!(tailϕ, y.ϕ)
        push!(taillabels, gmm.labels[j])
    end
    return TIVGMM(tivgaussians, headσ, headϕ, headlabels, tailσ, tailϕ, taillabels)
end

function tivgmm(mgmm::AbstractIsotropicMultiGMM, c = Inf)
    gmms = Dict{Symbol, IsotropicGMM{dims(mgmm), numbertype(mgmm)}}()
    for key in keys(mgmm.gmms)
        push!(gmms, Pair(key, tivgmm(mgmm.gmms[key], c)))
    end
    return IsotropicMultiGMM(gmms)
end

"""
    tivpσ, tivpϕ = tiv_pairwise_consts(tivx, tivy, interactions)

Pairwise widths and weights for the TIV rotation stage. For generic TIV models with
`interactions === nothing`, this is just `pairwise_consts(tivx, tivy)`, which leaves the
unlabeled and `IsotropicMultiGMM` paths unchanged.

For a pair of [`TIVGMM`](@ref)s, each TIV pair is scored as the *sum* of a head-head and a
tail-tail feature overlap, matching the additive structure of the interaction-weighted model
overlap: `pσ[i,j]` and `pϕ[i,j]` hold the two terms' widths and weights as length-2 vectors,
consumed termwise by `overlap` and `gauss_l2_bounds`. With `interactions === nothing`, only
endpoint pairs with equal labels contribute, each with coefficient 1, mirroring the labeled
`pairwise_consts` default.

The two terms arise by apportioning the mismatch between matched TIVs to their endpoints. Any
shared translation splits the mismatch `D` between two TIVs into head and tail feature
displacements with `δ_head - δ_tail = D`; taking the variance-proportional split
`δ_head = (s_h/S)D`, `δ_tail = -(s_t/S)D` (where `s_h` and `s_t` are the summed squared widths
of the two head and the two tail features, and `S = s_h + s_t`) makes each endpoint overlap a
Gaussian in `‖D‖` with width `S²/s_h` (resp. `S²/s_t`) and weight equal to the endpoints'
interaction coefficient times their weight product.

Because the terms are summed rather than multiplied, a repulsive endpoint pair penalizes the
match (two repulsive pairs doubly so), and a match whose other endpoint pair does not interact
still contributes its interacting half.
"""
tiv_pairwise_consts(tivx::AbstractGMM, tivy::AbstractGMM, ::Nothing) = pairwise_consts(tivx, tivy)

function tiv_pairwise_consts(tivx::TIVGMM{N, T, K}, tivy::TIVGMM{N, S, K}, ::Nothing) where {N, T, S, K}
    t = promote_type(T, S)
    xlabels = unique!(vcat(tivx.headlabels, tivx.taillabels))
    ylabels = unique!(vcat(tivy.headlabels, tivy.taillabels))
    self_interactions = Dict{Tuple{K, K}, t}()
    for label in xlabels ∩ ylabels
        self_interactions[(label, label)] = one(t)
    end
    return tiv_pairwise_consts(tivx, tivy, self_interactions)
end

function tiv_pairwise_consts(
        tivx::TIVGMM{N, T, K}, tivy::TIVGMM{N, S, K}, interactions::Dict{Tuple{K, K}, V}
    ) where {N, T, S, K, V <: Number}
    validate_interactions(interactions) || throw(ArgumentError("Interactions must not include redundant key pairs (i.e. (k1,k2) and (k2,k1))"))
    t = promote_type(T, S, V)

    # each label pair's coefficient is resolved once into `coefs` and thereafter indexed,
    # rather than hashed again for every TIV pair (see `pairwise_consts`)
    uxs = unique!(vcat(tivx.headlabels, tivx.taillabels))
    uys = unique!(vcat(tivy.headlabels, tivy.taillabels))
    coefs = t[interaction_coefficient(interactions, kx, ky) for kx in uxs, ky in uys]
    hix = [findfirst(isequal(l), uxs)::Int for l in tivx.headlabels]
    tix = [findfirst(isequal(l), uxs)::Int for l in tivx.taillabels]
    hiy = [findfirst(isequal(l), uys)::Int for l in tivy.headlabels]
    tiy = [findfirst(isequal(l), uys)::Int for l in tivy.taillabels]

    pσ = Matrix{SVector{2, t}}(undef, length(tivx), length(tivy))
    pϕ = Matrix{SVector{2, t}}(undef, length(tivx), length(tivy))
    for i in eachindex(tivx.gaussians)
        for j in eachindex(tivy.gaussians)
            s_h = tivx.headσ[i]^2 + tivy.headσ[j]^2
            s_t = tivx.tailσ[i]^2 + tivy.tailσ[j]^2
            s_sum = s_h + s_t
            pσ[i, j] = SVector(s_sum^2 / s_h, s_sum^2 / s_t)
            pϕ[i, j] = SVector(
                coefs[hix[i], hiy[j]] * tivx.headϕ[i] * tivy.headϕ[j],
                coefs[tix[i], tiy[j]] * tivx.tailϕ[i] * tivy.tailϕ[j]
            )
        end
    end
    return pσ, pϕ
end
