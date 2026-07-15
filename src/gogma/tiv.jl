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

Build TIVs for a labeled GMM, returning a `LabeledIsotropicGMM` whose label type is the
ordered pair `Tuple{K,K}`: the TIV connecting feature `i` (head) to feature `j` (tail) carries
the label `(gmm.labels[i], gmm.labels[j])`. TIV selection is identical to the unlabeled method.

The paired labels let a TIV rotation search weight TIV matches by their endpoints' interactions
(see `tiv_pairwise_consts`).
"""
function tivgmm(gmm::AbstractLabeledIsotropicGMM{N, T, K}, c = Inf) where {N, T, K}
    npts = length(gmm)
    n = ceil(c * npts)
    if npts^2 < n
        n = npts^2
    end
    scores = fill(zero(T), npts, npts)
    for i in 1:npts
        for j in i:npts
            scores[i, j] = scores[j, i] = norm(gmm.gaussians[i].μ - gmm.gaussians[j].μ) * √(gmm.gaussians[i].ϕ * gmm.gaussians[j].ϕ)
        end
    end

    tivgaussians = IsotropicGaussian{N, T}[]
    tivlabels = Tuple{K, K}[]
    order = sortperm(vec(scores), rev = true)
    for idx in order[1:Int(n)]
        i = Int(floor((idx - 1) / npts) + 1)
        j = mod(idx - 1, npts) + 1
        x, y = gmm.gaussians[i], gmm.gaussians[j]
        push!(tivgaussians, IsotropicGaussian(x.μ - y.μ, √(x.σ * y.σ), √(x.ϕ * y.ϕ)))
        push!(tivlabels, (gmm.labels[i], gmm.labels[j]))
    end
    return LabeledIsotropicGMM(tivgaussians, tivlabels)
end

function tivgmm(mgmm::AbstractIsotropicMultiGMM, c = Inf)
    gmms = Dict{Symbol, IsotropicGMM{dims(mgmm), numbertype(mgmm)}}()
    for key in keys(mgmm.gmms)
        push!(gmms, Pair(key, tivgmm(mgmm.gmms[key], c)))
    end
    return IsotropicMultiGMM(gmms)
end

"""
    tiv_interactions(interactions::Dict{Tuple{K,K},V}, plx, ply)

Derive the interaction coefficients between paired-label TIVs from the per-label `interactions`.
`plx` and `ply` are the distinct TIV labels (pairs `(head, tail)`) of the two TIV models. The
coefficient between TIV labels `(la, lb)` and `(lc, ld)` is the product of the endpoint
interactions `interaction_coefficient(la, lc) * interaction_coefficient(lb, ld)`, i.e. the
compatibility of matching both endpoints simultaneously.

Only one ordering of each unordered TIV-label pair is stored, as required by
[`pairwise_consts`](@ref); the coefficient is symmetric under swapping the two TIV labels, so
the omitted ordering resolves to the same value.
"""
function tiv_interactions(interactions::Dict{Tuple{K, K}, V}, plx, ply) where {K, V <: Number}
    out = Dict{Tuple{Tuple{K, K}, Tuple{K, K}}, V}()
    for p in plx
        for q in ply
            has_interaction(out, p, q) && continue
            out[(p, q)] = interaction_coefficient(interactions, p[1], q[1]) *
                interaction_coefficient(interactions, p[2], q[2])
        end
    end
    return out
end

"""
    tivpσ, tivpϕ = tiv_pairwise_consts(tivx, tivy, interactions)

Pairwise widths and weights for the TIV rotation stage. With `interactions === nothing`, this is
just `pairwise_consts(tivx, tivy)`. With a per-label `interactions` dictionary and paired-label
TIV models (from `tivgmm` on `AbstractLabeledIsotropicGMM`s), the TIV-level coefficients are
derived via [`tiv_interactions`](@ref) and applied through `pairwise_consts`.
"""
tiv_pairwise_consts(tivx::AbstractGMM, tivy::AbstractGMM, ::Nothing) = pairwise_consts(tivx, tivy)

function tiv_pairwise_consts(
        tivx::AbstractLabeledIsotropicGMM{N, T, Tuple{K, K}},
        tivy::AbstractLabeledIsotropicGMM{N, S, Tuple{K, K}},
        interactions::Dict{Tuple{K, K}, V}
    ) where {N, T, S, K, V <: Number}
    derived = tiv_interactions(interactions, unique(tivx.labels), unique(tivy.labels))
    return pairwise_consts(tivx, tivy, derived)
end
