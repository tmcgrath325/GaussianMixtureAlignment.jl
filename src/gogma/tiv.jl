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

"""
    tgmm = tivgmm(gmm::AbstractStackedLabeledIsotropicGMM, c=Inf)

Build TIVs for a stacked labeled GMM, returning a [`StackedTIVGMM`](@ref) that keeps the
slot-wise widths, amplitudes, and labels of each TIV's two endpoint stacks: the TIV
connecting stacked point `i` (head) to stacked point `j` (tail) has mean `μᵢ - μⱼ`, head
slots from point `i`, and tail slots from point `j`.

TIV selection mirrors the labeled method, scoring each candidate by its length times the
geometric mean of the endpoints' total amplitudes; `c` counts stacked points, so with
`c = Inf` this yields `n² - n` TIVs for `n` stacked points (zero-length TIVs, `i == j`, are
excluded as rotation-independent).
"""
function tivgmm(gmm::AbstractStackedLabeledIsotropicGMM{N, T, L, K}, c = Inf) where {N, T, L, K}
    npts = length(gmm)
    n = Int(min(ceil(c * npts), npts^2 - npts))
    ϕtot = weights(gmm)
    σagg = widths(gmm)
    scores = fill(zero(T), npts, npts)
    for i in 1:npts
        for j in i:npts
            scores[i, j] = scores[j, i] = norm(gmm.gaussians[i].μ - gmm.gaussians[j].μ) * √(ϕtot[i] * ϕtot[j])
        end
    end

    tivgaussians = IsotropicGaussian{N, T}[]
    headσ, headϕ, headlabels = SVector{L, T}[], SVector{L, T}[], SVector{L, K}[]
    tailσ, tailϕ, taillabels = SVector{L, T}[], SVector{L, T}[], SVector{L, K}[]
    order = sortperm(vec(scores), rev = true)
    for idx in order
        length(tivgaussians) == n && break
        i = Int(floor((idx - 1) / npts) + 1)
        j = mod(idx - 1, npts) + 1
        i == j && continue
        x, y = gmm.gaussians[i], gmm.gaussians[j]
        push!(tivgaussians, IsotropicGaussian(x.μ - y.μ, √(σagg[i] * σagg[j]), √(ϕtot[i] * ϕtot[j])))
        push!(headσ, x.σ)
        push!(headϕ, x.ϕ)
        push!(headlabels, x.labels)
        push!(tailσ, y.σ)
        push!(tailϕ, y.ϕ)
        push!(taillabels, y.labels)
    end
    return StackedTIVGMM{N, T, L, K}(tivgaussians, headσ, headϕ, headlabels, tailσ, tailϕ, taillabels)
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

function tiv_pairwise_consts(tivx::StackedTIVGMM{N, T, Lx, K}, tivy::StackedTIVGMM{N, S, Ly, K}, ::Nothing) where {N, T, S, Lx, Ly, K}
    t = promote_type(T, S)
    xlabels = unique!([l for ls in Iterators.flatten((tivx.headlabels, tivx.taillabels)) for l in ls])
    ylabels = unique!([l for ls in Iterators.flatten((tivy.headlabels, tivy.taillabels)) for l in ls])
    self_interactions = Dict{Tuple{K, K}, t}()
    for label in xlabels ∩ ylabels
        self_interactions[(label, label)] = one(t)
    end
    return tiv_pairwise_consts(tivx, tivy, self_interactions)
end

# For a stacked TIV pair, the head/tail variance split of the scalar TIV kernel is applied
# per pairing of endpoint slots: choosing head slots (a, c) and tail slots (b, d) fixes
# `s_h = σ_h[a]² + σ_h[c]²`, `s_t = σ_t[b]² + σ_t[d]²`, and `S = s_h + s_t`, contributing a
# head term of width `S²/s_h` and a tail term of width `S²/s_t`, each weighted by that
# endpoint slot pair's interaction coefficient and amplitude product. This enumerates
# `2⋅Lx²⋅Ly²` terms per TIV pair — exactly the terms a mean-duplicated model produces, so
# stacked and duplicated models give identical TIV overlaps, while the distance bounds are
# evaluated once per TIV pair instead of once per slot pairing.
#
# A zero-amplitude slot is an absent feature: a duplicated model has no TIV ending on one, so
# no term whose split involves that slot is emitted at all — not only the terms whose own
# amplitude product vanishes. Were such a term emitted, a head term would be counted once per
# padded tail pairing, with a width computed from the padding σ. Iterating only the real slots
# supplies that exclusion directly.
#
# Only terms with a nonzero weight are stored, so the `SVector` length is the largest such
# count over all pairs rather than `2⋅Lx²⋅Ly²`. Term position carries no meaning — every
# consumer reads `pσ[i,j][m]` and `pϕ[i,j][m]` together and sums over `m` — but the two must
# be filled from the same terms, or a weight is silently paired with the wrong variance.
function tiv_pairwise_consts(
        tivx::StackedTIVGMM{N, T, Lx, K}, tivy::StackedTIVGMM{N, S, Ly, K}, interactions::Dict{Tuple{K, K}, V}
    ) where {N, T, S, Lx, Ly, K, V <: Number}
    validate_interactions(interactions) || throw(ArgumentError("Interactions must not include redundant key pairs (i.e. (k1,k2) and (k2,k1))"))
    t = promote_type(T, S, V)

    uxs = unique!([l for ls in Iterators.flatten((tivx.headlabels, tivx.taillabels)) for l in ls])
    uys = unique!([l for ls in Iterators.flatten((tivy.headlabels, tivy.taillabels)) for l in ls])
    coefs = t[interaction_coefficient(interactions, kx, ky) for kx in uxs, ky in uys]
    hix = [map(l -> findfirst(isequal(l), uxs)::Int, ls) for ls in tivx.headlabels]
    tix = [map(l -> findfirst(isequal(l), uxs)::Int, ls) for ls in tivx.taillabels]
    hiy = [map(l -> findfirst(isequal(l), uys)::Int, ls) for ls in tivy.headlabels]
    tiy = [map(l -> findfirst(isequal(l), uys)::Int, ls) for ls in tivy.taillabels]

    rxh = [findall(!iszero, ϕ) for ϕ in tivx.headϕ]
    rxt = [findall(!iszero, ϕ) for ϕ in tivx.tailϕ]
    ryh = [findall(!iszero, ϕ) for ϕ in tivy.headϕ]
    ryt = [findall(!iszero, ϕ) for ϕ in tivy.tailϕ]

    # Each head term is emitted once per real tail pairing and each tail term once per real
    # head pairing, so the stored count follows from two `O(L²)` tallies rather than an
    # `O(L⁴)` enumeration.
    Q = 0
    for i in eachindex(tivx.gaussians)
        xhϕ, xtϕ, chx, ctx = tivx.headϕ[i], tivx.tailϕ[i], hix[i], tix[i]
        for j in eachindex(tivy.gaussians)
            yhϕ, ytϕ, chy, cty = tivy.headϕ[j], tivy.tailϕ[j], hiy[j], tiy[j]
            nh = 0
            for a in rxh[i], c in ryh[j]
                iszero(coefs[chx[a], chy[c]] * xhϕ[a] * yhϕ[c]) || (nh += 1)
            end
            nt = 0
            for b in rxt[i], d in ryt[j]
                iszero(coefs[ctx[b], cty[d]] * xtϕ[b] * ytϕ[d]) || (nt += 1)
            end
            Q = max(Q, nh * length(rxt[i]) * length(ryt[j]) + nt * length(rxh[i]) * length(ryh[j]))
        end
    end
    return stacked_tiv_consts(Val(Q), tivx, tivy, coefs, hix, tix, hiy, tiy, rxh, rxt, ryh, ryt, t)
end

function stacked_tiv_consts(::Val{Q}, tivx, tivy, coefs, hix, tix, hiy, tiy, rxh, rxt, ryh, ryt, ::Type{t}) where {Q, t}
    pσ = Matrix{SVector{Q, t}}(undef, length(tivx), length(tivy))
    pϕ = Matrix{SVector{Q, t}}(undef, length(tivx), length(tivy))
    sbuf = MVector{Q, t}(undef)
    wbuf = MVector{Q, t}(undef)
    for i in eachindex(tivx.gaussians)
        xhσ, xtσ, xhϕ, xtϕ = tivx.headσ[i], tivx.tailσ[i], tivx.headϕ[i], tivx.tailϕ[i]
        chx, ctx = hix[i], tix[i]
        for j in eachindex(tivy.gaussians)
            yhσ, ytσ, yhϕ, ytϕ = tivy.headσ[j], tivy.tailσ[j], tivy.headϕ[j], tivy.tailϕ[j]
            chy, cty = hiy[j], tiy[j]
            m = 0
            for a in rxh[i], c in ryh[j]
                s_h = xhσ[a]^2 + yhσ[c]^2
                w_h = coefs[chx[a], chy[c]] * xhϕ[a] * yhϕ[c]
                for b in rxt[i], d in ryt[j]
                    s_t = xtσ[b]^2 + ytσ[d]^2
                    s_sum = s_h + s_t
                    w_t = coefs[ctx[b], cty[d]] * xtϕ[b] * ytϕ[d]
                    if !iszero(w_h)
                        m += 1
                        sbuf[m] = s_sum^2 / s_h
                        wbuf[m] = w_h
                    end
                    if !iszero(w_t)
                        m += 1
                        sbuf[m] = s_sum^2 / s_t
                        wbuf[m] = w_t
                    end
                end
            end
            # unused slots must carry a positive variance: they are never read, but a zero
            # would divide by zero if one ever were
            for k in (m + 1):Q
                sbuf[k] = one(t)
                wbuf[k] = zero(t)
            end
            pσ[i, j] = SVector(sbuf)
            pϕ[i, j] = SVector(wbuf)
        end
    end
    return pσ, pϕ
end

# mixed TIV models lift the unstacked side to a single-slot StackedTIVGMM; Nothing and Dict
# methods keep these more specific than the `(AbstractGMM, AbstractGMM, Nothing)` fallback
tiv_pairwise_consts(tivx::StackedTIVGMM{N, T, L, K}, tivy::TIVGMM{N, S, K}, interactions::Nothing) where {N, T, L, K, S} =
    tiv_pairwise_consts(tivx, StackedTIVGMM(tivy), interactions)
tiv_pairwise_consts(tivx::StackedTIVGMM{N, T, L, K}, tivy::TIVGMM{N, S, K}, interactions::Dict{Tuple{K, K}, V}) where {N, T, L, K, S, V <: Number} =
    tiv_pairwise_consts(tivx, StackedTIVGMM(tivy), interactions)
tiv_pairwise_consts(tivx::TIVGMM{N, T, K}, tivy::StackedTIVGMM{N, S, L, K}, interactions::Nothing) where {N, T, K, S, L} =
    tiv_pairwise_consts(StackedTIVGMM(tivx), tivy, interactions)
tiv_pairwise_consts(tivx::TIVGMM{N, T, K}, tivy::StackedTIVGMM{N, S, L, K}, interactions::Dict{Tuple{K, K}, V}) where {N, T, K, S, L, V <: Number} =
    tiv_pairwise_consts(StackedTIVGMM(tivx), tivy, interactions)
