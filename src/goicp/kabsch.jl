# All matrices are DxN, where N is the number of positions and D is the dimensionality

# Here, P is the probe (to be rotated) and Q is the refereence
# https://en.wikipedia.org/wiki/Kabsch_algorithm
# This has been generalized to support weighted points:
# https://igl.ethz.ch/projects/ARAP/svd_rot.pdf

# assuming P and Q are already centered at the origin
# returns the rotation for alignment
function kabsch_centered(P,Q,w)
    @assert size(P) == size(Q)
    W = diagm(w/sum(w)) # here, the weights are assumed to sum to 1
    H = P*W*Q'
    D = Matrix{Float64}(I,size(H,1), size(H,2))
    # U,Σ,V = svd(H)
    U,Σ,V = GenericLinearAlgebra.svd(H)
    D[end] = sign(det(V*U'))
    return LinearMap(V * D * U')
end

function kabsch_centered(P,Q,matches::AbstractVector{<:Tuple{Int,Int}},wp=ones(size(P,2)),wq=ones(size(Q,2)))
    matchedP, matchedQ = matched_points(P,Q,matches)
    w = [wp[i]*wq[j] for (i,j) in matches]
    return kabsch_centered(matchedP, matchedQ, w)
end

kabsch_centered(P::PointSet, Q::PointSet) = kabsch_centered(P.coords, Q.coords, P.weights .* Q.weights);
kabsch_centered(P::PointSet, Q::PointSet, matches::AbstractVector{<:Tuple{Int,Int}}) = kabsch_centered(P.coords, Q.coords, matches, P.weights, Q.weights);

# transform DxN matrices
function (tform::Translation)(A::AbstractMatrix)
    return hcat([tform(A[:,i]) for i=1:size(A,2)]...)
end

# P and Q are not necessarily centered
# returns the transformation for alignment
function kabsch(P, Q, w::AbstractVector=ones(size(P,2)))
    @assert !any(w.<0) && sum(w)>0
    wn = w/sum(w) # weights should sum to 1 for computing the centroid
    centerP, centerQ = center_translation(P,wn), center_translation(Q,wn)
    R = kabsch_centered(centerP(P), centerQ(Q), wn)
    return inv(centerQ) ∘ R ∘ centerP
end

function kabsch(P,Q,matches::AbstractVector{<:Tuple{Int,Int}},wp=ones(size(P,2)),wq=ones(size(Q,2)))
    matchedP, matchedQ = matched_points(P,Q,matches)
    w = [wp[i]*wq[j] for (i,j) in matches]
    return kabsch(matchedP, matchedQ, w)
end

kabsch(P::PointSet, Q::PointSet) = kabsch(P.coords, Q.coords, P.weights .* Q.weights);
kabsch(P::PointSet, Q::PointSet, matches::AbstractVector{<:Tuple{Int,Int}}) = kabsch(P.coords, Q.coords, matches, P.weights, Q.weights);

function kabsch(P::AbstractMultiPointSet{N,T,K}, Q::AbstractMultiPointSet{N,T,K}, matchesdict, wp = weights(P), wq = weights(Q)) where {N,T,K}
    matchedP, matchedQ = matched_points(P,Q,matchesdict)
    w = Vector{T}()

    for (key, matches) in matchesdict
        for m in matches
            push!(w, wp[key][m[1]] * wq[key][m[2]])
        end
    end

    return kabsch(matchedP, matchedQ, w)
end

# centroid of positions in A, weighted by weights in w (assumed to sum to 1)
centroid(A, w=fill(1/size(A,2), size(A,2))) = A*w

# translation moving centroid to origin
center_translation(A, w=fill(1/size(A,2), size(A,2))) = Translation(-centroid(A,w))



# align via translation only (no rotation)
function translation_align(P,Q,w)
    @assert size(P) == size(Q)
    wnormalized = w/sum(w)
    dists = (Q - P) .* wnormalized'
    return Translation(sum(dists; dims=2))
end

function translation_align(P,Q,matches::AbstractVector{<:Tuple{Int,Int}},wp=ones(size(P,2)),wq=ones(size(Q,2)))
    matchedP, matchedQ = matched_points(P,Q,matches)
    w = [wp[i]*wq[j] for (i,j) in matches]
    return translation_align(matchedP, matchedQ, w)
end

translation_align(P::PointSet, Q::PointSet) = translation_align(P.coords, Q.coords, P.weights .* Q.weights)
translation_align(P::PointSet, Q::PointSet, matches::AbstractVector{<:Tuple{Int,Int}}) = translation_align(P.coords, Q.coords, matches, P.weights .* Q.weights)