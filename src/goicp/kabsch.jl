# The implementation that was originally developed here got upstreamed into CoordinateTransformations:
#    https://github.com/JuliaGeometry/CoordinateTransformations.jl/pull/97
# This now extends the upstream implementation.

CoordinateTransformations.kabsch_centered(P::PointSet, Q::PointSet) = kabsch_centered(P.coords, Q.coords, P.weights .* Q.weights);
CoordinateTransformations.kabsch(P::PointSet, Q::PointSet) = kabsch(P.coords => Q.coords, P.weights .* Q.weights);

function kabsch_centered_matches(P,Q,matches::AbstractVector{<:Tuple{Int,Int}},wp=ones(size(P,2)),wq=ones(size(Q,2)))
    matchedP, matchedQ = matched_points(P,Q,matches)
    w = [wp[i]*wq[j] for (i,j) in matches]
    return kabsch_centered(matchedP, matchedQ, w)
end

kabsch_centered_matches(P::PointSet, Q::PointSet, matches::AbstractVector{<:Tuple{Int,Int}}) = kabsch_centered_matches(P.coords, Q.coords, matches, P.weights, Q.weights);

# transform DxN matrices
function transform_columns(tform::Translation, A::AbstractMatrix)
    return reduce(hcat, [tform(A[:,i]) for i=1:size(A,2)])
end
function transform_columns(tform::AffineMap, A::AbstractMatrix)
    l = LinearMap(tform.linear)
    t = Translation(tform.translation)
    transform_columns(t, l(A))
end
transform_columns(tform::Union{Translation,AffineMap}, P::PointSet) = PointSet(transform_columns(tform, P.coords), P.weights)


function kabsch_matches(P,Q,matches::AbstractVector{<:Tuple{Int,Int}},wp=ones(size(P,2)),wq=ones(size(Q,2)))
    matchedP, matchedQ = matched_points(P,Q,matches)
    w = [wp[i]*wq[j] for (i,j) in matches]
    return kabsch(matchedP => matchedQ, w)
end

kabsch_matches(P::PointSet, Q::PointSet, matches::AbstractVector{<:Tuple{Int,Int}}) = kabsch_matches(P.coords, Q.coords, matches, P.weights, Q.weights);

function kabsch_matches(P::AbstractMultiPointSet{N,T,K}, Q::AbstractMultiPointSet{N,T,K}, matchesdict::Dict, wp = weights(P), wq = weights(Q)) where {N,T,K}
    matchedP, matchedQ = matched_points(P,Q,matchesdict)
    w = Vector{T}()

    for (key, matches) in matchesdict
        for m in matches
            push!(w, wp[key][m[1]] * wq[key][m[2]])
        end
    end

    return kabsch(matchedP => matchedQ, w)
end

# align via translation only (no rotation)
function translation_align(P,Q,w)
    @assert size(P) == size(Q)
    wnormalized = w/sum(w)
    dists = (Q - P) .* wnormalized'
    return Translation(sum(dists; dims=2))
end

function translation_align_matches(P,Q,matches::AbstractVector{<:Tuple{Int,Int}},wp=ones(size(P,2)),wq=ones(size(Q,2)))
    matchedP, matchedQ = matched_points(P,Q,matches)
    w = [wp[i]*wq[j] for (i,j) in matches]
    return translation_align(matchedP, matchedQ, w)
end

translation_align(P::PointSet, Q::PointSet) = translation_align(P.coords, Q.coords, P.weights .* Q.weights)
translation_align_matches(P::PointSet, Q::PointSet, matches::AbstractVector{<:Tuple{Int,Int}}) = translation_align_matches(P.coords, Q.coords, matches, P.weights .* Q.weights)