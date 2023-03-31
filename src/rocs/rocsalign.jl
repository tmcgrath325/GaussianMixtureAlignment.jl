struct ROCSAlignmentResult{D,S,T,F<:AbstractAffineMap,X<:AbstractGMM{D,S},Y<:AbstractGMM{D,T}} <: AlignmentResults
    x::X
    y::Y
    minimum::T
    tform::F
end

""" 
    m = second_moment(gmm, center, dim1, dim2)

Returns the second order moment of `gmm`
"""
function mass_matrix(positions::AbstractMatrix{<:Real}, 
                     weights=ones(eltype(positions),size(positions,2)), 
                     widths=zeros(eltype(positions),size(positions,2)), 
                     center=centroid(positions,weights))
    t = eltype(positions)
    npts = size(positions,2)
    M = fill(zero(t), 3, 3)
    dists = [positions[:,i].-center for i=1:npts]
    # diagonal terms
    for i=1:3
        for j=1:npts
            M[i,i] += weights[j] * (dists[j][i]^2 + widths[j]^2)
        end
    end
    # off-diagonal terms
    for i=1:3
        for j=i+1:3
            for k=1:npts
                inc = weights[k] * dists[k][i] * dists[k][j]
                M[i,j] += inc
                M[j,i] += inc
            end
        end
    end
    return M
end


"""
    tforms = inertial_transforms(gmm)

Returns 10 transformations to put `gmm` in an inertial frame. That is, the mass matrix of the GMM
is made diagonal, and the GMM center of mass is made the origin.
"""
function inertial_transforms(positions::AbstractMatrix{<:Real},
                             weights=ones(eltype(positions),size(positions,2)),
                             widths=zeros(eltype(positions),size(positions,2)); 
                             invert = false)
    com = centroid(positions, weights / sum(weights))
    massmat = mass_matrix(positions, weights, widths, com)
    evecs = eigvecs(massmat)

    T = eltype(com)
    N = length(com)

    # obtain all rotations that align the mass matrix to the coordinate system
    tforms = AffineMap{SMatrix{N,N,T,N^2},SVector{N,T}}[]

    # first, align the the eigenvectors to the coordinate system axes
    # make sure that a reflection is not performed
    if det(evecs) < 0 
        evecs[:,end] = -evecs[:,end]
    end

    push!(tforms, inv(LinearMap(SMatrix{length(com),length(com)}(evecs))) ∘ Translation(-com))

    # then consider unique rotations made up of 180 degree rotations about the coordinate axes (out of all 2^3, there are 4 unique poses)
    for i=1:3
        axis = zeros(T,N)
        axis[i] = one(T)
        push!(tforms, LinearMap(AngleAxis(π, axis...)) ∘  tforms[1])
    end
    if invert
        return [inv(tform) for tform in tforms]
    else
        return tforms
    end
end

inertial_transforms(x::AbstractModel; kwargs...) = inertial_transforms(coords(x), weights(x), widths(x); kwargs...)

"""
    score, tform, nevals = rocs_align_gmms(gmmfixed, gmmmoving; maxevals=1000)

Finds the optimal alignment between the two supplied GMMs using steric multipoles,
based on the [ROCS alignment algorithm.](https://docs.eyesopen.com/applications/rocs/theory/shape_shape.html)
"""
function rocs_align(gmmmoving::AbstractGMM, gmmfixed::AbstractGMM; kwargs...)
    # align both GMMs to their inertial frames
    tformfixed = inertial_transforms(gmmfixed)[1]    # Only need one alignment for the fixed GMM
    tformsmoving = inertial_transforms(gmmmoving)    # Take all 4 rotations for the GMM to be aligned

    # perform local alignment starting at each inertial rotation for the "moving" GMM
    results = Tuple{Float64, NTuple{6,Float64}}[]
    for tformm in tformsmoving
        push!(results, local_align(tformm(gmmmoving), tformfixed(gmmfixed); kwargs...))
    end

    # combine the inertial transform with the subsequent alignment transform for the best result
    minoverlap, mindex = findmin([r[1] for r in results])
    tformmoving = AffineMap(results[mindex][2]) ∘ tformsmoving[mindex]

    # Apply the inverse of `tformfixed` to the optimized transformation 
    alignment_tform = inv(tformfixed) ∘ tformmoving
    
    # return the result
    return ROCSAlignmentResult(gmmmoving, gmmfixed, minoverlap, alignment_tform)
end