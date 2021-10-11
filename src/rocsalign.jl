"""
    com = center_of_mass(gmm)

Returns the center of mass of `gmm`, where its first order moments are equal to 0.
"""
function center_of_mass(positions::AbstractMatrix{<:Real}, weights=ones(eltype(positions),size(positions,2)))
    return SVector{size(positions,1)}([sum(hcat([weights[i]*positions[:,i] for i=1:size(positions,2)]...), dims=2)...]/sum(weights))
end

center_of_mass(gaussians::AbstractVector{<:AbstractIsotropicGaussian}) = 
    return center_of_mass(hcat([g.μ for g in gaussians]...), [g.ϕ for g in gaussians])

center_of_mass(gmm::AbstractIsotropicGMM) = center_of_mass(gmm.gaussians)

center_of_mass(mgmm::AbstractMultiGMM) = center_of_mass(collect(Iterators.flatten([gmm.gaussians for (k,gmm) in mgmm])))

""" 
    m = second_moment(gmm, center, dim1, dim2)

Returns the second order moment of `gmm`
"""
function mass_matrix(positions::AbstractMatrix{<:Real}, 
                     widths=zeros(eltype(positions),size(positions,2)), 
                     weights=ones(eltype(positions),size(positions,2)), 
                     center=center_of_mass(positions,weights))
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
                             widths=zeros(eltype(positions),size(positions,2)), 
                             weights=ones(eltype(positions),size(positions,2)); 
                             invert = false)
    com = center_of_mass(positions, weights)
    massmat = mass_matrix(positions, widths, weights, com)
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

    # then rotate the first transformation about each coordinate axis by -π/2, π/2, and π
    for i=1:3
        axis = zeros(3)
        axis[i] = 1.
        for angle in (π)
            push!(tforms, LinearMap(AngleAxis(angle, axis...)) ∘  tforms[1])
        end
    end
    if invert
        return [inv(tform) for tform in tforms]
    else
        return tforms
    end
end

inertial_transforms(gaussians; kwargs...
    ) = inertial_transforms(hcat([g.μ for g in gaussians]...), [g.σ for g in gaussians], [g.ϕ for g in gaussians]; kwargs...)

inertial_transforms(gmm::AbstractIsotropicGMM; kwargs...
    ) = inertial_transforms(gmm.gaussians; kwargs...)

inertial_transforms(mgmm::AbstractMultiGMM; kwargs...
    ) = inertial_transforms(collect(Iterators.flatten([gmm.gaussians for (k,gmm) in mgmm])); kwargs...)

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
    tformmoving = AffineMap(results[mindex][2]...) ∘ tformsmoving[mindex]

    # Apply the inverse of `tformfixed` to the optimized transformation 
    alignment_tform = inv(tformfixed) ∘ tformmoving
    
    # return the objective value, best transformation
    return minoverlap, alignment_tform
end