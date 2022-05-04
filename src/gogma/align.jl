function overlap_alignment_objective(X::NTuple{6}, gmmx::AbstractGMM, gmmy::AbstractGMM, block::SearcRegion, pσ=nothing, pϕ=nothing)
    return -overlap(AffineMap(X...)(gmmx), gmmy, pσ, pϕ)
end

# alignment objective for rigid Ration (i.e. the first stage of TIV-GOGMA)
function overlap_rot_alignment_objective(X::NTuple{3}, gmmx::AbstractGMM, gmmy::AbstractGMM, block::RotationRegion, args...)
    return overlap_alignment_objective((X..., block.T...), gmmx, gmmy, block, args...)
end

# alignment objective for translation (i.e. the second stage of TIV-GOGMA)
function overlap_trl_alignment_objective(X::NTuple{3}, gmmx::AbstractGMM, gmmy::AbstractGMM, block::TranslationRegion, args...)
    return overlap_alignment_objective((block.R..., X...), gmmx, gmmy, block, args...)
end
