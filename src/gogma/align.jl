function gogma_align(gmmx::AbstractGMM, gmmy::AbstractGMM; kwargs...)
    pσ, pϕ = pairwise_consts(gmmx,gmmy)
    boundsfun(x,y,block) = gauss_l2_bounds(x,y,block,pσ,pϕ)
    localfun(x,y,block) = local_align(x,y,block,pσ,pϕ)
    return branchbound(gmmx, gmmy; boundsfun=boundsfun, localfun=localfun, kwargs...)
end
function rot_gogma_align(gmmx::AbstractGMM, gmmy::AbstractGMM; kwargs...)
    pσ, pϕ = pairwise_consts(gmmx,gmmy)
    boundsfun(x,y,block) = gauss_l2_bounds(x,y,block,pσ,pϕ)
    localfun(x,y,block) = local_align(x,y,block,pσ,pϕ)
    rot_branchbound(gmmx, gmmy; boundsfun=boundsfun, localfun=localfun, kwargs...)
end
function trl_gogma_align(gmmx::AbstractGMM, gmmy::AbstractGMM; kwargs...) 
    pσ, pϕ = pairwise_consts(gmmx,gmmy)
    boundsfun(x,y,block) = gauss_l2_bounds(x,y,block,pσ,pϕ)
    localfun(x,y,block) = local_align(x,y,block,pσ,pϕ)
    trl_branchbound(gmmx, gmmy; boundsfun=boundsfun, localfun=localfun, kwargs...)
end
function tiv_gogma_align(gmmx::AbstractGMM, gmmy::AbstractGMM, cx=Inf, cy=Inf; kwargs...)
    pσ, pϕ = pairwise_consts(gmmx,gmmy)
    boundsfun(x,y,block) = gauss_l2_bounds(x,y,block,pσ,pϕ)
    localfun(x,y,block) = local_align(x,y,block,pσ,pϕ)
    tiv_branchbound(gmmx, gmmy, tivgmm(gmmx, cx), tivgmm(gmmy, cy); boundsfun=boundsfun, localfun=localfun, kwargs...)
end