

UncertaintyRegion(box::SearchBox) = UncertaintyRegion(RotationVec(mid(box[1]), mid(box[2]), mid(box[3])), SVector{3}(mid(box[4]), mid(box[5]), mid(box[6])), wid(box[1])/2, wid(box[4])/2)
RotationRegion(box::SearchBox, T=zero(SVector{3})) = RotationRegion(RotationVec(box[1], box[2], box[3]), T, wid(box[1])/2)
TranslationRegion(box::SearchBox, R=RotationVec(0.0,0.0,0.0)) = TranslationRegion(R, SVector{3}(mid(box[1]), mid(box[2]), mid(box[3])), wid(box[1])/2)

searchbox(sr::UncertaintyRegion{T}) where {T} = SearchBox{6,T,Interval{T}}([map(r -> lohi(Interval, r - sr.σᵣ, r + sr.σᵣ), (sr.R.sx, sr.R.sy, sr.R.sz))..., map(r -> lohi(Interval, r - sr.σₜ, r + sr.σₜ), sr.T)...])
searchbox(sr::RotationRegion{T}) where {T} = SearchBox{3,T,Interval{T}}(map(r -> lohi(Interval, r - sr.σᵣ, r + sr.σᵣ), (sr.R.sx, sr.R.sy, sr.R.sz))...)
searchbox(sr::TranslationRegion{T}) where {T} = SearchBox{3,T,Interval{T}}(map(r -> lohi(Interval, r - sr.σₜ, r + sr.σₜ), sr.T)...)

# function getbounds(blockfun, boundsfun, box::SearchBox, pσ, pϕ, interactions = nothing)
#     return boundsfun(blockfun(box), pσ, pϕ, interactions)
# end

function globalalign(xinput::AbstractModel, yinput::AbstractModel;
    searchspace=nothing, R = RotationVec(0.0,0.0,0.0), T = SVector{3}(0.0,0.0,0.0),
    blockfun = UncertaintyRegion,
    boxsplitter = bisectall,
    objfun=overlapobj,
    boundsfun=gauss_l2_bounds, 
    kwargs...
) 
    x = xinput
    y = yinput
    if dims(x) != dims(y)
        throw(ArgumentError("Dimensionality of the GMMs must be equal"))
    end
    t = promote_type(numbertype(x), numbertype(y))

    if isnothing(searchspace)
        searchspace = blockfun(x, y, R, T)
    end

    box = searchbox(searchspace)
    thickres = thicksearch(objfun, box; boxsplitter=boxsplitter, boundsfun=boundsfun, kwargs...)
    return thickres
end

function thick_gogma_align(gmmx::AbstractGMM, gmmy::AbstractGMM; interactions=nothing, kwargs...)
    pσ, pϕ = pairwise_consts(gmmx,gmmy,interactions)
    boundsfun = box -> begin
        lb, ub = gauss_l2_bounds(gmmx, gmmy, UncertaintyRegion(box), pσ, pϕ)
        return lb, ub
    end
    objfun = X -> overlapobj(X, gmmx, gmmy, pσ, pϕ)
    return globalalign(gmmx, gmmy; boxsplitter=bisectall, boundsfun=boundsfun, objfun=objfun, kwargs...)
end