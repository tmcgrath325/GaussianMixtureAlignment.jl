

UncertaintyRegion(box::SearchBox) = UncertaintyRegion(RotationVec(mid(box[1]), mid(box[2]), mid(box[3])), SVector{3}(mid(box[4]), mid(box[5]), mid(box[6])), wid(box[1])/2, wid(box[4])/2)
RotationRegion(box::SearchBox, T=zero(SVector{3})) = RotationRegion(RotationVec(box[1], box[2], box[3]), T, wid(box[1])/2)
TranslationRegion(box::SearchBox, R=RotationVec(0.0,0.0,0.0)) = TranslationRegion(R, SVector{3}(mid(box[1]), mid(box[2]), mid(box[3])), wid(box[1])/2)

searchbox(sr::UncertaintyRegion{T}) where {T} = SearchBox{6,T,Interval{T}}([map(r -> lohi(Interval, r - sr.σᵣ, r + sr.σᵣ), (sr.R.sx, sr.R.sy, sr.R.sz))..., map(r -> lohi(Interval, r - sr.σₜ, r + sr.σₜ), sr.T)...])
searchbox(sr::RotationRegion{T}) where {T} = SearchBox{3,T,Interval{T}}(map(r -> lohi(Interval, r - sr.σᵣ, r + sr.σᵣ), (sr.R.sx, sr.R.sy, sr.R.sz))...)
searchbox(sr::TranslationRegion{T}) where {T} = SearchBox{3,T,Interval{T}}(map(r -> lohi(Interval, r - sr.σₜ, r + sr.σₜ), sr.T)...)

searchbox(srs::AbstractVector{<:UncertaintyRegion{T}}) where {T} = SearchBox{6*length(srs), T, Interval{T}}(vcat([[map(r -> lohi(Interval, r - sr.σᵣ, r + sr.σᵣ), (sr.R.sx, sr.R.sy, sr.R.sz))..., map(r -> lohi(Interval, r - sr.σₜ, r + sr.σₜ), sr.T)...] for sr in srs]...))

# function getbounds(blockfun, boundsfun, box::SearchBox, pσ, pϕ, interactions = nothing)
#     return boundsfun(blockfun(box), pσ, pϕ, interactions)
# end

# results in 2^n sub boxes
function bisect_random_chunk(box::SearchBox{N,T,TN}, n::Int) where {N,T,TN}
    idx = Int(floor(rand() * length(box) / n)) * n + 1
    r = idx:(idx+n-1)
    return bisect(box, r)
end

function bisect_largest_rigid(box::SearchBox{N,T,TN}) where {N,T,TN}
    tformidxs = 1:Int(length(box)/6)
    maxtformidx = findmax(i -> wid(box[(i-1)*6+1]) * wid(box[(i-1)*6+4]), tformidxs)[2]
    idx = 6*(maxtformidx - 1) + 1
    r = idx:(idx+5)
    return bisect(box, r)
end

function globalalign(fixedinput::AbstractModel, mobileinputs::AbstractVector{<:AbstractModel};
    searchspace=nothing,
    blockfun = UncertaintyRegion,
    boxsplitter = bisectall,
    objfun=overlapobj,
    boundsfun=gauss_l2_bounds, 
    kwargs...
) 
    y = fixedinput
    xs = mobileinputs
    ntforms = length(xs)
    if !all(x->dims(x) == dims(y), xs)
        throw(ArgumentError("Dimensionality of the GMMs must be equal"))
    end
    t = promote_type(numbertype.(xs)..., numbertype(y))

    if isnothing(searchspace)
        tlimit = translation_limit(y, xs...)
        searchspace = [blockfun(tlimit) for x in xs]
    end

    box = searchbox(searchspace)
    thickres = thicksearch(objfun, box; boxsplitter=boxsplitter, boundsfun=boundsfun, kwargs...)
    return thickres
end


# globalalign(models::Vararg{<:AbstractModel}; kwargs...) = gloablalign(models[1], models[2:end]; kwargs...);

function thick_gogma_align(y::AbstractGMM, xs::AbstractVector{<:AbstractGMM}; interactions=nothing, lohifun=lohi_interval, kwargs...)
    t = promote_type(numbertype(y), numbertype.(xs)...)
    pσ, pϕ = pairwise_consts(y, xs, interactions)
    blocks = [UncertaintyRegion(RotationVec(zeros(t,3)...), SVector{3,t}(0.,0.,0.), 0., 0.) for i in 1:length(xs)]
    tformedxs = [xs...]
    boundsfun = box -> begin
        for i in 0:length(blocks)-1
           blocks[i+1] = UncertaintyRegion(SearchBox{6, t, eltype(first(box))}(box[(6*i+1):(6*i+6)]))
        end
        return gauss_l2_bounds!(tformedxs, y, xs, blocks, pσ, pϕ; lohifun=lohifun)
    end
    objfun = X -> begin
        Rs = [RotationVec(X[(6*i+1):(6*i+3)]...) for i in 0:length(xs)-1]
        Ts = [SVector{3}(X[(6*i+4):(6*i+6)]) for i in 0:length(xs)-1]
        local_tformedxs = [R*x+T for (x,R,T) in zip(xs,Rs,Ts)]
        return -overlap(y, local_tformedxs, pσ, pϕ, interactions)
    end
    return globalalign(y, xs; boxsplitter=bisect_largest_rigid, boundsfun=boundsfun, objfun=objfun, kwargs...)
end

thick_gogma_align(models::Vararg{M}; kwargs...) where M<:AbstractModel = thick_gogma_align(models[1], [models[2:end]...]; kwargs...);