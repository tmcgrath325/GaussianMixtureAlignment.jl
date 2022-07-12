import CoordinateTransformations.AffineMap
import CoordinateTransformations.LinearMap
import CoordinateTransformations.Translation

AffineMap(params::NTuple{6}) = AffineMap(RotationVec(params[1:3]...), SVector{3}(params[4:6]...))
LinearMap(params::NTuple{3}) = LinearMap(RotationVec(params...))
Translation(params::NTuple{3}) = Translation(SVector{3}(params))

function affinemap_to_params(tform::AffineMap)
    R = RotationVec(tform.linear)
    T = SVector{3}(tform.translation)
    return (R.sx, R.sy, R.sz, T...)
end
