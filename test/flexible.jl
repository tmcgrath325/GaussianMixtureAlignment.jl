using Random: MersenneTwister

@testset "flexible: ArticulatedGMM and forward kinematics" begin
    V(x, y, z) = SVector(x, y, z)
    mk(x, y, z) = IsotropicGaussian(V(x, y, z), 1.0, 1.0)

    # single joint: z-axis through (1,0,0), moving Gaussians 2 and 3
    gs = [mk(0, 0, 0), mk(1, 0, 0), mk(2, 0, 0)]
    single = GMA.ArticulatedGMM(gs, [GMA.Joint(V(0, 0, 1.0), V(1.0, 0, 0), [2, 3], Int[])])
    @test single isa GMA.AbstractIsotropicGMM{3, Float64}   # a rigid GMM in its base pose
    @test length(single) == 3
    @test GMA.njoints(single) == 1
    @test GMA.joint_axis(single, 1) ≈ V(0, 0, 1)            # normalized on construction
    @test GMA.joint_features(single, 1) == [2, 3]

    flexed = GMA.flex(single, [π / 2])
    @test flexed isa GMA.ArticulatedGMM{3, Float64}
    @test flexed.gaussians[1].μ ≈ V(0, 0, 0)               # not moved by the joint
    @test flexed.gaussians[2].μ ≈ V(1, 0, 0)               # on the axis: fixed
    @test flexed.gaussians[3].μ ≈ V(1, 1, 0)               # (2,0,0) → 90° about z@(1,0,0)
    # the neutral conformation reproduces the base model exactly
    @test all(GMA.flex(single, [0.0]).gaussians[i].μ == gs[i].μ for i in 1:3)

    # nested chain: joint 1 (root) reframes joint 2 (its child) as well as moving features
    chain = GMA.ArticulatedGMM(
        gs,
        [
            GMA.Joint(V(0, 0, 1.0), V(0, 0, 0.0), [2, 3], [2]),
            GMA.Joint(V(0, 0, 1.0), V(1.0, 0, 0), [3], Int[]),
        ]
    )
    g3(φ) = GMA.flex(chain, φ).gaussians[3].μ
    @test g3([0.0, 0.0]) ≈ V(2, 0, 0)
    @test g3([π / 2, 0.0]) ≈ V(0, 2, 0)                    # ancestor only
    @test g3([0.0, π / 2]) ≈ V(1, 1, 0)                    # child only
    # both joints: the ancestor's rotation must carry the child's axis frame along,
    # so the child then rotates about the *moved* axis at (0,1,0) rather than (1,0,0)
    @test g3([π / 2, π / 2]) ≈ V(-1, 1, 0)

    # flex is smooth through φ = 0 (a fixed-axis rotation has no identity singularity)
    target = V(1.0, 1.0, 0.0)
    obj(φ) = sum(abs2, GMA.flex(chain, φ).gaussians[3].μ - target)
    h = 1.0e-6
    fd = [(obj([h, 0.0]) - obj([-h, 0.0])) / (2h), (obj([0.0, h]) - obj([0.0, -h])) / (2h)]
    @test all(isfinite, fd)

    # construction guards
    @test_throws DimensionMismatch GMA.flex(single, [0.0, 0.0])
    @test_throws "nonzero" GMA.Joint(V(0, 0, 0.0), V(0, 0, 0.0), Int[], Int[])
    @test_throws "outside" GMA.ArticulatedGMM(gs, [GMA.Joint(V(0, 0, 1.0), V(0, 0, 0.0), [9], Int[])])
    # a joint must precede its descendants (child index strictly greater than the parent's)
    @test_throws "descendant" GMA.ArticulatedGMM(
        gs,
        [
            GMA.Joint(V(0, 0, 1.0), V(0, 0, 0.0), [3], [2]),
            GMA.Joint(V(0, 0, 1.0), V(1.0, 0, 0), [3], [1]),
        ]
    )
end

@testset "flexible: FlexibleRegion and its splitter" begin
    ur = UncertaintyRegion(Float64(π), 2.0)              # σᵣ = π, σₜ = 2
    fr = GMA.FlexibleRegion(ur, [0.0, 0.5], [Float64(π), 0.3])   # K = 2

    @test GMA.njoints(fr) == 2
    @test length(GMA.center(fr)) == 8                    # 6 rigid + 2 joints
    @test GMA.center(fr) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5)
    @test UncertaintyRegion(fr) === ur                   # projection to the rigid box

    # the K-argument form covers the full angular range and, at K = 0, is purely rigid
    frfull = GMA.FlexibleRegion(ur, 2)
    @test frfull.φ == SVector(0.0, 0.0) && frfull.σφ == SVector(Float64(π), Float64(π))
    fr0 = GMA.FlexibleRegion(ur, 0)
    @test length(GMA.center(fr0)) == 6
    @test UncertaintyRegion(fr0) === ur

    # the splitter subdivides only the widest scaled group, bounding the branching factor
    kids_rot = GMA.subregions(fr, 2)                     # σᵣ = π dominates
    @test length(kids_rot) == 8                          # nsplits^3 rotation split
    @test all(k.σφ == fr.σφ && k.φ == fr.φ for k in kids_rot)   # joints untouched
    @test all(k.rigid.σᵣ ≈ ur.σᵣ / 2 && k.rigid.σₜ == ur.σₜ for k in kids_rot)

    kids_trl = GMA.subregions(fr, 2; rotscale = 0.0, trlscale = 1.0, jointscales = (0.0, 0.0))
    @test length(kids_trl) == 8                          # nsplits^3 translation split
    @test all(k.rigid.σₜ ≈ ur.σₜ / 2 && k.rigid.σᵣ == ur.σᵣ for k in kids_trl)

    kids_j1 = GMA.subregions(fr, 2; rotscale = 0.0, trlscale = 0.0, jointscales = (1.0, 0.0))
    @test length(kids_j1) == 2                           # a joint splits in two
    @test all(k.σφ[1] ≈ fr.σφ[1] / 2 for k in kids_j1)   # its interval is halved
    @test Set(k.φ[1] for k in kids_j1) == Set((-π / 2, π / 2))   # ...and it tiles [-π, π]
    @test all(k.φ[2] == 0.5 && k.σφ[2] == 0.3 for k in kids_j1)  # the other joint is untouched
    @test all(UncertaintyRegion(k) === ur for k in kids_j1)      # rigid box unchanged
end

# Samplers for a search block and the exact objective at a concrete (R, T, φ). The objective
# sums the signed per-pair -overlap the bounds bound, so it accepts explicit (pσ, pϕ) — with
# some pϕ negative it also exercises the repulsive (w < 0) branch.
_randR(rng, R, σᵣ) = RotationVec(ntuple(i -> (R.sx, R.sy, R.sz)[i] + σᵣ * (2rand(rng) - 1), 3)...)
_randT(rng, T, σₜ) = SVector{3}(ntuple(i -> T[i] + σₜ * (2rand(rng) - 1), 3))
_randφ(rng, φ, σφ) = [φ[k] + σφ[k] * (2rand(rng) - 1) for k in eachindex(φ)]

function _objective(x, y, R, T, φ, pσ, pϕ)
    tx = R * IsotropicGMM(GMA.flex(x, φ)) + T
    tot = 0.0
    for (i, gx) in enumerate(tx.gaussians), (j, gy) in enumerate(y.gaussians)
        tot += -overlap(sum(abs2, gx.μ - gy.μ), pσ[i, j], pϕ[i, j])
    end
    return tot
end

@testset "flexible: bounds validity (Monte-Carlo)" begin
    V3(x, y, z) = SVector(x, y, z)
    gs = [
        IsotropicGaussian(V3(0, 0, 0), 1.0, 1.0), IsotropicGaussian(V3(1, 0, 0), 1.0, 1.0),
        IsotropicGaussian(V3(2, 0, 0), 1.0, 1.0), IsotropicGaussian(V3(2, 1, 0), 1.0, 1.0),
        IsotropicGaussian(V3(3, 0, 0), 1.0, 1.0),
    ]
    # a chain of two joints: the root moves features 2..5 and reframes the distal joint
    js = [
        GMA.Joint(V3(0, 0, 1.0), V3(1.0, 0, 0), [2, 3, 4, 5], [2]),
        GMA.Joint(V3(0, 1.0, 0), V3(2.0, 0, 0), [4, 5], Int[]),
    ]
    x = GMA.ArticulatedGMM(gs, js)
    y = RotationVec(0.3, -0.2, 0.5) * IsotropicGMM(GMA.flex(x, [0.7, -0.4])) + V3(1.0, -2.0, 0.5)
    pσ, pϕ = GMA.pairwise_consts(x, y)

    # δ_g must upper-bound the true internal displacement over every sampled sub-box
    rng = MersenneTwister(20260706)
    δ_ok = true
    for _ in 1:60
        φc = [2π * rand(rng) - π for _ in 1:2]
        σφ = [0.9 * rand(rng) for _ in 1:2]
        block = GMA.FlexibleRegion(UncertaintyRegion(), φc, σφ)
        xc, δ = GMA.flex_displacements(x, block)
        for _ in 1:60
            xf = GMA.flex(x, _randφ(rng, φc, σφ))
            for g in 1:length(x)
                δ_ok &= norm(xf.gaussians[g].μ - xc.gaussians[g].μ) <= δ[g] + 1.0e-9
            end
        end
    end
    @test δ_ok

    # the lower bound must not exceed the objective at any sampled feasible (R, T, φ), the
    # upper bound must equal the objective at the block center, and lb ≤ ub — for both
    # attractive weights and a sign-flipped mix that turns some pairs repulsive
    for pϕtest in (pϕ, (m = copy(pϕ); m[1:2, :] .*= -1; m))
        lb_ok = true
        ub_ok = true
        order_ok = true
        for _ in 1:60
            R0 = _randR(rng, RotationVec(0, 0, 0), 0.6)
            T0 = _randT(rng, V3(0, 0, 0), 1.0)
            σᵣ = 0.3 + 0.5rand(rng)
            σₜ = 0.3 + rand(rng)
            φc = [2π * rand(rng) - π for _ in 1:2]
            σφ = [0.8 * rand(rng) for _ in 1:2]
            block = GMA.FlexibleRegion(UncertaintyRegion(R0, T0, σᵣ, σₜ), φc, σφ)
            lb, ub = GMA.flex_gauss_l2_bounds(x, y, block, pσ, pϕtest)
            order_ok &= lb <= ub + 1.0e-9
            ub_ok &= isapprox(ub, _objective(x, y, R0, T0, φc, pσ, pϕtest); atol = 1.0e-8, rtol = 1.0e-8)
            for _ in 1:60
                obj = _objective(x, y, _randR(rng, R0, σᵣ), _randT(rng, T0, σₜ), _randφ(rng, φc, σφ), pσ, pϕtest)
                lb_ok &= lb <= obj + 1.0e-9
            end
        end
        @test lb_ok
        @test ub_ok
        @test order_ok
    end

    # reductions: frozen joints match the rigid bounds on the flexed model, K = 0 matches the
    # rigid bounds on the base model, and the loose distance bound is no tighter than the tight one
    rigid = UncertaintyRegion(RotationVec(0.2, -0.1, 0.3), V3(0.5, -1.0, 0.2), 0.4, 0.7)
    φ = [0.6, -0.9]
    lbf, ubf = GMA.flex_gauss_l2_bounds(x, y, GMA.FlexibleRegion(rigid, φ, [0.0, 0.0]), pσ, pϕ)
    lbr, ubr = gauss_l2_bounds(IsotropicGMM(GMA.flex(x, φ)), y, rigid, pσ, pϕ)
    @test lbf ≈ lbr && ubf ≈ ubr

    x0 = GMA.ArticulatedGMM(collect(x.gaussians), GMA.Joint{3, Float64}[])
    lb0, ub0 = GMA.flex_gauss_l2_bounds(x0, y, GMA.FlexibleRegion(rigid, 0), pσ, pϕ)
    lbb, ubb = gauss_l2_bounds(IsotropicGMM(collect(x.gaussians)), y, rigid, pσ, pϕ)
    @test lb0 ≈ lbb && ub0 ≈ ubb

    block = GMA.FlexibleRegion(rigid, φ, [0.5, 0.3])
    lb_tight = GMA.flex_gauss_l2_bounds(x, y, block, pσ, pϕ; distance_bound_fun = GMA.tight_distance_bounds)[1]
    lb_loose = GMA.flex_gauss_l2_bounds(x, y, block, pσ, pϕ; distance_bound_fun = GMA.loose_distance_bounds)[1]
    @test lb_loose <= lb_tight + 1.0e-12
end

@testset "flexible: flex_gogma_align" begin
    V3(a, b, c) = SVector(a, b, c)

    # a model with no joints reduces exactly to rigid GOGMA alignment
    xpts = [[0.0, 0, 0], [3.0, 0, 0], [0, 4.0, 0]]
    ypts = [[1.0, 1, 1], [1.0, -2, 1], [1, 1, -3.0]]
    gx = [IsotropicGaussian(SVector{3}(p), 1.0, 1.0) for p in xpts]
    gy = IsotropicGMM([IsotropicGaussian(SVector{3}(p), 1.0, 1.0) for p in ypts])
    x0 = GMA.ArticulatedGMM(gx, GMA.Joint{3, Float64}[])
    rig = gogma_align(IsotropicGMM(gx), gy; maxsplits = 1.0e3)
    flx0 = GMA.flex_gogma_align(x0, gy; maxsplits = 1.0e3)
    @test flx0.upperbound ≈ rig.upperbound atol = 1.0e-8
    @test GMA.joint_angles(flx0) == ()

    # a jointed model aligned to a planted flexible transform of itself
    gsf = [
        IsotropicGaussian(V3(0, 0, 0), 0.7, 1.0), IsotropicGaussian(V3(1.0, 0, 0), 0.7, 1.0),
        IsotropicGaussian(V3(2.0, 0, 0), 0.7, 1.0), IsotropicGaussian(V3(2.0, 1, 0), 0.7, 1.0),
        IsotropicGaussian(V3(3.0, 0, 0), 0.7, 1.0), IsotropicGaussian(V3(2.0, -1, 0), 0.7, 1.0),
    ]
    jsf = [
        GMA.Joint(V3(0, 0, 1.0), V3(1.0, 0, 0), [2, 3, 4, 5, 6], [2]),
        GMA.Joint(V3(0, 1.0, 0), V3(2.0, 0, 0), [4, 5], Int[]),
    ]
    xf = GMA.ArticulatedGMM(gsf, jsf)
    Rstar = RotationVec(0.5, -0.3, 0.9)
    planted = (Rstar.sx, Rstar.sy, Rstar.sz, 1.0, -1.5, 0.7, 0.8, -0.6)
    yf = IsotropicGMM(GMA.flex_pose(planted, xf))
    ideal = overlap(yf, yf)

    # posing by the planted parameters reproduces the target overlap exactly
    @test overlap(GMA.flex_pose(planted, xf), yf) ≈ ideal atol = 1.0e-8

    res = GMA.flex_gogma_align(xf, yf; maxsplits = 300)
    # the search is at least as good as the (feasible) planted conformation, and its bounds
    # bracket the objective. An unlabeled flexible model may exceed the target's self-overlap
    # by folding, so the invariant is `≥ ideal`, not exact recovery.
    @test -res.upperbound >= ideal - 1.0e-6
    @test res.lowerbound <= res.upperbound

    # flexibility does at least as well as a rigid alignment of the same model
    rigf = gogma_align(IsotropicGMM(gsf), yf; maxsplits = 300)
    @test -res.upperbound >= -rigf.upperbound - 1.0e-6

    # result interface
    @test length(GMA.joint_angles(res)) == 2
    @test GMA.aligned(res) isa GMA.ArticulatedGMM{3, Float64}
    @test length(GMA.aligned(res)) == length(xf)
    @test GMA.tform(res) isa AffineMap
    @test GMA.upperbound(res) === res.upperbound
    @test GMA.lowerbound(res) === res.lowerbound
    @test GMA.num_splits(res) isa Int
    @test GMA.num_blocks(res) isa Int
    @test occursin("FlexibleAlignmentResult", sprint(show, MIME"text/plain"(), res))

    early = GMA.flex_gogma_align(xf, yf; maxsplits = 1)
    @test early.terminated_by == "terminated early"
    @test !GMA.converged(early)
end

# objective at a concrete (R, T, φ, ψ) with both models articulated: x is flexed and posed,
# y is flexed in place
function _objective2(x, y, R, T, φ, ψ, pσ, pϕ)
    tx = R * IsotropicGMM(GMA.flex(x, φ)) + T
    ty = GMA.flex(y, ψ)
    tot = 0.0
    for (i, gx) in enumerate(tx.gaussians), (j, gy) in enumerate(ty.gaussians)
        tot += -overlap(sum(abs2, gx.μ - gy.μ), pσ[i, j], pϕ[i, j])
    end
    return tot
end

@testset "flexible: articulated target (both models jointed)" begin
    V3(a, b, c) = SVector(a, b, c)

    # a rigid GMM is an articulated model with no joints
    plain = IsotropicGMM([IsotropicGaussian(V3(0, 0, 0), 1.0, 1.0), IsotropicGaussian(V3(1, 0, 0), 1.0, 1.0)])
    @test GMA.njoints(plain) == 0
    @test GMA.flex(plain, ()) === plain
    @test_throws DimensionMismatch GMA.flex(plain, [0.0])

    gs = [
        IsotropicGaussian(V3(0, 0, 0), 1.0, 1.0), IsotropicGaussian(V3(1, 0, 0), 1.0, 1.0),
        IsotropicGaussian(V3(2, 0, 0), 1.0, 1.0), IsotropicGaussian(V3(2, 1, 0), 1.0, 1.0),
        IsotropicGaussian(V3(3, 0, 0), 1.0, 1.0),
    ]
    js = [
        GMA.Joint(V3(0, 0, 1.0), V3(1.0, 0, 0), [2, 3, 4, 5], [2]),
        GMA.Joint(V3(0, 1.0, 0), V3(2.0, 0, 0), [4, 5], Int[]),
    ]
    x = GMA.ArticulatedGMM(gs, js)
    # the target: the same tree, differently placed, with one joint dropped (Kx = 2, Ky = 1)
    ygs = [IsotropicGaussian(RotationVec(0.3, -0.2, 0.5) * g.μ + V3(1.0, -2.0, 0.5), g.σ, g.ϕ) for g in gs]
    yj = GMA.Joint(RotationVec(0.3, -0.2, 0.5) * V3(0, 0, 1.0), RotationVec(0.3, -0.2, 0.5) * V3(1.0, 0, 0) + V3(1.0, -2.0, 0.5), [2, 3, 4, 5], Int[])
    y = GMA.ArticulatedGMM(ygs, [yj])
    pσ, pϕ = GMA.pairwise_consts(x, y)

    # the block's joint intervals split between the models in order: x's first, then y's
    fr = GMA.FlexibleRegion(UncertaintyRegion(), [0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    xφ, xσφ, yφ, yσφ = GMA.joint_intervals(x, y, fr)
    @test xφ == SVector(0.1, 0.2) && xσφ == SVector(0.4, 0.5)
    @test yφ == SVector(0.3) && yσφ == SVector(0.6)
    @test_throws DimensionMismatch GMA.joint_intervals(x, y, GMA.FlexibleRegion(UncertaintyRegion(), 1))
    @test_throws DimensionMismatch GMA.flex_displacements(x, fr)   # 3 intervals, 2 joints

    # lb ≤ objective at every sampled feasible (R, T, φ, ψ); ub == objective at the block
    # center; lb ≤ ub — for attractive weights and a sign-flipped repulsive mix
    rng = MersenneTwister(20260830)
    for pϕtest in (pϕ, (m = copy(pϕ); m[1:2, :] .*= -1; m))
        lb_ok = true
        ub_ok = true
        order_ok = true
        for _ in 1:60
            R0 = _randR(rng, RotationVec(0, 0, 0), 0.6)
            T0 = _randT(rng, V3(0, 0, 0), 1.0)
            σᵣ = 0.3 + 0.5rand(rng)
            σₜ = 0.3 + rand(rng)
            φc = [2π * rand(rng) - π for _ in 1:3]
            σφ = [0.8 * rand(rng) for _ in 1:3]
            block = GMA.FlexibleRegion(UncertaintyRegion(R0, T0, σᵣ, σₜ), φc, σφ)
            lb, ub = GMA.flex_gauss_l2_bounds(x, y, block, pσ, pϕtest)
            order_ok &= lb <= ub + 1.0e-9
            ub_ok &= isapprox(ub, _objective2(x, y, R0, T0, φc[1:2], φc[3:3], pσ, pϕtest); atol = 1.0e-8, rtol = 1.0e-8)
            for _ in 1:60
                angles = _randφ(rng, φc, σφ)
                obj = _objective2(x, y, _randR(rng, R0, σᵣ), _randT(rng, T0, σₜ), angles[1:2], angles[3:3], pσ, pϕtest)
                lb_ok &= lb <= obj + 1.0e-9
            end
        end
        @test lb_ok
        @test ub_ok
        @test order_ok
    end

    # reductions: a frozen target joint matches the one-sided bounds against the flexed
    # target, and a rigid target matches the one-sided bounds exactly
    rigid = UncertaintyRegion(RotationVec(0.2, -0.1, 0.3), V3(0.5, -1.0, 0.2), 0.4, 0.7)
    φ = [0.6, -0.9]
    ψ = 0.4
    lb2, ub2 = GMA.flex_gauss_l2_bounds(x, y, GMA.FlexibleRegion(rigid, [φ; ψ], [0.5, 0.3, 0.0]), pσ, pϕ)
    lb1, ub1 = GMA.flex_gauss_l2_bounds(x, IsotropicGMM(GMA.flex(y, [ψ])), GMA.FlexibleRegion(rigid, φ, [0.5, 0.3]), pσ, pϕ)
    @test lb2 ≈ lb1 && ub2 ≈ ub1
    yr = IsotropicGMM(ygs)
    lbr, ubr = GMA.flex_gauss_l2_bounds(x, yr, GMA.FlexibleRegion(rigid, φ, [0.5, 0.3]), pσ, pϕ)
    lb0, ub0 = GMA.flex_gauss_l2_bounds(x, y, GMA.FlexibleRegion(rigid, [φ; 0.0], [0.5, 0.3, 0.0]), pσ, pϕ)
    @test lbr ≈ lb0 && ubr ≈ ub0
    # widening the target's joint interval can only loosen the lower bound
    lbw, ubw = GMA.flex_gauss_l2_bounds(x, y, GMA.FlexibleRegion(rigid, [φ; ψ], [0.5, 0.3, 0.7]), pσ, pϕ)
    @test lbw <= lb2 + 1.0e-12 && ubw ≈ ub2

    # parameter layout: (rigid, x angles, y angles)
    params = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    @test GMA.flex_target(params, x, y).gaussians[3].μ ≈ GMA.flex(y, [0.9]).gaussians[3].μ
    @test GMA.flex_pose(params, x).gaussians[3].μ ≈ GMA.flex_pose(params[1:8], x).gaussians[3].μ
    @test_throws DimensionMismatch GMA.flex_target(params[1:7], x, y)

    # alignment against a target that is itself a flexed, posed copy of x with one joint
    planted = (0.5, -0.3, 0.9, 1.0, -1.5, 0.7, 0.8, -0.6, 0.5)
    posed = GMA.flex_pose(planted, x)
    target = GMA.flex_target(planted, x, y)
    ideal = overlap(posed, target)
    res = GMA.flex_gogma_align(x, y; flextarget = true, maxsplits = 300)
    # the search is at least as good as the (feasible) planted configuration, its bounds
    # bracket the objective, and the reported parameters reproduce the reported objective
    @test -res.upperbound >= ideal - 1.0e-6
    @test res.lowerbound <= res.upperbound
    @test overlap(GMA.aligned(res), GMA.aligned_target(res)) ≈ -res.upperbound atol = 1.0e-8
    @test length(GMA.joint_angles(res)) == 2
    @test length(GMA.target_joint_angles(res)) == 1
    @test GMA.aligned_target(res) isa GMA.ArticulatedGMM{3, Float64}
    @test length(res.tform_params) == 9
    @test occursin("target joint angles", sprint(show, MIME"text/plain"(), res))

    # a rigid target leaves the target-side interface empty
    res1 = GMA.flex_gogma_align(x, yr; maxsplits = 50)
    @test GMA.target_joint_angles(res1) == ()
    @test GMA.aligned_target(res1) === yr
    @test length(res1.tform_params) == 8

    # by default an articulated target is held in its base conformation: the search is the
    # one-sided search against the rigid base model, split for split
    frozen = GMA.flex_gogma_align(x, y; maxsplits = 50)
    @test GMA.target_joint_angles(frozen) == ()
    @test length(frozen.tform_params) == 8
    @test frozen.upperbound == res1.upperbound && frozen.lowerbound == res1.lowerbound
    @test GMA.aligned_target(frozen).gaussians[3].μ == y.gaussians[3].μ
    # a region with only x's joint intervals freezes the target in the bounds too
    fz = GMA.FlexibleRegion(UncertaintyRegion(), [0.1, 0.2], [0.4, 0.5])
    xφ, xσφ, yφ, yσφ = GMA.joint_intervals(x, y, fz)
    @test yφ == SVector(0.0) && yσφ == SVector(0.0)
    @test GMA.flex_gauss_l2_bounds(x, y, fz, pσ, pϕ) == GMA.flex_gauss_l2_bounds(x, yr, fz, pσ, pϕ)
    @test GMA.flex_target((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), x, y).gaussians[3].μ == y.gaussians[3].μ
end

@testset "flexible: self-overlap penalty" begin
    V3(a, b, c) = SVector(a, b, c)
    gs = [
        IsotropicGaussian(V3(0, 0, 0), 1.0, 1.0), IsotropicGaussian(V3(1, 0, 0), 1.0, 1.0),
        IsotropicGaussian(V3(2, 0, 0), 1.0, 1.0), IsotropicGaussian(V3(2, 1, 0), 1.0, 1.0),
        IsotropicGaussian(V3(3, 0, 0), 1.0, 1.0),
    ]
    # joint 1 moves 2..5 and reframes joint 2, which moves 4 and 5: three rigid fragments,
    # {1}, {2, 3} and {4, 5}
    js = [
        GMA.Joint(V3(0, 0, 1.0), V3(1.0, 0, 0), [2, 3, 4, 5], [2]),
        GMA.Joint(V3(0, 1.0, 0), V3(2.0, 0, 0), [4, 5], Int[]),
    ]
    x = GMA.ArticulatedGMM(gs, js)

    @test GMA.feature_joints(x) == [Int[], [1], [1], [1, 2], [1, 2]]
    p = GMA.SelfOverlap(x; weight = 2.0)
    # pairs within a fragment are omitted: 10 pairs total minus (2,3) and (4,5)
    @test length(p) == 8
    @test (2, 3) ∉ p.pairs && (4, 5) ∉ p.pairs
    pathof(g, h) = p.paths[findfirst(==((g, h)), p.pairs)]
    @test pathof(1, 3) == [1]             # only joint 1 lies between fragments {1} and {2,3}
    @test pathof(1, 4) == [2, 1]          # from feature 4 toward feature 1: joint 2, then 1
    @test pathof(2, 4) == [2]             # the shared joint 1 drops out
    @test p.s == fill(2.0, 8) && p.w == fill(1.0, 8)

    # the penalty is the weighted self-overlap over those pairs, and ignores the rigid pose
    φ = [0.7, -0.4]
    xc = GMA.flex(x, φ)
    expected = 2.0 * sum(overlap(sum(abs2, xc.gaussians[g].μ - xc.gaussians[h].μ), 2.0, 1.0) for (g, h) in p.pairs)
    @test GMA.penalty(p, xc) ≈ expected
    @test GMA.penalty(p, RotationVec(0.3, 0.2, -0.5) * xc + V3(1, 2, 3)) ≈ expected
    @test GMA.penalty(nothing, xc) == 0
    @test GMA.penalty_bounds(nothing, x, φ, [0.1, 0.1]) == (0, 0)

    # over any joint box, each pair's distance stays within its path chord sum of the center
    # distance, and the penalty bounds bracket the penalty at every sampled conformation
    rng = MersenneTwister(20260831)
    dist_ok = true
    lb_ok = true
    ub_ok = true
    for _ in 1:60
        φc = [2π * rand(rng) - π for _ in 1:2]
        σφ = [0.9 * rand(rng) for _ in 1:2]
        xcen = GMA.flex(x, φc)
        lb, ub = GMA.penalty_bounds(p, x, φc, σφ)
        ub_ok &= isapprox(ub, GMA.penalty(p, xcen); atol = 1.0e-10)
        for _ in 1:60
            xf = GMA.flex(x, _randφ(rng, φc, σφ))
            lb_ok &= lb <= GMA.penalty(p, xf) + 1.0e-9
            for (k, (g, h)) in enumerate(p.pairs)
                dc = norm(xcen.gaussians[h].μ - xcen.gaussians[g].μ)
                df = norm(xf.gaussians[h].μ - xf.gaussians[g].μ)
                δ = GMA.chord_sum(xcen, xcen.gaussians[h].μ, p.paths[k], σφ, Float64)
                dist_ok &= abs(df - dc) <= δ + 1.0e-9
            end
        end
    end
    @test dist_ok
    @test lb_ok
    @test ub_ok

    # frozen joints: bounds collapse to the penalty value; a repulsive pair flips the bound
    lbz, ubz = GMA.penalty_bounds(p, x, φ, [0.0, 0.0])
    @test lbz ≈ ubz ≈ GMA.penalty(p, xc)
    prep = GMA.SelfOverlap{Float64, Float64, Float64}(1.0, p.pairs, p.s, -p.w, p.paths)
    lbr, ubr = GMA.penalty_bounds(prep, x, φ, [0.3, 0.3])
    @test lbr <= ubr && ubr ≈ GMA.penalty(prep, xc)

    # the penalized objective bounds still bracket the penalized objective; the penalty adds
    # exactly its own bounds to the overlap bounds
    y = RotationVec(0.3, -0.2, 0.5) * IsotropicGMM(GMA.flex(x, [0.7, -0.4])) + V3(1.0, -2.0, 0.5)
    pσ, pϕ = GMA.pairwise_consts(x, y)
    rigid = UncertaintyRegion(RotationVec(0.2, -0.1, 0.3), V3(0.5, -1.0, 0.2), 0.4, 0.7)
    block = GMA.FlexibleRegion(rigid, φ, [0.5, 0.3])
    lb0, ub0 = GMA.flex_gauss_l2_bounds(x, y, block, pσ, pϕ)
    lbp, ubp = GMA.flex_gauss_l2_bounds(x, y, block, pσ, pϕ; penalties = (p, nothing))
    plb, pub = GMA.penalty_bounds(p, x, φ, [0.5, 0.3])
    @test lbp ≈ lb0 + plb && ubp ≈ ub0 + pub
    params = (0.2, -0.1, 0.3, 0.5, -1.0, 0.2, 0.7, -0.4)
    @test GMA.flex_overlapobj(params, x, y, pσ, pϕ; penalties = (p, nothing)) ≈ GMA.flex_overlapobj(params, x, y, pσ, pϕ) + GMA.penalty(p, xc)

    # end to end: the reported objective is the penalized objective at the reported
    # parameters, and it is bracketed by the bounds; weight 0 is the unpenalized search
    res = GMA.flex_gogma_align(x, y; selfoverlap = 1.0, maxsplits = 200)
    xt = GMA.aligned(res)
    p1 = GMA.SelfOverlap(x; weight = 1.0)
    @test res.upperbound ≈ -overlap(xt, y) + GMA.penalty(p1, xt) atol = 1.0e-8
    @test res.lowerbound <= res.upperbound
    res0 = GMA.flex_gogma_align(x, y; selfoverlap = 0, maxsplits = 200)
    resnone = GMA.flex_gogma_align(x, y; maxsplits = 200)
    @test res0.upperbound == resnone.upperbound
    @test_throws "nonnegative" GMA.flex_gogma_align(x, y; selfoverlap = -1)
    # a rigid model gets no penalty object: the search is unchanged by the weight
    xr = GMA.ArticulatedGMM(gs, GMA.Joint{3, Float64}[])
    @test GMA.flex_gogma_align(xr, y; selfoverlap = 5.0, maxsplits = 20).upperbound ≈ GMA.flex_gogma_align(xr, y; maxsplits = 20).upperbound
end

# objective for a stacked articulated model against a stacked target with given constants
_objective_stacked(x, y, R, T, φ, pσ, pϕ) = -overlap(R * GMA.flex(x, φ) + T, y, pσ, pϕ)

@testset "flexible: stacked articulated models" begin
    V3(a, b, c) = SVector(a, b, c)
    # two slots per feature: an :a slot everywhere, and a :b slot on features 2, 4 and 5
    # (padded elsewhere with ϕ = 0, σ = 1)
    SG(μ, ϕb) = StackedLabeledGaussian(V3(μ...), SVector(1.0, 0.8), SVector(1.0, ϕb), SVector(:a, :b))
    gs = [SG((0, 0, 0), 0.0), SG((1, 0, 0), 0.6), SG((2, 0, 0), 0.0), SG((2, 1, 0), 0.9), SG((3, 0, 0), 0.4)]
    js = [
        GMA.Joint(V3(0, 0, 1.0), V3(1.0, 0, 0), [2, 3, 4, 5], [2]),
        GMA.Joint(V3(0, 1.0, 0), V3(2.0, 0, 0), [4, 5], Int[]),
    ]
    x = GMA.ArticulatedStackedGMM(gs, js)
    @test x isa GMA.AbstractStackedLabeledIsotropicGMM{3, Float64, 2, Symbol}
    @test GMA.njoints(x) == 2

    # kinematics match the isotropic model's, and the type survives flexing and posing
    xiso = GMA.ArticulatedGMM([IsotropicGaussian(g.μ, 1.0, 1.0) for g in gs], js)
    φ = [0.7, -0.4]
    xf = GMA.flex(x, φ)
    @test xf isa GMA.ArticulatedStackedGMM{3, Float64, 2, Symbol}
    @test all(xf.gaussians[i].μ ≈ GMA.flex(xiso, φ).gaussians[i].μ for i in 1:5)
    @test all(xf.gaussians[i].σ == gs[i].σ && xf.gaussians[i].ϕ == gs[i].ϕ && xf.gaussians[i].labels == gs[i].labels for i in 1:5)
    R0 = RotationVec(0.3, -0.2, 0.5)
    posed = R0 * xf + V3(1.0, -2.0, 0.5)
    @test posed isa GMA.ArticulatedStackedGMM{3, Float64, 2, Symbol}
    @test GMA.joint_origin(posed, 2) ≈ R0 * GMA.joint_origin(xf, 2) + V3(1.0, -2.0, 0.5)
    # flexing by duals promotes the number type
    @test GMA.flex(x, ForwardDiff.Dual.(φ, 1.0)) isa GMA.ArticulatedStackedGMM{3, <:ForwardDiff.Dual, 2, Symbol}

    y = StackedLabeledIsotropicGMM(collect(posed.gaussians))
    # bounds validity against the label-aware objective, with equal-label interactions and
    # with a repulsive cross-label term so some slot pairings carry negative weight
    rng = MersenneTwister(20260901)
    for interactions in (nothing, Dict((:a, :a) => 1.0, (:b, :b) => 1.0, (:a, :b) => -0.5))
        pσ, pϕ = GMA.pairwise_consts(x, y, interactions)
        @test eltype(pσ) <: SVector                     # stacked constants are per-pair term lists
        lb_ok = true
        ub_ok = true
        for _ in 1:40
            R = _randR(rng, RotationVec(0, 0, 0), 0.6)
            T = _randT(rng, V3(0, 0, 0), 1.0)
            σᵣ = 0.3 + 0.5rand(rng)
            σₜ = 0.3 + rand(rng)
            φc = [2π * rand(rng) - π for _ in 1:2]
            σφ = [0.8 * rand(rng) for _ in 1:2]
            block = GMA.FlexibleRegion(UncertaintyRegion(R, T, σᵣ, σₜ), φc, σφ)
            lb, ub = GMA.flex_gauss_l2_bounds(x, y, block, pσ, pϕ)
            ub_ok &= isapprox(ub, _objective_stacked(x, y, R, T, φc, pσ, pϕ); atol = 1.0e-8, rtol = 1.0e-8)
            for _ in 1:40
                obj = _objective_stacked(x, y, _randR(rng, R, σᵣ), _randT(rng, T, σₜ), _randφ(rng, φc, σφ), pσ, pϕ)
                lb_ok &= lb <= obj + 1.0e-9
            end
        end
        @test lb_ok
        @test ub_ok
    end

    # the stacked bounds equal the isotropic bounds of the mean-duplicated model: each slot
    # becomes its own Gaussian at the shared mean, with only equal labels interacting
    dup = IsotropicGMM([IsotropicGaussian(g.μ, g.σ[k], g.ϕ[k]) for g in gs for k in 1:2 if g.ϕ[k] != 0])
    duplabels = [g.labels[k] for g in gs for k in 1:2 if g.ϕ[k] != 0]
    dupy = IsotropicGMM([IsotropicGaussian(g.μ, g.σ[k], g.ϕ[k]) for g in y.gaussians for k in 1:2 if g.ϕ[k] != 0])
    dupylabels = [g.labels[k] for g in y.gaussians for k in 1:2 if g.ϕ[k] != 0]
    # a mean-duplicated articulated model: joint feature lists map each stacked feature to
    # its surviving slots
    slotowner = [i for (i, g) in enumerate(gs) for k in 1:2 if g.ϕ[k] != 0]
    dupjs = [GMA.Joint(j.axis, j.origin, findall(in(j.features), slotowner), j.children) for j in js]
    dupx = GMA.ArticulatedGMM(collect(dup.gaussians), dupjs)
    pσd = [a.σ^2 + b.σ^2 for a in dup.gaussians, b in dupy.gaussians]
    pϕd = [duplabels[i] == dupylabels[j] ? a.ϕ * b.ϕ : 0.0 for (i, a) in enumerate(dup.gaussians), (j, b) in enumerate(dupy.gaussians)]
    pσ, pϕ = GMA.pairwise_consts(x, y)
    block = GMA.FlexibleRegion(UncertaintyRegion(RotationVec(0.2, -0.1, 0.3), V3(0.5, -1.0, 0.2), 0.4, 0.7), φ, [0.5, 0.3])
    lbs, ubs = GMA.flex_gauss_l2_bounds(x, y, block, pσ, pϕ)
    lbd, ubd = GMA.flex_gauss_l2_bounds(dupx, dupy, block, pσd, pϕd)
    @test lbs ≈ lbd && ubs ≈ ubd

    # self-overlap on a stacked model: per pair, the stacked Gaussian-pair overlap; and its
    # bounds bracket the penalty over sampled conformations
    p = GMA.SelfOverlap(x; weight = 1.5)
    @test length(p) == 8
    xc = GMA.flex(x, φ)
    @test GMA.penalty(p, xc) ≈ 1.5 * sum(overlap(xc.gaussians[g], xc.gaussians[h]) for (g, h) in p.pairs)
    prep = GMA.SelfOverlap(x; interactions = Dict((:a, :a) => 1.0, (:b, :b) => 1.0, (:a, :b) => -0.5))
    @test GMA.penalty(prep, xc) ≈ sum(overlap(xc.gaussians[g], xc.gaussians[h]; interactions = prep_int) for (g, h) in prep.pairs for prep_int in (Dict((:a, :a) => 1.0, (:b, :b) => 1.0, (:a, :b) => -0.5),))
    @test_throws "labeled" GMA.SelfOverlap(xiso; interactions = Dict((:a, :a) => 1.0))
    for pen in (p, prep)
        lb_ok = true
        ub_ok = true
        for _ in 1:40
            φc = [2π * rand(rng) - π for _ in 1:2]
            σφ = [0.9 * rand(rng) for _ in 1:2]
            lb, ub = GMA.penalty_bounds(pen, x, φc, σφ)
            ub_ok &= isapprox(ub, GMA.penalty(pen, GMA.flex(x, φc)); atol = 1.0e-10)
            for _ in 1:40
                lb_ok &= lb <= GMA.penalty(pen, GMA.flex(x, _randφ(rng, φc, σφ))) + 1.0e-9
            end
        end
        @test lb_ok
        @test ub_ok
    end

    # end to end on stacked models with the penalty: reported objective is the penalized
    # objective at the reported parameters, bracketed by the bounds
    res = GMA.flex_gogma_align(x, y; selfoverlap = 1.0, maxsplits = 150)
    xt = GMA.aligned(res)
    @test xt isa GMA.ArticulatedStackedGMM
    @test res.upperbound ≈ -overlap(xt, y) + GMA.penalty(GMA.SelfOverlap(x), xt) atol = 1.0e-8
    @test res.lowerbound <= res.upperbound
    # a stacked model cannot be paired with an unlabeled target
    @test_throws "stacked" GMA.flex_gogma_align(x, IsotropicGMM(collect(dupy.gaussians)))
end
