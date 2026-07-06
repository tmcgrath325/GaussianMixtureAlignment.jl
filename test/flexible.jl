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
    @test length(center(fr)) == 8                        # 6 rigid + 2 joints
    @test center(fr) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5)
    @test UncertaintyRegion(fr) === ur                   # projection to the rigid box

    # the K-argument form covers the full angular range and, at K = 0, is purely rigid
    frfull = GMA.FlexibleRegion(ur, 2)
    @test frfull.φ == SVector(0.0, 0.0) && frfull.σφ == SVector(Float64(π), Float64(π))
    fr0 = GMA.FlexibleRegion(ur, 0)
    @test length(center(fr0)) == 6
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
