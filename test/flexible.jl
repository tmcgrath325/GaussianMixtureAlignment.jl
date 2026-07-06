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
