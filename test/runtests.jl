using LowLevelParticleFiltersMTK
using LowLevelParticleFilters
using LowLevelParticleFilters: SimpleMvNormal
using ModelingToolkit
using SeeToDee
using Test
using Plots
using StaticArrays
using LinearAlgebra

# @testset "LowLevelParticleFiltersMTK.jl" begin
    t = ModelingToolkit.t_nounits
D = ModelingToolkit.D_nounits

@mtkmodel SimpleSys begin
    @variables begin
        x(t) = 0
        u(t) = 0
        y(t)
        w(t), [disturbance = true, input = true]
    end
    @equations begin
        D(x) ~ -x + u + w # Explicitly encode where dynamics noise enters the system with w
        y ~ x
    end
end

@named model = SimpleSys()
cmodel = complete(model)
inputs = [cmodel.u]
outputs = [cmodel.y]
disturbance_inputs = [cmodel.w]


nw = length(disturbance_inputs)
ny = length(outputs)
R1 = SMatrix{nw,nw}(0.1I(nw))
R2 = SMatrix{ny,ny}(0.1I(ny))

df = SimpleMvNormal(R1)
dg = SimpleMvNormal(R2)

Ts = 0.1
discretization = (f,Ts,x_inds,a_inds,nu)->SeeToDee.Rk4(f, Ts)

prob = StateEstimationProblem(cmodel, inputs, outputs; disturbance_inputs, df, dg, discretization, Ts)
ekf = get_filter(prob, ExtendedKalmanFilter)
ukf = get_filter(prob, UnscentedKalmanFilter)

u = [randn(1) for _ in 1:10]
x,u,y = simulate(ekf, u, dynamics_noise=true, measurement_noise=true)


fsole = forward_trajectory(ekf, u, y)
fsolu = forward_trajectory(ukf, u, y)
sole = StateEstimationSolution(fsole, prob)
solu = StateEstimationSolution(fsolu, prob)

@test fsole.xt ≈ fsolu.xt
@test fsole.Rt ≈ fsolu.Rt

using Plots
plot(fsole, size=(1000, 1000))
plot!(fsole.t, reduce(hcat, x)')

plot(fsolu, size=(1000, 1000))
plot!(fsolu.t, reduce(hcat, x)')


plot(sole)
plot!(solu)


plot(sole, idxs=cmodel.y^2 + 0.1*sin(cmodel.u))
plot!(solu, idxs=cmodel.y^2 + 0.1*sin(cmodel.u))



##
@test sole[cmodel.x] == sole[cmodel.y]

## DAE UKF on a constrained pendulum — INDEX-1 Baumgarte form. The position
## constraint x² + y² = 1 is replaced by its twice-differentiated,
## dynamics-substituted (acceleration-level) form augmented with Baumgarte
## stabilization. The unstabilized acceleration-level constraint is neutrally
## stable in (x²+y²−1): discretization error drifts the state off the unit
## circle with no restoring force. Adding 2α·(x·vx + y·vy) + (β²/2)·(x²+y²−1)
## gives the constraint a critically-damped restoring law (α = β = 50), so
## violations decay exponentially with timescale 1/α. With this reformulation
## MTK does not need to do index reduction and the full predict + correct +
## `forward_trajectory` pipeline keeps the covariance positive-definite.
@testset "DAE UKF" begin
    using SimpleNonlinearSolve
    using SciMLBase

    @mtkmodel Pendulum begin
        @parameters begin
            g_pend = 9.82
            α_bg   = 50.0
            β_bg   = 50.0
        end
        @variables begin
            x(t)  = 1.0
            y(t)  = 0.0
            vx(t) = 0.0
            vy(t) = 0.0
            λ(t)  = 0.0
            f1(t) = 0.0
            f2(t) = 0.0
            meas_x(t)
            meas_λ(t)
            w1(t), [disturbance = true, input = true]
            w2(t), [disturbance = true, input = true]
            w3(t), [disturbance = true, input = true]
            w4(t), [disturbance = true, input = true]
        end
        @equations begin
            D(x)  ~ vx + w1
            D(y)  ~ vy + w2
            D(vx) ~ -λ*x + f1 + w3
            D(vy) ~ -λ*y - g_pend + f2 + w4
            0     ~ vx^2 + vy^2 + x*(-λ*x + f1 + w3) + y*(-λ*y - g_pend + f2 + w4) +
                    2*α_bg*(x*vx + y*vy) +
                    (β_bg^2 / 2)*(x^2 + y^2 - 1)
            meas_x ~ x
            meas_λ ~ λ
        end
    end

    @named pendulum = Pendulum()
    cpend = complete(pendulum)
    pend_inputs   = [cpend.f1, cpend.f2]
    pend_outputs  = [cpend.meas_x, cpend.meas_λ]
    pend_dist_in  = [cpend.w1, cpend.w2, cpend.w3, cpend.w4]

    Ts_dae = 0.01
    R1_dae = SMatrix{4,4}(Diagonal([1e-3, 1e-3, 1e-3, 1e-3]))
    R2_dae = SMatrix{2,2}(1e-2*I)
    df_dae = SimpleMvNormal(R1_dae)
    dg_dae = SimpleMvNormal(R2_dae)

    dae_disc = (f, Ts, xi, ai, nu) ->
        SeeToDee.Trapezoidal(f, Ts, xi, ai, nu; inplace=false)

    x0map_pend = [cpend.x => 1.0, cpend.y => 0.0,
                  cpend.vx => 0.0, cpend.vy => 0.0,
                  cpend.λ => 0.0]

    prob_pend = StateEstimationProblem(cpend, pend_inputs, pend_outputs;
                                       disturbance_inputs = pend_dist_in,
                                       df = df_dae, dg = dg_dae,
                                       discretization = dae_disc,
                                       Ts = Ts_dae,
                                       x0map = x0map_pend, σ0 = 0.1)

    # Index-1 form: 4 differential states + 1 algebraic equation, nw == nx_diff.
    @test prob_pend.na == 1
    @test length(prob_pend.x_inds) == 4
    @test prob_pend.nw == length(prob_pend.x_inds)

    csolver = LowLevelParticleFilters.scimlbase_solver(SimpleNewtonRaphson(); reltol=1e-12)
    daeukf  = get_filter(prob_pend, DAEUnscentedKalmanFilter; constraint_solver = csolver)

    @test daeukf isa DAEUnscentedKalmanFilter
    @test length(daeukf.x) == 4
    @test length(daeukf.xz) == 5
    @test size(daeukf.R) == (4, 4)
    @test daeukf.nu == prob_pend.nu
    @test daeukf.ny == prob_pend.ny

    # Auto-generated descriptor split round-trips, and the residual at the
    # initial on-manifold state vanishes.
    let
        xz0 = daeukf.xz
        x_part, z_part = daeukf.get_x_z(xz0)
        @test daeukf.build_xz(x_part, z_part) ≈ xz0
        r = daeukf.residual(x_part, z_part, SA[0.0, 0.0], daeukf.p, 0.0)
        @test maximum(abs, r) < 1e-8
    end

    # End-to-end forward_trajectory under a driven input: the filter completes
    # the trajectory and the covariance stays positive-definite throughout.
    T_steps = 500
    t_vec   = (0:T_steps-1) .* Ts_dae
    u_drive = [SA[sin(tt^2), sin(tt^2 + 1.0)] for tt in t_vec]
    _, u_sim, y_sim = simulate(daeukf, u_drive)

    daeukf2 = get_filter(prob_pend, DAEUnscentedKalmanFilter; constraint_solver = csolver)
    fsol = forward_trajectory(daeukf2, u_sim, y_sim)

    @test length(fsol.xt) == T_steps
    @test all(R -> minimum(eigen(Symmetric(Matrix(R))).values) > 0, fsol.Rt)
    @test all(xt -> all(isfinite, xt), fsol.xt)
end

## Test static keyword argument
@testset "static keyword argument" begin
    # Test default behavior (static=true)
    prob_static = StateEstimationProblem(cmodel, inputs, outputs; disturbance_inputs, df, dg, discretization, Ts, static=true)
    @test prob_static.d0.μ isa SVector
    @test prob_static.d0.Σ isa SMatrix

    # Test with static=false
    prob_dynamic = StateEstimationProblem(cmodel, inputs, outputs; disturbance_inputs, df, dg, discretization, Ts, static=false)
    @test prob_dynamic.d0.μ isa Vector
    @test prob_dynamic.d0.Σ isa Matrix

    # Test backward compatibility (default should be static=true)
    prob_default = StateEstimationProblem(cmodel, inputs, outputs; disturbance_inputs, df, dg, discretization, Ts)
    @test prob_default.d0.μ isa SVector
    @test prob_default.d0.Σ isa SMatrix

    # Verify filters work with both array types
    ekf_static = get_filter(prob_static, ExtendedKalmanFilter)
    ekf_dynamic = get_filter(prob_dynamic, ExtendedKalmanFilter)

    # Test that both filters can process the same data
    fsole_static = forward_trajectory(ekf_static, u, y)
    fsole_dynamic = forward_trajectory(ekf_dynamic, u, y)

    # Results should be approximately equal regardless of array type
    @test fsole_static.xt[end] ≈ fsole_dynamic.xt[end]
    @test fsole_static.Rt[end] ≈ fsole_dynamic.Rt[end]
end

# end
