using OrdinaryDiffEqBDF, Test, LinearAlgebra
using OrdinaryDiffEqBDF: _qndf_restart!

# QNDF's restart at a discontinuity: the first backward difference is seeded from the
# vector field on the new side (dt u'(t⁺)) instead of zero, algebraic rows of a diagonal
# mass matrix are left out of the first step's error estimate, a non-diagonal mass matrix
# keeps the unseeded restart, and the order the controller sees is the restart's order 1.

# u' = g(u) switches from g1 to g2 where u crosses 1/2 (u' jumps)
function switching_problem(; mass_matrix = nothing, tspan = (0.0, 6.0))
    mode = Ref(1)
    function f!(du, u, p, t)
        du[1] = mode[] == 1 ? 1.0 - u[1] : 3.0 - u[1]
        length(u) > 1 && (du[2] = u[2] - 2.0 * u[1])      # 0 = z - 2x on the algebraic row
        nothing
    end
    cb = ContinuousCallback(
        (u, t, i) -> mode[] == 1 ? u[1] - 0.5 : 1.0,
        i -> (mode[] = 2; derivative_discontinuity!(i, true)); save_positions = (false, false)
    )
    u0 = mass_matrix === nothing ? [0.0] : [0.0, 0.0]
    fun = mass_matrix === nothing ? ODEFunction(f!) : ODEFunction(f!; mass_matrix)
    return ODEProblem(fun, u0, tspan), cb, mode
end

# Step until the callback has fired; return the integrator right after that step
function step_to_event(prob, cb, mode; tol = 1.0e-8)
    integ = init(prob, QNDF(); callback = cb, abstol = tol, reltol = tol, dense = false)
    while mode[] == 1
        step!(integ)
    end
    return integ
end

@testset "seeded restart" begin
    @testset "D[1] = dt u'(t⁺), prevD alike, dtprev = dt" begin
        prob, cb, mode = switching_problem()
        integ = step_to_event(prob, cb, mode)
        cache = integ.cache
        dtprev = _qndf_restart!(cache, integ)
        rate = [3.0 - integ.uprev[1]]
        @test dtprev == integ.dt
        @test cache.D[1] ≈ integ.dt .* rate
        @test cache.prevD[1] ≈ integ.dt .* rate
        @test cache.dtprev == integ.dt
        @test cache.restart
    end

    @testset "diagonal mass matrix: algebraic rows unseeded" begin
        prob, cb, mode = switching_problem(mass_matrix = Diagonal([1.0, 0.0]))
        integ = step_to_event(prob, cb, mode)
        cache = integ.cache
        _qndf_restart!(cache, integ)
        @test cache.D[1][1] ≈ integ.dt * (3.0 - integ.uprev[1])
        @test cache.D[1][2] == 0
        @test cache.restart
    end

    @testset "non-diagonal mass matrix: unseeded restart" begin
        prob, cb, mode = switching_problem(mass_matrix = [1.0 0.0; 1.0 0.0])
        integ = step_to_event(prob, cb, mode)
        cache = integ.cache
        @test _qndf_restart!(cache, integ) == one(integ.dt)
        @test !cache.restart
    end

    # Step until the first step accepted after the event; return the integrator, the number
    # of rejections on the way, and the length of that step
    function first_step_after_event(prob, cb, mode; tol)
        integ = step_to_event(prob, cb, mode; tol)
        acc0, rej0 = integ.stats.naccept, integ.stats.nreject
        while integ.stats.naccept == acc0
            step!(integ)
        end
        return integ, integ.stats.nreject - rej0, integ.t - integ.tprev
    end

    @testset "the first step is not limited by dt |u'|" begin
        # With the zero restart the first step is rejected down to about tol / |u'|
        # (the rate jumps to 3 - 1/2 = 5/2): the seeded one takes steps thousands of
        # times longer, in a few rejections
        tol = 1.0e-10
        for mm in (nothing, Diagonal([1.0, 0.0]))
            prob, cb, mode = switching_problem(mass_matrix = mm)
            integ, nrej, dt1 = first_step_after_event(prob, cb, mode; tol)
            @test dt1 > 1000 * tol / 2.5
            @test nrej <= 4    # the step size carried over from before the event may need cutting
            @test !integ.cache.restart
        end
    end

    @testset "the order after the restart step is the restart's" begin
        prob, cb, mode = switching_problem()
        integ, _, _ = first_step_after_event(prob, cb, mode; tol = 1.0e-10)
        @test integ.cache.order <= 2    # order 1 stepped, the controller raises it by at most one
    end
end
