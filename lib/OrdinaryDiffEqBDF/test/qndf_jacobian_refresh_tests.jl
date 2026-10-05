using OrdinaryDiffEqBDF, Test

# QNDF takes a new J and W at the first step after a discontinuity: the J of the other
# side of it is stale (the shared test `isJcurrent` compares only `t`), which let a state
# held at its value (rate exactly zero after the transition) drift by about the tolerance
# under the old W.

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

@testset "J and W taken anew after a discontinuity" begin
    prob, cb, mode = switching_problem()
    integ = init(prob, QNDF(); callback = cb, abstol = 1.0e-8, reltol = 1.0e-8, dense = false)
    njacs_before = 0
    while mode[] == 1
        njacs_before = integ.stats.njacs
        step!(integ)
    end
    # the step that located the event, then the first step after it
    njacs_event = integ.stats.njacs
    step!(integ)
    @test integ.stats.njacs == njacs_event + 1
    step!(integ)    # no discontinuity: no new J when the step is accepted unchanged
    @test integ.stats.njacs <= njacs_event + 2
end

@testset "a held state does not drift" begin
    # a' = 0 exactly after the event, b' = -b + a: a keeps its event value; with the J of the
    # other side it drifted by about the tolerance
    mode = Ref(1)
    function f!(du, u, p, t)
        a, b = u
        du[1] = mode[] == 1 ? 1.0 - a - 0.3 * b : 0.0
        du[2] = -b + a
        nothing
    end
    cb = ContinuousCallback(
        (u, t, i) -> mode[] == 1 ? u[1] - 0.5 : 1.0,
        i -> (mode[] = 2; derivative_discontinuity!(i, true)); save_positions = (false, false)
    )
    for tol in (1.0e-6, 1.0e-10)
        mode[] = 1
        sol = solve(ODEProblem(f!, [0.0, 0.0], (0.0, 6.0)), QNDF(); callback = cb, abstol = tol, reltol = tol)
        t_event = sol.t[findfirst(u -> u[1] >= 0.5 - 1.0e-12, sol.u)]
        held = [u[1] for (t, u) in zip(sol.t, sol.u) if t > t_event]
        @test maximum(abs.(held .- 0.5)) < 1.0e-12
    end
end
