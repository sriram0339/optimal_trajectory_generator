include("./generate_trajectory.jl")
include("./trajectory_evaluation.jl")
using LinearAlgebra
using Random
#using OptimizationMetaheuristics
#using OptimizationOptimJL
#using OptimizationCMAEvolutionStrategy
#using Optim 
#using OptimizationBBO

#using OptimizationPRIMA
using BlackBoxOptim
using Optimization
using OptimizationOptimJL
function optimize_trajectory_bbo(workspace_info::Workspace, sys::DynamicalSystem, ctrl_dt::Float64, ctrl_limits::Vector{Tuple{Float64, Float64}}, n_steps::Int, weights::Vector{Float64})
   
    n_inputs = size(ctrl_limits, 1)
    function cost_fun(u)::Float64
        U = reshape(u, n_steps, n_inputs)
        X, _ = generate_trajectory(sys,  U, ctrl_dt)
        cost = evaluate_trajectory_cost(X, U, workspace_info)
        return weights ⋅ cost
    end

    upper_bounds = vcat([repeat([tup[2]], n_steps) for tup in ctrl_limits]...)
    lower_bounds = vcat([repeat([tup[1]], n_steps) for tup in ctrl_limits]...)
    u0 = zeros(n_inputs * n_steps, 1)
    opt_free = bbsetup(cost_fun; Method=:adaptive_de_rand_1_bin_radiuslimited, SearchRange = (collect(zip(lower_bounds,upper_bounds))),
               NumDimensions = n_steps*n_inputs, TraceMode= :compact, TraceInterval=30, MaxFuncEvals = 20000, NumRestarts= 4 )
    bb_res = bboptimize(opt_free)
    opt_cost = best_fitness(bb_res)
    print(bb_res)
    u_res= best_candidate(bb_res)
    U_res = reshape(u_res, n_steps, n_inputs)
    X_res, T_res = generate_trajectory(sys,  U_res, ctrl_dt)
    cost = evaluate_trajectory_cost(X_res, U_res, workspace_info)
    println("Cost of optimization: $(cost)")
    return X_res, U_res, T_res, opt_cost
end
function optimize_trajectory(workspace_info::Workspace, sys::DynamicalSystem, ctrl_dt::Float64, ctrl_limits::Vector{Tuple{Float64, Float64}}, n_steps::Int, weights::Vector{Float64})

    n_inputs = size(ctrl_limits, 1)
    function cost_fun(u, _)::Float64
        U = reshape(u, n_steps, n_inputs)
        X, _ = generate_trajectory(sys,  U, ctrl_dt)
        cost = evaluate_trajectory_cost(X, U, workspace_info)
        return weights ⋅ cost
    end

    #optfun = OptimizationFunction(cost_fun, Optimization.AutoForwardDiff())
    upper_bounds = vcat([repeat([tup[2]], n_steps) for tup in ctrl_limits]...)
    lower_bounds = vcat([repeat([tup[1]], n_steps) for tup in ctrl_limits]...)
    u0 = zeros(n_inputs * n_steps, 1) 
    #println(reshape(lower_bounds, n_steps, n_inputs))
    p = [10.0, 11.0, 12.0]
    opt_problem = Optimization.OptimizationProblem(cost_fun, u0, p, lb= lower_bounds, ub = upper_bounds)
    #opt_result = solve(opt_problem,  SA(), maxiters=1000)
    opt_result = Optimization.solve(opt_problem, SAMIN(), maxiters=10000)
    #opt_result= solve(opt_problem, BBO_xnes())
    #opt_result = solve(opt_problem, GCMAESOpt())
    u_res = opt_result.minimizer
    opt_cost = opt_result.minimum
    U_res = reshape(u_res, n_steps, n_inputs)
    X_res, T_res = generate_trajectory(sys,  U_res, ctrl_dt)
    cost = evaluate_trajectory_cost(X_res, U_res, workspace_info)
    println("Cost of optimization: $(cost)")
    # return the best trajectory, the control inputs, the time vector and the cost
    return X_res, U_res, T_res, opt_cost
end
