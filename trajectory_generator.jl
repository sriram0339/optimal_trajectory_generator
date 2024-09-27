include("./trajectory_optimization.jl")

using Plots
using ArgParse 

function print_trajectory(i::Int, X_res::Matrix{Float64}, U_res::Matrix{Float64}, T_res::Vector{Float64}, ctrl_dt::Float64, workspace_info::Workspace)
    file_name = "trajectory_$(i).csv"
    open(file_name, "w") do file
        println(file, "t, x, y, omega, v")
        n_steps = length(T_res)
        @assert size(X_res, 1) == n_steps 
        n = size(X_res, 2)
        for j in 1:n_steps 
            print(file, "$j, ")
            print(file, "$(T_res[j]), ")
            for k in 1:n
                print(file, "$(X_res[j, k]), ")
            end
            println(file, "")
        end
    end

    file_name = "control_$(i).csv"
    open(file_name, "w") do file
        println(file, "t, accel, throttle")
        n_steps  = size(U_res, 1)
        n_ctrls = size(U_res, 2)
        for j in 1:n_steps 
            print(file, "$(j*ctrl_dt), ")
            for k in 1:n_ctrls
                print(file, "$(U_res[j, k]), ")
            end
            println(file, "")
        end
    end

    plot(X_res[:, 1], X_res[:, 2], label="Trajectory",  xlabel="x", ylabel="y",marker=:circle, markersize=0.5)
    plot!([workspace_info.waypoints[i][1] for i in 1:length(workspace_info.waypoints)], [workspace_info.waypoints[i][2] for i in 1:length(workspace_info.waypoints)], marker=:cross, label="Waypoints", xlabel="x", ylabel="y", linestyle=:dash)
    # plot each obstacle as a circle of radius 0.5
    for obs in workspace_info.obstacle_locations
        plot!([obs[1]], [obs[2]], marker=:circle, label="Obstacle", xlabel="x", ylabel="y", linestyle=:dash, markersize=7.5)
    end
    savefig("trajectory_$(i).png")
end 

function generate_random_weights(n::Int)
    function gen_weight()
        (a, b) = rand([(0.0, 0.5),  (2.0, 6.0),  (8.0, 10.0)])
        return a + (b - a) * rand()
    end
    weights = [gen_weight() for _ in 1:n]
    weights[7] = 20.0 # ensure that the weight associated with reaching the goal is high.
    #weights = 10.0 * weights/maximum([abs(wi) for wi in weights]) # normalize by L_inf norm.
    return weights
end

function generate_trajectory(sys::DynamicalSystem, workspace_info::Workspace, ctrl_dt::Float64, ctrl_limits::Vector{Tuple{Float64, Float64}}, n_steps::Int, num_trajectories::Int; use_simulated_annealing::Bool=false)
    filename_weights = "trajectory_weights.csv"
    filename_costs = "trajectory_costs.csv"
    open(filename_weights, "w") do file_weights
        open(filename_costs, "w") do file_costs
            println(file_costs, "id, avg_obstacle_distance, max_obstacle_distance , avg_waypoints_dev, max_waypoints_dev, avg_velocity_dev, avg_control_effort, dev_from_tgt_pos, backsliding_cost")
            println(file_weights, "id, avg_obstacle_distance, max_obstacle_distance , avg_waypoints_dev, max_waypoints_dev, avg_velocity_dev, avg_control_effort, dev_from_tgt_pos, backsliding_cost")
            for i in 1:num_trajectories
                println("Generating trajectory $i")
                weights = generate_random_weights(8)
                

                if use_simulated_annealing
                    X_res, U_res, T_res, obj_cost = optimize_trajectory(workspace_info, sys, ctrl_dt, ctrl_limits, n_steps, weights)
                    println("SAMin obj cost: $obj_cost")
                else
                    X_res, U_res, T_res, obj_cost = optimize_trajectory_bbo(workspace_info, sys, ctrl_dt, ctrl_limits, n_steps, weights)
                    println("BBO obj cost: $obj_cost")
                end
                
                
                print_trajectory(i, X_res, U_res, T_res, ctrl_dt, workspace_info) 
                costs = evaluate_trajectory_cost(X_res, U_res, workspace_info)
                println("Weights: $weights")
                println("Costs: $costs")
                print(file_costs, "$i, ")
                for cost in costs
                    print(file_costs, "$cost, ")
                end
                println(file_costs, "")

                print(file_weights, "$i, ")
                for w in weights
                    print(file_weights, "$w, ")
                end
                println(file_weights, "")
            end
        end
    end
end

function main()
    # prase command line arguments 
    ap = ArgParse.ArgumentParser()
    ArgParse.add_argument(ap, "--use_simulated_annealing"; action="store_true", help="Use simulated annealing to optimize the trajectory")
    ArgParse.add_argument(ap, "--num_trajectories"; type=Int, default=10, help="Number of trajectories to generate")
    args = ArgParse.parse_args(ap)  


    dyn_sys = DynamicalSystem(vehicle_model, [2.0], [0., 0. , 0.35 + 0.1*rand(), 5*rand()])
    obstacle_locations = [
    [2.0, 1.0], 
    [6.0, 2.5],
    [10.0, 3.8]
    ]
    waypoints =[ [i*0.5 + 0.005 * i^2, i*0.25] for i in 0:25]

    target_velocity = 2.5
    winfo = Workspace(obstacle_locations, waypoints, target_velocity)
    # control time step is 0.25 
    # acceleration limit is 1.0 m/s^2
    # steering rate limit is 0.4 rad/s
    # number of steps is 40
    # number of trajectories is 10
    generate_trajectory(dyn_sys, winfo, 0.25, [(-1.0, 1.0), (-0.4, 0.4)], 40, args.num_trajectories, use_simulated_annealing=args.use_simulated_annealing)
end

main()