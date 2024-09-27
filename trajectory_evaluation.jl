
struct Workspace
    # obstacle locations 
    obstacle_locations::Vector{Vector{Float64}}
    # waypoints 
    waypoints::Vector{Vector{Float64}}
    # target velocity 
    target_velocity::Float64
    # obstacle distance threshold
    obstacle_distance_threshold::Float64
    # waypoint tracking threshold
    waypoint_tracking_threshold::Float64
    
end

# Define a constructor with default values
Workspace(obstacle_locations::Vector{Vector{Float64}}, waypoints::Vector{Vector{Float64}}, target_velocity::Float64;
          obstacle_distance_threshold::Float64=1.5, waypoint_tracking_threshold::Float64=0.1) = 
    Workspace(obstacle_locations, waypoints, target_velocity, obstacle_distance_threshold, waypoint_tracking_threshold)

function obstacle_cost_fun(d::Float64, workspace_info::Workspace)::Float64
   #if d < workspace_info.obstacle_distance_threshold
        500.0 * exp( -0.5 * d^2/0.1)
   #lse
    #    0.0
   #end

end 

function wp_cost_fun(d::Float64, workspace_info::Workspace)::Float64
   if d < workspace_info.waypoint_tracking_threshold
        0.0 
   else
        5 * (d-workspace_info.waypoint_tracking_threshold)^2
   end
end 

function compute_obstacle_cost(trajectory::Matrix{Float64}, workspace_info::Workspace; sum_cost::Bool=true)::Float64
    # trajectory is a vector of states 
    # workspace_info is a struct with the workspace setup
    # return a scalar cost 
    # for each row in trajectory matrix, 
    # find minimum distance to any obstacle
    # if minimum distance is less than a threshold, accumulate to the cost.  
    # return the cost 
    @assert size(trajectory, 1) >= 1 "Trajectory must have at least one state"
    min_distances = [ minimum(norm(trajectory[i, 1:2] - obs) for obs in workspace_info.obstacle_locations) for i in 1:size(trajectory, 1)]
    if sum_cost 
        cost = sum( [obstacle_cost_fun(d, workspace_info) for d in min_distances])/(size(trajectory, 1))
    else
        cost = maximum( [obstacle_cost_fun(d, workspace_info) for d in min_distances]) 
    end
    return cost
end 

function point_to_segment_distance(point::Vector{Float64}, segment_start::Vector{Float64}, segment_end::Vector{Float64})::Float64
    # point is a vector of the point coordinates
    # segment_start is a vector of the start point coordinates
    # segment_end is a vector of the end point coordinates
    # return a scalar distance

    # compute the projection of the point onto the line segment
    # compute the distance from the point to the line segment
    # return the distance

    @assert length(point) == 2 "Point must have 2 coordinates"
    @assert length(segment_start) == 2 "Segment start must have 2 coordinates"
    @assert length(segment_end) == 2 "Segment end must have 2 coordinates"
    (x, y) = point
    (x1, y1) = segment_start
    (x2, y2) = segment_end
    if abs(x2 - x1) < 1e-6
        return abs(x - x1)
    else    
        m = (y2 - y1)/(x2 - x1)
        c = y1 - m*x1
        y_proj = m*x + c
        return abs(y-y_proj)/(sqrt(1+m^2)) 
    end
end

function compute_waypoint_deviation_cost_old(trajectory::Matrix{Float64}, workspace_info::Workspace; sum_cost::Bool=true)::Float64
    # trajectory is a vector of states 
    # workspace_info is a struct with the workspace setup
    # return a scalar cost 
    @assert size(trajectory, 1) >= 1 "Trajectory must have at least one state"
    wps = workspace_info.waypoints
    cost = 0.0 
    for i in 1:size(trajectory,1)
        # first find the waypoints 
        (xi, yi) = trajectory[i, 1:2]
        d = 0.0 
        if (xi < wps[1][1])
            d= norm(trajectory[i, 1:2] - wps[1])

        elseif (xi >= wps[end][1])
            d = norm(trajectory[i, 1:2] - wps[end])
        else
            for j in 2:length(wps)
                if (xi < wps[j][1])
                    d = min(norm(trajectory[i, 1:2] - wps[j-1]), norm(trajectory[i, 1:2] - wps[j]))
                    break
                end
            end
        end
        # cost += wp_cost_fun(d, workspace_info)
        if sum_cost 
            cost += wp_cost_fun(d, workspace_info)
        else    
            cost = max(cost, wp_cost_fun(d, workspace_info))
        end
    end
    if sum_cost 
        cost = cost /(size(trajectory, 1))
    end
    return cost 
end 

function compute_waypoint_deviation_cost(trajectory::Matrix{Float64}, workspace_info::Workspace; sum_cost::Bool=true)::Float64
    # trajectory is a vector of states 
    # workspace_info is a struct with the workspace setup
    # 1. calculate the distance from current state to the current waypoint 
    cur_wp_idx = 1
    cost = 0.0
    num_wp = length(workspace_info.waypoints)
    
    for i in 1:size(trajectory, 1)
        @assert cur_wp_idx <= num_wp - 1 "Waypoint index out of bounds"
        cur_wp = workspace_info.waypoints[cur_wp_idx]
        next_wp = workspace_info.waypoints[cur_wp_idx + 1]
        dist = point_to_segment_distance(trajectory[i, 1:2], cur_wp, next_wp)
        if sum_cost 
            cost += wp_cost_fun(dist, workspace_info)
        else    
            cost = max(cost, wp_cost_fun(dist, workspace_info))
        end
        if cur_wp_idx < num_wp - 1
            next_to_next_wp = workspace_info.waypoints[cur_wp_idx + 2]
            if (norm(trajectory[i, 1:2] - next_to_next_wp) < norm(trajectory[i, 1:2] - cur_wp))
                cur_wp_idx += 1
            end
        end
    end
    if sum_cost 
        cost = cost /(size(trajectory, 1))
    end
    return cost 
end


function check_waypoint_deviation_threshold(trajectory::Matrix{Float64}, workspace_info::Workspace)::Bool
    # trajectory is a vector of states 
    # workspace_info is a struct with the workspace setup
    # 1. calculate the distance from current state to the current waypoint 
    cur_wp_idx = 1
    num_wp = length(workspace_info.waypoints)
    for i in 1:size(trajectory, 1)
        @assert cur_wp_idx <= num_wp - 1 "Waypoint index out of bounds"
        cur_wp = workspace_info.waypoints[cur_wp_idx]
        next_wp = workspace_info.waypoints[cur_wp_idx + 1]
        dist = point_to_segment_distance(trajectory[i, 1:2], cur_wp, next_wp)
        if dist < workspace_info.waypoint_tracking_threshold
            return true
        end 
        if cur_wp_idx < num_wp - 1
            next_to_next_wp = workspace_info.waypoints[cur_wp_idx + 2]
            if (norm(trajectory[i, 1:2] - next_to_next_wp) < norm(trajectory[i, 1:2] - cur_wp))
                cur_wp_idx += 1
            end
        end
    end
    return false
end

function compute_target_velocity_deviation_cost(trajectory::Matrix{Float64}, workspace_info::Workspace)::Float64
    # trajectory is a vector of states 
    # workspace_info is a struct with the workspace setup
    # return a scalar cost 
    # for each row in trajectory matrix, 
    # find minimum distance to any waypoint
    # if minimum distance is less than a threshold, accumulate to the cost.  
    # return the cost
    @assert size(trajectory, 1) >= 1 "Trajectory must have at least one state"
    cost = sum((trajectory[i, 4] - workspace_info.target_velocity)^2 for i in 1:size(trajectory, 1))/(size(trajectory, 1))
    return cost 
end  

function compute_control_effort_cost(control_inputs::Matrix{Float64})::Float64
    # control_inputs is a vector of control inputs 
    # return a scalar cost 
    # for each row in control_inputs matrix, 
    # find the norm of the control input 
    # return the cost 
    cost = sum(norm(control_inputs[i, :]) for i in 1:size(control_inputs, 1))/(size(control_inputs, 1))
    return cost
end  

function compute_monotonicity_of_heading(trajectory::Matrix{Float64})::Float64
    # trajectory is a vector of states 
    # return a scalar cost 
    # for each row in trajectory matrix, 
    # find the difference in heading between the current state and the next state
    # return the cost 
    cost = sum([ 10.0 * abs(trajectory[i, 4] * cos(trajectory[i, 3])) for i in 1:size(trajectory, 1) if trajectory[i, 4] * cos(trajectory[i, 3]) < 0.0 ])/(size(trajectory, 1))
    return cost
end

# function to evaluate the trajectory cost
function evaluate_trajectory_cost(trajectory::Matrix{Float64}, control_inputs::Matrix{Float64}, workspace_info::Workspace)::Vector{Float64}
    # trajectory is a vector of states 
    # control_inputs is a vector of control inputs 
    # workspace_info is a struct with the workspace setup
    # return a vector of costs 

    # A. compute cost of collision with obstacles 
    v1 = compute_obstacle_cost(trajectory, workspace_info; sum_cost = true )
    v1_max = compute_obstacle_cost(trajectory, workspace_info; sum_cost = false)
    # cost of deviating from the waypoints 
    v2 = compute_waypoint_deviation_cost(trajectory, workspace_info; sum_cost = true)
    v2_max = compute_waypoint_deviation_cost(trajectory, workspace_info; sum_cost = false)  
    # cost of deviating from the target velocity 
    v3 = compute_target_velocity_deviation_cost(trajectory, workspace_info)
    # control effort
    v4 = compute_control_effort_cost(control_inputs)
    # TODO: Compute cost of last state from last waypoint 
    v5 = 10.0* norm(trajectory[end, 1:2] - workspace_info.waypoints[end])
    # cost of monotonicity of heading 
    v6 = compute_monotonicity_of_heading(trajectory)
    # return the total cost 
    return [v1, v1_max, v2, v2_max, v3, v4, v5, v6]
end

# function evaluate trajectory violation 
function evaluate_trajectory_violation(trajectory::Matrix{Float64}, workspace_info::Workspace)::Vector{Bool}
    # trajectory is a vector of states 
    # workspace_info is a struct with the workspace setup
    # return a vector of violations
    # violation is a struct with the following fields:
    # - collision_with_obstacles: Bool
    # - waypoint_deviation: Bool
    
    # First, check if the trajectory is in collision with any obstacles
    distance_from_obstacles = [minimum(norm(trajectory[i, :] - obs) for obs in workspace_info.obstacle_locations) for i in 1:size(trajectory, 1)]
    collision_with_obstacles = any(d < workspace_info.obstacle_distance_threshold for d in distance_from_obstacles)
    # Second, check if the trajectory is in violation of the waypoint tracking algorithm
    waypoint_deviation = check_waypoint_deviation_threshold(trajectory, workspace_info)

    return [collision_with_obstacles, waypoint_deviation]
end
