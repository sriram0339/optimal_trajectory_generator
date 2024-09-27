# Generate a trajectory for given initial conditions and control inputs 
using DifferentialEquations

# Create a struct for the dynamical system
struct DynamicalSystem
    f::Function
    params::Vector{Float64}
    x0::Vector{Float64}
end

# Define the function for the dynamical system
function (sys::DynamicalSystem)(dx, x, u, t)
    sys.f(dx, x, u, t, sys.params)
end

# Define the function to generate the trajectory
function generate_trajectory(sys::DynamicalSystem,  u, ctrl_dt::Float64 = 0.1; dt::Float64 =ctrl_dt/10.0)::Tuple{Matrix{Float64}, Vector{Float64}}

    # x0: initial condition
    # u: control inputs
    # ctrldt: control input frequency time step

    # find the dimensions of the control input matrix
    n_inputs = size(u, 2)
    n_steps = size(u, 1)
    n = size(sys.x0, 1)

    # initialize the state vector
    x = sys.x0

    # initialize the time vector
    t = 0.0

    # initialize the trajectory matrix
    X::Matrix{Float64} = sys.x0'
    T::Vector{Float64} = [0.0]

    # loop over the control inputs
    for i in 1:n_steps
        u_i = u[i, :]
        # update the state vector
        # create an ODE problem with current state as initial state and current control input 
        prob = ODEProblem(sys, x, (t, t + ctrl_dt), u_i)
        # solve the ODE problem
        sol = solve(prob, Tsit5(), dt=dt, reltol=1e-8, abstol=1e-8)
        # update the state vector
        x = sol[end]
        for τ in t+dt:dt:t+ctrl_dt
            x::Vector{Float64} = sol(τ)
            X = vcat(X, x')
            T = [T;τ]
        end
        # update the time vector
        t = t + ctrl_dt
    end

    # return the trajectory matrix and the time vector
    return X, T 
end

# This is a generic model of a ground vehicle that we can use to generate trajectories
function vehicle_model(dx, x, u, t, params)
    # x: state vector [x, y, theta, v]
    # u: control input vector [a, omega]
    # t: time
    # params: parameters [L]
    # L: length of the vehicle
    # extract the state variables
    x_pos = x[1]
    y_pos = x[2]
    theta = x[3]
    v = x[4]
    # parameter 
    L = params[1]
    # control onputs 
    accel = u[1]
    omega = u[2]
    # calculate the derivatives of the state variables
    dx[1] = v * cos(theta)
    dx[2] = v * sin(theta)
    dx[3] = v * tan(omega) / L
    dx[4] = accel  
end

