# =================================================================================================#
# Description: Produces the run times results
# Author: Ryan Thompson
# =================================================================================================#

include("Estimators/contextual_dag.jl")

import BenchmarkTools, CausalInference, CSV, CUDA, DataFrames, Distributions, Graphs, 
    LinearAlgebra, ProgressMeter, Random, Statistics

#==================================================================================================#
# Function to generate data
#==================================================================================================#

function gendata(par)

    # Save scenario parameters
    n, p, m, ne, s, graph_type, id = par

    # Generate stochastic disturbances
    ε_train = rand(Distributions.Normal(0, 1), n, p)
    ε_valid = rand(Distributions.Normal(0, 1), n, p)

    # Generate contextual features
    z_train = rand(Distributions.Uniform(- 1, 1), n, m)
    z_valid = rand(Distributions.Uniform(- 1, 1), n, m)
    z_test = rand(Distributions.Uniform(- 1, 1), n, m)

    # Generate graph
    if graph_type == "erdos-renyi"
        g = Graphs.erdos_renyi(p, ne)
    elseif graph_type == "scale-free"
        g = Graphs.static_scale_free(p, ne, 2)
    end

    # Generate a center for each graphical feature
    c = rand(Distributions.Uniform(- 1, 1), p, m)

    # Estimate thresholds for varying sparsity
    all_dists = []
    for i in 1:n
        dist = [LinearAlgebra.norm(z_train[i, :] - c[j, :], 2) for j in 1:p]
        for j in Graphs.vertices(g)
            for k in Graphs.outneighbors(g, j)
                push!(all_dists, abs(dist[k] - dist[j]))
            end
        end
    end
    threshold = Statistics.quantile(all_dists, 1 - s)

    # Create weighted adjacency matrix function
    function w(z)
        g_z = Graphs.SimpleDiGraph(p)
        dist = [LinearAlgebra.norm(z - c[j, :], 2) for j in 1:p]
        W = zeros(p, p)
        @inbounds for j in Graphs.vertices(g)
            for k in Graphs.outneighbors(g, j)
                if  dist[j] - dist[k] >  threshold
                    W[j, k] = dist[j] - dist[k]
                end
            end
        end
        W
    end

    # Generate weighted adjacency matrices
    w_train = stack([w(z_train[i, :]) for i in 1:n])
    w_valid = stack([w(z_valid[i, :]) for i in 1:n])
    w_test = stack([w(z_test[i, :]) for i in 1:n])

    # Generate graphical features
    x_train = vcat([ε_train[i, :]' * inv(LinearAlgebra.I(p) - w_train[:, :, i]) for i in 1:n]...)
    x_valid = vcat([ε_valid[i, :]' * inv(LinearAlgebra.I(p) - w_valid[:, :, i]) for i in 1:n]...)

    # Return generated data
    x_train, z_train, x_valid, z_valid, z_test, w_test

end

#==================================================================================================#
# Function to evaluate a model
#==================================================================================================#

function evaluate!(result, estimator, time, par)

    # Save scenario parameters
    n, p, m, ne, s, graph_type, id = par

    # Update results
    push!(result, [estimator, time, n, p, m, ne, s, graph_type, id])

end

#==================================================================================================#
# Function to run a given simulation design
#==================================================================================================#

function runsim(par)
    
    # Set aside space for results
    result = DataFrames.DataFrame(
        estimator = [], time = [], n = [], p = [], m = [], ne = [], s = [], graph_type = [], 
        id = []
    )
    
    # Generate data
    x_train, z_train, x_valid, z_valid, z_test, w_test = gendata(par)

    # Evaluate contextual DAG
    time = BenchmarkTools.@belapsed ContextualDAG.cdag($x_train, $z_train, $x_valid, $z_valid, 
        lambda = Inf, epoch_max = 10, verbose = false)
    evaluate!(result, "Contextual DAG", time, par)
        
    result

end

#==================================================================================================#
# Run simulations
#==================================================================================================#

# Specify simulation parameters
simulations =
vcat(
    DataFrames.DataFrame(
        (n = n, p = p, m = m, ne = ne, s = s, graph_type = graph_type, id = id) for
        n = [1000, 3250, 5500, 7750, 10000], # Number of samples
        p = 20, # Number of graphical features
        m = 2, # Number of contextual features
        ne = 10, # Number of edges
        s = 0.5,
        graph_type = ["erdos-renyi"], # Type of graph
        id = 1:10 # Simulation run ID
    ),
    DataFrames.DataFrame(
        (n = n, p = p, m = m, ne = ne, s = s, graph_type = graph_type, id = id) for
        n = 1000, # Number of samples
        p = [10, 20, 30, 40, 50], # Number of graphical features
        m = 2, # Number of contextual features
        ne = 10, # Number of edges
        s = 0.5,
        graph_type = ["erdos-renyi"], # Type of graph
        id = 1:10 # Simulation run ID
    )
)

rng = Random.MersenneTwister(2023); Random.default_rng() = rng
result = ProgressMeter.@showprogress map(runsim, eachrow(simulations))
result = reduce(vcat, result)
CSV.write("Results/timings.csv", result)