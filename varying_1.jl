# =================================================================================================#
# Description: Produces the experimental results for the varying graphs
# Author: Ryan Thompson
# =================================================================================================#

using Distributed

Distributed.addprocs(2)

Distributed.@sync Distributed.@everywhere begin

include("Estimators/contextual_dag.jl")
include("Estimators/sorted_contextual_dag.jl")
include("Estimators/fixed_dag.jl")

import CausalInference, CSV, CUDA, DataFrames, Distributions, Graphs, LinearAlgebra, ProgressMeter,
    Random, Statistics, Clustering

#==================================================================================================#
# Functions for metrics
#==================================================================================================#

# Create structural Hamming distance function
function struct_hamming_dist(w, ŵ)
    w_active = w .!= 0
    ŵ_active = ŵ .!= 0
    diff = abs.(w_active - ŵ_active)
    diff = diff + transpose(diff)
    diff[diff .> 1] .= 1
    sum(diff) / 2
end

# Create completed partially structural Hamming distance function
function cp_struct_hamming_dist(w, ŵ)
    w_cp = Graphs.adjacency_matrix(CausalInference.cpdag(Graphs.SimpleDiGraph(w)))
    ŵ_cp = Graphs.adjacency_matrix(CausalInference.cpdag(Graphs.SimpleDiGraph(ŵ)))
    struct_hamming_dist(w_cp, ŵ_cp)
end

# Create F1 score function
function f1_score(w, ŵ)
    tp = sum((w .≠ 0) .& (ŵ .≠ 0))
    fp = sum((w .== 0) .& (ŵ .≠ 0))
    fn = sum((w .≠ 0) .& (ŵ .== 0))
    if tp == 0 && fp == 0 && fn == 0
        1.0
    else
        2 * tp / (2 * tp + fp + fn)
    end
end

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
    x_train, z_train, w_train, x_valid, z_valid, w_valid, z_test, w_test

end

#==================================================================================================#
# Function to evaluate a model
#==================================================================================================#

function evaluate!(result, estimator, ŵ, w, par)

    # Save scenario parameters
    n, p, m, ne, s, graph_type, id = par

    # Compute estimation error
    est_error = sum((w - ŵ) .^ 2) / sum(w .^ 2)

    # Compute structural Hamming distance
    shd = sum(map(i -> struct_hamming_dist(w[:, :, i], ŵ[:, :, i]), 1:n)) / n

    # Compute completed partially structural Hamming distance
    cpshd = sum(map(i -> cp_struct_hamming_dist(w[:, :, i], ŵ[:, :, i]), 1:n)) / n

    # Compute completed partially structural Hamming distance
    f1score = sum(map(i -> f1_score(w[:, :, i], ŵ[:, :, i]), 1:n)) / n

    # Compute sparsity levels
    sparsity = sum(ŵ .!= 0) / n

    # Check if all graphs are DAGs
    dag_rate = sum(map(w -> !Graphs.is_cyclic(Graphs.SimpleDiGraph(w)), eachslice(ŵ, dims = 3))) / n

    # Update results
    push!(result, [estimator, est_error, shd, cpshd, f1score, sparsity, dag_rate, n, p, m, ne, s, 
        graph_type, id])

end

#==================================================================================================#
# Function to run a given simulation design
#==================================================================================================#

function runsim(par)

    CUDA.device!((Distributed.myid() - 1) % 2)

    # Set aside space for results
    result = DataFrames.DataFrame(
        estimator = [], est_error = [], shd = [], cpshd = [], f1score = [], sparsity = [], 
        dag_rate = [], n = [], p = [], m = [], ne = [], s = [], graph_type = [], id = []
    )

    Random.seed!(hash(par))

    # Generate data
    x_train, z_train, w_train, x_valid, z_valid, w_valid, z_test, w_test = gendata(par)

    # Evaluate contextual DAG
    fit = ContextualDAG.cdag(x_train, z_train, x_valid, z_valid, verbose = false)
    lambda = fit.lambda[argmin(abs.(fit.val_nonzero .- par.ne * par.s))]
    ŵ_test = ContextualDAG.coef(fit, z_test, lambda = lambda)
    evaluate!(result, "Contextual DAG", ŵ_test, w_test, par)
    fit = nothing

    # Evaluate fixed DAG
    fit = FixedDAG.fixed_dag(x_train)
    ŵ_test = fit[1][argmin(abs.(fit[2] .- par.ne * par.s))]
    order = Graphs.topological_sort(Graphs.SimpleDiGraph(ŵ_test))
    ŵ_test = repeat(ŵ_test, outer = (1, 1, par.n))
    evaluate!(result, "Fixed DAG", ŵ_test, w_test, par)
    fit = nothing

    # Evaluate sorted DAG (fixed)
    fit = ContextualDAG.cdag(x_train, z_train, x_valid, z_valid, order = order, verbose = false)
    lambda = fit.lambda[argmin(abs.(fit.val_nonzero .- par.ne * par.s))]
    ŵ_test = ContextualDAG.coef(fit, z_test, lambda = lambda)
    evaluate!(result, "Sorted DAG (fixed)", ŵ_test, w_test, par)
    fit = nothing

    # Evaluate sorted DAG (truth)
    order_train = dropdims(mapslices(w -> Graphs.topological_sort(Graphs.SimpleDiGraph(w)), 
        w_train, dims = (1, 2)), dims = 2)
    order_valid = dropdims(mapslices(w -> Graphs.topological_sort(Graphs.SimpleDiGraph(w)), 
        w_valid, dims = (1, 2)), dims = 2)
    order_test = dropdims(mapslices(w -> Graphs.topological_sort(Graphs.SimpleDiGraph(w)), 
        w_test, dims = (1, 2)), dims = 2)
    n, p = size(x_train)
    function create_dag_matrix(order::Vector{Int})
        p = length(order)
        w = zeros(p, p)
        for i = 1:p
            for j = i + 1:p
                w[order[i], order[j]] = 1
            end
        end
        w
    end
    mask_train = zeros(p, p, n)
    mask_valid = zeros(p, p, n)
    mask_test = zeros(p, p, n)
    for i in 1:n
        mask_train[:, :, i] = create_dag_matrix(order_train[:, i])
        mask_valid[:, :, i] = create_dag_matrix(order_valid[:, i])
        mask_test[:, :, i] = create_dag_matrix(order_test[:, i])
    end
    fit = SortedContextualDAG.cdag(x_train, z_train, x_valid, z_valid, mask_train = mask_train, 
        mask_valid = mask_valid, verbose = false)
    lambda = fit.lambda[argmin(abs.(fit.val_nonzero .- par.ne * par.s))]
    ŵ_test = SortedContextualDAG.coef(fit, z_test, lambda = lambda, mask = mask_test)
    evaluate!(result, "Sorted DAG (truth)", ŵ_test, w_test, par)
    fit = nothing

    # Evaluate clustered DAG
    k = ceil(Int, par.n / 100)
    cluster = Clustering.kmeans(z_train', k)
    assignment = cluster.assignments
    model = Vector{Tuple}(undef, k)
    for i in 1:k
        indices = findall(x -> x == i, assignment)
        model[i] = FixedDAG.fixed_dag(x_train[indices, :])
    end
    function predict_cluster(new_data, centers)
        distances = [LinearAlgebra.norm(new_point - center) for new_point in eachrow(new_data), 
            center in eachcol(centers)]
        dropdims(mapslices(argmin, distances, dims = 2), dims = 2)
    end
    new_assignment = predict_cluster(z_test, cluster.centers)
    ŵ_test = Array{Float64}(undef, par.p, par.p, par.n)
    for i in 1:par.n
        j = new_assignment[i]
        sparsity = sum(w_train[:, :, findall(x -> x == j, assignment)] .≠ 0) / 
            length(findall(x -> x == j, assignment))
        ŵ_test[:, :, i] = model[j][1][argmin(abs.(model[j][2] .- sparsity))]
    end
    evaluate!(result, "Clustered DAG", ŵ_test, w_test, par)

    CUDA.reclaim()

    result

end

end

#==================================================================================================#
# Run simulations
#==================================================================================================#

# Specify simulation parameters
simulations =
vcat(
    DataFrames.DataFrame(
        (n = n, p = p, m = m, ne = ne, s = s, graph_type = graph_type, id = id) for
        n = round.(Int, exp.(range(log(100), log(10000), 5))), # Number of samples
        p = 20, # Number of graphical features
        m = [2, 5], # Number of contextual features
        ne = 10, # Number of edges
        s = 0.5,
        graph_type = ["erdos-renyi", "scale-free"], # Type of graph
        id = 1:10 # Simulation run ID
    ),
    DataFrames.DataFrame(
        (n = n, p = p, m = m, ne = ne, s = s, graph_type = graph_type, id = id) for
        n = 1000, # Number of samples
        p = round.(Int, range(10, 50, 5)), # Number of graphical features
        m = [2, 5], # Number of contextual features
        ne = 10, # Number of edges
        s = 0.5,
        graph_type = ["erdos-renyi", "scale-free"], # Type of graph
        id = 1:10 # Simulation run ID
    )
)

result = ProgressMeter.@showprogress pmap(runsim, eachrow(simulations))
result = reduce(vcat, result)
CSV.write("Results/varying.csv", result)

Distributed.rmprocs(Distributed.workers())