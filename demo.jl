# =================================================================================================#
# Description: Produces the illustrative figures in the introduction
# Author: Ryan Thompson
# =================================================================================================#

import Distributions, Graphs, LaTeXStrings, LinearAlgebra, Random, Statistics, TikzGraphs

include("Estimators/contextual_dag.jl")
include("Estimators/fixed_dag.jl")

rng = Random.MersenneTwister(80); Random.default_rng() = rng

# Set parameters
n = 10000
p = 5
m = 2
ne = 6
s = 0.5

# Generate stochastic disturbances
ε_train = rand(Distributions.Normal(0, 1), n, p)
ε_valid = rand(Distributions.Normal(0, 1), n, p)

# Generate contextual features
z_train = rand(Distributions.Uniform(- 1, 1), n, m)
z_valid = rand(Distributions.Uniform(- 1, 1), n, m)
z_test = rand(Distributions.Uniform(- 1, 1), n, m)

# Generate graph
g = Graphs.erdos_renyi(p, ne)

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

# Graph labels
labels = [LaTeXStrings.L"$\mathsf{x}_1$", LaTeXStrings.L"$\mathsf{x}_2$", 
    LaTeXStrings.L"$\mathsf{x}_3$", LaTeXStrings.L"$\mathsf{x}_4$", LaTeXStrings.L"$\mathsf{x}_5$"]

# Plot true DAG
g_true_1 = TikzGraphs.plot(Graphs.SimpleDiGraph(w_test[:, :, 1]), labels = labels, 
    node_style = "circle, draw, fill = green!50")
g_true_2 = TikzGraphs.plot(Graphs.SimpleDiGraph(w_test[:, :, 2]), labels = labels, 
    node_style = "circle, draw, fill = yellow!50")

# Plot fitted contextual DAG
fit = ContextualDAG.cdag(x_train, z_train, x_valid, z_valid)
lambda = fit.lambda[argmin(abs.(fit.val_nonzero .- ne * s))]
ŵ_test = ContextualDAG.coef(fit, z_test, lambda = lambda)
g_contextual_1 = TikzGraphs.plot(Graphs.SimpleDiGraph(ŵ_test[:, :, 1]), labels = labels, 
    node_style = "circle, draw, fill = green!50")
g_contextual_2 = TikzGraphs.plot(Graphs.SimpleDiGraph(ŵ_test[:, :, 2]), labels = labels, 
    node_style = "circle, draw, fill = yellow!50")

# Plot fitted fixed DAG
fit = FixedDAG.fixed_dag(x_train)
ŵ_test = fit[1][argmin(abs.(fit[2] .- ne * s))]
ŵ_test = repeat(ŵ_test, outer = (1, 1, n))
g_fixed_1 = TikzGraphs.plot(Graphs.SimpleDiGraph(ŵ_test[:, :, 1]), labels = labels, 
    node_style = "circle, draw, fill = green!50")
g_fixed_2 = TikzGraphs.plot(Graphs.SimpleDiGraph(ŵ_test[:, :, 2]), labels = labels, 
    node_style = "circle, draw, fill = yellow!50")

# Export plots
TikzGraphs.save(TikzGraphs.TEX("Figures/g_true_1", limit_to = :picture), g_true_1)
TikzGraphs.save(TikzGraphs.TEX("Figures/g_true_2", limit_to = :picture), g_true_2)
TikzGraphs.save(TikzGraphs.TEX("Figures/g_contextual_1", limit_to = :picture), g_contextual_1)
TikzGraphs.save(TikzGraphs.TEX("Figures/g_contextual_2", limit_to = :picture), g_contextual_2)
TikzGraphs.save(TikzGraphs.TEX("Figures/g_fixed_1", limit_to = :picture), g_fixed_1)
TikzGraphs.save(TikzGraphs.TEX("Figures/g_fixed_2", limit_to = :picture), g_fixed_2)