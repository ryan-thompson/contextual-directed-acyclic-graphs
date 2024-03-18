# =================================================================================================#
# Description: Produces the experimental results for the drug data
# Author: Ryan Thompson
# =================================================================================================#

include("Estimators/contextual_dag.jl")

import Cairo, ColorSchemes, CSV, CUDA, DataFrames, Fontconfig, Gadfly, Graphs, Random, Statistics, 
    StatsBase, TikzGraphs

rng = Random.MersenneTwister(2023); Random.default_rng() = rng

# Load data
data = CSV.read("Data/drug_consumption.csv", DataFrames.DataFrame)
data[data[:, 31] .== "CL0", :]

# Save data dimension
n = size(data, 1)

# Extract contextual features
z = data[:, [7, 13]]
z = Matrix(z)

# Extract graphical features
x = data[:, [14:30..., 32]]
x = DataFrames.transform!(x, names(x) .=> (col -> parse.(Float64, replace.(col, r"[^\d.]" => ""))) 
    .=> names(x))
x = Matrix(x)

# Generate indices of training, validation, and testing sets
id = 1:n
train_id = StatsBase.sample(id, round(Int64, n * 0.8), replace = false)
id = setdiff(id, train_id)
valid_id = setdiff(id, train_id)

# Generate training, validation, and testing sets
x_train = Matrix(x[train_id, :])
z_train = Matrix(z[train_id, :])
x_valid = Matrix(x[valid_id, :])
z_valid = Matrix(z[valid_id, :])

# Remove intercepts
x_mean = mapslices(Statistics.mean, x_train, dims = 1)
x_train .-= x_mean
x_valid .-= x_mean

# Save graphical feature labels
labels = ["Alcohol", "Amphet", "Amyl", "Benzos", "Caffeine", "Cannabis", "Choc", "Coke", "Crack", 
          "Ecstasy", "Heroin", "Ketamine", "Legal highs", "LSD", "Meth", "Mushrooms", "Nicotine", 
          "VSA"]

# Fit contextual DAG
fit = ContextualDAG.cdag(x_train, z_train, x_valid, z_valid, lambda_n = 100, 
    hidden_layers = [32, 32])

# Plot graphs for low- and high-scoring individuals
z_new = [Statistics.quantile(data[:, 7], 0.1) Statistics.quantile(data[:, 13], 0.1); 
         Statistics.quantile(data[:, 7], 0.9) Statistics.quantile(data[:, 13], 0.9)]
ŵ = ContextualDAG.coef(fit, z_new, lambda = fit.lambda[argmin(abs.(fit.val_nonzero .- 5))])
g1 = TikzGraphs.plot(Graphs.SimpleDiGraph(ŵ[:, :, 1]), TikzGraphs.SimpleNecklace(), labels = labels)
g2 = TikzGraphs.plot(Graphs.SimpleDiGraph(ŵ[:, :, 2]), TikzGraphs.SimpleNecklace(), labels = labels) 
TikzGraphs.save(TikzGraphs.TEX("Figures/g_drug_1", limit_to = :picture), g1)
TikzGraphs.save(TikzGraphs.TEX("Figures/g_drug_2", limit_to = :picture), g2)

# Plot sparsity as a function of scores
z1_min, z1_max = minimum(z_train[:, 1]), maximum(z_train[:, 1])
z2_min, z2_max = minimum(z_train[:, 2]), maximum(z_train[:, 2])
z_new = hcat(repeat(range(z1_min, z1_max, length = 100), 100), 
    repeat(range(z2_min, z2_max, length = 100), inner = [100]))
ŵ = ContextualDAG.coef(fit, z_new, lambda = fit.lambda[argmin(abs.(fit.val_nonzero .- 5))])
sparsity = sum(ŵ .≠ 0, dims = (1, 2))[:]
function java_colormap(x)
    index = clamp(ceil(Int, x * 100), 1, 100)
    return [ColorSchemes.Java[1 - i] for i in range(0, stop = 1, length = 100)][index]
end
Gadfly.plot(
    x = z_new[:, 1], 
    y = z_new[:, 2], 
    color = sparsity, 
    Gadfly.Geom.rectbin,
    Gadfly.Coord.cartesian(xmin = z1_min, xmax = z1_max, ymin = z2_min, ymax = z2_max),
    Gadfly.Guide.xlabel("Neuroticism"), 
    Gadfly.Guide.ylabel("Sensation seeking"),
    Gadfly.Scale.color_continuous(colormap = java_colormap),
    Gadfly.Guide.colorkey(title = ""),
    Gadfly.Theme(plot_padding = [0Gadfly.mm])
    ) |> 
Gadfly.PDF("Figures/drug_sparsity.pdf", 3.7Gadfly.inch, 2.5Gadfly.inch)