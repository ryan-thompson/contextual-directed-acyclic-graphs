# =================================================================================================#
# Description: Plots the experimental results for the fixed graphs
# Author: Ryan Thompson
# =================================================================================================#

import Cairo, ColorSchemes, CSV, DataFrames, Fontconfig, Gadfly, Pipe, Statistics

#==================================================================================================#
# Graph recovery as a function of sample size for Erdos-Renyi graphs
#==================================================================================================#

# Summarise results for plotting
result = Pipe.@pipe CSV.read("Results/fixed.csv", DataFrames.DataFrame) |>
         DataFrames.filter(:graph_type => x -> x == "erdos-renyi", _) |>
         DataFrames.stack(_, [:shd, :f1score]) |>
         DataFrames.groupby(_, [:n, :m, :p, :graph_type, :estimator, :variable]) |>
         DataFrames.combine(
        _, 
        :value => Statistics.mean => :mean, 
        :value => (x -> Statistics.mean(x) - Statistics.std(x) / sqrt(size(x, 1))) => :low,
        :value => (x -> Statistics.mean(x) + Statistics.std(x) / sqrt(size(x, 1))) => :high
         ) |>
         DataFrames.transform(_, [:m] => DataFrames.ByRow(m -> "m = $m") => :m) |>
         DataFrames.transform(_, :variable => 
            DataFrames.ByRow(x -> x == "shd" ? "Struct. Hamming dist." : x) => :variable) |>
         DataFrames.transform(_, :variable => 
            DataFrames.ByRow(x -> x == "f1score" ? "F1-score" : x) => :variable)

# Plot structural Hamming distance
p1 = Gadfly.plot(
    DataFrames.filter(:variable => x -> x == "Struct. Hamming dist.", result),
    x = :n,
    y = :mean,
    ymin = :low,
    ymax = :high,
    color = :estimator,
    xgroup = :m,
    ygroup = :variable,
    Gadfly.Geom.subplot_grid(Gadfly.Geom.point, Gadfly.Geom.line, Gadfly.Geom.yerrorbar),
    Gadfly.Guide.xlabel(""), 
    Gadfly.Guide.ylabel(""),
    Gadfly.Guide.colorkey(title = ""),
    Gadfly.Scale.x_log10,
    Gadfly.Scale.DiscreteColorScale(p -> ColorSchemes.Java[1:p]),
    Gadfly.Theme(key_position = :top, plot_padding = [0Gadfly.mm], line_width = 1.5Gadfly.pt)
    )

# Plot F1-score
p2 = Gadfly.plot(
    DataFrames.filter(:variable => x -> x == "F1-score", result),
    x = :n,
    y = :mean,
    ymin = :low,
    ymax = :high,
    color = :estimator,
    xgroup = :m,
    ygroup = :variable,
    Gadfly.Geom.subplot_grid(Gadfly.Geom.point, Gadfly.Geom.line, Gadfly.Geom.yerrorbar),
    Gadfly.Guide.xlabel(""), 
    Gadfly.Guide.ylabel(""),
    Gadfly.Guide.colorkey(title = ""),
    Gadfly.Scale.x_log10,
    Gadfly.Scale.DiscreteColorScale(p -> ColorSchemes.Java[1:p]),
    Gadfly.Theme(key_position = :none, plot_padding = [0Gadfly.mm, 0Gadfly.mm, 11.5Gadfly.mm, 
        0Gadfly.mm], line_width = 1.5Gadfly.pt)
    )

Gadfly.hstack(p1, p2) |> 
    Gadfly.PDF("Figures/fixed-n-er.pdf", 10Gadfly.inch, 3Gadfly.inch)