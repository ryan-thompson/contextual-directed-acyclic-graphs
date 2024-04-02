# =================================================================================================#
# Description: Plots the run times results
# Author: Ryan Thompson
# =================================================================================================#

import Cairo, CSV, DataFrames, Fontconfig, Gadfly, Pipe, Statistics

#==================================================================================================#
# Run time as a function of sample size
#==================================================================================================#

# Summarise results for plotting
result = Pipe.@pipe CSV.read("Results/timings.csv", DataFrames.DataFrame)[1:50, :] |>
         DataFrames.stack(_, [:time]) |>
         DataFrames.groupby(_, [:n, :m, :p, :estimator, :variable]) |>
         DataFrames.combine(
        _, 
        :value => Statistics.mean => :mean, 
        :value => (x -> Statistics.mean(x) - Statistics.std(x) / sqrt(size(x, 1))) => :low,
        :value => (x -> Statistics.mean(x) + Statistics.std(x) / sqrt(size(x, 1))) => :high
         )

# Plot timings as a function of n
Gadfly.plot(
    result, 
    x = :n,
    y = :mean,
    ymin = :low,
    ymax = :high,
    Gadfly.Geom.line, 
    Gadfly.Geom.point, 
    Gadfly.Geom.yerrorbar,
    Gadfly.Coord.cartesian(xmin = 1000),
    Gadfly.Guide.xlabel("Sample size"), 
    Gadfly.Guide.ylabel("Run time"),
    Gadfly.Theme(default_color = "black", plot_padding = [0Gadfly.mm], line_width = 1.5Gadfly.pt)
    ) |> 
    Gadfly.PDF("Figures/timings-n.pdf", 5Gadfly.inch, 2.45Gadfly.inch)

#==================================================================================================#
# Run time as a function of number of nodes
#==================================================================================================#

# Summarise results for plotting
result = Pipe.@pipe CSV.read("Results/timings.csv", DataFrames.DataFrame)[51:100, :] |>
        DataFrames.stack(_, [:time]) |>
        DataFrames.groupby(_, [:n, :m, :p, :estimator, :variable]) |>
        DataFrames.combine(
        _, 
        :value => Statistics.mean => :mean, 
        :value => (x -> Statistics.mean(x) - Statistics.std(x) / sqrt(size(x, 1))) => :low,
        :value => (x -> Statistics.mean(x) + Statistics.std(x) / sqrt(size(x, 1))) => :high
        )

# Plot timings as a function of n
Gadfly.plot(
    result,
    x = :p,
    y = :mean,
    ymin = :low,
    ymax = :high,
    Gadfly.Geom.line, 
    Gadfly.Geom.point, 
    Gadfly.Geom.yerrorbar,
    Gadfly.Coord.cartesian(xmin = 10),
    Gadfly.Guide.xlabel("Number of nodes"), 
    Gadfly.Guide.ylabel("Run time"),
    Gadfly.Theme(default_color = "black", plot_padding = [0Gadfly.mm], line_width = 1.5Gadfly.pt)
    ) |> 
    Gadfly.PDF("Figures/timings-p.pdf", 5Gadfly.inch, 2.45Gadfly.inch)