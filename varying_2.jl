# =================================================================================================#
# Description: Plots the experimental results for the varying graphs
# Author: Ryan Thompson
# =================================================================================================#

import Cairo, ColorSchemes, CSV, DataFrames, Fontconfig, Gadfly, Pipe, Statistics

#==================================================================================================#
# Graph recovery as a function of sample size for Erdos-Renyi graphs
#==================================================================================================#

# Summarise results for plotting
result = Pipe.@pipe CSV.read("Results/varying.csv", DataFrames.DataFrame) |>
         DataFrames.filter(:graph_type => x -> x == "erdos-renyi", _)[1:500, :] |>
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
    Gadfly.Theme(key_position = :top, key_label_font_size = 9Gadfly.pt, plot_padding = [0Gadfly.mm], 
        line_width = 1.5Gadfly.pt)
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
    Gadfly.PDF("Figures/varying-n-er.pdf", 10Gadfly.inch, 3Gadfly.inch)

#==================================================================================================#
# Graph recovery as a function of sample size for scale-free graphs
#==================================================================================================#

# Summarise results for plotting
result = Pipe.@pipe CSV.read("Results/varying.csv", DataFrames.DataFrame) |>
         DataFrames.filter(:graph_type => x -> x == "scale-free", _)[1:500, :] |>
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
    Gadfly.Theme(key_position = :top, key_label_font_size = 9Gadfly.pt, plot_padding = [0Gadfly.mm], 
        line_width = 1.5Gadfly.pt)
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
    Gadfly.PDF("Figures/varying-n-sf.pdf", 10Gadfly.inch, 3Gadfly.inch)

#==================================================================================================#
# Graph recovery as a function of number of nodes for Erdos-Renyi graphs
#==================================================================================================#

# Summarise results for plotting
result = Pipe.@pipe CSV.read("Results/varying.csv", DataFrames.DataFrame) |>
         DataFrames.filter(:graph_type => x -> x == "erdos-renyi", _)[501:1000, :] |>
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
    x = :p,
    y = :mean,
    ymin = :low,
    ymax = :high,
    color = :estimator,
    xgroup = :m,
    ygroup = :variable,
    Gadfly.Geom.subplot_grid(Gadfly.Geom.point, Gadfly.Geom.line, Gadfly.Geom.yerrorbar,
        Gadfly.Coord.cartesian(xmin = 10)),
    Gadfly.Guide.xlabel(""), 
    Gadfly.Guide.ylabel(""),
    Gadfly.Guide.colorkey(title = ""),
    Gadfly.Scale.DiscreteColorScale(p -> ColorSchemes.Java[1:p]),
    Gadfly.Theme(key_position = :top, key_label_font_size = 9Gadfly.pt, plot_padding = [0Gadfly.mm], 
        line_width = 1.5Gadfly.pt)
    )

# Plot F1-score
p2 = Gadfly.plot(
    DataFrames.filter(:variable => x -> x == "F1-score", result),
    x = :p,
    y = :mean,
    ymin = :low,
    ymax = :high,
    color = :estimator,
    xgroup = :m,
    ygroup = :variable,
    Gadfly.Geom.subplot_grid(Gadfly.Geom.point, Gadfly.Geom.line, Gadfly.Geom.yerrorbar,
        Gadfly.Coord.cartesian(xmin = 10)),
    Gadfly.Guide.xlabel(""), 
    Gadfly.Guide.ylabel(""),
    Gadfly.Guide.colorkey(title = ""),
    Gadfly.Scale.DiscreteColorScale(p -> ColorSchemes.Java[1:p]),
    Gadfly.Theme(key_position = :none, plot_padding = [0Gadfly.mm, 0Gadfly.mm, 11.5Gadfly.mm, 
        0Gadfly.mm], line_width = 1.5Gadfly.pt)
    )

Gadfly.hstack(p1, p2) |> 
    Gadfly.PDF("Figures/varying-p-er.pdf", 10Gadfly.inch, 3Gadfly.inch)

#==================================================================================================#
# Graph recovery as a function of number of nodes for scale-free graphs
#==================================================================================================#

# Summarise results for plotting
result = Pipe.@pipe CSV.read("Results/varying.csv", DataFrames.DataFrame) |>
         DataFrames.filter(:graph_type => x -> x == "scale-free", _)[501:1000, :] |>
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

# Plot F1-score
p1 = Gadfly.plot(
    DataFrames.filter(:variable => x -> x == "Struct. Hamming dist.", result),
    x = :p,
    y = :mean,
    ymin = :low,
    ymax = :high,
    color = :estimator,
    xgroup = :m,
    ygroup = :variable,
    Gadfly.Geom.subplot_grid(Gadfly.Geom.point, Gadfly.Geom.line, Gadfly.Geom.yerrorbar,
        Gadfly.Coord.cartesian(xmin = 10)),
    Gadfly.Guide.xlabel(""), 
    Gadfly.Guide.ylabel(""),
    Gadfly.Guide.colorkey(title = ""),
    Gadfly.Scale.DiscreteColorScale(p -> ColorSchemes.Java[1:p]),
    Gadfly.Theme(key_position = :top, key_label_font_size = 9Gadfly.pt, plot_padding = [0Gadfly.mm], 
        line_width = 1.5Gadfly.pt)
    )

# Plot regression results
p2 = Gadfly.plot(
    DataFrames.filter(:variable => x -> x == "F1-score", result),
    x = :p,
    y = :mean,
    ymin = :low,
    ymax = :high,
    color = :estimator,
    xgroup = :m,
    ygroup = :variable,
    Gadfly.Geom.subplot_grid(Gadfly.Geom.point, Gadfly.Geom.line, Gadfly.Geom.yerrorbar,
        Gadfly.Coord.cartesian(xmin = 10)),
    Gadfly.Guide.xlabel(""), 
    Gadfly.Guide.ylabel(""),
    Gadfly.Guide.colorkey(title = ""),
    Gadfly.Scale.DiscreteColorScale(p -> ColorSchemes.Java[1:p]),
    Gadfly.Theme(key_position = :none, plot_padding = [0Gadfly.mm, 0Gadfly.mm, 11.5Gadfly.mm, 
        0Gadfly.mm], line_width = 1.5Gadfly.pt)
    )

Gadfly.hstack(p1, p2) |> 
    Gadfly.PDF("Figures/varying-p-sf.pdf", 10Gadfly.inch, 3Gadfly.inch)