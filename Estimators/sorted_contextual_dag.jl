# =================================================================================================#
# Description: Implementation of the sorted contextual DAG
# Author: Ryan Thompson
# =================================================================================================#

module SortedContextualDAG

import CUDA, Flux, Graphs, LinearAlgebra, Printf, Statistics, Zygote

export cdag, coef

#==================================================================================================#
# Function that reimplements Flux.early_stopping with <= instead of <
#==================================================================================================#

function early_stopping(f, delay; distance = -, init_score = 0, min_dist = 0)
    trigger = let best_score = init_score
      (args...; kwargs...) -> begin
        score = f(args...; kwargs...)
        Δ = distance(best_score, score)
        best_score = Δ < 0 ? best_score : score
        return Δ <= min_dist
      end
    end
    return Flux.patience(trigger, delay)
  end

#==================================================================================================#
# Functions that performs batched matrix computations
#==================================================================================================#

# Adapted from https://github.com/JuliaGPU/CUDA.jl/blob/master/lib/cublas/wrappers.jl
for (fname1, fname2, elty) in
    ((:cublasDgetrfBatched, :cublasDgetriBatched, :Float64),
     (:cublasSgetrfBatched, :cublasSgetriBatched, :Float32),
     (:cublasZgetrfBatched, :cublasZgetriBatched, :ComplexF64),
     (:cublasCgetrfBatched, :cublasCgetriBatched, :ComplexF32))
    @eval begin
        function matinv_batched!(A::Vector{<:CUDA.StridedCuMatrix{$elty}}, 
            C::Vector{<:CUDA.StridedCuMatrix{$elty}})
            batchSize = length(A)
            n = size(A[1], 1)
            lda = max(1, stride(A[1], 2))
            ldc = max(1, stride(C[1], 2))
            Aptrs = CUDA.CUBLAS.unsafe_batch(A)
            Cptrs = CUDA.CUBLAS.unsafe_batch(C)
            info = CUDA.zeros(Cint, batchSize)
            CUDA.CUBLAS.$fname1(CUDA.CUBLAS.handle(), n, Aptrs, lda, CUDA.CU_NULL, info, batchSize)
            CUDA.CUBLAS.$fname2(CUDA.CUBLAS.handle(), n, Aptrs, lda, CUDA.CU_NULL, Cptrs, ldc, info,
                batchSize)
            CUDA.CUBLAS.unsafe_free!(Aptrs)
            CUDA.CUBLAS.unsafe_free!(Cptrs)
        end
    end
end

function xw_mult(w; x)
    p, n = size(x)
    reshape(CUDA.CUBLAS.gemm_strided_batched('T', 'T', 1, w, reshape(x, 1, :, n)), p, n)
end

function Zygote.rrule(::typeof(xw_mult), w; x)
    y = xw_mult(w; x)
    function y_pullback(dy)
        p, n = size(x)
        dw = CUDA.CUBLAS.gemm_strided_batched('N', 'T', 1.0, reshape(dy, p, 1, n), 
            reshape(x, p, 1, n))
        dw = permutedims(dw, (2, 1, 3))
        (Zygote.NoTangent(), dw)
    end
    y, y_pullback
end

#==================================================================================================#
# Function to project weighted adjancency matrix onto ℓ1-ball
#==================================================================================================#

function project_l1(w, lambda)

    # Save input dims
    dims = size(w)
    nlambda = dims[3] * lambda

    # If already in ℓ1 ball or λ=0 then exit early
    if sum(abs.(w)) <= nlambda
        return w, 0
    elseif isapprox(lambda, zero(lambda), atol = sqrt(eps(lambda)))
        return zero(w), Inf
    end

    # Flatten to vector
    w = vec(w)

    # Remove signs
    w_abs = abs.(w)

    # Run algorithm for projection onto simplex by Duchi et al. (2008, ICML)
    ind = sortperm(w_abs, rev = true)
    w_sort = w_abs[ind]
    csum = cumsum(w_sort)
    indices = Flux.gpu(collect(1:prod(dims)))
    max_j = maximum((w_sort .* indices .> csum .- nlambda) .* indices)
    theta = (sum(w_sort[1:max_j]) - nlambda) / max_j

    # Threshold w
    w_abs = max.(w_abs .- theta, 0)
    w = w_abs .* sign.(w)

    # Return to original shape
    reshape(w, dims), theta

end

#==================================================================================================#
# Function to project weighted adjancency matrix onto set of DAGs
#==================================================================================================#

function project_dag(w̃, params)

    # Save params
    s, mu, alpha, tol, max_step, max_iter, threshold = params

    # Save dims
    p = size(w̃, 1)

    # Compute scaling constant
    max_ = maximum(abs.(w̃), dims = (1, 2))
    max_ = ifelse.(max_ .> 0.0f0, max_, 1.0f0)

    # Set learning rate
    lr = 1 / p

    # Initialise variables
    w = zero(w̃)
    I = zero(w̃) .+ LinearAlgebra.Diagonal(one(w̃[:, :, 1]))

    # matinv_batched! does in-place inverse
    sIw2 = similar(w̃)
    sIw2_inv = similar(w̃)

    # matinv_batched! expects a vector of matrices rather than a 3d array
    sIw2_batch = collect(eachslice(sIw2, dims = 3))
    sIw2_inv_batch = collect(eachslice(sIw2_inv, dims = 3))

    # Perform DAG projection
    for step in 1:max_step

        # Run gradient descent
        for iter in 1:max_iter

            # Compute gradients
            sIw2 .= permutedims(s * I - w .^ 2, (2, 1, 3))
            matinv_batched!(sIw2_batch, sIw2_inv_batch)
            grad = 2 * sIw2_inv .* w + mu * (w - w̃ ./ max_)

            # Take gradient descent step
            w .-= lr .* grad

            # Check for convergence
            if maximum(abs.(grad)) <= tol
                break
            end

        end

        # Update mu
        mu *= alpha

    end

    # Rescale weights
    w .*= max_

    # Threshold small weights
    w[abs.(w) .<= threshold] .= 0
    w[w .≠ 0] = w̃[w .≠ 0]

    w

end

#==================================================================================================#
# Function to project weighted adjancency matrix onto intersection of sets
#==================================================================================================#

function project(w̃; par, params, inference)

    # Project onto DAG set
    ŵ = project_dag(w̃, params)

    # Project onto ℓ1-ball
    if inference
        ŵ, theta = Flux.softshrink(ŵ, par), par
    else
        ŵ, theta = project_l1(ŵ, par)
    end

    ŵ, theta

end

function Zygote.rrule(::typeof(project), w̃; par, params, inference)

    # Compute optimal solution
    ŵ, theta = project(w̃, par = par, params = params, inference = inference)

    # Configure vector-Jacobian product (vJp)
    function ŵ_pullback(dŵ_tuple)
        dŵ = dŵ_tuple[1]
        dims = size(dŵ)
        dŵ = vec(dŵ)
        ŵ = vec(ŵ)
        if iszero(theta)
            dw̃ = (ŵ .≠ 0) .* dŵ
        elseif isinf(theta)
            dw̃ = zero(dŵ)
        else
            A = ŵ .≠ 0
            alpha =  1 / sum(A) * sum(sign.(ŵ) .* dŵ)
            dw̃ =  A .* dŵ .- alpha .* sign.(ŵ)
        end
        dw̃ = reshape(dw̃, dims)
        (Zygote.NoTangent(), dw̃, Zygote.NoTangent())
    end

    (ŵ, theta), ŵ_pullback

end

#==================================================================================================#
# Function to generate neural network architecture
#==================================================================================================#

function gennet(hidden_layers, p, m, activation_fun)

    # Determine number of layers
    nlayer = length(hidden_layers) + 1
    layer = Vector{Flux.Dense}(undef, nlayer)

    # Build layers
    for i in 1:nlayer
        if i == 1 == nlayer
            layer[i] = Flux.Dense(m, p ^ 2)
        elseif i == 1
            layer[i] = Flux.Dense(m, hidden_layers[i], activation_fun)
        elseif i == nlayer
            layer[i] = Flux.Dense(hidden_layers[i - 1], p ^ 2)
        else
            layer[i] = Flux.Dense(hidden_layers[i - 1], hidden_layers[i], activation_fun)
        end
    end
    
    # Unroll layers into chain
    Flux.Chain(layer..., x -> reshape(x, p, p, :))

end

#==================================================================================================#
# Function to train model
#==================================================================================================#

function train(model, lambda, params, x, z, x_val, z_val, optimiser, epoch_max, early_stop, patience, 
    verbose, verbose_freq, zero_inds, mask_train, mask_valid)

    # Save data dimensions
    p, n_val = size(x_val)

    # Instantiate optimiser
    optim = optimiser()
    optim_state = Flux.setup(optim, model)

    theta = Ref(0.0)
    
    # Create objective function
    function objective(model; x = x, z = z, par = lambda, params = params, inference = false, 
        mask = mask_train)
        p, n = size(x)
        ŵ, theta[] = project(model(z) .* mask, par = par, params = params, inference = inference)
        x̂ = xw_mult(ŵ; x = x)
        sum((x - x̂) .^ 2) / (n * p)
    end

    # Initialise variables
    train_loss = objective(model)
    val_loss = objective(model, x = x_val, z = z_val, par = theta[], inference = true, 
        mask = mask_valid)
    epochs = 0

    # Set convergence criterion
    if early_stop
        model_best = deepcopy(model)
        theta_best = theta[]
        train_loss_best = train_loss
        val_loss_best = val_loss
        epochs_best = 0
        converge = early_stopping(x -> x, patience, init_score = val_loss_best)
    else
        converge = early_stopping(x -> x, patience, init_score = Inf)
    end

    # Run optimisation
    for epoch in 1:epoch_max

        # Record training loss and gradients
        train_loss, grad = Flux.withgradient(objective, model)

        # Set gradients to zero for diagonal elements
        grad[1].layers[end - 1].weight[zero_inds, :] .= 0
        grad[1].layers[end - 1].bias[zero_inds] .= 0

        # Record validation loss
        val_loss = objective(model, x = x_val, z = z_val, par = theta[], inference = true, 
            mask = mask_valid)

        # Print staus update
        if verbose && epoch % verbose_freq == 0
            Printf.@printf("\33[2K\rEpoch: %i, Train loss: %.4f, Valid loss: %.4f", epoch, 
                train_loss, val_loss)
        end

        # Check for improvement
        if early_stop && val_loss < val_loss_best
            model_best = deepcopy(model)
            theta_best = theta[]
            train_loss_best = train_loss
            val_loss_best = val_loss
            epochs_best = epochs
        end

        # Check for convergence
        if early_stop 
            converge(val_loss) && break
        else
            converge(train_loss) && break
        end

        # Update parameters
        if epoch < epoch_max
            epochs = epoch
            Flux.update!(optim_state, model, grad[1])
        end

    end

    # Update model to best model if using early stopping
    if early_stop
        model = model_best
        theta = theta_best
        train_loss = train_loss_best
        val_loss = val_loss_best
        epochs = epochs_best
    else
        theta = theta[]
    end

    # Compute validation standard errors and sparsity levels for plotting
    val_nonzero = sum(project(model(z_val) .* mask_valid, par = theta, params = params, 
        inference = true)[1] .≠ 0) / n_val

    model, theta, train_loss, val_loss, val_nonzero, epochs

end

#==================================================================================================#
# Function to initialise model at a simple directed graph
#==================================================================================================#

function init(model, x, z, x_val, z_val, optimiser, epoch_max, early_stop, patience, verbose, 
    verbose_freq, zero_inds, mask_train, mask_valid)

    # Save data dimensions
    p, n_val = size(x_val)

    # Instantiate optimiser
    optim = optimiser()
    optim_state = Flux.setup(optim, model)

    # Create objective function
    function objective(model; x = x, z = z, mask = mask_train)
        p, n = size(x)
        ŵ = model(z) .* mask
        x̂ = xw_mult(ŵ; x = x)
        sum((x - x̂) .^ 2) / (n * p)
    end

    # Initialise variables
    val_loss = objective(model, x = x_val, z = z_val, mask = mask_valid)

    # Set convergence criterion
    if early_stop
        model_best = deepcopy(model)
        val_loss_best = val_loss
        converge = early_stopping(x -> x, patience, init_score = val_loss_best)
    else
        converge = early_stopping(x -> x, patience, init_score = Inf)
    end

    # Run optimisation
    for epoch in 1:epoch_max

        # Record training loss and gradients
        train_loss, grad = Flux.withgradient(objective, model)

        # Set gradients to zero for diagonal elements
        grad[1].layers[end - 1].weight[zero_inds, :] .= 0
        grad[1].layers[end - 1].bias[zero_inds] .= 0

        # Record validation loss
        val_loss = objective(model, x = x_val, z = z_val, mask = mask_valid)

        # Print staus update
        if verbose && epoch % verbose_freq == 0
            Printf.@printf("\33[2K\rEpoch: %i, Train loss: %.4f, Valid loss: %.4f", epoch, 
                train_loss, val_loss)
        end

        # Check for improvement
        if early_stop && val_loss < val_loss_best
            model_best = deepcopy(model)
            val_loss_best = val_loss
        end

        # Check for convergence
        if early_stop 
            converge(val_loss) && break
        else
            converge(train_loss) && break
        end

        # Update parameters
        if epoch < epoch_max
            Flux.update!(optim_state, model, grad[1])
        end

    end

    model

end

#==================================================================================================#
# Type for contextual DAG
#==================================================================================================#

struct ContextualDAGFit
    model::Vector{Flux.Chain}
    lambda::Vector{Float32}
    lambda_min::Float32
    theta::Vector{Float32}
    z_mean::Matrix{Float32}
    z_sd::Matrix{Float32}
    train_loss::Vector{Float32}
    val_loss::Vector{Float32}
    val_nonzero::Vector{Float32}
    epochs::Vector{Int}
    params::Tuple
end

#==================================================================================================#
# Function to perform a contextual DAG fit
#==================================================================================================#

"""
cdag(x, z, x_val, z_val; <keyword arguments>)

Performs a contextual DAG fit to variables `x` and contextual features `z`. The training data are \
    `x` and `z`, and the validation data are `x_val` and `z_val`.

# Arguments
- `lambda = nothing`: an optional sequence of regularisation parameters; if empty will be computed \
as an equispaced spaced sequence of length `lambda_n` from `lambda_max` to zero, where \
`lambda_max` is automatically computed from the data to ensure the maximum number of edges.
- `lambda_n = 20`: the number of regularisation parameters to evaluate.
- `optimiser = Flux.Adam`: an optimiser from Flux to use for training.
- `epoch_max = 10000`: the maximum number of training epochs.
- `early_stop = true`: whether to use early stopping; if `true` convergence is monitored on the \
validation set or if `false` convergence is monitored on the training set.
- `patience = 10`: the number of epochs to wait before declaring convergence.
- `hidden_layers = [128, 128]`: the configuration of the feedforward neural network; by default \
produces a network with two hidden dense layers of 128 neurons each.
- `initialise = "warm"`: how to initialise the optimiser; `"warm"` to warm start the optimiser \
using the previous solution along the regularisation path or `"cold"` to cold start the optimiser \
with a random initialisation.
- `verbose = true`: whether to print status updates during training.
- `verbose_freq = 10`: the number of epochs to wait between status updates.
- `standardise_z = true`: whether to standardise the contextual features to have zero mean and \
unit variance; helps during training.
- `activation_fun = Flux.relu`: an activation function to use in the hidden layers.
- `params = (1, 1, 0.5, 1e-2, 10, 10000, 0.1)`: parameters for the acyclicity projection in the \
following order: log det parameter `s`, path coefficient `μ`, decay factor `c`, convergence \
tolerance `tol`, step count `T`, maximum gradient descent iterations `max_iter`, thresholding \
parameter `threshold`
- `order = nothing`: an optional topological ordering of the variables; if `nothing` the \
topological ordering will be learned and allowed to vary with `z`.
``

See also [`coef`](@ref).
"""
function cdag(x::Matrix{<:Real}, z::Matrix{<:Real}, x_val::Matrix{<:Real}, 
    z_val::Matrix{<:Real}; lambda::Union{Real, Vector{<:Real}, Nothing} = nothing, 
    lambda_n::Int = 20, optimiser::DataType = Flux.Adam, epoch_max::Integer = 10000, 
    early_stop::Bool = true, patience::Integer = 10, hidden_layers::Vector{<:Any} = [128, 128], 
    initialise::String = "warm", verbose::Bool = true, verbose_freq::Integer = 1, 
    standardise_z::Bool = true, activation_fun::Function = Flux.relu, 
    params::Tuple = (1, 1, 0.5, 1e-2, 10, 10000, 0.1), 
    order::Union{Vector{<:Int}, Nothing} = nothing, mask_train, mask_valid)

    mask_train = Flux.gpu(mask_train)
    mask_valid = Flux.gpu(mask_valid)

    # Validate arguments
    initialise in ["warm", "cold"] || error("""initialise should be "warm" or "cold".""")

    # Save data dimensions
    n, p = size(x)
    m = size(z, 2)

    # Standardise contextual features
    if standardise_z
        z_mean = mapslices(Statistics.mean, z, dims = 1)
        z_sd = mapslices(z -> Statistics.std(z, corrected = false), z, dims = 1)
    else
        z_mean = zeros(1, m)
        z_sd = ones(1, m)
    end
    if any(z_sd .== 0)
        z_sd[z_sd .== 0] .= 1 # Handles constants
    end
    z = (z .- z_mean) ./ z_sd
    z_val = (z_val .- z_mean) ./ z_sd

    # Transpose because Flux expects features in rows and samples in columns
    x, z = transpose(x), transpose(z)
    x_val, z_val = transpose(x_val), transpose(z_val)

    # Flux defaults to f32 model parameters
    x, z = Flux.f32(x), Flux.f32(z)
    x_val, z_val = Flux.f32(x_val), Flux.f32(z_val)

    # Move data to correct device
    x, z = Flux.gpu(x), Flux.gpu(z)
    x_val, z_val = Flux.gpu(x_val), Flux.gpu(z_val)

    # Compute lambda sequence
    if !isnothing(lambda)
        lambda_n = length(lambda)
        if !isa(lambda, Vector)
            lambda = [lambda]
        end
        lambda = Float32.(lambda)
    end

    # Zero out the indices to impose a provided topological order
    if !isnothing(order)
        p = length(order)
        zero_indices = Int[]
        order_dict = Dict(order[i] => i for i in 1:p)
        for j = 1:p
            for i = 1:p
                if order_dict[i] >= order_dict[j]
                    push!(zero_indices, (j - 1) * p + i)
                end
            end
        end
        zero_inds = sort(zero_indices)
    else
        zero_inds = [(i - 1) * p + i for i in 1:p]
    end

    # Allocate space for models, objectives, and losses
    model = Vector{Flux.Chain}(undef, lambda_n)
    theta = Vector{Float32}(undef, lambda_n)
    train_loss = Vector{Float32}(undef, lambda_n)
    val_loss = Vector{Float32}(undef, lambda_n)
    val_nonzero = Vector{Float32}(undef, lambda_n)
    epochs = Vector{Int}(undef, lambda_n)

    # Loop over lambda values
    for i in 1:lambda_n

        # Print status update
        if verbose && i == 1
            print("Training with regularisation parameter $i of $lambda_n...\n")
        elseif verbose
            print("\r\033[1ATraining with regularisation parameter $i of $lambda_n...\n")
        end

        # Save lambda or initialise lambda_max
        if isnothing(lambda)
            lambda_i = Inf
        else
            lambda_i = lambda[i]
        end

        # Create neural network model
        if initialise == "cold" || i == 1
            model_i = gennet(hidden_layers, p, m, activation_fun)
        else
            model_i = deepcopy(model[i - 1])
        end

        # Zero-out weights corresponding to diagonal elements (ensures diagonal elements are zero)
        Flux.params(model_i[end - 1])[1][zero_inds, :] .= 0

        # Move model to training device
        model_i = Flux.gpu(model_i)
        
        # Initialise by training model as a simple directed graph
        if i == 1
            model_i = init(model_i, x, z, x_val, z_val, optimiser, epoch_max, early_stop, patience, 
                verbose, verbose_freq, zero_inds, mask_train, mask_valid)
        end

        # Train model
        model_i, theta[i], train_loss[i], val_loss[i], val_nonzero[i], epochs[i] = train(model_i, 
            lambda_i, params, x, z, x_val, z_val, optimiser, epoch_max, early_stop, patience, 
            verbose, verbose_freq, zero_inds, mask_train, mask_valid)

        # If no lambda, compute lambda
        if isnothing(lambda)
            lambda_max = sum(abs.(project(model_i(z), par = 0, params = params, 
                inference = true)[1])) / n
            lambda = range(lambda_max, zero(lambda_max), lambda_n)
        end

        # Move model back to CPU
        model[i] = Flux.cpu(model_i)

    end

    # Save min value of lambda
    lambda_min = lambda[argmin(val_loss)]

    # Set type to ContextualDAGFit
    ContextualDAGFit(model, lambda, lambda_min, theta, z_mean, z_sd, train_loss, val_loss, 
        val_nonzero, epochs, params)

end

#==================================================================================================#
# Function to extract coefficients from a fitted contextual DAG
#==================================================================================================#

"""
    coef(fit::ContextualDAGFit, z; <keyword arguments>)

Produce the coefficients of a weighted adjacency matrix from a contextual DAG fit using contextual \
features `z`. Set `lambda = "lambda_min"` for coefficients from the model with minimum validation \
loss or specify a value of `lambda` for predictions from a particular model. Set \
`gaurantee_dag = true` to threshold the adjacency matrix to guarantee that all cycles are removed.

See also [`fit`](@ref).
"""
function coef(fit::ContextualDAGFit, z::Matrix{<:Real}; lambda::Union{String, Real} = "lambda_min", 
    gaurantee_dag = true, mask)

    mask = Flux.gpu(mask)

    n = size(z, 1)

    # Standardise contextual features before they enter neural net
    z = (z .- fit.z_mean) ./ fit.z_sd

    # Transpose because Flux expects features in rows and samples in columns
    z = transpose(z)

    # Flux defaults to f32 model parameters
    z = Flux.f32(z)

    # Move data to correct device
    z = Flux.gpu(z)

    # Find correct lambda
    if lambda == "lambda_min"
        index_lambda = findall(fit.lambda .== fit.lambda_min)[1]
    else
        index_lambda = argmin(abs.(fit.lambda .- lambda))
    end

    # Compute coefficients by performing a forward pass
    model = Flux.gpu(fit.model[index_lambda])
    w = project(model(z) .* mask, par = fit.theta[index_lambda], params = fit.params, inference = true)[1]
    w = Flux.cpu(w)

    # Gaurantee output is a DAG
    if gaurantee_dag
        for i in 1:n
            while Graphs.is_cyclic(Graphs.SimpleDiGraph(w[:, :, i]))
                nonzero_ind = findall(!iszero, w[:, :, i])
                if isempty(nonzero_ind)
                    break
                end
                nonzero_value = [w[idx[1], idx[2], i] for idx in nonzero_ind]
                min_idx = argmin(abs.(nonzero_value))
                w[nonzero_ind[min_idx][1], nonzero_ind[min_idx][2], i] = 0.0
            end
        end
    end

    w

end

end