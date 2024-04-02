# =================================================================================================#
# Description: Implementation of the fixed DAG
# =================================================================================================#

module FixedDAG

import Graphs, LinearAlgebra

export fixed_dag

function fixed_dag(x; lambda = nothing, lambda_n = 20, lambda_max = 1, lambda_min = 1e-3, s = 1, 
    mu = 1, alpha = 0.5, converge_tol = 1e-6, max_step = 10, max_iter = 1000000, lr = 0.001)

    # Save data dimensions
    n, p = size(x)

    # Allocate space for adjacency matrix
    w = zeros(p, p)
    w_old = zeros(p, p)

    # Precompute x^T x matrix upfront
    xtx = transpose(x) * x / n

    # Compute lambda sequence
    if isnothing(lambda)
        lambda = exp.(range(log(lambda_max), log(lambda_min), lambda_n))
    else
        lambda_n = length(lambda)
    end
    
    # Allocate space for models
    model = Vector{Matrix}(undef, lambda_n)
    nonzero = Vector{Int}(undef, lambda_n)

    # Loop over lambda values
    for i in 1:lambda_n

        # Save lambda and mu
        lambda_i = lambda[i]
        mu_i = mu

        # Run DAGMA
        for step in 1:max_step
            for iter in 1:max_iter

                # Compute gradients
                loss_grad = - xtx + xtx * w# + lambda_i .* sign.(w)
                h_grad = 2 * inv(transpose(s * LinearAlgebra.I - w .^ 2)) .* w
                grad = mu_i * loss_grad + h_grad
                grad.-= LinearAlgebra.Diagonal(grad)

                # Take gradient descent step
                w = w .- lr * grad
                w = sign.(w) .* max.(abs.(w) .- lambda_i * mu_i * lr, 0)
                
                # Check for convergence
                if maximum(abs.(w - w_old)) <= converge_tol
                    break
                else
                    w_old = deepcopy(w)
                end
            end

            # Update mu
            mu_i *= alpha

        end

        # Gaurantee output is a DAG
        while Graphs.is_cyclic(Graphs.SimpleDiGraph(w))
            nonzero_idxs = findall(!iszero, w)
            if isempty(nonzero_idxs)
                break
            end
            nonzero_values = [w[idx[1], idx[2]] for idx in nonzero_idxs]
            min_idx = argmin(abs.(nonzero_values))
            w[nonzero_idxs[min_idx][1], nonzero_idxs[min_idx][2]] = 0.0
        end

        # Save model
        model[i] = deepcopy(w)
        nonzero[i] = sum(w .≠ 0)

    end

    (model, nonzero)

end

end