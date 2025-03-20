module DataDrivenOptimization

using JuMP
using HiGHS  
using Distances
using Statistics
using Statistics: mean
using LinearAlgebra
using PyCall
using DataFrames


const AffOrQuadExpr = Union{AffExpr, QuadExpr}

tree = pyimport("sklearn.tree")
ensemble = pyimport("sklearn.ensemble")
model_selection = pyimport("sklearn.model_selection")
linear_model = pyimport("sklearn.linear_model")
neighbors = pyimport("sklearn.neighbors")

"""
The struct Problem contains all of the information needed to describe a two-stage optimization problem with constraints and an uncertain cost. If the problem is only one-stage, then dtwo should be set to zero (see below).

# Fields
- 'obj': the objective function; a function of the form c(z, y), where z is the decision and y is an uncertain parameter
- 'objmax': the max() terms in the objective function; an array of array of functions of the form c_k(z, y)
    Example: an array of the form [[c_1(z,y), c_2(z,y)], [c_3(z,y), c_4(z,y)]] will add the terms max(c_1(z,y), c_2(z,y)) + max(c_3(z,y), c_4(z,y)) to the objective
- 'objtwo': the second-stage objective function; a function of the form cs(f, z, y), where f is the second-stage decision, z is the first-stage decision, and y is the uncertain parameter
- 'objtwocons': the second-stage constraints; an array of functions of the form gs_k(f, z, y), where f is the second-stage decision, z is the first-stage decision, and y is the uncertain parameter
    Example: an array of the form [gs_1(f, z, y), gs_2(f, z, y)] will add the second-stage constraints gs_1(f, z, y) <= 0 and gs_2(f, z, y) <= 0
- 'dtwo': the dimension of the second-stage decision f
- 'cons': the first-stage constraints; an array of functions of the form g_k(z), where z is the first-stage decision
    Example: an array of the form[g_1(z), g_2(z)] will add the constraints g_1(z) and g_2(z) to the optimization problem
- 'dz': the dimension of the first-stage decision z
- 'name': a name given to the optimization problem, e.g. "newsvendor"
"""
struct Problem
    obj::Function
    objmax::Array{Array{Function, 1}, 1}
    objtwo::Function
    objtwocons::Array{Function, 1}
    dtwo::Int
    cons::Array{Function, 1}
    dz::Int
    name::String
end

"""
The struct ObjectiveLearnerCache is used internally to store optimization problems to be able to warm-start them when the parameters change.
"""
struct ObjectiveLearnerCache
    problem::Problem
    Y::Array{Float64, 2}
    model::Model
    objexpr::Vector{AffOrQuadExpr}  # (2) GenericAffOrQuadExpr 대신
end

"""
Each of the various methods used to solve the data-driven optimization problem is represented as a mutable struct of type Learner.
"""
abstract type Learner end

"""
Methods that can be interpreted as approximating E[c(z; y) | x = x^0] as a weighted combination of c(z; y^i) have subtype ObjectiveLearner.
"""
abstract type ObjectiveLearner <: Learner end

"""
PointPredictionLearner does point prediction (predict y, then optimize) using linear regression.
"""
mutable struct PointPredictionLearner <: Learner
    X::Array{Float64, 2}
    Y::Array{Float64, 2}
    coef::Array{Float64, 2}
    intercept::Array{Float64, 1}
    PointPredictionLearner() = new()
end

"""
PointPredictionNearestNeighborsLearner does point prediction (predict y, then optimize) using nearest neighbors.
"""
mutable struct PointPredictionNearestNeighborsLearner <: Learner
    sklearn_model::PyObject
    X::Array{Float64, 2}
    Y::Array{Float64, 2}
    PointPredictionNearestNeighborsLearner() = new()
    PointPredictionNearestNeighborsLearner(sklearn_model) = new(sklearn_model)
end

"""
PointPredictionSimulationOptimalLearner finds the true value of E[y | x = x^0] by sampling from the conditional distribution, and then optimizes c(z; E[y | x = x^0]).
"""
mutable struct PointPredictionSimulationOptimalLearner <: Learner
    y_sampler::Function
    num_y_samples::Int
    X::Array{Float64, 2}
    Y::Array{Float64, 2}
    PointPredictionSimulationOptimalLearner() = new()
    PointPredictionSimulationOptimalLearner(y_sampler, num_y_samples) = new(y_sampler, num_y_samples)
end

"""
LinearProbabilisticLearner uses linear regression to predict ŷ, and then optimizes ∑_i c(z; ŷ + ϵ_i), where ϵ_i are the residuals from the linear regression.
"""
mutable struct LinearProbabilisticLearner <: Learner
    X::Array{Float64, 2}
    Y::Array{Float64, 2}
    coef::Array{Float64, 2}
    intercept::Array{Float64, 1}
    residuals::Array{Float64, 2}
    LinearProbabilisticLearner() = new()
end

"""
SAALearner is the sample average approximation, optimizing ∑_i c(z; y^i).
"""
mutable struct SAALearner <: ObjectiveLearner
    cached::Bool
    X::Array{Float64, 2}
    Y::Array{Float64, 2}
    cache::ObjectiveLearnerCache
    SAALearner() = new(false)
end

"""
NearestNeighborsLearner is the nearest neighbors method from Bertsimas and Kallus (2018).
"""
mutable struct NearestNeighborsLearner <: ObjectiveLearner
    cached::Bool
    k::Int
    X::Array{Float64, 2}
    Y::Array{Float64, 2}
    cache::ObjectiveLearnerCache
    NearestNeighborsLearner() = new(false)
    NearestNeighborsLearner(k) = new(false, k)
end

"""
CARTLearner is the CART method from Bertsimas and Kallus (2018).
"""
mutable struct CARTLearner <: ObjectiveLearner
    cached::Bool
    sklearn_tree::PyObject
    X::Array{Float64, 2}
    Y::Array{Float64, 2}
    leaf_indices::Array{Int64, 1}
    cache::ObjectiveLearnerCache
    CARTLearner() = new(false)
    CARTLearner(sklearn_tree) = new(false, sklearn_tree)
end

"""
RandomForestLearner is the random forest method from Bertsimas and Kallus (2018).
"""
mutable struct RandomForestLearner <: ObjectiveLearner
    cached::Bool
    sklearn_forest::PyObject
    X::Array{Float64, 2}
    Y::Array{Float64, 2}
    leaf_indices::Array{Int64, 2}
    cache::ObjectiveLearnerCache
    RandomForestLearner() = new(false)
    RandomForestLearner(sklearn_forest) = new(false, sklearn_forest)
end

"""
KernelObjectiveLearner is the kernel objective prediction method developed in the accompanying paper.
"""
mutable struct KernelObjectiveLearner <: ObjectiveLearner
    cached::Bool
    γ::Float64
    λ::Float64
    X::Array{Float64, 2}
    Y::Array{Float64, 2}
    K_inv_chol::Cholesky{Float64, Array{Float64, 2}}
    cache::ObjectiveLearnerCache
    KernelObjectiveLearner() = new(false)
    KernelObjectiveLearner(γ, λ) = new(false, γ, λ)
end

"""
KernelOptimizerLearner is the kernel optimizer prediction method developed in the accompanying paper.
"""
mutable struct KernelOptimizerLearner <: Learner
    γ::Float64
    λ::Float64
    ψ::Float64
    spec_λ::Float64
    X::Array{Float64, 2}
    Y::Array{Float64, 2}
    a::Array{Float64, 2}
    KernelOptimizerLearner() = new()
    KernelOptimizerLearner(γ, λ, ψ) = new(γ, λ, ψ)
end

"""
SimulationOptimalLearner finds the true objective E[c(z; y) | x = x^0] by sampling from the conditional distribution and optimizes over this objective. This is effectively the sample average approximation where the data comes from the true conditional distribution.
"""
mutable struct SimulationOptimalLearner <: Learner
    y_sampler::Function
    num_y_samples::Int
    X::Array{Float64, 2}
    Y::Array{Float64, 2}
    SimulationOptimalLearner() = new()
    SimulationOptimalLearner(y_sampler, num_y_samples) = new(y_sampler, num_y_samples)
end

"""
ExPostOptimalLearner optimizes over c(z; y^0), where y^0 is the ex-post value of y corresponding to x^0.
"""
mutable struct ExPostOptimalLearner <: Learner
end

"""
    make_optimization_objective!(problem, model, z, Y)

Using the JuMP model 'model', return an array containing the expressions c(z; Y[i, :]), where c(z; y) is the objective from 'problem' (including max terms and second-stage terms), adding the necessary constraints to 'model' to reformulate as a linear optimization problem.

For example, if the objective is of the form c(z; y) = max(y[1]*z[1], z[2]), the function will return an array of the form [v[1], v[2], ..., v[n]], adding the constraints v[i] >= Y[i, 1]*z[1] and v[i] >= z[2] to 'model'.
"""
function make_optimization_objective!(problem::Problem, model::Model, z, Y)
    n = size(Y, 1)

    # JuMP 1.x: Vector{AffOrQuadExpr}(undef, n)
    objexpr = Vector{AffOrQuadExpr}(undef, n)

    for i = 1:n
        objexpr[i] = @expression(model, problem.obj(z, Y[i, :]))
    end

    if size(problem.objmax, 1) > 0
        vmax = @variable(model, [i = 1:n, jout = 1:size(problem.objmax, 1)])
        for i = 1:n
            for jout = 1:size(problem.objmax, 1)
                add_to_expression!(objexpr[i], vmax[i, jout])
                for jin = 1:size(problem.objmax[jout], 1)
                    @constraint(model, vmax[i, jout] >= problem.objmax[jout][jin](z, Y[i, :]))
                end
            end
        end
    end

    if problem.dtwo > 0
        f = @variable(model, [i = 1:n, k = 1:problem.dtwo])
        for i = 1:n
            add_to_expression!(objexpr[i], problem.objtwo(f[i, :], z, Y[i, :]))
            if size(problem.objtwocons, 1) > 0
                @constraint(model, [j = 1:size(problem.objtwocons, 1)], problem.objtwocons[j](f[i, :], z, Y[i, :]) <= 0)
            end
        end
    end

    return objexpr
end

"""
    make_optimization_constraints!(problem, model, z)

Add (first-stage) constraints from 'problem' to JuMP 'model', using the JuMP variable 'z'.
"""
function make_optimization_constraints!(problem::Problem, model::Model, z)
    if size(problem.cons, 1) > 0
        @constraint(model, [j = 1:size(problem.cons, 1)], problem.cons[j](z) <= 0)
    end
end

"""
    make_optimization_constraints_relaxed!(problem, model, z)

Using the JuMP model 'model', return an array containing the expressions max(g_k(z), 0)^2, where g_k are the constraints from 'problem', adding the necessary constraints to 'model' to reformulate as a quadratic optimization problem.

See make_optimization_objective! documentation for analagous example.
"""
function make_optimization_constraints_relaxed!(problem::Problem, model::Model, z)
    if size(problem.cons, 1) > 0
        vmaxcon = @variable(model, [j = 1:size(problem.cons, 1)])
        objexpr_con = @expression(model, sum(vmaxcon[j]^2 for j = 1:size(problem.cons, 1)))
        for j = 1:size(problem.cons, 1)
            @constraint(model, vmaxcon[j] >= problem.cons[j](z))
            @constraint(model, vmaxcon[j] >= 0)
        end
        return objexpr_con
    else
        return @expression(model, 0.0)
    end
end

"""
    make_optimization_constraints_relaxed_linear!(problem, model, z)

Using the JuMP model 'model', return an array containing the expressions max(g_k(z), 0), where g_k are the constraints from 'problem', adding the necessary constraints to 'model' to reformulate as a linear optimization problem.

See make_optimization_objective! documentation for analagous example.
"""
function make_optimization_constraints_relaxed_linear!(problem::Problem, model::Model, z)
    if size(problem.cons, 1) > 0
        vmaxcon = @variable(model, [j = 1:size(problem.cons, 1)])
        objexpr_con = @expression(model, sum(vmaxcon[j] for j = 1:size(problem.cons, 1)))
        for j = 1:size(problem.cons, 1)
            @constraint(model, vmaxcon[j] >= problem.cons[j](z))
            @constraint(model, vmaxcon[j] >= 0)
        end
        return objexpr_con
    else
        return @expression(model, 0.0)
    end
end

"""
    project(problem, z)

Projects z onto the feasible set of the constraints from 'problem'.
"""
function project(problem, z)
    model = Model(HiGHS.Optimizer)
    set_optimizer_attribute(model, "log_to_console", false)

    @variable(model, v[1:problem.dz])
    make_optimization_constraints!(problem, model, v)
    @objective(model, Min, sum((v[k] - z[k])^2 for k = 1:problem.dz))

    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        return value.(v)
    else
        error("Unable to project")
    end
    return zeros(problem.dz)
end

"""
    cost(problem, Z, Y)

Computes a vector of costs c(Z[i, :]; Y[i, :]), where c(z; y) is the objective from 'problem' (including max terms and second-stage terms).
"""
function cost(problem::Problem, Z::Array{Float64, 2}, Y::Array{Float64, 2})
    n = size(Z, 1)
    d = problem.dz

    costs = zeros(n)
    for i = 1:n
        model = Model(HiGHS.Optimizer)
        set_optimizer_attribute(model, "log_to_console", false)

        @variable(model, z[1:d])
        objexpr = make_optimization_objective!(problem, model, z, Y[i:i, :])
        v = project(problem, Z[i, :])
        @constraint(model, [k = 1:d], z[k] == v[k])
        @objective(model, Min, objexpr[1])

        optimize!(model)
        if termination_status(model) == MOI.OPTIMAL
            costs[i] = objective_value(model)
        else
            error("Unable to evaluate cost")
        end
    end
    return costs
end

"""
    split(X, Y)

Splits X and Y into training and test sets, with the first half of observations going into the training set and second half going into the test set.
"""
function split(X::Array{Float64, 2}, Y::Array{Float64, 2})
    n = size(X, 1)
    tr_X = X[1:convert(Int64, ceil(n/2)), :]
    tr_Y = Y[1:convert(Int64, ceil(n/2)), :]
    te_X = X[convert(Int64, (ceil(n/2) + 1)):n, :]
    te_Y = Y[convert(Int64, (ceil(n/2) + 1)):n, :]
    return tr_X, tr_Y, te_X, te_Y
end

"""
    cv!(problem, learner, tr_X, tr_Y, va_X, va_Y)

Selects hyperparameters of 'learner' (represented as fields of the corresponding mutable struct) using cross-validation, with training set (tr_X, tr_Y) and validation set (va_X, va_Y).

This function is implemented for each specific subtype of Learner, unless the subtype has no hyperparameters.
"""
function cv!(problem::Problem, learner::Learner, tr_X::Array{Float64, 2}, tr_Y::Array{Float64, 2}, va_X::Array{Float64, 2}, va_Y::Array{Float64, 2})
end

"""
    fit!(problem, learner, X, Y)

Fits 'learner' to training data (X, Y), doing whatever computations and storing whatever objects needed to be able to produce decisions on a test set. Also stores training data X in learner.X and Y in learner.Y.

This function is implemented for each specific subtype of Learner.
"""
function fit!(problem::Problem, learner::Learner, X::Array{Float64, 2}, Y::Array{Float64, 2})
end

"""
    prescribe(problem, learner, X)

Prescribes a decision for 'problem' using 'learner' for each X[i, :] in X. Returns a matrix Z, where Z[i, :] is the decision corresponding to X[i, :].

This function is implemented for each specific subtype of Learner.
"""
function prescribe(problem::Problem, learner::Learner, X::Array{Float64, 2})
    return zeros(size(X, 1), problem.dz)
end

"""
    cv!(problem, learner, tr_X, tr_Y, va_X, va_Y, param_values)

When 'param_values' is provided as a Dict containing fields of 'learner' and arrays of possible values, trains 'learner' with training set and computes cost on validation set for every combination of values and sets fields to the combination minimizing cost.
"""
function cv!(problem::Problem, learner::Learner,
              tr_X::Array{Float64, 2}, tr_Y::Array{Float64, 2},
              va_X::Array{Float64, 2}, va_Y::Array{Float64, 2},
              param_values::Dict)
    kv = collect(zip(collect(param_values)...))
    params = kv[1]
    vs = kv[2]
    best_cost = Inf
    best_v = Nothing
    for v_tuple in Iterators.product(vs...)
        for i in 1:length(params)
            setfield!(learner, params[i], v_tuple[i])
        end
        fit!(problem, learner, tr_X, tr_Y)
        Z = prescribe(problem, learner, va_X)
        c = mean(cost(problem, Z, va_Y))
        if c < best_cost
            best_cost = c
            best_v = v_tuple
        end
    end
    for i in 1:length(params)
        setfield!(learner, params[i], best_v[i])
    end
end

function fit!(problem::Problem, learner::PointPredictionLearner, 
              X::Array{Float64, 2}, Y::Array{Float64, 2})
    learner.X = X
    learner.Y = Y
    sklearn_model = linear_model.LinearRegression()
    sklearn_model.fit(X, Y)
    coef = sklearn_model.coef_
    if ndims(coef) == 1
        learner.coef = reshape(coef, 1, size(coef, 1))
    else
        learner.coef = coef
    end
    learner.intercept = sklearn_model.intercept_
end

function prescribe(problem::Problem, learner::PointPredictionLearner, X::Array{Float64, 2})
    nte = size(X, 1)
    d = problem.dz
    Z = zeros(nte, problem.dz)

    for ite = 1:nte
        x = X[ite, :]
        y = transpose(learner.coef * x + learner.intercept)

        model = Model(HiGHS.Optimizer)
        set_optimizer_attribute(model, "log_to_console", false)

        @variable(model, z[1:d])
        objexpr = make_optimization_objective!(problem, model, z, y)
        make_optimization_constraints!(problem, model, z)
        @objective(model, Min, objexpr[1])

        optimize!(model)
        if has_values(model)
            Z[ite, :] = value.(z)
        else
            Z[ite, :] = 0.0
        end
    end
    return Z
end

function cv!(problem::Problem, learner::PointPredictionNearestNeighborsLearner, 
             tr_X::Array{Float64, 2}, tr_Y::Array{Float64, 2}, 
             va_X::Array{Float64, 2}, va_Y::Array{Float64, 2})
    X = vcat(tr_X, va_X)
    Y = vcat(tr_Y, va_Y)
    train_indices = collect(0:(size(tr_X, 1) - 1))
    test_indices = collect(size(tr_X, 1):(size(X, 1) - 1))
    cv_indices = [[train_indices, test_indices]]

    k_space = collect(1:convert(Int64, ceil(size(tr_X, 1)/2)))
    clf = neighbors.KNeighborsRegressor()

    # (3) iid=true 제거
    grid = model_selection.GridSearchCV(
        clf,
        Dict("n_neighbors" => k_space),
        scoring="r2",
        cv=cv_indices
    )

    grid.fit(X, Y)
    learner.sklearn_model = grid.best_estimator_
end

function fit!(problem::Problem, learner::PointPredictionNearestNeighborsLearner, 
              X::Array{Float64, 2}, Y::Array{Float64, 2})
    learner.X = X
    learner.Y = Y
    learner.sklearn_model.fit(X, Y)
end

function prescribe(problem::Problem, learner::PointPredictionNearestNeighborsLearner, 
                   X::Array{Float64, 2})
    nte = size(X, 1)
    d = problem.dz
    Z = zeros(nte, problem.dz)

    Y_hat = learner.sklearn_model.predict(X)
    for ite = 1:nte
        model = Model(HiGHS.Optimizer)
        set_optimizer_attribute(model, "log_to_console", false)

        @variable(model, z[1:d])
        objexpr = make_optimization_objective!(problem, model, z, Y_hat[ite:ite, :])
        make_optimization_constraints!(problem, model, z)
        @objective(model, Min, objexpr[1])

        optimize!(model)
        if has_values(model)
            Z[ite, :] = value.(z)
        else
            Z[ite, :] = 0.0
        end
    end
    return Z
end

function fit!(problem::Problem, learner::PointPredictionSimulationOptimalLearner,
              X::Array{Float64, 2}, Y::Array{Float64, 2})
    learner.X = X
    learner.Y = Y
end

function prescribe(problem::Problem, learner::PointPredictionSimulationOptimalLearner,
                   X::Array{Float64, 2})
    nte = size(X, 1)
    Z = zeros(nte, problem.dz)

    for ite = 1:nte
        x = X[ite, :]
        Y_mean = zeros(1, size(learner.Y, 2))
        for i = 1:learner.num_y_samples
            Y_mean[1, :] += learner.y_sampler(x)
        end
        Y_mean[1, :] /= learner.num_y_samples
        saa_learner = SAALearner()
        fit!(problem, saa_learner, X[ite:ite, :], Y_mean)
        Z[ite, :] = prescribe(problem, saa_learner, X[ite:ite, :])[1, :]
    end
    return Z
end

function fit!(problem::Problem, learner::LinearProbabilisticLearner,
              X::Array{Float64, 2}, Y::Array{Float64, 2})
    learner.X = X
    learner.Y = Y
    sklearn_model = linear_model.LinearRegression()
    sklearn_model.fit(X, Y)
    coef = sklearn_model.coef_
    if ndims(coef) == 1
        learner.coef = reshape(coef, 1, size(coef, 1))
    else
        learner.coef = coef
    end
    learner.intercept = sklearn_model.intercept_
    learner.residuals = Y .- sklearn_model.predict(X)
end

function prescribe(problem::Problem, learner::LinearProbabilisticLearner, X::Array{Float64, 2})
    nte = size(X, 1)
    Z = zeros(nte, problem.dz)

    for ite = 1:nte
        x = X[ite, :]
        y = learner.coef * x .+ learner.intercept
        Y_pert = zeros(size(learner.residuals, 1), size(learner.residuals, 2))
        for i = 1:size(Y_pert, 1)
            Y_pert[i, :] = learner.residuals[i, :] + y
        end
        saa_learner = SAALearner()
        fit!(problem, saa_learner, X[ite:ite, :], Y_pert)
        Z[ite, :] = prescribe(problem, saa_learner, X[ite:ite, :])[1, :]
    end
    return Z
end

function fit!(problem::Problem, learner::SAALearner, X::Array{Float64, 2}, Y::Array{Float64, 2})
    learner.X = X
    learner.Y = Y
end

function prescribe(problem::Problem, learner::SAALearner, X::Array{Float64, 2})
    tr_Y = learner.Y
    ntr = size(tr_Y, 1)
    nte = size(X, 1)

    Z = zeros(nte, problem.dz)
    w = fill(1.0/ntr, ntr)
    z = prescribe_objhelper(problem, learner, w)

    for ite = 1:nte
        Z[ite, :] = z
    end
    return Z
end

function cv!(problem::Problem, learner::NearestNeighborsLearner,
             tr_X::Array{Float64, 2}, tr_Y::Array{Float64, 2},
             va_X::Array{Float64, 2}, va_Y::Array{Float64, 2})
    cv!(problem, learner, tr_X, tr_Y, va_X, va_Y,
        Dict(:k => unique(round.(Int, collect(range(1, stop=ceil(size(tr_X, 1)/2), length=50))))))
end

function fit!(problem::Problem, learner::NearestNeighborsLearner,
              X::Array{Float64, 2}, Y::Array{Float64, 2})
    learner.X = X
    learner.Y = Y
end

function prescribe(problem::Problem, learner::NearestNeighborsLearner, X::Array{Float64, 2})
    tr_X = learner.X
    tr_Y = learner.Y
    k = learner.k
    ntr = size(tr_X, 1)
    nte = size(X, 1)

    Z = zeros(nte, problem.dz)
    dists = max.(pairwise(Euclidean(), transpose(tr_X), transpose(X), dims=2), 0)

    for ite = 1:nte
        w = zeros(ntr)
        dist = dists[:, ite]
        near = sortperm(vec(dist))
        for i = 1:k
            w[near[i]] = 1.0
        end
        Z[ite, :] = prescribe_objhelper(problem, learner, w)
    end
    return Z
end

function cv!(problem::Problem, learner::CARTLearner, tr_X::Array{Float64, 2}, tr_Y::Array{Float64, 2},
             va_X::Array{Float64, 2}, va_Y::Array{Float64, 2})
    X = vcat(tr_X, va_X)
    Y = vcat(tr_Y, va_Y)
    train_indices = collect(0:(size(tr_X, 1) - 1))
    test_indices = collect(size(tr_X, 1):(size(X, 1) - 1))
    cv_indices = [[train_indices, test_indices]]

    max_depth_space = collect(1:(convert(Int, floor(log(2, size(tr_X, 1)))) - 1))
    min_samples_split_space = vcat([0.0001], collect(range(0.1, stop=1.0, length=5)))
    min_samples_leaf_space = vcat([0.0001], collect(range(0.1, stop=0.5, length=3)))

    function prescription_scoring(estimator, X, Y)
        cv_learner = CARTLearner(estimator)
        fit!(problem, cv_learner, X, hcat(Y))
        Z = prescribe(problem, cv_learner, X)
        return -1.0 * mean(cost(problem, Z, hcat(Y)))
    end

    clf = tree.DecisionTreeRegressor()
    # (3) iid=true 제거
    grid = model_selection.GridSearchCV(
        clf,
        Dict(
            "max_depth" => max_depth_space,
            "min_samples_split" => min_samples_split_space,
            "min_samples_leaf" => min_samples_leaf_space
        ),
        scoring=prescription_scoring,
        cv=cv_indices
    )
    grid.fit(X, Y)
    learner.sklearn_tree = grid.best_estimator_
end

function fit!(problem::Problem, learner::CARTLearner, X::Array{Float64, 2}, Y::Array{Float64, 2})
    learner.X = X
    learner.Y = Y
    learner.sklearn_tree.fit(X, Y)
    learner.leaf_indices = learner.sklearn_tree.apply(X)
end

function prescribe(problem::Problem, learner::CARTLearner, X::Array{Float64, 2})
    ntr = size(learner.X, 1)
    nte = size(X, 1)
    train_leaf_indices = learner.leaf_indices
    test_leaf_indices = learner.sklearn_tree.apply(X)

    Z = zeros(nte, problem.dz)
    for ite = 1:nte
        w = zeros(ntr)
        one_indices = findall(train_leaf_indices .== test_leaf_indices[ite])
        for i in one_indices
            w[i] = 1.0
        end
        Z[ite, :] = prescribe_objhelper(problem, learner, w)
    end
    return Z
end

function cv!(problem::Problem, learner::RandomForestLearner,
             tr_X::Array{Float64, 2}, tr_Y::Array{Float64, 2},
             va_X::Array{Float64, 2}, va_Y::Array{Float64, 2})
    X = vcat(tr_X, va_X)
    Y = vcat(tr_Y, va_Y)
    train_indices = collect(0:(size(tr_X, 1) - 1))
    test_indices = collect(size(tr_X, 1):(size(X, 1) - 1))
    cv_indices = [[train_indices, test_indices]]

    max_depth_space = collect(1:(convert(Int, floor(log(2, size(tr_X, 1)))) - 1))
    min_samples_split_space = vcat([0.0001], collect(range(0.1, stop=1.0, length=5)))
    min_samples_leaf_space = vcat([0.0001], collect(range(0.1, stop=0.5, length=3)))

    function prescription_scoring(estimator, X, Y)
        cv_learner = RandomForestLearner(estimator)
        fit!(problem, cv_learner, X, hcat(Y))
        Z = prescribe(problem, cv_learner, X)
        return -1.0 * mean(cost(problem, Z, hcat(Y)))
    end

    clf = ensemble.RandomForestRegressor(n_estimators=100)
    # (3) iid=true 제거
    grid = model_selection.GridSearchCV(
        clf,
        Dict(
            "max_depth" => max_depth_space,
            "min_samples_split" => min_samples_split_space,
            "min_samples_leaf" => min_samples_leaf_space
        ),
        scoring=prescription_scoring,
        cv=cv_indices
    )
    if size(Y, 2) == 1
        grid.fit(X, vec(Y))
    else
        grid.fit(X, Y)
    end
    learner.sklearn_forest = grid.best_estimator_
end

function fit!(problem::Problem, learner::RandomForestLearner, 
              X::Array{Float64, 2}, Y::Array{Float64, 2})
    learner.X = X
    learner.Y = Y
    if size(Y, 2) == 1
        learner.sklearn_forest.fit(X, vec(Y))
    else
        learner.sklearn_forest.fit(X, Y)
    end
    learner.leaf_indices = learner.sklearn_forest.apply(X)
end

function prescribe(problem::Problem, learner::RandomForestLearner, X::Array{Float64, 2})
    ntr = size(learner.X, 1)
    nte = size(X, 1)
    train_leaf_indices = learner.leaf_indices
    test_leaf_indices = learner.sklearn_forest.apply(X)

    Z = zeros(nte, problem.dz)
    for ite = 1:nte
        w_mat = zeros(ntr, size(train_leaf_indices, 2))
        for iest = 1:size(train_leaf_indices, 2)
            one_indices = findall(train_leaf_indices[:, iest] .== test_leaf_indices[ite, iest])
            for i in one_indices
                w_mat[i, iest] = 1.0
            end
        end
        w = zeros(ntr)
        for i in 1:size(w, 1)
            w[i] = mean(w_mat[i, :])
        end
        Z[ite, :] = prescribe_objhelper(problem, learner, w)
    end
    return Z
end

function cv!(problem::Problem, learner::KernelObjectiveLearner,
             tr_X::Array{Float64, 2}, tr_Y::Array{Float64, 2},
             va_X::Array{Float64, 2}, va_Y::Array{Float64, 2})
    sq_dist_mat = max.(pairwise(SqEuclidean(), transpose(tr_X), transpose(tr_X), dims=2), 0)
    sig_mean = mean(sqrt.(sq_dist_mat))

    sig_try = [sig_mean/16.0; sig_mean*50.0]
    lam_try = [1.0; 1e-11]
    try_len = 10
    if problem.name == "product" && size(tr_X, 1) > 200
        sig_try = [sig_mean/16.0; sig_mean*5.0]
        lam_try = [1.0e-2; 1.0e-10]
        try_len = 7
    end

    gam_try = 1.0 ./ (2.0 .* (sig_try .^ 2))
    gam_try = exp.(range(log(gam_try[1]), stop=log(gam_try[end]), length=try_len))
    lam_try = exp.(range(log(lam_try[1]), stop=log(lam_try[end]), length=try_len))

    cv!(problem, learner, tr_X, tr_Y, va_X, va_Y,
        Dict(:γ => collect(gam_try), :λ => collect(lam_try)))

    println(sqrt.(1.0 ./ (2.0 .* gam_try))./ sig_mean, "|", lam_try)
    println(sqrt(1.0/(2.0*learner.γ))/sig_mean, "|", learner.λ)
end

function fit!(problem::Problem, learner::KernelObjectiveLearner, 
              X::Array{Float64, 2}, Y::Array{Float64, 2})
    learner.X = X
    learner.Y = Y

    dist_mat = max.(pairwise(SqEuclidean(), transpose(X), transpose(X), dims=2), 0)
    K̂ = Symmetric(exp.(-learner.γ*dist_mat))
    evmin = eigmin(K̂)
    eviter = 0
    while evmin < 0 && eviter < 100
        K̂ = Symmetric(K̂) + abs(evmin)*2.0 * I
        evmin = eigmin(K̂)
        eviter += 1
    end

    learner.K_inv_chol = cholesky(K̂ + learner.λ * size(K̂, 1) * I)
end

function prescribe(problem::Problem, learner::KernelObjectiveLearner, X::Array{Float64, 2})
    tr_X = learner.X
    tr_Y = learner.Y
    M = learner.K_inv_chol
    γ = learner.γ
    λ = learner.λ
    ntr = size(tr_X, 1)
    nte = size(X, 1)

    Z = zeros(nte, problem.dz)
    for ite = 1:nte
        x = X[ite, :]
        dist = max.(pairwise(SqEuclidean(), transpose(tr_X), reshape(x, length(x), 1), dims=2), 0)
        kx = exp.(-γ .* dist)
        w = vec(max.(M \ kx, 0))
        Z[ite, :] = prescribe_objhelper(problem, learner, w)
    end
    return Z
end

"""
    prescribe_objhelper(problem, learner, w)

Produces a decision by minimizing ∑_i w[i] * c(z; learner.Y[i, :]) subject to 'problem' constraints.

Uses learner.cache to enable warm starts when w changes and everything else stays the same.
"""
function prescribe_objhelper(problem::Problem, learner::ObjectiveLearner, w::Array{Float64, 1})
    tr_Y = learner.Y
    ntr = size(tr_Y, 1)
    d = problem.dz

    if learner.cached && learner.cache.problem == problem && learner.cache.Y == tr_Y
        model = learner.cache.model
        objexpr = learner.cache.objexpr
        z = model[:z]
    else
        model = Model(HiGHS.Optimizer)
        set_optimizer_attribute(model, "log_to_console", false)

        @variable(model, z[1:d])
        objexpr = make_optimization_objective!(problem, model, z, tr_Y)
        make_optimization_constraints!(problem, model, z)
    end

    objexpr_final = @expression(model, 0.0*z[1])
    for i = 1:ntr
        add_to_expression!(objexpr_final, w[i]*objexpr[i])
    end
    @objective(model, Min, objexpr_final)

    optimize!(model)
    learner.cache = ObjectiveLearnerCache(problem, tr_Y, model, objexpr)
    learner.cached = true

    if has_values(model)
        return value.(z)
    end
    return zeros(d)
end

function cv!(problem::Problem, learner::KernelOptimizerLearner,
             tr_X::Array{Float64, 2}, tr_Y::Array{Float64, 2},
             va_X::Array{Float64, 2}, va_Y::Array{Float64, 2})
    sq_dist_mat = max.(pairwise(SqEuclidean(), transpose(tr_X), transpose(tr_X), dims=2), 0)
    sig_mean = mean(sqrt.(sq_dist_mat))

    sig_try = [sig_mean*1.0; sig_mean*100]
    lam_try = [1.0e-2; 1.0e-11]
    psi_try = [0.0]
    gam_try_len = 10
    lam_try_len = 10
    speclam_try = [1.0e-10]

    if problem.name == "product"
        sig_try = [sig_mean*50; sig_mean*100]
        lam_try = [1.0e-5; 1.0e-7]
        gam_try_len = 5
        lam_try_len = 2
        speclam_try = [1.0e-8]
        if size(tr_X, 1) > 200
            sig_try = [sig_mean*50.0]
            lam_try = [1.0e-7]
            gam_try_len = 1
            lam_try_len = 1
            speclam_try = [1.0e-8]
        end
    end
    if problem.name == "newsvendor"
        sig_try = [sig_mean*50.0]
        lam_try = [1.0e-8; 1.0e-10]
        psi_try = [0.0]
        gam_try_len = 1
        lam_try_len = 2
        speclam_try = [1.0e-10]
    end

    gam_try = 1 ./ (2 .* (sig_try .^ 2))
    gam_try = exp.(range(log(gam_try[1]), stop=log(gam_try[end]), length=gam_try_len))
    lam_try = exp.(range(log(lam_try[1]), stop=log(lam_try[end]), length=lam_try_len))
    psi_try = range(psi_try[1], stop=psi_try[end], length=1)

    cv!(problem, learner, tr_X, tr_Y, va_X, va_Y,
        Dict(:γ => collect(gam_try),
             :λ => collect(lam_try),
             :ψ => collect(psi_try),
             :spec_λ => collect(speclam_try)))

    println(sqrt.(1.0 ./ (2*gam_try)) ./ sig_mean, "|", lam_try, "|", psi_try)
    println(sqrt(1.0/(2.0*learner.γ))/sig_mean, "|", learner.λ, "|", learner.ψ)
end

function fit!(problem::Problem, learner::KernelOptimizerLearner, 
              X::Array{Float64, 2}, Y::Array{Float64, 2})
    learner.X = X
    learner.Y = Y
    n = size(X, 1)
    d = problem.dz

    dist_mat = max.(pairwise(SqEuclidean(), transpose(X), transpose(X), dims=2), 0)
    K̂ = Symmetric(exp.(-learner.γ*dist_mat))
    K̂ = K̂ + learner.spec_λ*I
    spec_frac = 1.0
    eigd = eigen(K̂)
    eig_keep = 0
    for i in 1:size(eigd.values, 1)
        if (eigd.values[end - i + 1] > learner.spec_λ) ||
           ((i/size(eigd.values, 1) < spec_frac) && (eigd.values[end - i + 1] > 0))
            eig_keep += 1
        end
    end
    eig_keep = max(eig_keep, 1)

    eigvec_keep = eigd.vectors[:, (size(eigd.vectors, 2) - eig_keep + 1):end]
    eigval_keep = eigd.values[(size(eigd.values, 1) - eig_keep + 1):end]
    F = eigvec_keep * Diagonal(1.0 ./ sqrt.(eigval_keep))
    dtrunc = size(F, 2)

    model = Model(HiGHS.Optimizer)
    set_optimizer_attribute(model, "log_to_console", false)

    @variable(model, z[1:n, 1:d])
    objexpr = Array{Any}(undef, n)
    objexpr_con = Array{Any}(undef, n)
    for i = 1:n
        objexpr[i] = make_optimization_objective!(problem, model, z[i, :], learner.Y[i:i, :])[1]
        objexpr_con[i] = make_optimization_constraints_relaxed_linear!(problem, model, z[i, :])
    end
    @variable(model, θ[i = 1:dtrunc, k = 1:d])
    @constraint(model, θconstr[i = 1:dtrunc, k = 1:d], θ[i, k] == sum(F[j, i]*z[j, k] for j = 1:n))

    @objective(
        model, 
        Min, 
        (1.0/n)*(sum(objexpr[i] for i = 1:n) + learner.ψ*sum(objexpr_con[i] for i = 1:n)) +
        learner.λ*sum(θ[i, k]*θ[i, k] for i = 1:dtrunc, k = 1:d)
    )

    optimize!(model)

    learner.a = zeros(n, d)
    if has_values(model)
        learner.a = F * transpose(F) * value.(z)
    end
end

function prescribe(problem::Problem, learner::KernelOptimizerLearner, X::Array{Float64, 2})
    tr_X = learner.X
    a = learner.a
    γ = learner.γ
    λ = learner.λ
    nte = size(X, 1)

    Z = zeros(nte, problem.dz)
    for ite = 1:nte
        x = X[ite, :]
        dist = max.(pairwise(SqEuclidean(), transpose(tr_X), reshape(x, length(x), 1), dims=2), 0)
        kx = exp.(-γ .* dist)
        Z[ite, :] = vec(transpose(kx)*a)
    end
    return Z
end

function fit!(problem::Problem, learner::SimulationOptimalLearner, 
              X::Array{Float64, 2}, Y::Array{Float64, 2})
    learner.X = X
    learner.Y = Y
end

function prescribe(problem::Problem, learner::SimulationOptimalLearner, 
                   X::Array{Float64, 2})
    nte = size(X, 1)
    Z = zeros(nte, problem.dz)

    for ite = 1:nte
        x = X[ite, :]
        Y_samp = zeros(learner.num_y_samples, size(learner.Y, 2))
        for i = 1:size(Y_samp, 1)
            Y_samp[i, :] = learner.y_sampler(x)
        end
        saa_learner = SAALearner()
        fit!(problem, saa_learner, X[ite:ite, :], Y_samp)
        Z[ite, :] = prescribe(problem, saa_learner, X[ite:ite, :])[1, :]
    end
    return Z
end

"""
    prescribe_expost(problem, learner, Y)

Produces decisions by minimizing c(z; Y[i, :]) subject to 'problem' constraints. Returns a matrix Z where Z[i, :] is the decision that minimizes c(z; Y[i, :]).

Needs argument Y unlike other prescribe methods.
"""
function prescribe_expost(problem::Problem, learner::ExPostOptimalLearner, Y::Array{Float64, 2})
    nte = size(Y, 1)
    Z = zeros(nte, problem.dz)

    for ite = 1:nte
        model = Model(HiGHS.Optimizer)
        set_optimizer_attribute(model, "log_to_console", false)

        @variable(model, z[1:problem.dz])
        objexpr = make_optimization_objective!(problem, model, z, Y[ite:ite, :])
        make_optimization_constraints!(problem, model, z)
        @objective(model, Min, objexpr[1])

        optimize!(model)
        if termination_status(model) == MOI.OPTIMAL
            Z[ite, :] = value.(z)
        else
            println(ite)
            println(model)
            error("Unable to evaluate ex-post optimal")
        end
    end
    return Z
end

"""
    simulate(problem, learners, learners_names, train_sampler, val_sampler, test_sampler)

Uses 'train_sampler', 'val_sampler', and 'test_sampler' to generate training, validation, and test data. Then trains, validates, and tests each of the learners in 'learners', and returns a DataFrame with costs and times.
"""
function simulate(problem::Problem, learners::Array{Learner, 1}, learners_names::Array{String, 1},
                  train_sampler, val_sampler, test_sampler)
    df = DataFrame(Method = String[], Cost = Float64[],
                   TrainTime = Float64[], ValTime = Float64[], TestTime = Float64[])
    tr_X, tr_Y = train_sampler()
    va_X, va_Y = val_sampler()
    te_X, te_Y = test_sampler()

    for il = 1:size(learners, 1)
        learner = learners[il]
        println("Learner: ", learners_names[il])

        cvt = @elapsed cv!(problem, learner, tr_X, tr_Y, va_X, va_Y)
        trt = @elapsed fit!(problem, learner, tr_X, tr_Y)

        if typeof(learner) == ExPostOptimalLearner
            Z, tet = @timed prescribe_expost(problem, learner, te_Y)
        else
            Z, tet = @timed prescribe(problem, learner, te_X)
        end

        costs = cost(problem, Z, te_Y)
        push!(df, (learners_names[il], mean(costs), trt, cvt, tet))
    end

    return df
end

"""
    r2(test_sampler, y_sampler, num_test_samples, num_y_samples)

Computes r2 of conditional expectation using 'test_sampler' to generate sets of observations (X, Y) and y_sampler(x) to generate samples of y from conditional distribution y | x. 'num_test_samples' is how many sets (X, Y) to generate, and 'num_y_samples' is how many samples of y to generate for each X[i, :] to estimate conditional expectation.
"""
function r2(test_sampler, y_sampler, num_test_samples, num_y_samples)
    X, Y = test_sampler()
    if num_test_samples > 1
        for i = 2:num_test_samples
            Xi, Yi = test_sampler()
            X = vcat(X, Xi)
            Y = vcat(Y, Yi)
        end
    end
    num = 0.0
    den = 0.0
    y_mean = vec(mean(Y, dims=1))
    for i = 1:size(X, 1)
        y_cond = zeros(size(Y, 2))
        for j = 1:num_y_samples
            y_cond += y_sampler(X[i, :])
        end
        y_cond /= num_y_samples
        num += sum((y_cond - Y[i, :]).^2)
        den += sum((y_mean - Y[i, :]).^2)
    end
    num /= size(X, 1)
    den /= size(X, 1)
    return 1 - num/den
end

end  # module DataDrivenOptimization
