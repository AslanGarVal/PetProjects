using Flux, DelimitedFiles, Plots, MLDataUtils, StatsBase, CUDA
using RDatasets

CUDA.allowscalar(false)

data = RDatasets.dataset("ISLR", "Default")

todigit(x) = x == "Yes" ? 1. : 0.
data[!, :Default] = map(todigit, data[:, :Default])
data[!, :Student] = map(todigit, data[:, :Student])

target = :Default
numerics = [:Balance, :Income]
features = [:Student, :Balance, :Income]
train, test = shuffleobs(data) |>
    d -> stratifiedobs(first, d, p=0.7)

for feature in numerics
    μ, σ = rescale!(train[!, feature], obsdim=1)
    rescale!(test[!, feature], μ, σ, obsdim=1)
end

prep_X(x) = Matrix(x)' |> gpu
prep_y(y) = reshape(y, 1, :) |> gpu
train_X, test_X = prep_X.((train[:, features], test[:, features]))
train_y, test_y = prep_y.((train[:, target], test[:, target]))

model = Chain(Dense(length(features), 50, relu), Dense(50, 1, σ)) |> gpu

predict(x; thres = .5) = model(x) .> thres
accuracy(x, y) = mean(cpu(predict(x)) .== cpu(y))

loss(x, y) = mean(Flux.Losses.binarycrossentropy(model(x), y))

evalcb() = @show(loss(train_X, train_y), accuracy(test_X, test_y))

train_dl = Flux.Data.DataLoader(gpu.(collect.((train_X, train_y))); batchsize = 128, shuffle = true)

Flux.@epochs 5 Flux.train!(loss, params(model), train_dl, ADAM(0.01), cb = evalcb)

sum(model(test_X) .> 0.5)