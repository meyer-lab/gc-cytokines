using Distributions
using BenchmarkTools

include("model.jl")

### Check that runCkine can run
params = [rand(LogNormal(0.1, 0.25)) for i=1:Nparams]

params[20] = tanh(params[20])*0.9

tps = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]

out, Sout = runCkine(tps, params, false, false)

### Check that runCkinePretreat can run
postStim = [rand(LogNormal(0.1, 0.25)) for i=1:6]

out = runCkinePretreat(10.0, 10.0, params, postStim)

### Check that at long times we come to equilibrium

