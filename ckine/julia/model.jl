using DifferentialEquations

include("reaction.jl")

function runCkine(tps, params, sensi, IL2case)
	if IL2case
		ILs, surface, endosome, trafP = IL2param(params)
		f = IL2Deriv
	else
		ILs, surface, endosome, trafP = fullParam(params)
		f = fullDeriv
	end

	u0 = solveAutocrine(trafP)

	if sensi
		prob = ODELocalSensitivityProblem(f, u0, (0.0, maximum(tps)), params)
	else
		prob = ODEProblem(f, u0, (0.0, maximum(tps)), params)
	end

	sol = solve(prob, CVODE_BDF())

	if sensi
		# TODO: Fix tps handling
		return extract_local_sensitivities(sol, tps[2])
	else
		return sol(tps), nothing
	end
end


function runCkinePretreat(pret, tt, params, postStim)
	ILs, surface, endosome, trafP = fullParam(params)

	u0 = solveAutocrine(trafP)

	prob = ODEProblem(fullDeriv, u0, (0.0, pret), params)

	sol = solve(prob, CVODE_BDF())

	# Set ILs in params
	params[1:6] = postStim

	prob = ODEProblem(fullDeriv, u0, (0.0, tt), params)

	sol = solve(prob, CVODE_BDF())

	return sol(tt)
end


function runCkineParallel(rxnRatesIn, tp, sensi)
	#u0 = solveAutocrine(trafP)
	#tspan = (0.0, max(tps))

	#if sensi
	#	prob = ODELocalSensitivityProblem(f, u0, tspan, p)
	#else
	#	prob = ODEProblem(f, u0, tspan, p)
	#end

	#sol = solve(prob, Tsit5())
	#t = collect(tps)

	return out, sensiOut
end
