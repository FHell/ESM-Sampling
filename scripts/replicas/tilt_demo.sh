#!/bin/sh
# Stress-tilt demonstration (ledger E24): after the high run, continue the low end state for 50
# steps at a fixed step ηw = 2e-4, with λ = 0 and λ = 2 (the same random streams).
cd "$(dirname "$0")/../.."
until grep -q "done:\|rror" results/replicas/rep_high.log 2>/dev/null; do sleep 15; done
sleep 5
for lam in 0.0 2.0; do
    nice -n 19 julia --project=. -t 4 scripts/replicas/run_replicas.jl low 50 results/replicas/tilt_lam$lam.jls 1 $lam \
        results/replicas/rep_low.jls 2e-4 > results/replicas/tilt_lam$lam.log 2>&1
    tail -1 results/replicas/tilt_lam$lam.log
done
