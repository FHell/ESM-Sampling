#!/bin/sh
cd /home/hellmann/Repos/ESM-Sampling
J="nice -n 19 julia --project=. -t 4"
$J scripts/mixing/run_a1.jl 150 300 > results/mixing/a1u.log 2>&1
$J test/runtests.jl > results/mixing/test_runtests.log 2>&1
$J test/test_exact.jl > results/mixing/test_exact.log 2>&1
$J scripts/mixing/run_exp.jl shared_kop_sms 200 kop,sms fine_weather=0.0 > results/mixing/shared_kop_sms.log 2>&1
$J scripts/mixing/run_exp.jl shared_kop 200 kop fine_weather=0.0 > results/mixing/shared_kop.log 2>&1
