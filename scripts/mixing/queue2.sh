#!/bin/sh
# Second batch: α = 1 (level-0 bias, frozen) with the utilization measure; tests of the kpeak
# slice-width fix (src/adaptive.jl); storage multiscale moves with kop at shared weather.
cd /home/hellmann/Repos/ESM-Sampling
J="nice -n 19 julia --project=. -t 4 scripts/mixing/run_exp.jl"
$J a1u_kop 200 kop use_bias=true measure=utilization > results/mixing/a1u_kop.log 2>&1
$J a1u_base 200 none use_bias=true measure=utilization > results/mixing/a1u_base.log 2>&1
nice -n 19 julia --project=. -t 4 test/runtests.jl > results/mixing/test_runtests.log 2>&1
nice -n 19 julia --project=. -t 4 test/test_exact.jl > results/mixing/test_exact.log 2>&1
$J shared_kop_sms 200 kop,sms fine_weather=0.0 > results/mixing/shared_kop_sms.log 2>&1
$J shared_kop 200 kop fine_weather=0.0 > results/mixing/shared_kop.log 2>&1
