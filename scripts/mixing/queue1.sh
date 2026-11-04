#!/bin/sh
# Capacity experiments, one after another (nice 19, 4 threads).
cd /home/hellmann/Repos/ESM-Sampling
J="nice -n 19 julia --project=. -t 4 scripts/mixing/run_exp.jl"
$J shared_base 300 none fine_weather=0.0 > results/mixing/shared_base.log 2>&1
$J shared_kop 300 kop fine_weather=0.0 > results/mixing/shared_kop.log 2>&1
$J own_kop 300 kop > results/mixing/own_kop.log 2>&1
$J shared_util_kop 300 kop fine_weather=0.0 measure=utilization > results/mixing/shared_util_kop.log 2>&1
