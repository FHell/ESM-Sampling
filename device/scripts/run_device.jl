# Usage: julia --project=device -t 4 device/scripts/run_device.jl configs/device40.toml
#   FAMILIES=all (default: the fast plain families), ADAPT=1 (host adaptation: learned modes, shapes and
#   coefficients, the dense angle families), FLOAT=32 (the Float32 build, `ESMDevice.F32`)
using ESMSampling, ESMDevice
cfg = ESMSampling.load_config(ARGS[1])
D = get(ENV, "FLOAT", "64") == "32" ? ESMDevice.F32 : ESMDevice
fams = get(ENV, "FAMILIES", "fast") == "all" ? :all : D.FAMILIES_FAST
D.run_device(cfg; families=fams, adapt=get(ENV, "ADAPT", "0") == "1")
