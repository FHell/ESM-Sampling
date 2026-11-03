# Usage: julia --project=device -t 4 device/scripts/run_device.jl configs/device40.toml
using ESMSampling, ESMDevice
cfg = ESMSampling.load_config(ARGS[1])
ESMDevice.run_device(cfg)
