# CUDA Generator Build Notes

The C++/CUDA generator lives in this directory. It is the intended Phase 2 accelerator once the toolchain is compatible.

## Current blocker on this Windows host

- Visual Studio 2022 Build Tools **19.44** requires **CUDA >= 12.4** for the bundled STL.
- Installed CUDA toolkit with VS integration: **12.1** only.
- CUDA **12.6** is present but lacks `extras/visual_studio_integration`.

## Recommended build paths

### Option A: WSL2 + Ubuntu (preferred in design doc)

```bash
sudo apt install build-essential cmake
# Install CUDA 12.x for WSL from NVIDIA
cd generator/cuda
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
./build/goldbach_generate --output ../../data/goldbach_train_v1.bin
```

### Option B: Upgrade Windows CUDA to >= 12.4 with VS integration

Install a CUDA toolkit version that matches MSVC 19.44, then:

```powershell
cmake -B build-vs -G "Visual Studio 17 2022" -A x64 -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
cmake --build build-vs --config Release
```

## Immediate fallback

Until CUDA builds locally, use the Python generator:

```powershell
cd generator/python
python generate.py --phase phase2
```

This produces the same `.bin` format and final filenames:

- `data/goldbach_train_v1.bin`
- `data/goldbach_manifest_v1.json`
