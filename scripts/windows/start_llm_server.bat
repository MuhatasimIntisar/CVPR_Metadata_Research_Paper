@echo off
REM Starts llama.cpp's server for scripts/expand_search_pools.py (test_env laptop).
REM Edit the paths below for another machine. Leave this window open while expanding.
REM   -ngl 99   offload all layers to the GPU (CUDA build)
REM   --jinja   use the model's chat template (needed to switch off Qwen3/3.5 thinking)

set LLAMA_DIR=C:\msc_ds_ai\AI Engineering\Assignment\llama-b9999-bin-win-cuda-12.4-x64
REM Qwen3.5-4B Q8_0 (Feb 2026, near-lossless): ~4.5 GB weights + small context = fits 6 GB.
REM Text only here, so the vision projector (mmproj) is not needed.
REM Previous model: qwen3-1.7b.Q4_K_M.gguf (same folder).
set MODEL=C:\msc_ds_ai\AI Engineering\Assignment\Qwen3.5-4B-Q8_0.gguf
set PORT=8080

"%LLAMA_DIR%\llama-server.exe" -m "%MODEL%" --port %PORT% -c 4096 -ngl 99 --jinja
