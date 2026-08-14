@echo off
echo ===================================================
echo     SemiconDaAIR-v5 BENCHMARK & TEST SUITE
echo ===================================================
python tools/system_info.py
python tools/inspect_model.py
python tools/validate_checkpoint.py
python scripts/benchmark.py
python tests/test_inference.py
pause
