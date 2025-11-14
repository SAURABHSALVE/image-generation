@echo off
echo ========================================
echo  AI Image Demo - SMALLEST MODEL
echo ========================================
echo.
echo Model: 500MB only!
echo Download: 1-2 minutes
echo Generation: 15-30 seconds
echo.
echo PERFECT FOR DEMOS!
echo.

set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2

streamlit run app_demo.py

pause
