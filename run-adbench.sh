#!/bin/sh

set -e

cd ADBench
pwsh -command '& { ADBench/run-all.ps1 -buildtype "Release" '\
' -tools (echo Futhark Tapenade Manual) '\
' -gmm_sizes (echo 10k) -gmm_d_vals_param @(64) -gmm_k_vals_param @(200) '\
' -ba_min_n 5 -ba_max_n 5 '\
' -hand_sizes @("big") -hand_min_n 5 -hand_max_n 5 '\
' -lstm_l_vals @(4) -lstm_c_vals @(4096) '\
' }; exit $LASTEXITCODE'
