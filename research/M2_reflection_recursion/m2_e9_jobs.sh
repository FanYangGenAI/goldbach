#!/bin/bash
( for q in 101 307 1009 3001 10007; do case $q in 101) S=4000;; 307) S=3000;; 1009) S=2000;; 3001) S=800;; 10007) S=300;; esac; python3 m2_e8_trend.py $q eta0.25 $S $((q+1000)); done > m2_e9_out_eta025.txt ) &
( for q in 101 307 1009; do case $q in 101) S=3000;; 307) S=2000;; 1009) S=600;; esac; python3 m2_e8_trend.py $q eta0.5 $S $((q+1000)); done > m2_e9_out_eta05.txt ) &
wait
