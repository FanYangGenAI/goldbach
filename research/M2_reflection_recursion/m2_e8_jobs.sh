#!/bin/bash
( for q in 101 307 1009 3001 10007 30011; do case $q in 101) S=4000;; 307) S=3000;; 1009) S=2000;; 3001) S=1000;; 10007) S=400;; 30011) S=300;; esac; python3 m2_e8_trend.py $q 1 $S $((q+100)); done > m2_e8_out_A1.txt ) &
( for q in 101 307 1009 3001 10007; do case $q in 101) S=4000;; 307) S=3000;; 1009) S=2000;; 3001) S=1000;; 10007) S=400;; esac; python3 m2_e8_trend.py $q 2 $S $((q+200)); done > m2_e8_out_A2.txt ) &
wait
