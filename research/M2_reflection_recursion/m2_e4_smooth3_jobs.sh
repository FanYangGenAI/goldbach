#!/bin/bash
for A in 24 48; do
  python3 m2_e4_random_model.py smooth3 101 $A 2000 $((101+A+7))
  python3 m2_e4_random_model.py smooth3 307 $A 2000 $((307+A+7))
  python3 m2_e4_random_model.py smooth3 1009 $A 1000 $((1009+A+7))
  python3 m2_e4_random_model.py smooth3 3001 $A 300 $((3001+A+7))
done
