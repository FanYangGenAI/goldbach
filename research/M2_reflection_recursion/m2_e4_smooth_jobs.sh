#!/bin/bash
for A in 24 12 6; do
  python3 m2_e4_random_model.py smooth 101 $A 2000 $((101+A))
  python3 m2_e4_random_model.py smooth 307 $A 2000 $((307+A))
  python3 m2_e4_random_model.py smooth 1009 $A 1000 $((1009+A))
  python3 m2_e4_random_model.py smooth 3001 $A 300 $((3001+A))
done
