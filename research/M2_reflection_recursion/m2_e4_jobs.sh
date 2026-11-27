#!/bin/bash
# core 1: big q=10007 qa (slow)
( python3 m2_e4_random_model.py qa 10007 6 200 10007 > m2_e4_out_core1.txt ) &
# core 2: everything else
(
for q in 101 307; do python3 m2_e4_random_model.py qa $q 6 4000 $q; done
python3 m2_e4_random_model.py qa 1009 6 2000 1009
python3 m2_e4_random_model.py qa 3001 6 600 3001
for q in 101 307; do python3 m2_e4_random_model.py plain $q 6 4000 $q; done
python3 m2_e4_random_model.py plain 1009 6 2000 1009
python3 m2_e4_random_model.py plain 3001 6 600 3001
for A in 2 3 4 8; do python3 m2_e4_random_model.py qa 1009 $A 2000 1009; done
) > m2_e4_out_core2.txt &
wait
