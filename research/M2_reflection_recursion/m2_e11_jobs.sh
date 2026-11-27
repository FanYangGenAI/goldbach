#!/bin/bash
( python3 m2_e11_paths.py 101 0.5 1000 3101; python3 m2_e11_paths.py 307 0.5 600 3307; python3 m2_e11_paths.py 1009 0.5 300 4009 ) > m2_e11_out_eta05.txt &
( python3 m2_e11_paths.py 307 0.25 1000 3307; python3 m2_e11_paths.py 1009 0.25 600 4009; python3 m2_e11_paths.py 3001 0.25 400 6001; python3 m2_e11_paths.py 10007 0.25 150 13007 ) > m2_e11_out_eta025.txt &
wait
