#!/bin/bash

./levcuda ./assets/20k_a.txt ./assets/20k_b.txt
./levcuda ./assets/20k_a.txt ./assets/20k_b.txt -a ./assets/ascii_alphabet.txt
./levcuda ./assets/ascii20k_a.txt ./assets/ascii20k_b.txt -a ./assets/ascii_alphabet.txt
