#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 [1/10/20/30 - thousand target string length] [1/10/20/30]"
  exit 1
fi

./levcuda ./assets/"$1"k_a.txt ./assets/"$2"k_b.txt
