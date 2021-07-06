#!/bin/bash
# Runs subhalo analysis on all runfiles in folder

for file in runfile-*.py
do
    python3 subhalo_reach.py $(basename "$file" .py)
done
