#!/bin/bash
# Runs dark disk analysis on all runfiles in folder

for file in runfile-*.py
do
    python3 darkdisk_reach.py $(basename "$file" .py)
done
