#!/bin/bash

srun --comment="Integral" --input=0 --ntasks=$1 ./a.out
