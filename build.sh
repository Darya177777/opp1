#!/bin/bash
set -x

mpicc -std=gnu99 -O3 -Wpedantic -Wall -Werror -o itog.out par3.c
