#!/bin/bash

FILE=$1
tail $FILE -n +2 | head -n 100000000 | sort -k2,3 -t, -u | sort -k1 -n -t, > $(dirname $FILE)/unique_100M.csv
