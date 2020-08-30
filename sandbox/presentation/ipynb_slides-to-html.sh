#! /bin/bash

file_to_convert=$1
jupyter nbconvert $file_to_convert --to slides --post serve
