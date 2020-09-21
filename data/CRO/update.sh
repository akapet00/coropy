#! /bin/bash

python scrape_worldometers.py
git add *.dat
git commit -m "update CRD data - automated daily update"
git push

