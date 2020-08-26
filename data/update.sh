#! /bin/bash

python scrape_worldometers.py
git add *.dat
git commit -m "update data - automated daily update"
git push
