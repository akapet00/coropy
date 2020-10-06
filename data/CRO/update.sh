#! /bin/bash

python scrape_worldometers.py  # number of confirmed, recovered and deceased
python scrape_koronavirushr.py  # number of daily tests
git add *.dat
git commit -m "update CRDT data - automated daily update"
git push