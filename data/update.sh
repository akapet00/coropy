#! /bin/bash

python scrape_worldometers.py
git add .
git commit -m "update data - automated daily update"
git rebase origin/master 
git push
