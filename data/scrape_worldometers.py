import requests
from bs4 import BeautifulSoup

URL = 'https://www.worldometers.info/coronavirus/country/croatia/'

def main():
    response = requests.get(URL)
    soup = BeautifulSoup(response.text, 'html.parser')
    maincounters = soup.find_all('div', class_='maincounter-number')

    cdr = [] # list of number of confirmed, deceased and recovered cases
    for maincounter in maincounters:
        nmb = maincounter.find('span').get_text()
        nmb = nmb.replace(',', '') # remove commas
        nmb = nmb.replace(' ', '') # remove whitespaces
        cdr.append(nmb)

    files = ['confirmed_cases.dat', 'death_cases.dat', 'recovered_cases.dat']
    for idx, file in enumerate(files):
        with open(file, 'a') as f:
            f.write(f'\n{cdr[idx]}')
    
if __name__=='__main__':
    main()