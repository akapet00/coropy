import requests


URL = 'https://koronavirus.hr'
TARGET = 'testirano je'


def main():
    x = requests.get(URL)
    target_idx = x.text.find(TARGET)
    daily_tests_idx = (target_idx + len(TARGET), target_idx + len(TARGET) + 6)
    daily_tests_nmb = int(
        x.text[daily_tests_idx[0]:daily_tests_idx[1]].replace('.', ''))
    with open('tests.dat', 'a') as f:
            f.write(f'\n{daily_tests_nmb}')


if __name__ == "__main__":
    main()