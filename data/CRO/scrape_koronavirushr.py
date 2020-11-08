import requests


URL = 'https://koronavirus.hr'
TARGETS = ['testirano je njih', 'testirane su']


def main():
    x = requests.get(URL)
    for TARGET in TARGETS:
        target_idx = x.text.find(TARGET)
        if target_idx != -1:
            daily_tests_idx = (target_idx + len(TARGET), target_idx + len(TARGET) + 6)
            daily_tests_nmb = int(
                x.text[daily_tests_idx[0]:daily_tests_idx[1]].replace('.', ''))
            with open('tests.dat', 'a') as f:
                f.write(f'\n{daily_tests_nmb}')


if __name__ == "__main__":
    main()
