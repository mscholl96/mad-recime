from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import csv
FILE_DIR = '/Users/maximilian/Desktop/'

url_file = FILE_DIR + 'websites_8.pkl'
out_file = FILE_DIR + 'amounts_8.csv'

food_com = pd.read_pickle(url_file)

print(f'Scraping {len(food_com)} websites...')

i = 0
for idx, website in food_com.iterrows():
    try:
        page = requests.get(website['url'], timeout=5)
        if page.status_code == 200:
            dom = BeautifulSoup(page.content, 'html.parser')
            ingredient_quantities = dom.find_all("div", {"class": "recipe-ingredients__ingredient-quantity"})
            quantities = []
            for quantity in ingredient_quantities:
                quantities.append(quantity.get_text())
            with open(out_file, mode='a') as f:
                w = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                w.writerow([idx, quantities])
        else:
            print(f'Page {website["url"]} returned error code: {page.status_code} for index {idx}')
    except BaseException as e:
        print(f'Exception occured for index {idx}: {e}')
    i+=1
    if i % 100 == 0:
        print(f'Progress: {i}')