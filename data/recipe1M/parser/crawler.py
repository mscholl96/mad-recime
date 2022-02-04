from time import process_time_ns
from bs4 import BeautifulSoup
import requests

page = requests.get('https://www.food.com/recipe/crunchy-onion-potato-bake-479149')
if page.status_code == 200:
    content = page.content
else:
    print(f'Page returned error code: {page.status_code}')

dom = BeautifulSoup(content, 'html.parser')

print(f'Website title: {dom.title.string}')

ingredient_quantities = dom.find_all("div", {"class": "recipe-ingredients__ingredient-quantity"})
print(len(ingredient_quantities))

for quantity in ingredient_quantities:
    sup = quantity.find('sup')
    sub = quantity.find('sub')
    # sup and sub are the tags of the fraction numbers, if they are None, there is no fraction inside the field
    if sup and sub:
        # Normal fraction, such as 1/4
        if len(quantity.contents) == 3:
            print(f'Amount: {sup.string}/{sub.string}')
        # Number and fraction, such as 1 1/2
        elif len(quantity.contents) == 4:
            print(f'Amount: {quantity.contents[0]}{sup.string}/{sub.string}')