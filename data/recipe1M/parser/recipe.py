

from quantulum3 import parser as qparse
from pattern.text.en import singularize
import pandas as pd
       
class Recipe:
    """Class to represent a single recipe"""
    def __init__(self, id, title):
        self.id = id
        self.title = title
        self.ingredients = []
        self.instructions = []

    def __str__(self):
        return f"ID: {self.id} \nTitle: {self.title} \nIngredients: {str(self.ingredients)} \nInstructions {self.instructions}"
    
    def __repr__(self):
        return str(self)

    def to_dict(self):
        return {'id': self.id, 'title': self.title, 'ingredients': pd.DataFrame(self.ingredients), 'instructions': pd.Series(self.instructions)}

    def parse_ingredients(self, raw_ingredients):
        for elem in raw_ingredients:
            self.ingredients.append({'amount': 1, 'unit':'', 'ingredient': singularize(elem['text']).lower()})

    def parse_instructions(self, raw_instructions):
        for elem in raw_instructions:
            self.instructions.append(elem['text'])
    
    def get_ingredient_amounts(self, raw_ingredients):
        i = 0
        for ingredient in self.ingredients:
            # Find corresponding ingredient text in raw recipe and parse
            quants = qparse.parse(raw_ingredients[i]['text'].lower())
            if quants:
                entity = quants[0].unit.entity.name
                # Sometimes 0 is parsed, when there is no number in text, the implicit amount is 1 in these cases
                ingredient['amount'] = quants[0].value if quants[0].value > 0 else 1
                unit = quants[0].unit.name
                # Convert to millilitre
                if unit == 'cubic centimetre':
                    ingredient['unit'] = 'millilitre'
                # c. parses to centavo or cent, cup cubed to cubic cup
                elif 'cup' in unit:
                    ingredient['unit'] = 'cup'
                # pinch (Prise) parses to pint inch
                elif unit == 'pint inch':
                    ingredient['unit'] = 'pinch'
                # pkg. parses to peck gram or peck, ct (carton) parses to carat
                elif 'peck' in unit or unit == 'carat':
                    ingredient['unit'] = 'package'
                # parsed from "2T" (tablespoon)
                elif unit == 'tonne':
                    ingredient['unit'] = 'tablespoon'
                # parsed from "inch cubed ..."
                elif unit == 'cubic inch':
                    ingredient['unit'] = 'inch'
                elif not entity in 'volume, mass, length':
                    if 'centavo' in unit or 'cent' in unit:
                        ingredient['unit'] = 'cup'
                    ingredient['unit'] = ''
                else:
                  ingredient['unit'] = unit
            i+=1