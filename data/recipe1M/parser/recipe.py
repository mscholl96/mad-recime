

from quantulum3 import parser as qparse
from pattern.text.en import singularize
import pandas as pd
       
class Recipe:
    """Class to represent a single recipe"""
    def __init__(self, id):
        self.title = ""
        self.id = id
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
                ingredient['amount'] = quants[0].value
                unit = quants[0].unit.name
                if not entity in 'volume, mass, length':
                    # c. parses to centavo or cent, cup cubed to cubic cup
                    if 'centavo' in unit or 'cent' in unit or 'cup' in unit:
                        ingredient['unit']= 'cup'
                    # pinch (Prise) parses to pint inch
                    elif unit == 'pint inch':
                        ingredient['unit']= 'pinch'
                    # pkg. parses to peck gram
                    elif unit == 'peck gram':
                        ingredient['unit']= 'package'
                    else:
                        ingredient['unit']= ''
                else:
                    ingredient['unit']= unit
            i+=1