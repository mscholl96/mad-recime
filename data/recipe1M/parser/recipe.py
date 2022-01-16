

from quantulum3 import parser as qparse
import pandas as pd
       
class Recipe:
    """Class to represent a single recipe"""
    def __init__(self, id):
        self.title = ""
        self.id = id
        self.ingredients = pd.DataFrame(columns=['amount', 'unit', 'ingredient'])
        self.instructions = pd.Series(dtype=str)

    def __str__(self):
        return f"ID: {self.id} \nTitle: {self.title} \nIngredients: {str(self.ingredients)} \nInstructions {self.instructions}"
    
    def __repr__(self):
        return str(self)

    def parse_ingredients(self, raw_ingredients):
        # Ignore if more than 20
        if len(raw_ingredients) > 20:
            return False
        for elem in raw_ingredients:
            self.ingredients = self.ingredients.append({'amount': 1, 'unit':'', 'ingredient': elem['text']}, ignore_index=True)
        return True

    def parse_instructions(self, raw_instructions):
        # Ignore more than 30
        if len(raw_instructions) > 30:
            return False
        for elem in raw_instructions:
            self.instructions = self.instructions.append(pd.Series(elem['text'], dtype=str), ignore_index=True)
        return True
    
    def get_ingredient_amounts(self, raw_ingredients):
        for _, ingredient in self.ingredients.iterrows():
            # Find corresponding ingredient text in raw recipe
            for raw in raw_ingredients:
                text = raw['text'].lower()
                if ingredient.ingredient in text:
                    quants = qparse.parse(text)
                    if len(quants) != 0:
                        entity = quants[0].unit.entity.name
                        ingredient.amount = quants[0].value
                        unit = quants[0].unit.name
                        # c. parses to centavo or cent, cup cubed to cubic cup
                        if entity == 'currency' or unit == 'cubic cup':
                          ingredient.unit = 'cup'
                        # pinch (Prise) parses to pint inch
                        elif unit == 'pint inch':
                          ingredient.unit = 'pinch'
                        # pkg. parses to peck gram
                        elif unit == 'peck gram':
                          ingredient.unit = 'package'
                        # Everything else is mostly dimensionless
                        elif unit == 'dimensionless' or not entity in 'volume, mass, length':
                          ingredient.unit = ''
                        else:
                          ingredient.unit = unit