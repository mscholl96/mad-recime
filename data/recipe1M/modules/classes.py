

from quantulum3 import parser as qparse
import re
import json
# import nltk
# nltk.download('all')

ingredient_data = json.load(open('../det_ingrs.json'))

class Ingredient:
    """Class to represent a single ingredient"""
    def __init__(self):
        self.amount = 1
        self.unit = ""
        self.ingredient = ""

    def __str__(self):
        return f"({self.amount}; {self.unit}; {self.ingredient})"
    
    def __repr__(self):
        return str(self)

        
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

    def parse_ingredients(self, raw_ingredients):
        if len(raw_ingredients) <= 20:
            for elem in raw_ingredients:
                ingredient = Ingredient()
                ingredient.ingredient = elem['text']
                self.ingredients.append(ingredient)

    def parse_instructions(self, raw_instructions):
        if len(raw_instructions) <= 30:
            for elem in raw_instructions:
                self.instructions.append(elem['text'])
    
    def get_ingredient_amounts(self, raw_ingredients):
        for ingredient in self.ingredients:
            # Find corresponding ingredient text in raw recipe
            for raw in raw_ingredients:
                text = raw['text'].lower()
                if ingredient.ingredient in text:
                    quants = qparse.parse(text)
                    if len(quants) != 0:
                        ingredient.unit = "" if quants[0].unit.name == 'dimensionless' else quants[0].unit.name
                        ingredient.amount = quants[0].value
