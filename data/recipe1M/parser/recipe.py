

from quantulum3 import parser as qparse
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
            self.ingredients.append({'amount': 1, 'unit':'', 'ingredient': elem['text'].lower()})

    def parse_instructions(self, raw_instructions):
        for elem in raw_instructions:
            self.instructions.append(elem['text'])
    
    def get_ingredient_amounts(self, raw_ingredients):
        i = 0
        for ingredient in self.ingredients:
            # Find corresponding ingredient text in raw recipe and parse
            quants = qparse.parse(raw_ingredients[i]['text'].lower())
            i+=1
            
            if quants:
                # Sometimes there are several quantities filtered from string. If possible select one, that is not dimensionless
                quantity = quants[0]
                for elem in quants[1:]:
                    if quantity.unit.name == 'dimensionless' and elem.unit.name != 'dimensionless':
                        quantity = elem
                    else:
                        # Break if non dimensionless quantity has been found
                        break
                entity = quantity.unit.entity.name
                # Sometimes 0 is parsed, when there is no number in text, the implicit amount is 1 in these cases
                ingredient['amount'] = quantity.value if quantity.value > 0 else 1
                unit = quantity.unit.name
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
                    # c. parses to centavo or cent, cup cubed to cubic cup
                    if 'centavo' in unit or 'cent' in unit:
                        ingredient['unit'] = 'cup'
                    ingredient['unit'] = ''
                else:
                  ingredient['unit'] = unit