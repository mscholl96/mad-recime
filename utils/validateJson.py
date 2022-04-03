# Script to validate the json format across multiple scripts
# Json structure was aggreed and defined in https://github.com/mscholl96/mad-recime/issues/5
# [
#   {
#     "title": "string",
#     "ingredients": [
#       {
#         "ingredient": "string",
#         "amount": "number",
#         "unit": "string",
#         "instruction": "string"
#       }
#     ],
#     "instructions": [
#       {
#         "instruction": "string"
#       }
#     ]
#   }
# ]

import json
import jsonschema
from jsonschema import validate


recimeScema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "ingredients": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ingredient": {"type": "string"},
                        "amount": {"type": "number"},
                        "unit": {"type": "string"},
                        "instruction": {"type": "string"},
                    },
                },
            },
            "instructions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"instruction": {"type": "string"}},
                },
            },
        },
    },
}


def validateRecimeJson(jsonData):
    """Function to validate a loaded json file against the recime json schema.

    :param jsonData: Loaded Json string (i.e. return value from json.loads)

    :raised jsonschea.exceptions.ValidationError: Error is risen if the given input does not match the schema.
    :return: True in case the validation was sucessful, false otherwise.
    """
    try:
        validate(instance=jsonData, schema=recimeScema)
    except jsonschema.exceptions.ValidationError as err:
        return False
    return True
