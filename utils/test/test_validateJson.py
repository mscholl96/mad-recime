import json
import pytest
from utils import validateJson

@pytest.mark.monitor_skip_test
def test_validateJson_positive():
    testJson = """
    [
        {
            "title": "Pink Sangria",
            "ingredients": [
            {
                "ingredient": "rose wine",
                "amount": 750,
                "unit": "ml",
                "instruction": "chilled"
            },
            {
                "ingredient": "brandy",
                "amount": 14,
                "unit": "cup",
                "instruction": ""
            },
            {
                "ingredient": "orange liqueur",
                "amount": 14,
                "unit": "cup",
                "instruction": ""
            },
            {
                "ingredient": "white cranberry juice",
                "amount": 2,
                "unit": "cup",
                "instruction": ""
            },
            {
                "ingredient": "orange",
                "amount": 1,
                "unit": "",
                "instruction": "sliced"
            },
            {
                "ingredient": "lemon",
                "amount": 1,
                "unit": "",
                "instruction": "sliced"
            },
            {
                "ingredient": "Sprite",
                "amount": 355,
                "unit": "ml",
                "instruction": ""
            }
            ],
            "instructions": [
            {
                "instruction": "In a pitcher, combine all the ingredients except the soft drink and ice cubes."
            },
            {
                "instruction": "Refrigerate for 1 hour."
            },
            {
                "instruction": "Add the soft drink and ice just before serving."
            }
            ]
        }
    ]
    """
    jsonData = json.loads(testJson)
    assert validateJson.validateRecimeJson(jsonData) == True

@pytest.mark.monitor_skip_test
def test_validateJson_negative():
    testJson = """
    [
        {
            "title": "Pink Sangria",
            "ingredients": [
            {
                "ingredient": "rose wine",
                "amount": "750",
                "unit": "ml",
                "instruction": "chilled"
            },
            {
                "ingredient": "brandy",
                "amount": 14,
                "unit": "cup",
                "instruction": ""
            },
            {
                "ingredient": "orange liqueur",
                "amount": 14,
                "unit": "cup",
                "instruction": ""
            },
            {
                "ingredient": "white cranberry juice",
                "amount": 2,
                "unit": "cup",
                "instruction": ""
            },
            {
                "ingredient": "orange",
                "amount": 1,
                "unit": "",
                "instruction": "sliced"
            },
            {
                "ingredient": "lemon",
                "amount": 1,
                "unit": "",
                "instruction": "sliced"
            },
            {
                "ingredient": "Sprite",
                "amount": 355,
                "unit": "ml",
                "instruction": ""
            }
            ],
            "instructions": [
            {
                "instruction": "In a pitcher, combine all the ingredients except the soft drink and ice cubes."
            },
            {
                "instruction": "Refrigerate for 1 hour."
            },
            {
                "instruction": "Add the soft drink and ice just before serving."
            }
            ]
        }
    ]
    """
    jsonData = json.loads(testJson)
    assert validateJson.validateRecimeJson(jsonData) == False
