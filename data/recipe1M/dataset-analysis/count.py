import re
import nltk
nltk.download('all')

charsToRemove = "[,.*®©™()?!:;+]"
fractionRegex = re.compile("[0-9]+/[0-9]+")
# Counts words in a given list (ingredients/instructions) and returns their occurence in a dictionary
# Filters numbers (such as 1 apple or 1/2 apple) and puts them into 'numeric' category
def countWords(list, words_dict):
    for elem in list:
        text = elem['text']
        strippedText = re.sub(charsToRemove, "", text.lower())
        for word in strippedText.split():
            if word.isnumeric() or re.match(fractionRegex, word):
                words_dict['numeric'] = words_dict.setdefault('numeric', 0) + 1
            else:
                words_dict[word] = words_dict.setdefault(word, 0) + 1
    return words_dict



def countNouns(list, nouns_dict):
    for elem in list:
        tokenized_text = nltk.pos_tag(nltk.word_tokenize(elem['text']))
        for word, kind in tokenized_text:
            if kind == 'NN':
                nouns_dict[word] = nouns_dict.setdefault(word, 0) + 1
    return nouns_dict