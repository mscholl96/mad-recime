from network.CVAE.src.ReciMePreprocessor import ReciMePreprocessor
import pandas as pd
import pickle
from memory_profiler import profile

@profile
def main():
    preProcessor = ReciMePreprocessor("network/CVAE/data/vocab.bin")

    dataPath = 'network/CVAE/data/'
    with open(dataPath + 'recipes_valid.pkl', 'rb') as f:
        pklData = pd.DataFrame(pickle.load(f))

    output = preProcessor.preProcessInput(pklData.iloc[:1000]['ingredients'])

    print(len(output))

if __name__ == '__main__':
    main()