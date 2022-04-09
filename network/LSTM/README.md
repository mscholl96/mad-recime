***
# Title and Instruction Generation

## Network
LSTM (Long Short Term Memory) networks, same architecture used for both title and instruction generation.


## Project Structure

The initial implementation is done in PyTorch but due to the more intuitive framework, this is to be moved to Pytorch Lightning.
```
torch/
├─instrGen.ipynb
├─logs/
│ ├─instruction/
│ └─title/
├─model/
│ ├─instruction/
│ └─title/
├─old_nbs/
│ ├─instrGen.ipynb
│ └─titleGen.ipynb
├─preProc.ipynb
├─sample.ipynb
├─src/
│ ├─__init__.py
│ ├─__pycache__/
│ ├─data.py
│ └─models.py
└─titleGen.ipynb
```

The functional parts are located in <code><em>model.py</em></code> and <code><em>data.py</em></code>.

Execution of the training is handled in the respective notebooks <code><em>instrGen.ipynb</em></code> and <code><em>titleGen.ipynb</em></code>.

Preprocessing can be run with <code><em>preProc.ipynb</em></code>.

A prediction with an already trained model can be done using <code><em>sample.ipynb</em></code>.

## Training execution

```python
import src.data as dataLib
import src.models as modelLib

# hyperparams
hp = modelLib.HyperParams(epochs=30, batchSize=32, hiddenDim=256, numLayers=1, embeddingDim=300)

# dataset
dataSet = dataLib.InstructionSet(dataPath, setSize=2)

# model
model = modelLib.EmbedLSTM(hp, dataSet.tokenizer, device)

# train
modelLib.train(dataSet, model, hp, device, logDir)
```