***

<div align="center">    

# ReciMe   
[GoogleDrive](https://drive.google.com/drive/folders/1Tv-HiBBj2SkGkX0ekYiT8spX6w1MTOaO?usp=sharing) | 
[Documentation](https://www.overleaf.com/7246658912hbddqhwtqptp)

Code coverage: ![Badge](doc/coverage.svg)

</div>
 
## Description   
ReciMe is a fancy recipe generator based on AI. At some point it will offer an iOS App with remarkable user experience.

## How to setup   
<!-- cd deep-learning-project-template 
pip install -e .    -->
Use Python 3.6 due to colab support (preferrably in conda)
 
 ```bash
# clone project   
git clone https://github.com/mscholl96/mad-recime.git

# install project   
pip install -r requirements.txt
 ```   
To execute the tests run:

 ```bash
# run the shell script
sh runTests.sh 
```

## Structure

The generator is split in two functional parts 
* a [CVAE network](https://github.com/mscholl96/mad-recime/blob/master/network/CVAE) computing the ingredients for the recipe
* two [LSTM networks](https://github.com/mscholl96/mad-recime/blob/master/network/LSTM) for title and instruction generation

## Citation   
```
@article{Recime2022},
  title={Ai based generation of cooking recipes},
  author={Wagner, Marcel and Scholl, Maximilian and Schatz, Hannes},
  year={2022}
}
```   