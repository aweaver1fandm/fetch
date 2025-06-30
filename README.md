# PyTorch FETCH
## NOTE THIS IS A WORK IN PROGRESS

**6/7/2025: 
Currently cleaning up code, adding some additional features and such in the devel branch.  I hope to release a stable version of this code within the next week**

This is a `PyTorch` version of FETCH forked from the original version found [here](https://github.com/devanshkv/fetch).

There are several key differences between this version and the original version of FETCH

1. The code only works for pre-trained models available in Pytorch Vision
    - Currenlty only DenseNet and VGG models have been transfer-trained
    - TorchVision has some but not all of the other models used (e.g. XCeption)
    - There are guidelines on how to extend this code to use other models
2. Predictions can be done based on just frequency data, just DM data, or both
3. Models must be downloaded for local use and are available through Globus [here]()
    - If you do not currenlty have a Globus login, you can get one for free
    - All models here were trained and tested using the original FETCH data available at [astro.phys.wvu.edu/fetch](http://astro.phys.wvu.edu/fetch/).
4. Training and predicting is done on .h5 files as before but to be clear
    - Files can contain single or multiple observations
    - Files must contain at data labeled as data_freq_time and data_dm_time
    - It may also contain data_labels for training data to indicate an observation is or is not a pulsar
    - For more details on the structure of the data, see the code in pulsar_data.py
5. There are some differences in training procedures and the
   models themselves (minor difference as far as I can tell) mostly due to differences between Keras/Tensorflow and Pytorch

Installation
---
Code:
    git clone 
    cd fetch
    python -m pip install .

Models: 
    Must be downloaded for local use.
    Models are accessible through Globus [here]()

Training
---
    
Predicting
---

Extending Pytorch FECTH
---
I have tried to add comments to the code of the form ```# TODO:``` that indicate where and what needs to be done to extend the code here to use other models.  Roughly you will need to do the following in model.py

1. Add the name and parameter size of the model to the PARAMS variable
    - If using a pre-built model from PyTorch vision, use that name (and as the code is written, case matters)
    - The parameter size is ???
2. Create a function to unfreeze model layers based on specifics of the model you are working with and update the TorchVisionModel
constructor to call your function

If you want to use another model, not part of PyTorch Vision, you will have build the model yourself, do the same things as above, but also

1. Import your model
2. Make adjustments to the TorchVisionModel constructor to set self.model to call your model instead of loading a PytorchVision model

Citing this work
---
