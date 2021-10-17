# Multimodal Transformer With Learnable Frontend and Self Attention for Emotion Recognition 

This repo contains the code for detecting emotion from the conversational dataset IEMOCAP for the implementation of the paper "Multimodal Transformer With Learnable Frontend and Self Attention for Emotion Recognition" submitted to ICASSP 2022. This repository contains the code when Session 5 is conisdered as test and Session 1 as validation.

## Description of the code
- The implementation has three stages, namely, training the unimodal audio and text models, training the Bi-GRU with self-attention and the multimodal transformer
- With the **wav** files for the audio and the **csv** files for text, the first step would be to run **audio_model.py** and the notebook **sentiment_text.ipynb** for audio and text respectively
- The representations from the trained models in the step above are used to create pickle files for the entire dataset
- With these representations, two Bi-GRU models with self-attention (refer to **bigru_audio/text.ipynb**) is trained. The best models for both audio and text are already provided in the **unimodal_models** folder. 
- A multimodal transformer is trained on both the modalities of the dataset for the final accuracy results
- Please note that usage of **IEMOCAP** requires permission. Once this is done, we can share the dataset files. For permission please visit [IEMOCAP release](https://sail.usc.edu/iemocap/iemocap_release.htm)

## Running the code
- Clone the repository `https://github.com/iiscleap/multimodal_emotion_recognition.git`
- For the LEAF-CNN framework for audio sentiment classification, we use [this Pytorch implementation](https://github.com/denfed/leaf-audio-pytorch) for LEAF.
  - Run ```python3 -m venv .leaf_venv```
  - Run ```source .leaf_venv/bin/activate```
  - Run ```pip install -r requirements_leaf.txt```
  - Clone the repository ```https://github.com/denfed/leaf-audio-pytorch.git``` 
  - The files should be arranged as follows:
      ```
    Sess5
    └───leaf_wavs_train
        └───Ses01F_impro01_F000.wav
        └───Ses01F_impro01_F005.wav
        └───...
    └───leaf_wavs_test
        └───Ses05F_impro01_F000.wav
        └───Ses05F_impro01_F008.wav
        └───...
    label_dict.json
    leaf-audio-pytorch-main
    │
    └───__init__.py
    └───setup.py
    └───...
    └───audio_model.py
    └───leaf_audio_pytorch
        └───...
    ```
- Running **sentiment_text.ipynb** provides the text unimodal model
- For running the two Bi-GRU models with self-attention, run **bigru_audio.ipynb** to get ```best_model_aud0.tar``` and **bigru_text.ipynb** to get ```best_model_text0.tar```. These are to be placed in the folder **unimodal_models**.
- For running the multimodal transformer we create another environment
  - Run ```python3 -m venv .trans_venv```
  - Run ```source .trans_venv/bin/activate```
  - Run ```pip install -r requirements.txt```
  - With the config file provided, run ```python3 main.py```
  - The files at this stage should be arranged as follows:
  ```
    main.py
    config.py
    features
    └───<PICKLE_FILE>
    src
    └───model_lstm_tranformers.py
    └───read_data.py
    └───test_lstm_transformers.py
    └───...
    unimodal_models
    └───best_model_aud0.tar
    └───best_model_text0.tar
    ```




