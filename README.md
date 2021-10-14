# Multimodal Transformer With Learnable Frontend and Self Attention for Emotion Recognition 
Code for multimodal emotion recognition in IEMOCAP for Session 5 as test

This repo contains the code for detecting emotion from the conversational dataset IEMOCAP. For running the code, please refer to the **config.py** file. The important
contents of this file are as follows:

- pickle_path: contains the path of the pickle file for the dataset
- NUM_HIDDEN_LAYERS: number of hidden layers in the transformer
- NUM_ATTENTION_HEADS: number of attention heads in the transformer
- HIDDEN_SIZE: Dimension of hidden layers in transformer
- AUDIO_DIM: Dimension of the audio vectors from the pickle file
- TEXT_DIM: Dimension of the text vectors from the pickle file
- GRU_DIM: Dimension of the Bi-GRU with attention output
- USE_TEXT: Set to true if text modality is used
- USE_AUDIO: Set to true if audio modality is used
- USE_IMAGE: Set to true if image modality is used
- USE_GRU: Set to true for the required model
- LR: Learning rate
- EPOCHS: Number of epochs for training


After changing the config parameters according to requirement, run **main.py**. Ensure that **line 144 reads main(True, True, True)**. The model will be trained
and the most accurate model will be saved for testing. The two unimodal models for audio and text are provided in **unimodal_models** folder.


For getting the audio unimodal model using LEAF and CNN refer to **audio_model.py**. For the LEAF implementation, please refer to [Pytorch implementation](https://github.com/denfed/leaf-audio-pytorch). Please have a look at the requirements of this implementation for running the LEAF-CNN model.

For the transformer part of the code, please refer to **requirements.txt**. Please note that it is possible that there is a clash between the requirements of the LEAF-CNN network and the transformer model. It is advised to create separate environments for both of them.

A Colab notebook **sentiment_text.ipynb** is provided for creating the unimodal model for text

For the Bi-GRU with attention please refer to **class GRUModel** (lines 175-209) of **src/model_lstm_tranformers.py**.
