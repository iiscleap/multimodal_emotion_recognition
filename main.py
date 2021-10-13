import numpy as np
import logging
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
from src.read_data import create_data
from src.model import train, MyDataset
from src.test import test_model
from src.test_lstm_transformers import test_model as test_lstm_model
from src.test_lstm_transformers_ee import test_model as test_lstm_model_ee
from src.model_lstm_tranformers import train as train_lstm
from src.model_lstm_transformers_ee import train as train_lstm_ee
from src.model_hybrid import train as train_hybrid
#from src.model_transformers import train as train_trans
from src.model_lstms import train as train_lstms
#from src.test_transformers import test_model as test_trans
from src.test_lstms import test_model as test_lstms
from src.test_hybrid import test_model as test_hybrid
from config import pickle_path, train_npy_audio_path, train_npy_text_path
from config import train_npy_image_path, train_npy_label_path
from config import test_npy_audio_path, test_npy_image_path
from config import test_npy_label_path, test_npy_text_path
from config import length_train, length_test, train_text_csv_path
from config import test_text_csv_path, USE_GRU, USE_EE, USE_AUDIO, USE_TEXT, USE_TRANS
from config import USE_ONLY_LSTM, USE_HYBRID

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def create_train_data():
    """Creates a the train and validation dataloaders and stores in appropriate 
    dictionary. Also returns the maximum length of any video in the dataset.
    """
    train_data = dict()
    val_data = dict()
    batch_size = 32
    for fold in range(1):
        train_imgs = np.load(train_npy_image_path.replace('train', 'train'+str(fold)))
        train_aud = np.load(train_npy_audio_path.replace('train', 'train'+str(fold)))
        train_text = np.load(train_npy_text_path.replace('train', 'train'+str(fold)))
        train_labels = np.load(train_npy_label_path.replace('train', 'train'+str(fold)))
        train_lengths = np.load(length_train.replace('train', 'train'+str(fold)))
        val_imgs = np.load(train_npy_image_path.replace('train', 'val'+str(fold)))
        val_aud = np.load(train_npy_audio_path.replace('train', 'val'+str(fold)))
        val_text = np.load(train_npy_text_path.replace('train', 'val'+str(fold)))
        val_labels = np.load(train_npy_label_path.replace('train', 'val'+str(fold)))
        val_lengths = np.load(length_train.replace('train', 'val'+str(fold)))
        train_size, val_size = train_imgs.shape[0], val_imgs.shape[0]
        train_indices, val_indices = list(range(train_size)), list(range(val_size))
        #train_sampler = SubsetRandomSampler(train_indices)
        #valid_sampler = SubsetRandomSampler(val_indices)
        train_dataset = MyDataset(train_imgs, train_aud, train_text, train_labels,
                                train_lengths)
        val_dataset = MyDataset(val_imgs, val_aud, val_text, val_labels,
                                val_lengths)
        train_loader = DataLoader(train_dataset,
                        batch_size=batch_size,
                        pin_memory=True,
                        shuffle=False,
                        drop_last=False,
                        )
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                pin_memory=True,
                                shuffle=False,
                                drop_last=False,
                                )
        train_data[fold] = train_loader
        val_data[fold] = val_loader
    return train_data, val_data, val_lengths.shape[1]

def main(create_flag, train_flag, test_flag):
    '''Depending on the three input flags it creates the data, trains a model
    based on the same and then tests the trained model.
    '''
    if create_flag:
        '''Creates the data from the pickle at the pickle path provided. All the arguments
        come from the config file.
        '''
        create_data(pickle_path, train_npy_audio_path, train_npy_image_path, train_npy_text_path,
                    train_npy_label_path, test_npy_audio_path, test_npy_image_path,
                    test_npy_text_path, test_npy_label_path, length_train, length_test, 
                    train_text_csv_path, test_text_csv_path)
    if train_flag:
        train_data, val_data, max_length = create_train_data()
        train_lstm(train_data, val_data, max_length)
        '''
        if USE_EE:
            train_lstm_ee(train_data, val_data, max_length)
        elif USE_TRANS == False and USE_ONLY_LSTM == False and USE_HYBRID == False:
            if USE_GRU:#Trains a unimodal lstm followed by transformer architecture
                train_lstm(train_data, val_data, max_length)
            else:
                train(train_data, val_data, max_length)
        elif USE_TRANS:
            train_trans(train_data, val_data, max_length)
        elif USE_ONLY_LSTM:
            train_lstms(train_data, val_data, max_length)
        elif USE_HYBRID:
            train_hybrid(train_data, val_data, max_length)'''

    if test_flag:
        test_imgs = np.load(test_npy_image_path)
        test_aud = np.load(test_npy_audio_path)
        test_text = np.load(test_npy_text_path)
        test_labels = np.load(test_npy_label_path)
        test_lengths = np.load(length_test)
        max_length = test_lengths.shape[1]
        test_lstm_model('gru_best_model.tar', test_imgs, test_aud, test_text, test_labels,
                        test_lengths, test_lengths.shape[1])
        '''
        if USE_EE:
            test_lstm_model_ee('gru_best_model_ee.tar', test_imgs, test_aud, test_text, test_labels,
                    test_lengths, test_lengths.shape[1])
        elif USE_TRANS == False and USE_ONLY_LSTM == False and USE_HYBRID == False:
            if USE_GRU:#Tests the required model
                test_lstm_model('gru_best_model.tar', test_imgs, test_aud, test_text, test_labels,
                        test_lengths, test_lengths.shape[1])
            else:
                if USE_TEXT:
                    test_model('unimodal_trans_text.tar', test_imgs, test_aud, test_text, test_labels,
                            test_lengths, test_lengths.shape[1])
                elif USE_AUDIO:
                    test_model('unimodal_trans_audio.tar', test_imgs, test_aud, test_text, test_labels,
                            test_lengths, test_lengths.shape[1])
        elif USE_TRANS:
            test_trans('transformer_model.tar', test_imgs, test_aud, test_text, test_labels,
                        test_lengths, test_lengths.shape[1])
        elif USE_ONLY_LSTM:
            test_lstms('lstm_best_model.tar', test_imgs, test_aud, test_text, test_labels,
                        test_lengths, test_lengths.shape[1])
        elif USE_HYBRID:
            test_hybrid('hybrid_best_model.tar', test_imgs, test_aud, test_text, test_labels,
                        test_lengths, test_lengths.shape[1])'''

if __name__ == "__main__":
    main(True, True, True)
