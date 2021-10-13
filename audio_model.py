import torch
from leaf_audio_pytorch import frontend
import sys
import os
import logging
from scipy.io import wavfile
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
import torch.nn as nn
import random

#Logger set
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

np.random.seed(1234)
torch.manual_seed(1234)

#CUDA devices enabled
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()


class MyDataset(Dataset):
    '''Dataset class for audio which reads in the audio signals and prepares them for
    training. In particular it pads all of them to the maximum length of the audio
    signal that is present. All audio signals are padded/sampled upto 34s in this
    dataset.
    '''
    def __init__(self, list_audio, data_path):
        self.list_audio = list_audio
        self.data_path = str(data_path)
        self.duration = 34000
        self.sr = 16000
        self.channel = 1
        with open('../label_dict.json', 'r') as f:
            self.labels = json.load(f)

    def __len__(self):
        return len(self.list_audio) 
        
    def __getitem__(self, audio_ind):
        audio_file = os.path.join(self.data_path, self.list_audio[audio_ind])
        class_id = self.labels[self.list_audio[audio_ind]]
        
        (sr, sig) = wavfile.read(audio_file)

        aud = (sig, sr)
        '''
        if (sr == self.sr):
            resig = sig
        else:
            resig = librosa.resample(sig, sr, self.sr)'''

        reaud = (sig, self.sr)
        resig = sig
        sig_len = resig.shape[0]
        max_len = self.sr//1000 * self.duration
        if len(resig.shape) == 2:
            resig = np.mean(resig, axis = 1)

        if (sig_len > max_len):
            # Truncate the signal to the given length
            final_sig = resig[:max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = np.zeros((pad_begin_len))
            pad_end = np.zeros((pad_end_len))

            final_sig = np.float32(np.concatenate((pad_begin, resig, pad_end), 0))
            final_aud = (final_sig, self.sr)

        return final_sig, class_id


class Net(nn.Module):
    '''Defines the CNN network that works on the output from LEAF for audio
    classification.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.start = 16
        self.final = 100
        self.conv1 = nn.Conv2d(1, self.start*2, 3, 1)
        self.bn1 = nn.BatchNorm2d(self.start*2)

        self.conv2 = nn.Conv2d(self.start*2, self.start*2, 3, 1)
        self.bn2 = nn.BatchNorm2d(self.start*2)

        self.conv3 = nn.Conv2d(self.start*2, self.start*4, 3, 1)
        self.bn3 = nn.BatchNorm2d(self.start*4)

        self.conv4 = nn.Conv2d(self.start*4, self.start*4, 3, 1)
        self.bn4 = nn.BatchNorm2d(self.start*4)

        self.conv5= nn.Conv2d(self.start*4, self.start*8, 3, 1)
        self.bn5 = nn.BatchNorm2d(self.start*8)

        self.conv6 = nn.Conv2d(self.start*8, self.final, 3, 1)
        self.bn6 = nn.BatchNorm2d(self.final)

        self.pool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=False)
        self.fc = nn.Linear(self.final, 8)

    def forward(self, x):
        x = x.to(device)
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(self.bn1(x))

        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(self.bn3(x))

        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.pool(x)

        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        
        x = self.conv6(x)
        x = self.relu(self.bn6(x))
        x = self.pool(x)

        x = x.squeeze(-2)
        x1 = torch.mean(x, dim = 2)

        x = self.fc(x1)

        return x, x1

def compute_accuracy(output, labels):
    #Function for calculating accuracy
    pred = torch.argmax(output, dim = 1)
    correct_pred = (pred == labels).float()
    tot_correct = correct_pred.sum()

    return tot_correct

def compute_loss(output, labels):
    #Function for calculating loss
    loss = nn.CrossEntropyLoss()(output, labels.squeeze(-1).long())
    return loss


def train():
    '''Creates the dataset and the LEAF and CNN model. Trains the LEAF and CNN jointly
    for a total of 100 epochs. The best LEAF and CNN model based on validation loss
    is saved for the next components in the model.
    '''
    l = os.listdir('../Sess5/leaf_wavs_train')
    l = [x for x in l if '.wav' in x]

    myds = MyDataset(l, "../Sess5/leaf_wavs_train")
    # Random split of 80:20 between training and validation
    num_items = len(myds)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    print(num_train, num_val)
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    # Create training and validation data loaders

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

    conv_model = Net()
    a = frontend.Leaf()

    conv_model.to(device)
    a.to(device)

    base_lr = 0.00005
    optimizer = Adam([{'params':conv_model.parameters(), 'lr':base_lr},
                      {'params':a.parameters(), 'lr':base_lr}])

    final_val_loss = 99999

    for e in range(100):
        conv_model.train()
        a.train()
        tot_loss, tot_correct = 0.0, 0.0
        val_loss, val_acc = 0.0, 0.0
        val_correct = 0.0
        for i, data in enumerate(train_dl):
            conv_model.zero_grad()
            a.zero_grad()
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.unsqueeze(1)
            leaf_out = a(inputs.float())
            final_out, _ = conv_model(leaf_out)
            correct = compute_accuracy(final_out, labels)
            loss = compute_loss(final_out, labels)
            tot_loss += loss.item()
            tot_correct += correct.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        conv_model.eval()
        a.eval()
        with torch.no_grad():
            for i, data in enumerate(val_dl):
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs = inputs.unsqueeze(1)
                leaf_out = a(inputs.float())
                val_out, _ = conv_model(leaf_out)
                correct = compute_accuracy(val_out, labels)
                loss = compute_loss(val_out, labels)
                val_loss += loss.item()
                val_correct += correct.item()
        if val_loss < final_val_loss:
            torch.save({'model_state_dict': a.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),},
                        'best_model_leaf.tar')
            torch.save({'model_state_dict': conv_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),},
                        'best_model_cnn.tar')
            final_val_loss = val_loss
        train_loss = tot_loss/len(train_dl)
        train_acc = tot_correct/num_train
        val_loss_log = val_loss/len(val_dl)
        val_acc_log = val_correct/num_val
        e_log = e + 1
        logger.info(f"Epoch {e_log}, \
                    Training Loss {train_loss},\
                    Training Accuracy {train_acc}")
        logger.info(f"Epoch {e_log}, \
                    Validation Loss {val_loss_log},\
                    Validation Accuracy {val_acc_log}")

if __name__ == "__main__":
    train()
