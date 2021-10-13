import logging
from src.model import MyDataset
from src.model_lstms import compute_accuracy
from src.model_lstms import GRUModel
from config import AUDIO_DIM, TEXT_DIM, IMAGE_DIM, GRU_DIM
from config import USE_TEXT, USE_AUDIO, USE_IMAGE, unimodal_folder
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
import os
torch.backends.cudnn.enabled = False
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def test_model(PATH, test_imgs, test_aud, test_text, test_labels, test_lengths, max_length):

    test_dataset = MyDataset(test_imgs, test_aud, test_text, test_labels,
                            test_lengths)
    test_size = test_imgs.shape[0]
    indices = list(range(test_size))
    test_sampler = SubsetRandomSampler(indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    test_loader = DataLoader(test_dataset,
                            sampler = test_sampler,
                            batch_size=64,
                            num_workers=2,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=False,
                            )
    test_results = dict()
    for fold in range(1):
        test_model = GRUModel(GRU_DIM*4, GRU_DIM, 4, 0.2, 2, True).double()
        model_path = PATH.replace("model", "model_"+str(fold))
        checkpoint = torch.load(model_path)
        test_model.load_state_dict(checkpoint['model_state_dict'])
        test_model.to(device)
        test_model.eval()
        if USE_TEXT:
            text_model = GRUModel(TEXT_DIM, GRU_DIM, 4, 0.2, 2, True).double()
            text_model.to(device)
            text_model.eval()
            checkpoint_text = torch.load(os.path.join(unimodal_folder, 'best_model_text'+str(fold)+'.tar'))
            text_model.load_state_dict(checkpoint_text['model_state_dict'])
        if USE_AUDIO:
            aud_model = GRUModel(AUDIO_DIM, GRU_DIM, 4, 0.2, 2, True).double()
            aud_model.to(device)
            aud_model.eval()
            checkpoint_aud = torch.load(os.path.join(unimodal_folder, 'best_model_aud'+str(fold)+'.tar'))
            aud_model.load_state_dict(checkpoint_aud['model_state_dict'])
        for ind, test_dic in enumerate(test_loader):
            if USE_AUDIO:
                inp = test_dic['aud'].permute(0, 2, 1).double()
                out_aud, out = aud_model.forward(inp.to(device))
            if USE_TEXT:
                inp = test_dic['text'].permute(0, 2, 1).double()
                out_text, out = text_model.forward(inp.to(device))

            inp = torch.cat((out_aud, out_text), axis = 2)
            _, test_out = test_model.forward(inp.to(device))
            test_dic['target'][test_dic['target'] == -1] = 4
            test_results[fold] = test_out
        acc_fold = compute_accuracy(test_out.cpu(), test_dic).item()
        logger.info(f"Accuracy of fold-  {fold}- {acc_fold}")
