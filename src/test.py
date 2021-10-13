import logging
from src.model import MyDataset
from src.model import MultiModalTransformer, compute_accuracy
from config import NUM_ATTENTION_HEADS, NUM_HIDDEN_LAYERS, HIDDEN_SIZE
from config import AUDIO_DIM, TEXT_DIM, IMAGE_DIM
from config import USE_TEXT, USE_AUDIO, USE_IMAGE
from pytorch_transformers.modeling_bert import (BertConfig, BertEmbeddings,
                                                BertEncoder, BertLayerNorm,
                                                BertPreTrainedModel)
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
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
    config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    config.num_hidden_layers = NUM_HIDDEN_LAYERS
    config.num_attention_heads = NUM_ATTENTION_HEADS
    config.hidden_size = HIDDEN_SIZE
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
    for fold in range(4):
        test_model = MultiModalTransformer(config, AUDIO_DIM, TEXT_DIM, IMAGE_DIM, max_length)
        if USE_AUDIO:
            model_path = PATH.replace("audio", "audio"+str(fold))
        elif USE_TEXT:
            model_path = PATH.replace("text", "text"+str(fold))
        checkpoint = torch.load(model_path)
        test_model.load_state_dict(checkpoint['model_state_dict'])
        test_model.to(device)
        test_model.eval()
        for ind, test_dic in enumerate(test_loader):
            _, test_out = test_model.forward(test_dic, USE_TEXT, USE_AUDIO, USE_IMAGE)
            test_dic['target'][test_dic['target'] == -1] = 4
            test_results[fold] = test_out
        acc_fold = compute_accuracy(test_out.cpu(), test_dic).item()
        logger.info(f"Accuracy of fold-  {fold}- {acc_fold}")
