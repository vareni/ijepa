from ijepa_pretrained_model import FinetuningModel
from src.helper import init_model
from ijepa_pretrained_model import load_yaml
import torch
from dataset import load_data

def load_model(model_path):
    device = torch.device('cpu')
    patch_size, model_name, crop_size, pred_depth, pred_emb_dim = load_yaml()

    target_encoder, predictor = init_model(device, patch_size, model_name, crop_size, pred_depth, pred_emb_dim)
    del predictor

    model = FinetuningModel(pretrained_model=target_encoder, num_classes=2)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()

    return model


if __name__ == '__main__':
    model_path = '../data/TB_small/model_TB_3.pth'
    model = load_model(model_path)

    data_path = '../data/TB_small'
    train_loader, val_loader = load_data(data_path)
    inputs, classes = next(iter(val_loader))
    res = model(inputs)
    print(f'predicted {torch.argmax(res, dim=1)}, real {classes}')