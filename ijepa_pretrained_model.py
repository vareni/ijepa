import torch
import copy
from src.helper import init_model
import yaml
import torch.nn as nn
import torch.nn.functional as F
from dataset import load_data
import torch.optim as optim
# from csv_rna_seq import load_data_rna

import logging

logging.basicConfig(
    filename='../models/model_TB_log.log',
    filemode='a',
    level=logging.INFO,
    force=True
)
logger = logging.getLogger(__name__)


class FinetuningModel(nn.Module):
    def __init__(self, pretrained_model, num_classes, drop=0.2):
        super(FinetuningModel, self).__init__()
        self.pretrained_model = pretrained_model

        self.num_classes = num_classes
        self.head_drop = nn.Dropout(drop)

        self.mlp_head = nn.Linear(self.pretrained_model.embed_dim,
                                  self.num_classes)

    def forward(self, x):
        x = self.pretrained_model(x)

        x = torch.mean(x, dim=1)

        x = x.squeeze(1)

        x = F.layer_norm(x, (x.size(-1),))  # normalize over feature-dim

        x = self.head_drop(x)  # As in timm.models

        x = self.mlp_head(x)
        return x


def load_model(
        r_path,
        encoder,
        predictor,
        target_encoder
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # -- loading encoder
        pretrained_dict = checkpoint['encoder']
        for k, v in pretrained_dict.items():
            encoder.state_dict()[k[len("module."):]].copy_(v)

        # -- loading predictor
        pretrained_dict = checkpoint['predictor']
        for k, v in pretrained_dict.items():
            predictor.state_dict()[k[len("module."):]].copy_(v)

        # -- loading target_encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint['target_encoder']
            for k, v in pretrained_dict.items():
                target_encoder.state_dict()[k[len("module."):]].copy_(v)

        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return encoder, predictor, target_encoder, epoch


def new_model(target_encoder):
    for p in target_encoder.parameters():
        p.requires_grad = False
    new_model = FinetuningModel(target_encoder, 2)
    return new_model


def train_model(model, train_loader, val_loader, model_path, device='cpu', num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        i = 1
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track training loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            logger.info(f"Epoch [{epoch + 1}/{num_epochs}],  batch {i} / {len(train_loader)}, "
                  f"Partial Train Loss: {running_loss / len(train_loader):.4f}, "
                  f"Partial Train Acc: {correct / total:.4f}, ")
            i += 1

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / total

        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    logger.info("Model saved to ", model_path)


def load_yaml(y_path="configs/in1k_vith14_ep300.yaml"):
    with open(y_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    patch_size = data['mask']['patch_size']
    model_name = data['meta']['model_name']
    crop_size = data['data']['crop_size']
    pred_depth = data['meta']['pred_depth']
    pred_emb_dim = data['meta']['pred_emb_dim']

    return patch_size, model_name, crop_size, pred_depth, pred_emb_dim


if __name__ == '__main__':
    r_path = "../IN1K-vit.h.14-300e.pth.tar"
    device = torch.device('cpu')
    patch_size, model_name, crop_size, pred_depth, pred_emb_dim = load_yaml()

    logger.info('yaml loaded successfully')

    encoder, predictor = init_model(device, patch_size, model_name, crop_size, pred_depth, pred_emb_dim)
    target_encoder = copy.deepcopy(encoder)

    logger.info('init model successfully')

    encoder, predictor, target_encoder, epoch = load_model(r_path,
                                                           encoder,
                                                           predictor,
                                                           target_encoder)

    logger.info('loaded model successfully')

    # -- Remove from device once they are required for loading pretrained parameters only
    del encoder
    del predictor

    model = new_model(target_encoder)
    logger.info('new model created')

    data_path = 'data/TB_small'
    train_loader, val_loader = load_data(data_path)  #load_data_rna()
    logger.info('data loaded')

    model_path = 'models/model_TB_4.pth'
    logger.info('start training model')
    train_model(model, train_loader=train_loader, val_loader=val_loader, num_epochs=15, model_path=model_path)



