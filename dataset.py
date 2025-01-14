from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import logging
logger = logging.getLogger(__name__)


def load_data(dataset_path):
    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
    ])

    # Load the dataset using ImageFolder
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Define DataLoader for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    # Verify the dataset and dataloaders
    class_names = dataset.classes
    logger.info(f"Classes: {class_names}")
    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of validation samples: {len(val_dataset)}")

    # Example: Check the size of one batch
    for images, labels in train_loader:
        logger.info(f"Batch size: {images.size()}")
        logger.info(f"Labels: {labels}")
        break

    return train_loader, val_loader


if __name__ == '__main__':
    path = '../data/TB_small'
    t, v = load_data(path)
