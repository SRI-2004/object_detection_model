import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from torchvision.transforms import transforms
from utils import load_checkpoint, save_checkpoint, get_bboxes, mean_average_precision
from torch.quantization import QuantStub, DeQuantStub, quantize

# Configuration
DEVICE = "cuda"
BATCH_SIZE = 8
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

def quantize_model(model, train_loader, eval_loader, quantization_config):
    model.qconfig = quantization_config
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    # Calibration
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)

    # Evaluation with quantized model
    model.eval()
    pred_boxes, target_boxes = get_bboxes(
        eval_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
    )
    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
    )
    print(f"Quantized Model Evaluation - Test mAP: {mean_avg_prec}")

    return model

def main():
    # Load pre-trained model from .pth.tar file
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Define your optimizer
    checkpoint = torch.load("finaltransform.pth.tar")
    load_checkpoint(checkpoint, model, optimizer)
    model.eval()

    # Load datasets
    train_dataset = VOCDataset(
        "data/100examples.csv",
        transform=transforms.ToTensor(),
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    eval_dataset = VOCDataset(
        "data/test.csv", transform=transforms.ToTensor(), img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Quantization configuration
    quantization_config = torch.quantization.get_default_qconfig('fbgemm')

    # Quantize the model
    quantized_model = quantize_model(model, train_loader, eval_loader, quantization_config)

    # Save the quantized model
    save_checkpoint({"state_dict": quantized_model.state_dict(), "optimizer": optimizer.state_dict()}, filename="quantized_model.pth")

if __name__ == "__main__":
    main()
