import os
import csv
import cv2
import torch
import pydicom
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights, retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt

#Constants
DICOM_DIR = 'rsna_data/stage_2_train_images'
LABELS_CSV = 'rsna_data/stage_2_train_labels.csv'
MODEL_PATH = 'saved_model.pth'
RESIZE_TO = (768, 768)

#Loading Annotations
def load_annotations(csv_path):
    df = pd.read_csv(csv_path)
    df_pos = df[df['Target'] == 1]
    annotations = {}
    for _, row in df_pos.iterrows():
        pid = row['patientId']
        x, y, w, h = row['x'], row['y'], row['width'], row['height']
        box = [x, y, x + w, y + h]
        if pid not in annotations:
            annotations[pid] = []
        annotations[pid].append(box)
    print(f"Loaded {len(annotations)} annotated pneumonia cases.")
    return annotations

#Dataset
class RSNADataset(Dataset):
    def __init__(self, dicom_dir, annotations_dict):
        self.dicom_dir = dicom_dir
        self.annotations = annotations_dict
        self.image_ids = list(annotations_dict.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        pid = self.image_ids[idx]
        dcm_path = os.path.join(self.dicom_dir, f"{pid}.dcm")
        dcm = pydicom.dcmread(dcm_path)
        img = dcm.pixel_array.astype(np.float32)
        h_orig, w_orig = img.shape
        img = cv2.resize(img, RESIZE_TO)
        img = (img - img.min()) / (img.max() - img.min())
        img_tensor = torch.tensor(img).unsqueeze(0)

        boxes = []
        for box in self.annotations[pid]:
            x1, y1, x2, y2 = box
            x1 = x1 * RESIZE_TO[0] / w_orig
            x2 = x2 * RESIZE_TO[0] / w_orig
            y1 = y1 * RESIZE_TO[1] / h_orig
            y2 = y2 * RESIZE_TO[1] / h_orig
            boxes.append([x1, y1, x2, y2])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

#Model Loader
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.detection.anchor_utils import AnchorGenerator
from collections import OrderedDict

def get_model(num_classes=2):
    # Loading pretrained resnet101
    backbone = resnet101(weights=ResNet101_Weights.DEFAULT)

    # Removing the fully connected layer and avgpool
    modules = OrderedDict([
        ("conv1", backbone.conv1),
        ("bn1", backbone.bn1),
        ("relu", backbone.relu),
        ("maxpool", backbone.maxpool),
        ("layer1", backbone.layer1),
        ("layer2", backbone.layer2),
        ("layer3", backbone.layer3),
        ("layer4", backbone.layer4)
    ])
    backbone = torch.nn.Sequential(modules)

    # Creating the FPN
    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    in_channels_list = [256, 512, 1024, 2048]
    out_channels = 256
    fpn_backbone = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=LastLevelMaxPool())

    # Defining anchors
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # Final RetinaNet model
    model = RetinaNet(backbone=fpn_backbone, num_classes=num_classes, anchor_generator=anchor_generator)

    return model

#Training Function
def train_model(train_loader, val_loader, device, model_path=MODEL_PATH, epochs=40, patience=7):
    model = get_model()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            if not isinstance(loss_dict, dict):
                raise ValueError(f"Expected a dict of losses, got {type(loss_dict)}")

            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.train() 
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                if not isinstance(loss_dict, dict):
                    raise ValueError(f"Expected a dict of losses, got {type(loss_dict)} during validation")

                loss = sum(loss_dict.values())
                val_loss += loss.item() 

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print("Saved improved model.")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping.")
                break

#Evaluation with ground truth
from torchvision.ops import nms
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def evaluate_with_ground_truth(model, val_loader, device, iou_threshold=0.5, score_threshold=0.3, nms_threshold=0.4):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[iou_threshold])
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            predictions = model(images)

            for pred, target in zip(predictions, targets):
                boxes = pred['boxes']
                scores = pred['scores']
                labels = pred['labels']

                keep = scores > score_threshold
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                if boxes.shape[0] > 0:
                    keep_nms = nms(boxes, scores, nms_threshold)
                    boxes = boxes[keep_nms]
                    scores = scores[keep_nms]
                    labels = labels[keep_nms]

                pred_dict = {"boxes": boxes.cpu(), "scores": scores.cpu(), "labels": labels.cpu()}
                gt_dict = {"boxes": target["boxes"].cpu(), "labels": target["labels"].cpu()}

                metric.update([pred_dict], [gt_dict])

    results = metric.compute()
    print("\n📊 Evaluation on validation set:")
    print(f"mAP@{iou_threshold:.1f}: {results['map_50']:.4f}")
    print(f"Overall mAP: {results['map']:.4f}")
    print(f"Recall@100: {results['mar_100']:.4f}")


#Main
if __name__ == "__main__":
    annotations = load_annotations(LABELS_CSV)
    full_dataset = RSNADataset(DICOM_DIR, annotations)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(MODEL_PATH):
        print("Model already trained — loading saved model.")
        model = get_model()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
    else:
        print("No saved model found — training from scratch.")
        train_model(train_loader, val_loader, device)
        model = get_model()
        model.load_state_dict(torch.load(MODEL_PATH))
        model.to(device)
        model.eval()

    print("Model ready for evaluation or inference.")
    evaluate_with_ground_truth(model, val_loader, device)

