import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import sys 
from dataset import LaneDataset
from model import U_NET

# --- THIẾT LẬP CÁC THAM SỐ ---
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 25
NUM_WORKERS = 2 
NUM_CLASSES = 3
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True

# --- ĐƯỜNG DẪN ---
TRAIN_IMG_DIR = "/home/loipham/INTERN/28_07_project/dataset/train/"
TRAIN_MASK_DIR = "/home/loipham/INTERN/28_07_project/dataset/train_mask/"
VAL_IMG_DIR = "/home/loipham/INTERN/28_07_project/dataset/valid/"
VAL_MASK_DIR = "/home/loipham/INTERN/28_07_project/dataset/valid_mask/"

MODEL_SAVE_PATH = "train.pth"


def train_fn(loader, model, optimizer, loss, scaler, device):
    loop = tqdm(loader, desc="Training")
    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)

        with torch.amp.autocast(device_type=device, enabled=(device=="cuda")):
            predictions = model(data)
            loss = loss(predictions, targets)

        optimizer.zero_grad()
        if device == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loop.set_postfix(loss=loss.item())

# --- HÀM ĐÁNH GIÁ TRÊN TẬP VALIDATION ---
def evaluate_model(loader, model, loss, device="cuda"):
    print("Evaluating on validation set...")
    model.eval() 
    total_loss = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating"):
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            loss = loss(preds, y)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    
    model.train()
    return avg_loss

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ],
    )

    model = U_NET(classes=NUM_CLASSES).to(DEVICE)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scaler = torch.amp.GradScaler(enabled=(DEVICE=="cuda"))

    train_dataset = LaneDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transform=train_transform)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY, shuffle=True
    )

    val_dataset = LaneDataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, transform=val_transform)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY, shuffle=False
    )
    
    best_val_loss = float("inf") 

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_fn(train_loader, model, optimizer, loss, scaler, DEVICE)

        current_val_loss = evaluate_model(val_loader, model, loss, device=DEVICE)

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"==> Model mới tốt nhất được lưu tại epoch {epoch+1} với Validation Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()