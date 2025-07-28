import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import hàm tạo model từ file model.py của bạn
from model import U_NET

# --- THIẾT LẬP CÁC THAM SỐ (PHẢI GIỐNG VỚI KHI HUẤN LUYỆN) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 448 # hoặc 256 nếu là model cũ

# --- ĐƯỜNG DẪN ĐẾN MODEL ---
MODEL_PATH = "train.pth" # Sửa thành tên file model của bạn
# ----------------------------------------------------

class LaneApp:
    def __init__(self, root):
        self.root = root
        self.root.title("So sánh")
        self.root.geometry("1200x700") 

        # 1. Khởi tạo và nạp model
        self.model = self.load_model()

        # 2. Tạo các thành phần giao diện
        self.btn_select = tk.Button(root, text="Chọn ảnh để dự đoán", command=self.select_and_predict)
        self.btn_select.pack(pady=10)

        # Tạo một Frame chính để chứa 2 ảnh
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame cho ảnh gốc (bên trái)
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(left_frame, text="Ảnh Gốc", font=("Helvetica", 14)).pack()
        self.lbl_original = tk.Label(left_frame)
        self.lbl_original.pack(fill=tk.BOTH, expand=True)

        # Frame cho ảnh kết quả (bên phải)
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(right_frame, text="Kết Quả Dự Đoán", font=("Helvetica", 14)).pack()
        self.lbl_result = tk.Label(right_frame)
        self.lbl_result.pack(fill=tk.BOTH, expand=True)


    def load_model(self):
        try:
            model = U_NET(classes=NUM_CLASSES).to(DEVICE)
            model.load_state_dict(torch.load(MODEL_PATH))
            model.eval()
            return model
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể nạp model: {e}")
            self.root.destroy()
            return None

    def select_and_predict(self):
        file_path = filedialog.askopenfilename(
            title="Chọn một file ảnh",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return

        original_image = cv2.imread(file_path)
        if original_image is None:
            messagebox.showerror("Lỗi", "Không thể đọc file ảnh đã chọn.")
            return

        prediction_mask = self.predict(original_image)
        result_image = self.create_overlay(original_image, prediction_mask)
        
        # Hiển thị cả 2 ảnh
        self.display_images(original_image, result_image)

    def predict(self, image):
        transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ])
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = transform(image=image_rgb)
        input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            preds = self.model(input_tensor)
            preds = torch.argmax(preds, dim=1).squeeze(0)
            prediction_mask = preds.cpu().numpy().astype(np.uint8)
        
        return prediction_mask

    def create_overlay(self, original_image, prediction_mask):
        original_h, original_w, _ = original_image.shape
        resized_mask = cv2.resize(prediction_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        output_image = original_image.copy()

        SOLID_COLOR = [0, 0, 255]
        DASHED_COLOR = [0, 100, 0]

        output_image[resized_mask == 1] = SOLID_COLOR
        output_image[resized_mask == 2] = DASHED_COLOR
        
        return output_image

    def display_images(self, original_cv2, result_cv2):
        # Lấy kích thước của frame để resize ảnh cho vừa
        self.root.update_idletasks()
        frame_width = self.lbl_original.winfo_width()
        frame_height = self.lbl_original.winfo_height()

        # --- Hiển thị ảnh gốc ---
        image_rgb_orig = cv2.cvtColor(original_cv2, cv2.COLOR_BGR2RGB)
        pil_image_orig = Image.fromarray(image_rgb_orig)
        pil_image_orig.thumbnail((frame_width, frame_height)) # Resize cho vừa
        tk_image_orig = ImageTk.PhotoImage(image=pil_image_orig)
        
        self.lbl_original.config(image=tk_image_orig)
        self.lbl_original.image = tk_image_orig

        # --- Hiển thị ảnh kết quả ---
        image_rgb_result = cv2.cvtColor(result_cv2, cv2.COLOR_BGR2RGB)
        pil_image_result = Image.fromarray(image_rgb_result)
        pil_image_result.thumbnail((frame_width, frame_height)) # Resize cho vừa
        tk_image_result = ImageTk.PhotoImage(image=pil_image_result)

        self.lbl_result.config(image=tk_image_result)
        self.lbl_result.image = tk_image_result


if __name__ == "__main__":
    root = tk.Tk()
    app = LaneApp(root)
    root.mainloop()