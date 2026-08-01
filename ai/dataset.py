import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# 1. แผนผังสำหรับ 6 คลาส
LABEL_MAPPING_6CLS = {
    0: 3,     # Vandalism -> Vandalism (3)
    1: 1,     # Stealing -> Theft (1)
    2: 1,     # Shoplifting -> Theft (1)
    3: 5,     # Shooting -> Shooting (5)
    4: 1,     # Robbery -> Theft (1)
    5: None,  # Roadaccidents -> Skip
    6: 0,     # Normal -> Normal (0)
    7: 0,     # Walking -> Normal (0)
    8: 0,     # Walking_While_Using_Phone -> Normal (0)
    9: 0,     # Walking_While_Reading_Book -> Normal (0)
    10: 0,    # Standing_Still -> Normal (0)
    11: 0,    # Sitting -> Normal (0)
    12: 2,    # Fighting -> Violence (2)
    13: 4,    # Explosion -> Fire_Explosion (4)
    14: 0,    # Meet_and_Split -> Normal (0)
    15: 1,    # Burglary -> Theft (1)
    16: 0,    # Clapping -> Normal (0)
    17: 2,    # Assault -> Violence (2)
    18: 4,    # Arson -> Fire_Explosion (4)
    19: None, # Arrest -> Skip
    20: 2,    # Abuse -> Violence (2)
}

CLASS_NAMES_6CLS = {
    0: "Normal",
    1: "Theft",
    2: "Violence",
    3: "Vandalism",
    4: "Fire_Explosion",
    5: "Shooting"
}

# 2. แผนผังสำหรับ 4 คลาส
LABEL_MAPPING_4CLS = {
    0: 1,     # Vandalism -> Property_Crime (1)
    1: 1,     # Stealing -> Property_Crime (1)
    2: 1,     # Shoplifting -> Property_Crime (1)
    3: 2,     # Shooting -> Personal_Violence (2)
    4: 1,     # Robbery -> Property_Crime (1)
    5: None,  # Roadaccidents -> Skip
    6: 0,     # Normal -> Normal (0)
    7: 0,     # Walking -> Normal (0)
    8: 0,     # Walking_While_Using_Phone -> Normal (0)
    9: 0,     # Walking_While_Reading_Book -> Normal (0)
    10: 0,    # Standing_Still -> Normal (0)
    11: 0,    # Sitting -> Normal (0)
    12: 2,    # Fighting -> Personal_Violence (2)
    13: 3,    # Explosion -> Fire_Hazard (3)
    14: 0,    # Meet_and_Split -> Normal (0)
    15: 1,    # Burglary -> Property_Crime (1)
    16: 0,    # Clapping -> Normal (0)
    17: 2,    # Assault -> Personal_Violence (2)
    18: 3,    # Arson -> Fire_Hazard (3)
    19: None, # Arrest -> Skip
    20: 2,    # Abuse -> Personal_Violence (2)
}

CLASS_NAMES_4CLS = {
    0: "Normal",
    1: "Property_Crime",
    2: "Personal_Violence",
    3: "Fire_Hazard"
}

# 3. แผนผังสำหรับ 3 คลาสหลัก (ตัดคลาสไฟไหม้/ระเบิดออกตามคำแนะนำ)
LABEL_MAPPING_3CLS = {
    0: 1,     # Vandalism -> Property_Crime (1)
    1: 1,     # Stealing -> Property_Crime (1)
    2: 1,     # Shoplifting -> Property_Crime (1)
    3: 2,     # Shooting -> Personal_Violence (2)
    4: 1,     # Robbery -> Property_Crime (1)
    5: None,  # Roadaccidents -> Skip
    6: 0,     # Normal -> Normal (0)
    7: 0,     # Walking -> Normal (0)
    8: 0,     # Walking_While_Using_Phone -> Normal (0)
    9: 0,     # Walking_While_Reading_Book -> Normal (0)
    10: 0,    # Standing_Still -> Normal (0)
    11: 0,    # Sitting -> Normal (0)
    12: 2,    # Fighting -> Personal_Violence (2)
    13: None, # Explosion -> Skip (Fire)
    14: 0,    # Meet_and_Split -> Normal (0)
    15: 1,    # Burglary -> Property_Crime (1)
    16: 0,    # Clapping -> Normal (0)
    17: 2,    # Assault -> Personal_Violence (2)
    18: None, # Arson -> Skip (Fire)
    19: None, # Arrest -> Skip
    20: 2,    # Abuse -> Personal_Violence (2)
}

CLASS_NAMES_3CLS = {
    0: "Normal",
    1: "Property_Crime",
    2: "Personal_Violence"
}

# ค่าเริ่มต้นสำหรับความเข้ากันได้
CLASS_NAMES = CLASS_NAMES_3CLS

class ActionVideoDataset(Dataset):
    """
    โหลดและสุ่มตัด 32 เฟรมแบบ Dense Temporal Sampling รองรับแผนผัง 3, 4 และ 6 คลาส
    """
    def __init__(self, txt_path, dataset_dir, num_frames=32, is_training=False, num_classes_schema=3):
        self.dataset_dir = dataset_dir
        self.num_frames = num_frames
        self.is_training = is_training
        
        if num_classes_schema == 6:
            self.mapping = LABEL_MAPPING_6CLS
            self.class_names = CLASS_NAMES_6CLS
        elif num_classes_schema == 4:
            self.mapping = LABEL_MAPPING_4CLS
            self.class_names = CLASS_NAMES_4CLS
        else:
            self.mapping = LABEL_MAPPING_3CLS
            self.class_names = CLASS_NAMES_3CLS
            
        # 1. Data Augmentation สำหรับฝั่ง Training
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # 2. Transform มาตรฐานสำหรับฝั่ง Validation / Test
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # โหลดไฟล์รายการวิดีโอ
        self.video_list = []
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"ไม่พบไฟล์รายการข้อมูลที่: {txt_path}")
            
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().rsplit(maxsplit=1)
                if len(parts) >= 2:
                    video_file = parts[0]
                    orig_label = int(parts[1])
                    
                    new_label = self.mapping.get(orig_label)
                    if new_label is not None:
                        if os.path.isabs(video_file):
                            full_video_path = video_file
                        else:
                            split_folder = "train"
                            if "test" in txt_path.lower():
                                split_folder = "test"
                            elif "valid" in txt_path.lower():
                                split_folder = "valid"
                            full_video_path = os.path.join(dataset_dir, split_folder, video_file)
                        self.video_list.append((full_video_path, new_label))
                        
        print(f"โหลดข้อมูลสำเร็จจาก {os.path.basename(txt_path)}: รวมทั้งสิ้น {len(self.video_list)} วิดีโอ ({num_classes_schema} คลาส, 32 เฟรม/วิดีโอ)")

    def __len__(self):
        return len(self.video_list)

    def _load_video_frames(self, path):
        """
        สกัด 32 เฟรมเข้มข้น (Dense Sampling) และสุ่ม Flip กระจกทั้งคลิปแบบคร่อมทุกเฟรมพร้อมกัน
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return torch.zeros((self.num_frames, 3, 224, 224), dtype=torch.float32)
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return torch.zeros((self.num_frames, 3, 224, 224), dtype=torch.float32)
            
        if total_frames >= self.num_frames:
            if self.is_training and total_frames > (self.num_frames * 2):
                start_frame = np.random.randint(0, total_frames - (self.num_frames * 2))
                end_frame = start_frame + (self.num_frames * 2)
            else:
                start_frame = 0
                end_frame = total_frames - 1
                
            frame_indices = np.linspace(start_frame, end_frame, self.num_frames, dtype=int)
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                if len(frames) > 0:
                    frame = frames[-1].copy()
                else:
                    frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
            frames.append(frame)
            
        cap.release()
        
        do_flip = self.is_training and (np.random.rand() > 0.5)
        
        processed_frames = []
        active_transform = self.train_transform if self.is_training else self.val_transform
        
        for f in frames:
            if do_flip:
                f = cv2.flip(f, 1)
                
            tensor_frame = active_transform(f)
            processed_frames.append(tensor_frame)
            
        return torch.stack(processed_frames)

    def __getitem__(self, idx):
        video_path, label = self.video_list[idx]
        try:
            frames = self._load_video_frames(video_path)
        except Exception:
            frames = torch.zeros((self.num_frames, 3, 224, 224), dtype=torch.float32)
            
        return frames, label
