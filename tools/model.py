import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import os


# Dataset Class for Segmentation
class LungDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # print(f"Mask min: {mask.min().item()}, Mask max: {mask.max().item()}")
        # print(f"Image min: {image.min().item()}, Image max: {image.max().item()}")
        return image, mask


class ClassificationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        self.class_names = os.listdir(data_dir)
        self.class_to_label = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        
        for class_name, label in self.class_to_label.items():
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(label)
                
        # Get the size of the first image as a reference
        if self.images:
            with Image.open(self.images[0]) as img:
                self.image_size = img.size  # (width, height)
        else:
            self.image_size = None  # Handle case with no images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label




class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        def up_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
            )

        self.encoder1 = conv_block(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.up4 = up_block(1024, 512)
        self.up3 = up_block(1024, 256)
        self.up2 = up_block(512, 128)
        self.up1 = up_block(256, 64)

        self.final = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x) # dwon 1
        enc2 = self.encoder2(self.pool1(enc1)) # dwon 2 
        enc3 = self.encoder3(self.pool2(enc2)) # dwon 3
        enc4 = self.encoder4(self.pool3(enc3)) # dwon 4

        
        bottleneck = self.bottleneck(self.pool4(enc4)) # down 5

        dec4 = torch.cat((enc4, self.up4(bottleneck)), dim=1)
        dec3 = torch.cat((enc3, self.up3(dec4)), dim=1)
        dec2 = torch.cat((enc2, self.up2(dec3)), dim=1)
        dec1 = torch.cat((enc1, self.up1(dec2)), dim=1)
        final = self.final(dec1)

        return torch.sigmoid(final)
    
    
# Classifier Architecture
class DepthwiseClassifier(nn.Module):
    def __init__(self, input_channel, output_channel, img_size, classes_num):
        super(DepthwiseClassifier, self).__init__()
        
        # 定義卷積模塊
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=1),
                nn.ReLU(inplace=True),
            )
        
        def size_calculate(x, kernel_size = 2, padding = 1, stride = 1):
            return ((x + padding*2)- (kernel_size-1))//stride
            
            
        self.conv = conv_block(input_channel, output_channel)
        
        # 計算經過卷積後的特徵圖大小
        self.conv_output_size = img_size  # 假設 padding=1，stride=1，輸出大小與輸入相同
        aft_size = size_calculate(img_size) 
        

        # 計算全連接層的輸入大小
        self.fc_input_size = output_channel * aft_size * aft_size
        # self.fc = nn.Sequential(
        #     nn.Linear(self.fc_input_size, self.fc_input_size//1000),
        #     nn.Linear(self.fc_input_size//1000, classes_num),
        # )
        self.fc = nn.Linear(self.fc_input_size, classes_num)
    def forward(self, x):
        # 卷積操作
        x = self.conv(x)

        # 平展特徵圖
        x = x.view(x.size(0), -1)  # Batch size, flatten features

        # 全連接層
        output = self.fc(x)
        
        return output
