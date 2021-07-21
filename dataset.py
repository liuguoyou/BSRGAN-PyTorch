import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from degradation.degradation import degradation_pipeline
from utils import check_image_file


def degradation(image):
    image = Image.fromarray(degradation_pipeline(np.array(image)).astype(np.uint8))
    return image

# 데이터 셋 생성 클래스
class Dataset(object):
    # (이미지 디렉토리, 패치 사이즈, 스케일, 텐서플로우를 이용한 이미지 로더 사용 여부) 초기화
    def __init__(self, images_dir, image_size, upscale_factor):
        self.filenames = [os.path.join(images_dir, x) for x in os.listdir(images_dir) if check_image_file(x)]
        self.lr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size // upscale_factor, image_size // upscale_factor), interpolation=Image.BICUBIC),
            transforms.Lambda(degradation),
            transforms.ToTensor()
        ])
        self.hr_transforms = transforms.Compose([
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            #transforms.AutoAugment(),
            transforms.ToTensor()
        ])
    # lr & hr 이미지를 읽고 크롭하여 lr & hr 이미지를 반환하는 함수
    def __getitem__(self, idx):
        hr = self.hr_transforms(Image.open(self.filenames[idx]).convert("RGB"))
        lr = self.lr_transforms(hr)
        
        return lr, hr

    def __len__(self):
        return len(self.filenames)
