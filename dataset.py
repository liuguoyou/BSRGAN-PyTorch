import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from degradation.degradation import Degradation
from utils import check_image_file


# 데이터 셋 생성 클래스
class Dataset(object):
    # (이미지 디렉토리, 패치 사이즈, 스케일, 텐서플로우를 이용한 이미지 로더 사용 여부) 초기화
    def __init__(self, images_dir, image_size, upscale_factor):
        """Degrdation class 불러오기"""
        self.deg = Degradation()
        """이미지 파일 불러오기"""
        self.filenames = [
            os.path.join(images_dir, x)
            for x in os.listdir(images_dir)
            if check_image_file(x)
        ]
        self.lr_transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Lambda(self.deg.degradation_pipeline),
                transforms.Resize(
                    (image_size // upscale_factor, image_size // upscale_factor),
                    interpolation=Image.BICUBIC,
                ),
                transforms.ToTensor(),
            ]
        )
        self.hr_transforms = transforms.Compose(
            [
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # transforms.AutoAugment(),
                transforms.ToTensor(),
            ]
        )

    # lr & hr 이미지를 읽고 크롭하여 lr & hr 이미지를 반환하는 함수
    def __getitem__(self, idx):
        hr = self.hr_transforms(Image.open(self.filenames[idx]).convert("RGB"))
        lr = self.lr_transforms(hr)

        return lr, hr

    def __len__(self):
        return len(self.filenames)
