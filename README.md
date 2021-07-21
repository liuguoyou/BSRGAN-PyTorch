## [Abstract]

![](https://images.velog.io/images/heaseo/post/86e209c5-5284-4c39-a08f-f0117c682c31/BSRGAN_degradation.png)

Single Image Super-resolutoin (SISR)으로 현실에서 훼손된 이미지를 복원할려고 하면 잘 되지 않는 경우가 많다. 여러 종류의 열화기법을 사용해서 SISR 모델을 학습을 시켜도 현실에서 여러 이유로 훼손된 이미지를 복구하기란 쉽지않다. 이 문제를 해결하기 위해, 이 논문은 blur, downsampling 그리고 noise 열화기법을 임의적으로 섞어서 학습 된 모델을 소개한다. 특히 blur 같은 경우 isotropic과 anisotropic 2가지의 가우시안 커널을 사용했고, downsampling은 nearest, blilnear 그리고 bicubic 보간기법을 임의적으로 선택해 사용했다. 게다가 노이지 생성을 위해 가우시안 노이즈, JPEG 압축 노이즈 그리고 camera image signal processing (ISP) 파이프라인 모델을 사용했다. 제안한 모델의 효과성을 입증하기 위해 저자는 deep blind ESRGAN super-resolver를 이용하여 위의 여러가지 열화기법을 포함해서 모델을 학습시켰다. 
 [더보기](https://velog.io/@heaseo/BSRGAN-Designing-a-Practical-Degradation-Model-for-Deep-Blind-Image-Super-Resolution)
<br>

## Requirements
``` bash
- pip3 install pytorch
- pip3 install imgaug
- pip3 install tensorboard
- pip3 install tensorflow
- pip3 install scipy
- pip3 install opencv-python
```

## Train
``` bash
python3 train.py --train-file ${train_datasets} --eval-file ${valid_datasets} --outputs-dir ${save_model_dir} --scale ${2 or 4} --pretrained-net ${BSRNet.pth}
```

## Test
``` bash
python3 test.py --weights-file ${BSRGAN.pth} --image-file ${image file path} --scale ${2 or 4}
```

<br>

## Results
<table>
    <tr>
        <td><center>Original</center></td>
        <td><center>BSRGAN x4</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="https://images.velog.io/images/heaseo/post/6f2fa011-343a-4e27-ae86-000b44a24727/chip.png""></center>
    	</td>
    	<td>
    		<center><img src="https://images.velog.io/images/heaseo/post/a14905de-8a40-4cb6-aed1-a1eb9612c744/chip_BSRGAN.png"></center>
    	</td>
    </tr>
    <tr>
        <td><center>Original</center></td>
        <td><center>BSRGAN x4</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="https://images.velog.io/images/heaseo/post/77efe499-80b6-4c27-941c-c6571f07b8f1/oldphoto2.png""></center>
    	</td>
    	<td>
    		<center><img src="https://images.velog.io/images/heaseo/post/87eff938-198f-4be7-ad04-96ad1059700e/oldphoto2_BSRGAN.png"></center>
    	</td>
    </tr>
</table>