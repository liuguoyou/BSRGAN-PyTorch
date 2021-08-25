import cv2
import numpy as np
import random
import imgaug.augmenters as ia
from PIL import Image

from scipy import ndimage
from scipy.interpolate import interp2d
from degradation.isp import ISP


class Degradation:
    def __init__(self):
        self.isp = ISP()

    def get_degrade_seq(self):
        degrade_seq = []
        need_shift = False
        global_sf = None

        # -----------------------
        # isotropic gaussian blur
        # -----------------------
        B_iso = {
            "mode": "blur",
            "kernel_size": random.choice([7, 9, 11, 13, 15, 17, 19, 21]),
            "is_aniso": False,
            "sigma": random.uniform(0.1, 2.8),
        }
        degrade_seq.append(B_iso)

        # -------------------------
        # anisotropic gaussian blur
        # -------------------------
        B_aniso = {
            "mode": "blur",
            "kernel_size": random.choice([7, 9, 11, 13, 15, 17, 19, 21]),
            "is_aniso": True,
            "x_sigma": random.uniform(0.5, 8),
            "y_sigma": random.uniform(0.5, 8),
            "rotation": random.uniform(0, 180),
        }
        degrade_seq.append(B_aniso)

        # -----------
        # down sample
        # -----------
        B_down = {"mode": "down", "sf": random.uniform(2, 4)}
        mode = random.randint(1, 4)
        if mode == 1:
            B_down["down_mode"] = "nearest"
            B_down["sf"] = random.choice([2, 4])
            need_shift = True
        elif mode == 2:
            B_down["down_mode"] = "bilinear"
        elif mode == 3:
            B_down["down_mode"] = "bicubic"
        elif mode == 4:
            down_mode = random.choice(["bilinear", "bicubic"])
            up_mode = random.choice(["bilinear", "bicubic"])
            up_sf = random.uniform(0.5, B_down["sf"])
            B_down["down_mode"] = down_mode
            B_down["sf"] = B_down["sf"] / up_sf
            B_up = {"mode": "down", "sf": up_sf, "down_mode": up_mode}
            degrade_seq.append(B_up)
        degrade_seq.append(B_down)
        global_sf = B_down["sf"]

        # --------------
        # gaussian noise
        # --------------
        B_noise = {"mode": "noise", "noise_level": random.randint(1, 25)}
        degrade_seq.append(B_noise)

        # ----------
        # jpeg noise
        # ----------
        if random.randint(1, 4) <= 3:
            B_jpeg = {"mode": "jpeg", "qf": random.randint(30, 95)}
            degrade_seq.append(B_jpeg)

        # -------------------
        # Processed camera sensor noise
        # -------------------
        if random.randint(1, 4) <= 4:
            B_camera = {"mode": "camera"}
            degrade_seq.append(B_camera)

        # -------
        # shuffle
        # -------
        random.shuffle(degrade_seq)

        # ---------------
        # last jpeg noise
        # ---------------
        B_jpeg_last = {"mode": "jpeg", "qf": random.randint(30, 95)}
        degrade_seq.append(B_jpeg_last)

        # --------------------
        # restore correct size
        # --------------------
        B_restore = {"mode": "restore", "sf": global_sf, "need_shift": need_shift}
        degrade_seq.append(B_restore)
        return degrade_seq

    def degradation_pipeline(self, img):
        img = np.array(img).astype(np.float32)
        h, w, c = img.shape
        degrade_seq = self.get_degrade_seq()
        # print_degrade_seg(degrade_seq)
        for degrade_dict in degrade_seq:
            mode = degrade_dict["mode"]
            if mode == "blur":
                img = self.get_blur(img, degrade_dict)
            elif mode == "down":
                img = self.get_down(img, degrade_dict)
            elif mode == "noise":
                img = self.get_noise(img, degrade_dict)
            elif mode == "jpeg":
                img = self.get_jpeg(img, degrade_dict)
            elif mode == "camera":
                img = self.get_camera(img)
            elif mode == "restore":
                img = self.get_restore(img, h, w, degrade_dict)
        return Image.fromarray(img.clip(0, 255).astype(np.uint8))

    def get_blur(self, img, degrade_dict):
        k_size = degrade_dict["kernel_size"]
        if degrade_dict["is_aniso"]:
            sigma_x = degrade_dict["x_sigma"]
            sigma_y = degrade_dict["y_sigma"]
            angle = degrade_dict["rotation"]
        else:
            sigma_x = degrade_dict["sigma"]
            sigma_y = degrade_dict["sigma"]
            angle = 0

        kernel = np.zeros((k_size, k_size))
        d = k_size // 2
        for x in range(-d, d + 1):
            for y in range(-d, d + 1):
                kernel[x + d][y + d] = self.get_kernel_pixel(x, y, sigma_x, sigma_y)
        M = cv2.getRotationMatrix2D((k_size // 2, k_size // 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (k_size, k_size))
        kernel = kernel / np.sum(kernel)

        # kernel = kernel*255/np.max(kernel)
        # kernel = kernel.astype(np.uint8).reshape((k_size, k_size, 1))
        # cv2.imwrite("test.png", kernel)
        img = ndimage.filters.convolve(
            img, np.expand_dims(kernel, axis=2), mode="reflect"
        )

        return img.clip(0, 255)

    def get_down(self, img, degrade_dict):
        sf = degrade_dict["sf"]
        mode = degrade_dict["down_mode"]
        h, w, c = img.shape
        if mode == "nearest":
            img = img[0::sf, 0::sf, :]
        elif mode == "bilinear":
            new_h, new_w = int(h / sf) // 2 * 2, int(w / sf) // 2 * 2
            img = cv2.resize(img, (new_h, new_w), interpolation=cv2.INTER_LINEAR)
        elif mode == "bicubic":
            new_h, new_w = int(h / sf) // 2 * 2, int(w / sf) // 2 * 2
            img = cv2.resize(img, (new_h, new_w), interpolation=cv2.INTER_CUBIC)
        return img.clip(0, 255)

    def get_noise(self, img, degrade_dict):
        noise_level = degrade_dict["noise_level"]
        img = img + np.random.normal(0, noise_level, img.shape)
        return img.clip(0, 255)

    def get_jpeg(self, img, degrade_dict):
        qf = degrade_dict["qf"]
        trans = ia.JpegCompression(compression=qf)
        degrade_function = lambda x: trans.augment_image(x)
        img = degrade_function(img.astype(np.uint8))
        return img.clip(0, 255)

    def get_camera(self, img):
        deg_img, _ = self.isp.cbdnet_noise_generate_srgb(img)
        return deg_img.clip(0, 255)

    def get_restore(self, img, h, w, degrade_dict):
        need_shift = degrade_dict["need_shift"]
        sf = degrade_dict["sf"]
        img = cv2.resize(img, (h, w), interpolation=cv2.INTER_CUBIC)
        if need_shift:
            img = self.shift_pixel(img, int(sf))
        return img.clip(0, 255)

    def get_kernel_pixel(self, x, y, sigma_x, sigma_y):
        return (
            1
            / (2 * np.pi * sigma_x * sigma_y)
            * np.exp(
                -((x * x / (2 * sigma_x * sigma_x)) + (y * y / (2 * sigma_y * sigma_y)))
            )
        )

    def shift_pixel(self, x, sf, upper_left=True):
        """shift pixel for super-resolution with different scale factors
        Args:
            x: WxHxC or WxH
            sf: scale factor
            upper_left: shift direction
        """
        h, w = x.shape[:2]
        shift = (sf - 1) * 0.5
        xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
        if upper_left:
            x1 = xv + shift
            y1 = yv + shift
        else:
            x1 = xv - shift
            y1 = yv - shift

        x1 = np.clip(x1, 0, w - 1)
        y1 = np.clip(y1, 0, h - 1)

        if x.ndim == 2:
            x = interp2d(xv, yv, x)(x1, y1)
        if x.ndim == 3:
            for i in range(x.shape[-1]):
                x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

        return x

    def print_degrade_seg(self, degrade_seq):
        for degrade_dict in degrade_seq:
            print(degrade_dict)
