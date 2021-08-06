import os
import argparse
import math
import logging
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from torch.utils.data.dataloader import DataLoader
from torch import nn
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

from utils import (
    AverageMeter,
    ProgressMeter,
    calc_psnr,
    calc_ssim,
    calc_lpips,
    preprocess,
)
from dataset import Dataset
from PIL import Image
from models.loss import VGGLoss, GANLoss
from models.models import Generator, Discriminator
from lpips import LPIPS


if __name__ == "__main__":
    """로그 설정"""
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

    """ Argparse 설정 """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--eval-file", type=str, required=True)
    parser.add_argument("--outputs-dir", type=str, required=True)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--pretrained-net", type=str, default="BSRNet.pth")
    parser.add_argument("--gan-lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=100000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=288)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resume-g", type=str, default="generator.pth")
    parser.add_argument("--resume-d", type=str, default="discriminator.pth")
    parser.add_argument("--cuda", type=str, default="2")
    args = parser.parse_args()

    """ weight를 저장 할 경로 설정 """
    args.outputs_dir = os.path.join(args.outputs_dir, f"BSRGAN_x{args.scale}")
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    """ 텐서보드 설정 """
    writer = SummaryWriter(args.outputs_dir)

    """ GPU 디바이스 설정 """
    cudnn.benchmark = True
    cudnn.deterministic = True
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    """ Torch Seed 설정 """
    torch.manual_seed(args.seed)

    """ BSRGAN 모델 설정 """
    generator = Generator(scale_factor=args.scale).to(device)
    discriminator = Discriminator().to(device)

    """ Loss 설정 """
    pixel_criterion = nn.L1Loss().to(device)
    content_criterion = VGGLoss().to(device)
    adversarial_criterion = GANLoss().to(device)

    """ Optimizer 설정 """
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=args.gan_lr, betas=(0.9, 0.999)
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=args.gan_lr, betas=(0.9, 0.999)
    )

    """ Learning rate scheduler 설정 """
    interval_epoch = math.ceil(args.num_epochs // 8)
    epoch_indices = [
        interval_epoch,
        interval_epoch * 2,
        interval_epoch * 4,
        interval_epoch * 6,
    ]
    discriminator_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        discriminator_optimizer, milestones=epoch_indices, gamma=0.5
    )
    generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        generator_optimizer, milestones=epoch_indices, gamma=0.5
    )
    scaler = amp.GradScaler()

    """ epoch & PSNR 설정 """
    total_epoch = args.num_epochs
    g_epoch = 0
    d_epoch = 0
    best_lpips = 0

    """ lpips 설정 """
    lpips_metrics = LPIPS(net="vgg").to(device)

    """ 체크포인트 weight 불러오기 """
    if os.path.exists(args.resume_g) and os.path.exists(args.resume_d):
        """resume generator"""
        checkpoint_g = torch.load(args.resume_g)
        generator.load_state_dict(checkpoint_g["model_state_dict"])
        g_epoch = checkpoint_g["epoch"] + 1
        generator_optimizer.load_state_dict(checkpoint_g["optimizer_state_dict"])

        """ resume discriminator """
        checkpoint_d = torch.load(args.resume_d)
        discriminator.load_state_dict(checkpoint_d["model_state_dict"])
        d_epoch = checkpoint_g["epoch"] + 1
        discriminator_optimizer.load_state_dict(checkpoint_d["optimizer_state_dict"])
    elif os.path.exists(args.pretrained_net):
        """load BSRGAN pth if there are no pre-trained generator & discriminator"""
        # checkpoint = torch.load(args.pretrained_net)
        # generator.load_state_dict(checkpoint)
        state_dict = generator.state_dict()
        for n, p in torch.load(args.pretrained_net, map_location=device).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
    else:
        raise RuntimeError(
            "You need pre-trained BSRGAN.pth or generator & discriminator"
        )

    """ 로그 인포 프린트 하기 """
    logger.info(
        f"BSRGAN MODEL INFO:\n"
        f"\tScale:                         {args.scale}\n"
        f"BSRGAN TRAINING INFO:\n"
        f"\tTotal Epoch:                   {args.num_epochs}\n"
        f"\tStart generator Epoch:         {g_epoch}\n"
        f"\tStart discrimnator Epoch:      {d_epoch}\n"
        f"\tTrain directory path:          {args.train_file}\n"
        f"\tTest directory path:           {args.eval_file}\n"
        f"\tOutput weights directory path: {args.outputs_dir}\n"
        f"\tGAN learning rate:             {args.gan_lr}\n"
        f"\tPatch size:                    {args.patch_size}\n"
        f"\tBatch size:                    {args.batch_size}\n"
    )

    """ 데이터셋 & 데이터셋 설정 """
    train_dataset = Dataset(args.train_file, args.patch_size, args.scale)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_dataset = Dataset(args.eval_file, args.patch_size, args.scale)
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    """ 트레이닝 시작 & 테스트 시작"""
    for epoch in range(g_epoch, total_epoch):
        generator.train()
        discriminator.train()

        """ Losses average meter 설정 """
        d_losses = AverageMeter(name="D Loss", fmt=":.6f")
        g_losses = AverageMeter(name="G Loss", fmt=":.6f")
        pixel_losses = AverageMeter(name="Pixel Loss", fmt=":6.4f")
        content_losses = AverageMeter(name="Content Loss", fmt=":6.4f")
        adversarial_losses = AverageMeter(name="adversarial losses", fmt=":6.4f")

        """ 모델 평가 measurements 설정 """
        psnr = AverageMeter(name="PSNR", fmt=":.6f")
        lpips = AverageMeter(name="LPIPS", fmt=":.6f")
        ssim = AverageMeter(name="SSIM", fmt=":.6f")

        """ progress meter 설정 """
        progress = ProgressMeter(
            num_batches=len(eval_dataloader) - 1,
            meters=[
                psnr,
                lpips,
                ssim,
                d_losses,
                g_losses,
                pixel_losses,
                content_losses,
                adversarial_losses,
            ],
            prefix=f"Epoch: [{epoch}]",
        )

        """  트레이닝 Epoch 시작 """
        for i, (lr, hr) in enumerate(train_dataloader):
            """LR & HR 디바이스 설정"""
            lr = lr.to(device)
            hr = hr.to(device)

            """ 식별자 최적화 초기화 """
            discriminator_optimizer.zero_grad()

            with amp.autocast():
                """추론"""
                preds = generator(lr)
                """ 식별자 통과 후 loss 계산 """
                real_output = discriminator(hr)
                d_loss_real = adversarial_criterion(real_output, True)

                fake_output = discriminator(preds.detach())
                d_loss_fake = adversarial_criterion(fake_output, False)

                d_loss = (d_loss_real + d_loss_fake) / 2

            """ 가중치 업데이트 """
            scaler.scale(d_loss).backward()
            scaler.step(discriminator_optimizer)
            scaler.update()

            """ 생성자 최적화 초기화 """
            generator_optimizer.zero_grad()

            with amp.autocast():
                """추론"""
                preds = generator(lr)
                """ 식별자 통과 후 loss 계산 """
                real_output = discriminator(hr.detach())
                fake_output = discriminator(preds)
                pixel_loss = pixel_criterion(preds, hr.detach())
                content_loss = content_criterion(preds, hr.detach())
                adversarial_loss = adversarial_criterion(fake_output, True)
                g_loss = 1 * pixel_loss + 1 * content_loss + 0.1 * adversarial_loss

            """ 1 epoch 마다 테스트 이미지 확인 """
            if i == 0:
                vutils.save_image(
                    lr.detach(), os.path.join(args.outputs_dir, f"LR_{epoch}.jpg")
                )
                vutils.save_image(
                    hr.detach(), os.path.join(args.outputs_dir, f"HR_{epoch}.jpg")
                )
                vutils.save_image(
                    preds.detach(), os.path.join(args.outputs_dir, f"preds_{epoch}.jpg")
                )

            """ 가중치 업데이트 """
            scaler.scale(g_loss).backward()
            scaler.step(generator_optimizer)
            scaler.update()

            """ 생성자 초기화 """
            generator.zero_grad()

            """ loss 업데이트 """
            d_losses.update(d_loss.item(), lr.size(0))
            g_losses.update(g_loss.item(), lr.size(0))
            pixel_losses.update(pixel_loss.item(), lr.size(0))
            content_losses.update(content_loss.item(), lr.size(0))
            adversarial_losses.update(adversarial_loss.item(), lr.size(0))

        """ 스케줄러 업데이트 """
        discriminator_scheduler.step()
        generator_scheduler.step()

        """ 1 epoch 마다 텐서보드 업데이트 """
        writer.add_scalar("d_Loss/train", d_losses.avg, epoch)
        writer.add_scalar("g_Loss/train", g_losses.avg, epoch)
        writer.add_scalar("pixel_losses/train", pixel_losses.avg, epoch)
        writer.add_scalar("adversarial_losses/train", content_losses.avg, epoch)
        writer.add_scalar("adversarial_losses/train", adversarial_losses.avg, epoch)

        """  테스트 Epoch 시작 """
        generator.eval()
        with torch.no_grad():
            for i, (lr, hr) in enumerate(eval_dataloader):
                lr = lr.to(device)
                hr = hr.to(device)
                preds = generator(lr)
                psnr.update(calc_psnr(preds, hr), len(lr))
                ssim.update(calc_ssim(preds, hr).mean(), len(lr))
                lpips.update(calc_lpips(preds, hr, lpips_metrics), len(lr))
                if i == len(eval_dataset) // args.batch_size:
                    progress.display(i)

        """ 1 epoch 마다 텐서보드 업데이트 """
        writer.add_scalar("psnr/test", psnr.avg, epoch)
        writer.add_scalar("ssim/test", ssim.avg, epoch)
        writer.add_scalar("lpips/test", lpips.avg, epoch)

        """  Best 모델 저장 """
        if lpips.avg > best_lpips:
            best_lpips = lpips.avg
            torch.save(
                generator.state_dict(), os.path.join(args.outputs_dir, "best_g.pth")
            )

        """ Epoch 1000번에 1번 저장 """
        if epoch % 100 == 0:
            """Discriminator 모델 저장"""
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": discriminator.state_dict(),
                    "optimizer_state_dict": discriminator_optimizer.state_dict(),
                },
                os.path.join(args.outputs_dir, "d_epoch_{}.pth".format(epoch)),
            )

            """ Generator 모델 저장 """
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": generator.state_dict(),
                    "optimizer_state_dict": generator_optimizer.state_dict(),
                    "best_lpips": best_lpips,
                },
                os.path.join(args.outputs_dir, "g_epoch_{}.pth".format(epoch)),
            )

    """ 텐서보드 종료 """
    writer.close()
