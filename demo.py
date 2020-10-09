# encoding:utf-8
"""
单张效果测试
@author: libo
"""
import os
import argparse
import torch

from torchvision import transforms
from models.fast_scnn import get_fast_scnn
from PIL import Image
from utils.visualize import get_color_pallete

import time

parser = argparse.ArgumentParser(description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='fast_scnn', help='model name (default: fast_scnn)')
parser.add_argument('--dataset', type=str, default='citys', help='dataset name (default: citys)')
parser.add_argument('--weights-folder', default='./weights', help='Directory for saving checkpoint models')
parser.add_argument('--input-pic', type=str, default='./datasets/frankfurt_000001_058914_leftImg8bit.png', help='path to the input picture')
# parser.add_argument('--input-pic', type=str, default='./datasets/30048.jpg', help='path to the input picture')
parser.add_argument('--outdir', default='./test_result', type=str, help='path to save the predict result')
parser.add_argument('--cpu', dest='cpu', action='store_true')
parser.set_defaults(cpu=True)       # False

args = parser.parse_args()


def demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # image = Image.open(args.input_pic).convert('RGB')
    image = Image.open(args.input_pic).resize((2048, 1024)).convert('RGB')
    # img_name = os.path.splitext(os.path.split(args.input_pic)[-1])[0] + '.jpg'
    # image.save(os.path.join(args.outdir, img_name))
    print('image.shape:', image.size)       # (2048, 1024)
    image = transform(image).unsqueeze(0).to(device)
    print('image.shape2:', image.shape)     # [1, 3, 1024, 2048]
    model = get_fast_scnn(args.dataset, pretrained=True, root=args.weights_folder, map_cpu=args.cpu).to(device)        # pretrained=True
    print('Finished loading model!')
    model.eval()
    with torch.no_grad():
        start = time.time()
        outputs = model(image)
        print('outputs:', outputs[0].shape)     # [1, 19, 1024, 2048]
        print('cost_time:', str(time.time()-start))
    pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
    print('pred.shape:', pred.shape)
    print('pred:', pred)
    mask = get_color_pallete(pred, args.dataset)
    outname = os.path.splitext(os.path.split(args.input_pic)[-1])[0] + '.png'
    mask.save(os.path.join(args.outdir, outname))


if __name__ == '__main__':
    demo()
