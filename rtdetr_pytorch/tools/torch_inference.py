import argparse
from pathlib import Path
import sys
import time
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig 

import torch
from torch import nn
from PIL import Image, ImageDraw
from torchvision import transforms

class ImageReader:
    def __init__(self, resize=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = transforms.Compose([
            # transforms.Resize((resize, resize)) if isinstance(resize, int) else transforms.Resize(
            #     (resize[0], resize[1])),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
        ])
        self.resize = resize
        self.pil_img = None   

    def __call__(self, image_path, *args, **kwargs):
        self.pil_img = Image.open(image_path).convert('RGB').resize((self.resize, self.resize))
        return self.transform(self.pil_img).unsqueeze(0)


class Model(nn.Module):
    def __init__(self, confg=None, ckpt="") -> None:
        super().__init__()
        self.cfg = YAMLConfig(confg, resume=ckpt)
        if ckpt:
            checkpoint = torch.load(ckpt, map_location='cpu') 
            if 'ema' in checkpoint:
                state = checkpoint['ema']['module']
            else:
                state = checkpoint['model']
        else:
            raise AttributeError('only support resume to load model.state_dict by now.')

        # NOTE load train mode state -> convert to deploy mode
        self.cfg.model.load_state_dict(state)

        self.model = self.cfg.model.deploy()
        self.postprocessor = self.cfg.postprocessor.deploy()
        # print(self.postprocessor.deploy_mode)
        
    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)


def main(image_path, device, config, checkpoint):
    img_path = Path(image_path)
    device = torch.device(device)
    reader = ImageReader(resize=640)
    model = Model(confg=config, ckpt=checkpoint)
    model.to(device=device)

    img = reader(img_path).to(device)
    size = torch.tensor([[img.shape[2], img.shape[3]]]).to(device)
    
    start_time = time.time()
    output = model(img, size)
    inf_time = time.time() - start_time
    fps = float(1/inf_time)
    print("Inferece time = {} s".format(inf_time, '.4f'))
    print("FPS = {} ".format(fps, '.1f') )
    
    labels, boxes, scores = output
    
    im = reader.pil_img
    draw = ImageDraw.Draw(im)
    thrh = 0.6

    for i in range(img.shape[0]):

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]

        for b in box:
            draw.rectangle(list(b), outline='red', )
            draw.text((b[0], b[1]), text=str(lab[i]), fill='blue', )

    # save_path = Path(args.output_dir) / img_path.name
    file_dir = os.path.dirname(image_path)
    new_file_name = os.path.basename(image_path).split('.')[0] + '_torch'+ os.path.splitext(image_path)[1]
    new_file_path = file_dir + '/' + new_file_name
    print('new_file_path: ', new_file_path)
    im.save(new_file_path)
 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str, )
    parser.add_argument("--ckpt", '-w', type=str, ) # pth
    parser.add_argument("--image", '-i', type=str, ) 
    parser.add_argument("--device", '-d', default="cpu")
    args = parser.parse_args()

    main(args)