import argparse
from pathlib import Path
import sys
import time
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig 
from src.data.coco import coco_dataset

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

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def main(args):
    device = torch.device(args.device)
    reader = ImageReader(resize=640)
    model = Model(confg=args.config, ckpt=args.ckpt)
    model.to(device=device)
    img_path_list = []
    all_inf_time = []

    if args.imgpath != None:
        possible_img_extension = ['.jpg', '.jpeg', '.JPG', '.bmp', '.png']
        for (root, dirs, files) in os.walk(args.imgpath):
            if len(files) > 0:
                for file_name in files:
                    if os.path.splitext(file_name)[1] in possible_img_extension:
                        img_path = root + '/' + file_name
                        img_path_list.append(img_path)
    else:
        img_path_list.append(args.image)
    
    for path in img_path_list:
        img_path = Path(path)
        img = reader(img_path).to(device)
        size = torch.tensor([[img.shape[2], img.shape[3]]]).to(device)
        
        start_time = time.time()
        output = model(img, size)
        inf_time = time.time() - start_time
        fps = float(1/inf_time)
        print(f"Inferece time = {inf_time:.4f} s")
        print(f"FPS = {fps:.2f}")
        all_inf_time.append(inf_time)
        
        labels, boxes, scores = output
        
        im = reader.pil_img
        draw = ImageDraw.Draw(im)
        thrh = args.threshold

        for i in range(img.shape[0]):

            scr = scores[i]
            lab = labels[i][scr > thrh]
            box = boxes[i][scr > thrh]

            # Map the category ID to the class name
            # print('Model predict:', lab[i])
            if lab[i].size != 0:
                category_id = coco_dataset.mscoco_label2category[lab[i]]
                class_name = coco_dataset.mscoco_category2name[category_id]

                for b in box:
                    draw.rectangle(list(b), outline='red')
                    draw.text((b[0], b[1]), text=str(class_name), fill='yellow')
                
        file_dir = Path(img_path).parent.parent / 'torch_output'
        createDirectory(file_dir)
        new_file_name = os.path.basename(img_path).split('.')[0] + '_torch'+ os.path.splitext(img_path)[1]
        new_file_path = file_dir / new_file_name
        print('New File Path: ', new_file_path)
        print("================================================================================")
        im.save(new_file_path)

    avr_time = sum(all_inf_time) / len(img_path_list)
    avr_fps = float(1/avr_time)
    print('All images count: {}'.format(len(img_path_list)))
    print(f"Average Inference time = {avr_time:.4f} s")
    print(f"Average FPS = {avr_fps:.2f}")
 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str, ) #pth
    parser.add_argument("--ckpt", '-w', type=str, ) #pth
    parser.add_argument("--image", '-i', type=str, ) #pth
    parser.add_argument("--imgpath", '-ipth', type=str, default=None) #pth
    parser.add_argument("--threshold", '-t', type=float, default=0.6)
    parser.add_argument("--device", '-d', default="cpu")
    args = parser.parse_args()

    main(args)