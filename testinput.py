import os
import cv2
import torch
import numpy as np
import torch.nn as nn
bucket_name = "momentoon-cd88f.appspot.com"

from tqdm import tqdm
from torch.nn import functional as F
import inference_image



class ResBlock(nn.Module):
    def __init__(self, num_channel):
        super(ResBlock, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(num_channel, num_channel, 3, 1, 1),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channel, num_channel, 3, 1, 1),
            nn.BatchNorm2d(num_channel))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs):
        output = self.conv_layer(inputs)
        output = self.activation(output + inputs)
        return output


class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownBlock, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 2, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))


    def forward(self, inputs):
        output = self.conv_layer(inputs)
        return output


class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, is_last=False):
        super(UpBlock, self).__init__()
        self.is_last = is_last
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1))
        self.act = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))
        self.last_act = nn.Tanh()


    def forward(self, inputs):
        output = self.conv_layer(inputs)
        if self.is_last:
            output = self.last_act(output)
        else:
            output = self.act(output)
        return output



class SimpleGenerator(nn.Module):
    def __init__(self, num_channel=32, num_blocks=4):
        super(SimpleGenerator, self).__init__()
        self.down1 = DownBlock(3, num_channel)
        self.down2 = DownBlock(num_channel, num_channel*2)
        self.down3 = DownBlock(num_channel*2, num_channel*3)
        self.down4 = DownBlock(num_channel*3, num_channel*4)
        res_blocks = [ResBlock(num_channel*4)]*num_blocks
        self.res_blocks = nn.Sequential(*res_blocks)
        self.up1 = UpBlock(num_channel*4, num_channel*3)
        self.up2 = UpBlock(num_channel*3, num_channel*2)
        self.up3 = UpBlock(num_channel*2, num_channel)
        self.up4 = UpBlock(num_channel, 3, is_last=True)

    def forward(self, inputs):
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down4 = self.res_blocks(down4)
        up1 = self.up1(down4)
        up2 = self.up2(up1+down3)
        up3 = self.up3(up2+down2)
        up4 = self.up4(up3+down1)
        return up4



weight = torch.load('weight.pth', map_location='cpu')
model = SimpleGenerator()
model.load_state_dict(weight)
#torch.save(model.state_dict(), 'weight.pth')
model.eval()


path_UF = "C:/Users/k/Desktop/FLASK/uploads" #Unfiltered, original image
path_BU = "C:/Users/k/Desktop/FLASK/storage"
path_RS = "C:/Users/k/Desktop/FLASK/static/results"
codeToDownload = 'temp'
model_list = [] # TODO: Inserting model, currently we have only one temporary model.


path_filtered = "images/FILTERED/"
path_archive = "images/ARCHIVE/"


path_hayao = "C:/Users/k/Desktop/FLASK/UF_hayao"
path_mikoto = "C:/Users/k/Desktop/FLASK/UF_mikoto"
path_modelC = "C:/Users/k/Desktop/FLASK/UF_wind52"
path_windb = "C:/Users/k/Desktop/FLASK/UF_wind58"

#for i in range(0,2):
while 1:
                for file in os.listdir(path_UF):
                        #TODO : make image resizeable or models able to resize even if the picture isn't fit to model.
                        print(file)
                        if codeToDownload in file:
                            if(file[4] == 'N'):# previous prototype images 
                                load_path = os.path.join(path_UF, file)
                                save_path = os.path.join(path_RS, "Result"+file)
                                raw_image = cv2.imread(load_path)

                                tempx, tempy, tempz= raw_image.shape
                                raw_image = cv2.resize(raw_image ,(256,256), interpolation = cv2.INTER_AREA)
                                cv2.imwrite(save_path, raw_image)

                                image = raw_image/127.5 - 1
                                image = image.transpose(2, 0, 1)
                                image = torch.tensor(image).unsqueeze(0)
                                output = model(image.float())
                                output = output.squeeze(0).detach().numpy()
                                output = output.transpose(1, 2, 0)
                                output = (output + 1) * 127.5
                                output = np.clip(output, 0, 255).astype(np.uint8)
                                #output = np.concatenate([raw_image, output], axis=1)

                                cv2.imwrite(save_path, output)


                                os.remove(path_UF+"/"+file)

                                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                                sr.readModel('EDSR_x3.pb')
                                sr.setModel('edsr', 3)
                                load_path = os.path.join(path_RS, "Result"+file)
                                save_path = os.path.join(path_RS, "Result"+file)
                                print(save_path, ' ', tempy, ' ', tempx)
                                raw_image = cv2.imread(load_path)
                                result = sr.upsample(raw_image)
                                raw_image = cv2.resize(result ,(tempy ,tempx), interpolation = cv2.INTER_AREA)
                                cv2.imwrite(save_path, raw_image)

                            if(file[4] == '0'): # Hayao
                                load_path = os.path.join(path_UF, file)
                                save_path = os.path.join(path_RS, "Result"+file)
                                raw_image = cv2.imread(load_path)

                                tempx, tempy, tempz= raw_image.shape
                                raw_image = cv2.resize(raw_image ,(256,256), interpolation = cv2.INTER_AREA)
                                cv2.imwrite(save_path, raw_image) # save image

                                image = raw_image/127.5 - 1
                                image = image.transpose(2, 0, 1)
                                image = torch.tensor(image).unsqueeze(0)
                                output = model(image.float())
                                output = output.squeeze(0).detach().numpy()
                                output = output.transpose(1, 2, 0)
                                output = (output + 1) * 127.5
                                output = np.clip(output, 0, 255).astype(np.uint8)
                                #output = np.concatenate([raw_image, output], axis=1)

                                cv2.imwrite(save_path, output)


                                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                                sr.readModel('EDSR_x3.pb')
                                sr.setModel('edsr', 3)
                                load_path = os.path.join(path_RS, "Result"+file)
                                save_path = os.path.join(path_RS, "Result"+file)
                                print(save_path, ' ', tempy, ' ', tempx)
                                raw_image = cv2.imread(load_path)
                                result = sr.upsample(raw_image)
                                raw_image = cv2.resize(result ,(tempy ,tempx), interpolation = cv2.INTER_AREA)
                                cv2.imwrite(save_path, raw_image)
                            if(file[4] == '1'):
                                load_path = os.path.join(path_UF, file)
                                save_path = os.path.join(path_hayao, "Result"+file)
                                raw_image = cv2.imread(load_path)

                                tempx, tempy, tempz= raw_image.shape
                                raw_image = cv2.resize(raw_image ,(256,256), interpolation = cv2.INTER_AREA)
                                cv2.imwrite(save_path, raw_image)
                                inference_image.animeGAN('C:/Users/k/Desktop/FLASK/contents/checkpoint/generator_hayao.pth', path_hayao, path_RS)
                                os.remove(path_UF+"/"+file)
                                os.remove(path_hayao+"/"+'Result'+file)

                                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                                sr.readModel('EDSR_x3.pb')
                                sr.setModel('edsr', 3)
                                load_path = os.path.join(path_RS, "Result"+file)
                                save_path = os.path.join(path_RS, "Result"+file)
                                print(save_path, ' ', tempy, ' ', tempx)
                                raw_image = cv2.imread(load_path)
                                result = sr.upsample(raw_image)
                                raw_image = cv2.resize(result ,(tempy ,tempx), interpolation = cv2.INTER_AREA)
                                cv2.imwrite(save_path, raw_image)
                            if(file[4] == '2'):
                                load_path = os.path.join(path_UF, file)
                                save_path = os.path.join(path_mikoto, "Result"+file)
                                raw_image = cv2.imread(load_path)

                                tempx, tempy, tempz= raw_image.shape
                                raw_image = cv2.resize(raw_image ,(256,256), interpolation = cv2.INTER_AREA)
                                cv2.imwrite(save_path, raw_image)
                                inference_image.animeGAN('C:/Users/k/Desktop/FLASK/contents/checkpoint/generator_shinkai.pth', path_mikoto, path_RS)
                                os.remove(path_UF+"/"+file)
                                os.remove(path_mikoto+"/"+'Result'+file)


                                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                                sr.readModel('EDSR_x3.pb')
                                sr.setModel('edsr', 3)
                                load_path = os.path.join(path_RS, "Result"+file)
                                save_path = os.path.join(path_RS, "Result"+file)
                                print(save_path, ' ', tempy, ' ', tempx)
                                raw_image = cv2.imread(load_path)
                                result = sr.upsample(raw_image)
                                raw_image = cv2.resize(result ,(tempy ,tempx), interpolation = cv2.INTER_AREA)
                                cv2.imwrite(save_path, raw_image)

                            if(file[4] == '3'):
                                load_path = os.path.join(path_UF, file)
                                save_path = os.path.join(path_modelC, "Result"+file)
                                raw_image = cv2.imread(load_path)
                                tempx, tempy, tempz= raw_image.shape
                                raw_image = cv2.resize(raw_image ,(256,256), interpolation = cv2.INTER_AREA)
                                cv2.imwrite(save_path, raw_image)
                                inference_image.animeGAN('C:/Users/k/Desktop/FLASK/contents/checkpoint/generator_wind58.pth', path_modelC, path_RS)
                                os.remove(path_UF+"/"+file)
                                os.remove(path_modelC+"/"+'Result'+file)

                                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                                sr.readModel('EDSR_x3.pb')
                                sr.setModel('edsr', 3)
                                save_path = os.path.join(path_RS, "Result"+file)
                                raw_image = cv2.imread(save_path)
                                result = sr.upsample(raw_image)
                                raw_image = cv2.resize(result ,(tempy ,tempx), interpolation = cv2.INTER_AREA)
                                cv2.imwrite(save_path, raw_image)

                            if(file[4] == '4'):
                                load_path = os.path.join(path_UF, file)
                                save_path = os.path.join(path_windb, "Result"+file)
                                raw_image = cv2.imread(load_path)
                                tempx, tempy, tempz= raw_image.shape
                                print( "B:", tempx, tempy , tempz)
                                raw_image = cv2.resize(raw_image ,(256,256), interpolation = cv2.INTER_AREA)
                                cv2.imwrite(save_path, raw_image)
                                inference_image.animeGAN('C:/Users/k/Desktop/FLASK/contents/checkpoint/generator_be14.pth', path_windb, path_RS)
                                os.remove(path_UF+"/"+file)
                                os.remove(path_windb+"/"+'Result'+file)

                                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                                sr.readModel('EDSR_x3.pb')
                                sr.setModel('edsr', 3)
                                load_path = os.path.join(path_RS, "Result"+file)
                                save_path = os.path.join(path_RS, "Result"+file)
                                print(save_path, ' ', tempy, ' ', tempx)
                                raw_image = cv2.imread(load_path)
                                result = sr.upsample(raw_image)
                                raw_image = cv2.resize(result ,(tempy ,tempx), interpolation = cv2.INTER_AREA)
                                cv2.imwrite(save_path, raw_image)