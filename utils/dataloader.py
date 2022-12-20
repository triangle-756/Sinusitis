# 패키지 import
import imghdr
import re
import random
from unittest import result
import tqdm
from matplotlib.image import imread
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
import cv2
import numpy as np
import pydicom as pydcm
from PIL import Image, ImageChops
import torch.utils.data as data
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from utils.data_augumentation import Compose,Compose2, RandomScale, RandomRotation, RandomMirror, Resize, Normalize_Tensor, Crop, RandomShift


'''
여기에는 SinusitisDataset, SinusitisDataset2, DataTransform, DataTransform2, KeyDataTransform 존재. 
KeyDataTransform은 3DCNN keyslice 후 전처리과정 관련 datatransform 클래스.

dataload 과정은 크게 4 step으로 구성됨. 
make_img_dataframe은 현재 directory에 저장되어 있는 raw image들의 정보를 읽는 과정.
make_data_list는 읽어낸 raw image 중 학습에 필요한 이미지만 선택해 target과 대응시키는 과정.
이때 split_to_k 함수로 k개의 split으로 나눔.
train_val_select에서 학습 데이터들을 train과 val로 나눔. SinusitisDataset2의 경우 k_fold cross validation을 고려해 나눔.
이후 pull_item에서 원하는 index의 img, target tuple을 전처리 후 반환함.

지금 버그는 아마 make_data_list에서 학습에 필요한 이미지를 선택하는 과정에서 버그가 있는것으로 생각됨.

몇몇 debug 용 print문이 있는 것 유의.
'''






def target_vectorize(str, num_classes):
    clss = str.split('+')
    tmp = list(map(lambda x: F.one_hot(torch.tensor(int(x)), num_classes=num_classes), clss))
    label = torch.zeros(num_classes)
    for i in range(len(tmp)):
        label += tmp[i]
    return label

def build_KeySliceNet(path=None):
    print('Loading pretrained efficientnet-b0')
    net = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    def weights_init(m):
      if isinstance(m, nn.Linear):
          nn.init.xavier_normal_(m.weight.data)
          if m.bias is not None:  
              nn.init.constant_(m.bias, 0.0)
    net.apply(weights_init)
    if path != None:
        net.load_state_dict(torch.load(path))
    return net

class DataTransform():
    """
    화상과 어노테이션의 전처리 클래스. 훈련시와 검증시 다르게 동작한다.
    화상의 크기를 input_size x input_size로 한다.
    훈련시에 데이터 확장을 수행한다.


    Attributes
    ----------
    input_size : int
        리사이즈 대상 화상의 크기.
    color_mean : (R, G, B)
        각 색상 채널의 평균값.
    color_std : (R, G, B)
        각 색상 채널의 표준편차.
    """

    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train': Compose([
                Crop(0.8),
                RandomRotation(angle=[-10, 10]),  # 회전
                Resize(input_size),  # 리사이즈(input_size)
                Normalize_Tensor(color_mean, color_std)  # 색상 정보의 표준화와 텐서화
            ]),
            'val': Compose([
                Crop(0.8),
                Resize(input_size),  # 리사이즈(input_size)
                Normalize_Tensor(color_mean, color_std)  # 색상 정보의 표준화와 텐서화
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            전처리 모드를 지정.
        """
        return self.data_transform[phase](img, anno_class_img)


class SinusitisDataset(data.Dataset):
    """
    Dataset을 만드는 클래스. PyTorch의 Dataset 클래스를 상속받는다.

    Attributes
    ----------
    img_list : 리스트
        화상의 경로를 저장한 리스트
    anno_list : 리스트
        어노테이션의 경로를 저장한 리스트
    phase : 'train' or 'test'
        학습 또는 훈련을 설정한다.
    transform : object
        전처리 클래스의 인스턴스
    """

    def __init__(self, output_type, transform, rootpath=None, data=None, kfold=False):
        super(SinusitisDataset, self).__init__()
        self.output_type=output_type
        self.transform = transform
        self.k=kfold
        if rootpath!=None:
            self.input_type = 'path'
            self.imgrootpath, self.targetrootpath = rootpath
            self.make_img_dataframe()
            self.make_data_list()
        elif data!=None:
            # self.img = cv2.imread(img_path)
            self.input_type = 'data'
            self.img, self.anno = data
            self.img = np.reshape(self.img, (-1,)+self.img.shape[2:])
            # self.anno = cv2.imread(anno_path)
            self.anno = np.reshape(self.anno, (-1,)+self.anno.shape[2:])
        if rootpath!=None or data != None:
            self.train_val_select(index=0)

    def __len__(self):
        '''화상의 매수를 반환'''
        if self.input_type == 'data':
            return len(self.img_train_val[self.phase])
        elif self.input_type == 'path':
            return len(self.img_train_val_path_list[self.phase])

    def __getitem__(self, index):
        '''
        전처리한 화상의 텐서 형식 데이터와 어노테이션을 취득
        '''
        img, target, path = self.pull_item(index)
        return img, target, path
    def make_img_dataframe(self):
        #img 분산
        #dataframe화
        walk = os.listdir(self.imgrootpath)
        self.img_dataframe = pd.DataFrame(columns=['patientID', 'Date', 'Path'])
        for path in tqdm.tqdm(walk):
            files = os.listdir(self.imgrootpath+'/'+path)
            for File in files:
                dcm = pydcm.dcmread(self.imgrootpath+'/'+path+'/'+File)
                try:
                    patientID=int(dcm.PatientID)
                except ValueError:
                    continue
                date = dcm.StudyDate
                try:
                    dcm.ContrastBolusAgent
                except AttributeError:
                    if date not in self.img_dataframe[self.img_dataframe['patientID']==patientID]['Date']:
                        self.img_dataframe.loc[len(self.img_dataframe.index)] = [patientID, date, self.imgrootpath+'/'+path]
    def make_data_list(self):
        #input 분산, output 분산(mask) or 모아서(classification)
        if self.output_type == 'mask':
            # input dir, output dir&mask
            # Left, Right 폴더 안에 각 mask 데이터 존재
            left = self.targetrootpath+'/left'
            right = self.targetrootpath+'/right'
            left_list = list(map(int,os.listdir(left))); right_list = list(map(int,os.listdir(right)))
            self.img_path_list = list()
            self.mask_path_list = list()
            for i, patientID in tqdm.tqdm(enumerate(self.img_dataframe['patientID'])):
                mask_path = (right+f'/{patientID:08d}' if patientID in right_list else None, left+f'/{patientID:08d}' if patientID in left_list else None)
                if not all(mask_path):
                    print(f'{patientID:08d}의 mask data가 없습니다: {mask_path}')
                else:
                    imgs_path = self.imgrootpath+f'/{patientID:08d}'
                    masks_path = mask_path
                    slices_lists = sorted(os.listdir(imgs_path))
                    masks_lists = map(lambda x: sorted(os.listdir(x)), masks_path)
                    img_list = list()
                    mask_label = list()
                    data_image = list()
                    for r_path,l_path in zip(*masks_lists):
                        png_r = torch.tensor(cv2.imread(masks_path[0]+'/'+r_path))
                        if torch.sum(png_r)<10:
                            mask_label.insert(0,torch.tensor(0))
                        else:
                            mask_label.insert(0,torch.tensor(1))
                        # png = T.functional.hflip(T.functional.rotate(png,90))
                        # mask.insert(0,png)
                    #img_list = find_discrete(slices_lists, imgs_path)
                    for slice_path in slices_lists:
                        dcm = pydcm.dcmread(imgs_path+'/'+slice_path)
                        if self.PNS_filter(dcm):
                            img_list.append(imgs_path+'/'+slice_path) # [slice][색RGB][높이][폭] tensor
                    self.img_path_list.append(img_list)
                    self.mask_path_list.append(mask_label)
                    if len(img_list) != len(mask_label):
                        print(f'{patientID:08d}의 img와 mask 갯수가 다릅니다:{len(img_list)}, {len(mask_label)}')
                        break
                    #print(i)
            self.k_datasets = self.split_to_k((self.img_path_list, self.mask_path_list),5)
                    
        elif self.output_type == 'classification':
            #study date, ID, gender, age, RT, Lt, discrepency, True Rt, True Lt, pathology
            output = pd.read_excel(self.targetrootpath)
            output['study date'] = output['study date'].map(lambda x: x.strftime("%Y%m%d") if isinstance(x,datetime.datetime) else '20'+str(x)[-6:])
            self.img_path_list =list()
            self.label = list() 
            self.data_img = list()
            for index in tqdm.tqdm(range(len(output.index))):
                row=output.iloc[index]
                tmp = self.img_dataframe[self.img_dataframe['patientID'].astype(int)==int(row['ID'])]
                tmp = tmp[tmp['Date'].astype(int)==int(row['study date'])]
                if len(tmp.index) == 0:
                    print(f"no image of patientID:{row['ID']}, study date:{row['study date']}")
                else:
                    img_path = tmp['Path'].iloc[0]
                    slice_lists = sorted(os.listdir(img_path))   
                    images = list()
                    data_images = list()
                    for slice_path in slice_lists:
                        dcm = pydcm.dcmread(img_path+'/'+slice_path)
                        if self.PNS_filter(dcm):
                            img = dcm.pixel_array
                            data_images.append(img)
                            images.append(img_path+'/'+slice_path)
                    if len(img)==0:
                        print(tmp)
                        print(slice_lists)
                    self.img_path_list.append(images)
                    self.data_img.append(data_images)
                    self.label.append((target_vectorize(str(row['Rt']), 5), target_vectorize(str(row['Lt']), 5)))
            torch.save(self.data_img, '/content/drive/MyDrive/Resnet_data_img.pt')
            self.k_datasets = self.split_to_k((self.img_path_list, self.label),5)


    def split_to_k(self, datasets, k):
        inputs, outputs = datasets
        tmp = list(zip(*(inputs, outputs)))
        rand = random.Random(1234)
        rand.shuffle(tmp)
        inputs, outputs = list(zip(*tmp))
        k_inputs=list();k_outputs=list()
        for i in range(0,k):
            k_inputs.append(inputs[int(len(inputs)*i/k):int(len(inputs)*(i+1)/k)])
            k_outputs.append(outputs[int(len(outputs)*i/k):int(len(outputs)*(i+1)/k)])
        return (k_inputs, k_outputs)


    def set_phase(self, phase):
        self.phase=phase
        return self

    def train_val_select(self, ratio=(0.8,0.1,0.1), index=0):
        if self.k>0:
            self.img_train_val_path_list = {'train':[], 'val':[]}
            self.target_train_val = {'train':[], 'val':[]}
            for i in range(0,self.k):
                if i!=index:
                    self.img_train_val_path_list['train']+=self.k_datasets[0][i]
                    self.target_train_val['train']+=self.k_datasets[1][i]
                else:
                    self.img_train_val_path_list['val']+=self.k_datasets[0][i]
                    self.target_train_val['val']+=self.k_datasets[1][i]
            return 0
                    
        train, val, ext = ratio
        if self.input_type == 'data' and self.output_type== 'mask':
            self.randgen = torch.Generator()    
            ratio = int(self.img.shape[0]*(val+ext)) #should be int
            shuff = torch.randperm(self.img.shape[0],generator=self.randgen)
            self.img_train_val = {'train':self.img[shuff[ratio:],:], 'val':self.img[shuff[:ratio],:]}
            self.anno_train_val = {'train':self.anno[shuff[ratio:],:], 'val':self.anno[shuff[:ratio],:]}

        elif self.input_type == 'path' and self.output_type =='classification':
            self.rand = random.Random(1234)
            tmp = list(zip(*(self.img_path_list, self.label)))
            self.rand.shuffle(tmp)
            self.img_path_list, self.label = zip(*tmp)
            ratio = int(len(self.label)*(val+ext))
            self.img_train_val_path_list = {'train':self.img_path_list[ratio:], 'val':self.img_path_list[:ratio]}
            self.target_train_val = {'train':self.label[ratio:], 'val':self.label[:ratio]}

        elif self.input_type == 'path' and self.output_type == 'mask':
            self.rand = random.Random(1234)
            tmp = list(zip(*(self.img_path_list, self.mask_path_list)))
            self.rand.shuffle(tmp)
            self.img_path_list, self.mask_path_list = zip(*tmp)
            ratio = int(len(self.mask_path_list)*(val+ext))
            self.img_train_val_path_list = {'train':self.img_path_list[ratio:], 'val':self.img_path_list[:ratio]}
            self.target_train_val = {'train':self.mask_path_list[ratio:], 'val':self.mask_path_list[:ratio]}            
            
    def PNS_filter(self, dcm):
      for x in ['pns', 'non']:
          if (x not in dcm.SeriesDescription.lower()):
              return False
      if not int(self.img_dataframe[self.img_dataframe['patientID'].astype(int)==
              int(dcm.PatientID)]['Date'].max()) == int(dcm.SeriesDate):
        return False
      return True
            
    def pull_item(self, index):
        '''화상의 텐서 형식 데이터, 어노테이션을 취득한다'''
        if self.input_type == 'data' and self.output_type == 'mask':
            # 1. 화상 읽기
            image = torch.tensor(self.img_train_val[self.phase][index])   # [높이][폭][색RGB] ndarray

            # 2. 어노테이션 화상 읽기
            anno_class_img = torch.tensor(self.anno_train_val[self.phase][index])   # [높이][폭] ndarray

            # 3. 전처리 실시
            image = T.functional.to_pil_image(image[None,:,:].expand(3,-1,-1))
            anno_class_img = T.functional.to_pil_image(anno_class_img.to(torch.float64))
            

            img, anno_class_img = self.transform(self.phase, (image, anno_class_img))

            return img, anno_class_img

        elif self.input_type == 'path':
            imgs_path = self.img_train_val_path_list[self.phase][index]
            target = self.target_train_val[self.phase][index]
            image = list()  
            imgs = list()
            tmp = imgs_path if type(imgs_path)==type([]) else [imgs_path]
            for slice_path in tmp:
                dcm = pydcm.dcmread(slice_path)
                dcm_ = dcm.pixel_array
                dcmimage = (dcm_-np.min(dcm_))/(np.max(dcm_)-np.min(dcm_))
                image.append(T.functional.to_pil_image(torch.Tensor(dcm.pixel_array.astype('int16')).expand(3,-1,-1))) # [slice][색RGB][높이][폭] tensor
                imgs.append(torch.tensor(dcmimage))
            img,_ = self.transform(self.phase, image, image)
            return (imgs, img[0], target, imgs_path) if len(img)==1 else (imgs, img, target, imgs_path)

    def reduce_dim(self):
      def flatten(lis):
          ret = list()
          for l in lis:
            ret = ret + l
          return ret
      self.target_train_val = {k:flatten(v) for k,v in self.target_train_val.items()}
      self.img_train_val_path_list = {k:flatten(v) for k,v in self.img_train_val_path_list.items()}
      #self.target_train_val

'''
a = dl.SinusitisDataset2()
inport data of a
a.dataset_split() -> img_train_val_path_list2, target_train_val에 'R' 'L' tag 추가
'''
class SinusitisDataset2(SinusitisDataset):
    def __init__(self, output_type, transform, netpath ='/content/drive/MyDrive/weights and log/efficientnet-b0__120_1.0E-05_1.0E-05_192_-1_1.0E+00_1.0E+00.pth',
                efnetpath = '/content/drive/MyDrive/unet_modified_data/checkpoint/model_epoch29.pth', rootpath=None, data=None, kfold=False):
        super(SinusitisDataset2, self).__init__(output_type, KeyDataTransform(input_size = 300,color_mean = (0.485),color_std = (0.229)), rootpath=rootpath, data=data, kfold=kfold)
        if rootpath!=None or data!=None:
            self.train_val_select(index=0)
        self.transform_3D=transform
        self.netdict = torch.load(efnetpath)['net']
        self.net_seg = modified_unet()
        self.net_seg.load_state_dict(self.netdict)
        self.net_seg.to('cpu', dtype=torch.float)
        self.device = "cpu"
        print("사용 장치: ", self.device)
        self.net = build_KeySliceNet(path = netpath)
        self.net = self.net.to(self.device)
        self.net.eval()

    def __len__(self):
        '''화상의 매수를 반환'''
        return len(self.img_train_val_path_list[self.phase]*2)

    def keysliceCNN(self, imgs, targets):
        batch = 8
        outputs = torch.tensor([])
        for i in range(0, len(imgs)//batch+1):
            s = i*batch; e = (i+1)*batch if (i+1)*batch<=len(imgs) else len(imgs)
            bt_img = torch.tensor([])
            if s==e:
                break
            for img in imgs[s:e]:
                bt_img = torch.cat((bt_img, img.expand([1]+[-1 for x in img.shape])))

            bt_img = bt_img.to(self.device)
            output = F.softmax(self.net(bt_img), dim=1)
            output = [(x[0]<x[1]).item() for x in output]
            outputs=torch.cat((outputs,bt_img[output].to('cpu')))
        if len(outputs.size())==1:
            print(len(imgs))
            tmp = torch.tensor([])
            for i in range(0, len(imgs)):
                tmp = torch.cat((tmp,imgs[i].expand([1]+[-1 for x in imgs[i].shape])))
            outputs = tmp
        outputs = torch.einsum('ijkl->jikl',outputs).expand(1,-1,-1,-1,-1) # [batch][3][height][width] -> [3][batch][height][width]
        return outputs 

    def keysliceCNN2(self, imgs):
        batch = 1
        outputs = []
        for i in range(0, len(imgs)//batch+1):
            s = i*batch; e = (i+1)*batch if (i+1)*batch<=len(imgs) else len(imgs)
            bt_img = torch.tensor([])
            if s==e:
                break
            for img in imgs[s:e]:
                bt_img = torch.cat((bt_img, img.expand([1]+[-1 for x in img.shape])))
            bt_img = bt_img.to(self.device)
            output = F.softmax(self.net(bt_img), dim=1)
            output = [(x[0]<x[1]).item() for x in output]
            outputs += output
        return outputs 
    
    def detach_ROI(self, img):
        raw_img = img.reshape(512,512).detach().numpy()
        result_1 = self.net_seg(img.reshape(1,1,512,512).to('cpu', dtype=torch.float)).reshape(512,512).detach().numpy()
        result_1 = np.where(result_1<0, 0, result_1)
        result_1 = np.where(result_1!=0, 1, result_1)
        result = np.multiply(raw_img, result_1)
        return torch.Tensor(result)

    def pull_item(self, index):
        raw_imgs, imgs, targets, imgs_path = super().pull_item(index//2)
        key_ = self.keysliceCNN2(imgs)
        #result_img = self.detach_ROI(raw_imgs[key_.index(True)]).expand(3,-1,-1).reshape(1,3,1,512,512)
        #for i in range(len(key_)):
        #    if key_[i] == True:
        #        raw_img = self.detach_ROI(raw_imgs[i]).expand(3,-1,-1).reshape(1,3,1,512,512)
        #        result_img = torch.cat([result_img, raw_img], dim=2)
        result_img = raw_imgs[key_.index(True)].expand(3,-1,-1).reshape(1,3,1,512,512)
        for i in range(key_.index(True)+1, len(key_)):
            if key_[i] == True:
                raw_img = raw_imgs[i].expand(3,-1,-1).reshape(1,3,1,512,512)
                result_img = torch.cat([result_img, raw_img],dim=2)
        if index%2==0: #'R'
            result_img = result_img[:,:,:,:,:result_img.size(dim=-1)//2]
            targets = targets[0]
        if index%2==1: #'L'
            result_img = result_img[:,:,:,:,(result_img.size(dim=-1)//2):]
            targets = targets[1]
        result_img = F.interpolate(result_img, size=(160,160,160), mode = 'area')
        #img,_=self.transform_3D(self.phase, result_img, result_img)
        result_img = result_img.reshape(3,160,160,160)
        return result_img, targets, imgs_path
        
class KeyDataTransform():
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train': Compose([
                Crop(0.8),
                Resize(input_size),  # 리사이즈(input_size)
                Normalize_Tensor(color_mean, color_std)  # 색상 정보의 표준화와 텐서화
            ]),
            'val': Compose([
                Crop(0.8),
                Resize(input_size),  # 리사이즈(input_size)
                Normalize_Tensor(color_mean, color_std)  # 색상 정보의 표준화와 텐서화
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            전처리 모드를 지정.
        """
        return self.data_transform[phase](img, anno_class_img)

class DataTransform2():
    def __init__(self, color_mean, color_std):
        self.data_transform = {
            'train': Compose2([
                RandomScale(scale = [0.9, 1.1]),
                RandomShift(pixel = [-5, 5]),
                #Normalize_Tensor(color_mean, color_std)  # 색상 정보의 표준화와 텐서화
            ]),
            'val': Compose2([
                RandomScale(scale = [0.9, 1.1]),
                RandomShift(pixel = [-5, 5]),
                #Normalize_Tensor(color_mean, color_std)  # 색상 정보의 표준화와 텐서화
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            전처리 모드를 지정.
        """
        return self.data_transform[phase](img, anno_class_img)

class modified_unet(nn.Module):
    def __init__(self):
        super(modified_unet, self).__init__()
        
        self.maxpool = nn.MaxPool3d(kernel_size=2)
        self.relu = nn.ReLU()

        def conv(in_ch,out_ch,ks,s,p=1):
            layers = []
            layers += [
                       nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks, stride=s, padding=p),
                       nn.BatchNorm2d(num_features=out_ch),
                       nn.ReLU()
                       ]
            cbr = nn.Sequential(*layers)
            return cbr
        
        self.maxp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_1_1 = conv(1  ,64 , 3, 1)
        self.enc_1_2 = conv(64 ,64 , 3, 1)

        self.enc_2_1 = conv(64 ,128, 3, 1)
        self.enc_2_2 = conv(128,128, 3, 1)
        
        self.enc_3_1 = conv(128,256, 3, 1)
        self.enc_3_2 = conv(256,256, 3, 1)
        
        self.enc_4_1 = conv(256,512, 3, 1)
        self.enc_4_2 = conv(512,512, 3, 1)

        self.enc_5_1 = conv(512,1024, 3, 1, 1)
        self.enc_5_2 = conv(1024,1024, 3, 1, 1)

        self.unp_5 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec_5_1 = conv(1024, 512, 3, 1)
        self.dec_5_2 = conv(512, 512, 3, 1)

        self.unp_4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec_4_1 = conv(512, 256, 3, 1)
        self.dec_4_2 = conv(256, 256, 3, 1)

        self.unp_3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec_3_1 = conv(256, 128, 3, 1)
        self.dec_3_2 = conv(128, 128, 3, 1)

        self.unp_2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec_2_1 = conv(128, 64, 3, 1)
        self.dec_2_2 = conv(64, 64, 3, 1)

        self.dec_1 = nn.Conv2d(64,1,1,1,0)

    def forward(self, x):
        x = self.enc_1_1(x)
        x = self.enc_1_2(x)
        x1 = x
        x = self.maxp(x)

        x = self.enc_2_1(x)
        x = self.enc_2_2(x)
        x2 = x
        x = self.maxp(x)

        
        x = self.enc_3_1(x)
        x = self.enc_3_2(x)
        x3 = x
        x = self.maxp(x)
        
        x = self.enc_4_1(x)
        x = self.enc_4_2(x)
        x4 = x
        x = self.maxp(x)

        x = self.enc_5_1(x)
        x = self.enc_5_2(x)

        x = torch.cat([self.unp_5(x), x4], dim=1)
        x = self.dec_5_1(x)
        x = self.dec_5_2(x)

        x = torch.cat([self.unp_4(x), x3], dim=1)
        x = self.dec_4_1(x)
        x = self.dec_4_2(x)

        x = torch.cat([self.unp_3(x), x2], dim=1)
        x = self.dec_3_1(x)
        x = self.dec_3_2(x)

        x = torch.cat([self.unp_2(x), x1], dim=1)
        x = self.dec_2_1(x)
        x = self.dec_2_2(x)

        x = self.dec_1(x)

        torch.cuda.empty_cache()






        return x