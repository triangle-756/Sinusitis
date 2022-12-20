# 3장 시맨틱 분할의 데이터 확장
# 주의: 어노테이션 이미지는 색상 팔레트 형식(인덱스 컬러 이미지)로 되어 있음.

# 패키지 import
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFilter, ImageChops
import numpy as np
from skimage.transform import resize


class Compose(object):
    """transform 인수에 저장된 변형을 순차적으로 실행하는 클래스
       대상 화상과 어노테이션 화상을 동시에 변환합니다. 
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs, anno_class_imgs):
        def transform(i, img, anno_class_img):
            if i ==0:
                for t in self.transforms:
                    img, anno_class_img = t(img, anno_class_img)
            else:
                for t in self.transforms:
                    img, anno_class_img = t(img, anno_class_img, random_fix=True)
            return img, anno_class_img
        img=list(); anno_class_img=list()
        for i, (image, anno_class_image) in enumerate(zip(imgs, anno_class_imgs)):
              a,b = transform(i, image, anno_class_image)
              img.append(a);  anno_class_img.append(b)
        return img, anno_class_img

class Compose2(object):
    """transform 인수에 저장된 변형을 순차적으로 실행하는 클래스
       대상 화상과 어노테이션 화상을 동시에 변환합니다. 
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs, anno_class_imgs):
      for t in self.transforms:
        img,anno_class_img = t(imgs, anno_class_imgs)
      return img, anno_class_img

class Crop():
    def __init__(self, ratio = 0.8):
        self.ratio = ratio
    def __call__(self, img, anno_class_img, random_fix=False):
        #[slice][색RGB][높이][폭]
        width = img.size[0]
        height = img.size[1]
        img = transforms.CenterCrop((int(height*self.ratio), int(width*self.ratio)))(img)
        anno_class_img = transforms.CenterCrop((int(height*self.ratio), int(width*self.ratio)))(anno_class_img)
        return img, anno_class_img

class RandomScale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img, anno_class_img, random_fix=False):
        random_scale = np.random.uniform(self.scale[0], self.scale[1])
        size = [int(x*random_scale) for x in img.size()[len(img.size())-3:]]
        zero = img.min()
        # 화상 리사이즈
        img = F.interpolate(img, size=size, mode='area')
        # 어노테이션 리사이즈
        anno_class_img = F.interpolate(anno_class_img, size=size, mode='area')


        # 화상을 원래 크기로 잘라
        # 위치를 구한다
        if random_scale > 1.0:
            img=img[:,:,size[0]//2-60:size[0]//2+60,size[1]//2-60:size[1]//2+60,size[2]//2-60:size[2]//2+60]
            anno_class_img=anno_class_img[:,:,size[0]//2-60:size[0]//2+60,size[1]//2-60:size[1]//2+60,size[2]//2-60:size[2]//2+60]

        else:
            # input_size보다 짧으면 padding을 수행한다
            img=F.pad(img, [60-size[i//2]//2 for i in range(0,6)], value=zero)
            anno_class_img=F.pad(anno_class_img, [60-size[i//2]//2 for i in range(0,6)], value=zero)

        return img, anno_class_img


class RandomRotation(object):
    def __init__(self, angle):
        self.angle = angle
        self.rotate_angle = (np.random.uniform(self.angle[0], self.angle[1]))

    def __call__(self, img, anno_class_img, random_fix=False):
        if not random_fix:
            self.rotate_angle = (np.random.uniform(self.angle[0], self.angle[1]))

        # 회전 각도 결정

        # 회전
        img = img.rotate(self.rotate_angle, Image.BILINEAR)
        anno_class_img = anno_class_img.rotate(self.rotate_angle, Image.NEAREST)

        return img, anno_class_img

class RandomShift(object):
  def __init__(self, pixel):
      self.pixel = pixel
  
  def __call__(self,  imgs, anno_class_imgs, random_fix=False):
      shift_pixel = [np.random.randint(self.pixel[1]-self.pixel[0]+1)+self.pixel[0] for x in range(0,3)]
      dim = len(imgs.size())
      for i in range(0,3):
          imgs = torch.roll(imgs, shifts = shift_pixel[i], dims = dim-3+i)
          anno_class_imgs = torch.roll(anno_class_imgs,shift_pixel[i],dims=dim-3+i)
      
      return imgs,anno_class_imgs

class RandomMirror(object):
    """50% 확률로 좌우 반전시키는 클래스"""
    def __init(self, pixel):
        self.rand = np.random.randint(2)
    def __call__(self, img, anno_class_img, random_fix=False):
        if not random_fix:
           self.rand = np.rand.randint(2)
        if self.rand:
            img = ImageOps.mirror(img)
            anno_class_img = ImageOps.mirror(anno_class_img)
        return img, anno_class_img


class Resize(object):
    """input_size 인수의 크기를 변형하는 클래스"""

    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, img, anno_class_img, random_fix=False):

        # width = img.size[0]  # img.size=[폭][높이]
        # height = img.size[1]  # img.size=[폭][높이]

        img = img.resize((self.input_size, self.input_size),
                         Image.BICUBIC)
        anno_class_img = anno_class_img.resize(
            (self.input_size, self.input_size), Image.NEAREST)

        return img, anno_class_img


class Normalize_Tensor(object):
    def __init__(self, color_mean, color_std):
        self.color_mean = color_mean
        self.color_std = color_std

    def __call__(self, img, anno_class_img, random_fix=False):

        # PIL 이미지를 Tensor로 변환. 크기는 최대 1로 규격화된다
        if type(img)!=type(torch.tensor([])):
            img = transforms.functional.to_tensor(img)
        # 색상 정보의 표준화
        img = transforms.functional.normalize(
            img, self.color_mean, self.color_std)
        if len(img.size())<4:
            img = transforms.Grayscale(3)(img)

        # 어노테이션 화상을 Tensor로 변환
        if type(anno_class_img)!=type(torch.tensor([])):
            anno_class_img = (transforms.functional.to_tensor(anno_class_img)>0).type(torch.FloatTensor)

        return img, anno_class_img
