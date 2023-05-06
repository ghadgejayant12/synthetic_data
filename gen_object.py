import torch
import cv2
import numpy as np 
from PIL import Image
import os
import sys
import random
import time
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_b"
HOME = os.getcwd()
CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
class ObjectDataGenerate:
  def __init__(self,background,generator):
    # background is the background image on which the objects are to be placed
    # generator is the object of SamAutomaticMaskGenerator which is used for generation mask
    # bbox stores the [x_min,y_min,x_max,y_max,class_id] for every object added to the background
    # all changes will be made to the background image sent and changes will be made incrementally
    self.background=background
    if self.background.shape[0]<=256 or self.background.shape[1]<=256:
      self.background = cv2.resize(self.background,(1024,1024))
    self.generator=generator
    self.bbox=list()
    self.classes = {'car':0}
  
  def add_object(self,source,class1='car'):
    source=cv2.resize(source,(256,256))
    print('Working the sam model')
    result = self.generator.generate(source)
    print('Done with the SAM model')
    x=random.randint(0,self.background.shape[0]-256)
    y=random.randint(0,self.background.shape[1]-256)
    print('Here entering the for loop')
    for i in range(0,256):
      for j in range(0,256):
        if result[0]['segmentation'][i][j]==True:
          self.background[x+i,y+j,:]=source[i,j,:]
    print('Now done with the for loop')
    #print(x,y,x+256,y+256)
    self.bbox.append([y/self.background.shape[1],x/self.background.shape[0],(y+200)/self.background.shape[1],(x+200)/self.background.shape[0],self.classes[class1]])
  def get_data(self):
    return self.background, self.bbox

class DatasetGenerate:
  def __init__(self,source_images_dir,backgrounds_dir,max_objects_per_image,total_images_to_generate):
    self.source_dir = source_images_dir
    self.backgrounds_dir = backgrounds_dir
    self.generator = SamAutomaticMaskGenerator(sam)
    self.image_paths = list()
    self.bbox_lst = list()
    self.max_objects_per_image =max_objects_per_image
    self.total_images_to_generate = total_images_to_generate
    
  def generate_data(self,num_obj,bg_img_path,bg_img_name,bg_img_store_path):
    bg_img = cv2.imread(bg_img_path)
    gen  = ObjectDataGenerate(bg_img,self.generator)
    source_images = os.listdir(self.source_dir)
    for i in range(num_obj):
      choice = random.randint(0,len(source_images)-1)
      source_curr = os.path.join(self.source_dir,source_images[choice])
      object_img = cv2.imread(source_curr)
      gen.add_object(object_img)
    result_img, bbox = gen.get_data()
    print('Saving the image to,',os.path.join(bg_img_store_path,bg_img_name))
    cv2.imwrite(os.path.join(bg_img_store_path,bg_img_name),result_img)
    self.image_paths.append(os.path.join(bg_img_store_path,bg_img_name))
    bbox2 = list()
    for x in bbox:
      bbox2.append(x+[bg_img_name])
    self.bbox_lst.extend(bbox2)
    
  def build_dataset(self):
    for j in range(self.total_images_to_generate):
      st = time.time()
      i = os.listdir(self.backgrounds_dir)[0]
      path_im = os.path.join(self.backgrounds_dir,i)
      object_num = random.randint(1,self.max_objects_per_image)
      print('---------------------------')
      print('Working on',i,'background image')
      if 'dataset_images' not in os.listdir(os.getcwd()):
        os.mkdir('dataset_images')
      bg_img_name = i.split('.')[0]+'_'+str(j)+'ab1.jpeg'
      print(bg_img_name)
      self.generate_data(num_obj=object_num, bg_img_path=path_im, bg_img_name=i , bg_img_store_path=os.path.join(os.getcwd(),'dataset_images'))
      end = time.time()
      print(end-st,'Time for 1 image')
    return self.image_paths, self.bbox_lst

# Run the below two lines to get the code for 
# obj = DatasetGenerate(os.path.join(os.getcwd(),'source'), os.path.join(os.getcwd(),'bg'),3)
# bg_img,bbox = obj.build_dataset()
# print(bbox)