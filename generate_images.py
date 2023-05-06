import torch
import numpy as np
import models
import os
import random
import cv2
import io
from PIL import Image
import PIL
from pytorch_pretrained_biggan import (BigGAN,one_hot_from_int,truncated_noise_sample,convert_to_images)

class ImageGenerate(object):
	def __init__(self,generator_path):
		self.generator_path = generator_path
		self.generator = models.generator
		self.generator.load_state_dict(torch.load('genweights4.pth',map_location=torch.device('cpu')))
		if 'source' not in os.listdir(os.getcwd()):
			os.mkdir('source')
		self.results_dir = os.path.join(os.getcwd(),'source')

	def generate_image(self,image_name):
		test = torch.randn(1,128,1,1,device=torch.device('cpu'))
		image = self.generator(test)
		b = image[0].permute(1,2,0).detach().numpy()
		cv2.imwrite(os.path.join(self.results_dir,image_name),b)

	def generate_source(self,source_number):
		for i in range(source_number):
			img_name = 'car_'+str(i)+'.jpg'
			self.generate_image(img_name)
		pass

# obj = ImageGenerate('./weights/genweights4.pth')
# print(obj.generate_image())
class ImageGenerator(object):
	def __init__(self):
		self.generator = BigGAN.from_pretrained('biggan-deep-256',cache_dir='./weights')
		if 'source' not in os.listdir(os.getcwd()):
			os.mkdir('source')
		self.results_dir = os.path.join(os.getcwd(),'source')
	
	def save_image(self,img_list):
		for i in range(len(img_list)):
			cv2.imwrite(os.path.join(self.results_dir,'car_'+str(i)+'.jpg'), np.array(img_list[i]))
		return True
	
	def generate_source(self,source_number):
		choices = [817,751,717]
		choice = choices[random.randint(0,2)]
		truncation=1
		if source_number<1:
			source_number=1
		class_vector = one_hot_from_int(choice,batch_size=source_number)
		noise_vector = truncated_noise_sample(truncation=truncation, batch_size=source_number)
		noise_vector = torch.from_numpy(noise_vector)
		class_vector = torch.from_numpy(class_vector)
		with torch.no_grad():
			output = self.generator(noise_vector.float(), class_vector.float(), truncation)
		img_list=convert_to_images(output)
		self.save_image(img_list=img_list)