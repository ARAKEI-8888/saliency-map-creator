import os
import cv2
from pathlib import Path

def file_name_list(path_name):
	path = Path(os.getcwd()+'/'+path_name)
	path_list = list(path.glob("*"))
	return path_list

def craet_and_save_image_1(file_path, file_name, number):
	image = cv2.imread(file_path)
	saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
	(success, saliencyMap) = saliency.computeSaliency(image)
	saliencyMap = (saliencyMap * 255).astype("uint8")

	if success is True:
		cv2.imwrite(str(number)+'_saliency_1_'+file_name, saliencyMap)
		cv2.waitKey(0)

def craet_and_save_image_2(file_path, file_name, number):
	image = cv2.imread(file_path)
	saliency = cv2.saliency.StaticSaliencyFineGrained_create()
	(success, saliencyMap) = saliency.computeSaliency(image)
	saliencyMap = (saliencyMap * 255).astype("uint8")

	if success is True:
		cv2.imwrite(str(number)+'_saliency_2_'+file_name, saliencyMap)
		cv2.waitKey(0)

if __name__ == "__main__" :
	file_list = file_name_list("test_image")
	cnt = 0
	for file_path in file_list:
		file_name = file_path.as_posix().split("/")[-1]
		craet_and_save_image_1(file_path.as_posix(), file_name, cnt)
		craet_and_save_image_2(file_path.as_posix(), file_name, cnt)
		cnt = cnt +1