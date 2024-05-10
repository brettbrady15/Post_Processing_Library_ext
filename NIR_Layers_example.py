from PostProcessingLibrary import Layer_Creation, ImageProcessor
import os
import cv2

path = r"D:\FE24042401\FP24042401_NIR.hdf5"
calibration_image = r"Calibration_Images\NIR-Frame-1.tiff"
cal_img = cv2.imread(calibration_image, cv2.IMREAD_ANYDEPTH)
 
image_processor = ImageProcessor(path)

result_df = Layer_Creation.new_extract(65, path, cal_img)
