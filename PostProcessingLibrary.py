"""
PostProcessingLibrary

This library is used to manipulate data recorded using the Farsoon data setup

"""
__author__ = '{ENS Brett Brady}'
__credits__ = ['{Jared Research Group}']
__version__ = '{1}.{1}.{0}'
__maintainer__ = '{BB}'
__email__ = '{brettbrady15@outlook.com}'
__status__ = '{On Going}'

# TO-DO
# Ensure HDF5 and file path works for all functions
# write example scripts

import h5py
import os
from mpl_toolkits.mplot3d import Axes3D
from FLIRwrapperBB import FrameHandler_BB, EnvHandler_BB
import cv2
import csv
import numpy as np
from alive_progress import alive_bar
import pandas as pd 
from tkinter import filedialog
import matplotlib.pyplot as plt
import re
import tkinter as tk
print(f"PostProcessingLibrary version {__version__} -- Jared Research Group")
class HDF5_Handler:
    def create_HDF5(path, FO_num, NIR_Comp = 7, FLIR_Comp = 5):
        """
        Creates an HDF5 archive.

        Parameters:
            path (str): Path to the directory containing FLIR and NIR images.
            FO_num (str): Build Number. 
            NIR_Comp (int, optional): Compression level for NIR images. Defaults to 7.
            FLIR_Comp (int, optional): Compression level for FLIR images. Defaults to 5.

        Returns:
            str: Path to the created HDF5 archive.
        """
        print('Creating HDF5 Archive...')

        FLIR_path = os.path.join(path,"FLIR")
        NIR_path = os.path.join(path,"NIR")

        hdf5_path = path + f'/{FO_num}_FLIR.hdf5'
        H5F = h5py.File(hdf5_path, 'w') # Create HDF5 Archive
        # CHANGE AS NEEDED
        H5F.attrs['FLIR Camera'] = 'FLIR A50'
        H5F.create_group("FLIR")
        FLIR = H5F["FLIR"]


        images = sorted([img for img in os.listdir(FLIR_path) if img.endswith(".npy")], key=natural_sort_key)
        print("Starting FLIR Compression...")
        with alive_bar(len(images), bar="filling") as bar:
            for image in images:
                with h5py.File(hdf5_path, 'a') as H5F:
                        IR_String = os.path.join(FLIR_path, image)
                        ir_image_data = np.load(IR_String)
                        IR_dset = FLIR.create_dataset(image, data = ir_image_data, dtype = "f", compression="gzip", compression_opts=FLIR_Comp)            
                bar()

        H5F.close()


        hdf5_path = path + f'/{FO_num}_NIR.hdf5'
        H5F = h5py.File(hdf5_path, 'w') # Create HDF5 Archive
        H5F.attrs['NIR Camera'] = 'NIR'
        H5F.attrs['NIR Exposure Time'] = "200 ms"
        H5F.create_group("NIR")
        images = sorted([img for img in os.listdir(NIR_path) if img.endswith(".tiff")], key=natural_sort_key)
        print("Starting NIR Compression...")
        with alive_bar(len(images), bar="filling") as bar:
            for image in images:
                try:
                    with h5py.File(hdf5_path, 'a') as H5F:
                            NIR = H5F["NIR"]
                            NIR_String = os.path.join(NIR_path, image)
                            NIR_frame = cv2.imread(NIR_String, cv2.IMREAD_UNCHANGED)
                            NIR_String = os.path.basename(NIR_String)
                            NIR_dset = NIR.create_dataset(image, data = NIR_frame, compression="gzip", compression_opts=NIR_Comp, dtype='uint16')
                            NIR_dset.attrs['Maximum'] = np.max(NIR_frame)
                            NIR_dset.attrs['Minimum'] = np.min(NIR_frame)

                except TypeError as e:
                    print(f"Skipping image {image} due to error: {e}")
                bar()
                H5F.close()


        return hdf5_path
    
    def Extract_to_Images(hdf5_path, output_dir):
        pass

class Create_Videos:
    def Video_FLIR(input_path, output_video_path, json_file_path, calibration_image, fps=30):
        """
        Processes FLIR video.

        Parameters:
            input_path (str): Path to the images or HDF5 archive.
            output_video_path (str): Path to save the output video.
            json_file_path (str): Path to the JSON file containing camera calibration data for the recorded images.
            calibration_imgae (mat): image of calibration plate
            calibration_json (str): Path to the JSON file containting camera calibration data for the calibration images.
            fps (int, optional): Frames per second for the output video. Defaults to 30.

        Returns:
            bool: True if successful, False otherwise.
        """
        
        H, X, Y, base_img = Perspective_Transform.FLIR.get_transform(calibration_image)
        points = []
        env, calibration = EnvHandler_BB.load_JSON(json_file_path)
        if input_path.endswith(".hdf5"):
            # Handle HDF5 file
            with h5py.File(input_path, "r") as f:
                images = list(f["FLIR"].keys())
                frame = f["FLIR"][images[0]][:]


        else:
            images = [img for img in os.listdir(input_path) if img.endswith(".npy") or img.endswith(".npy")]
            frame = np.load(os.path.join(input_path, images[0]))
        
        frame = FrameHandler_BB.convert_to_C(frame, calibration, env)
        # ####################TEMP
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # pt1 = np.array([200, 168])
        # pt2 = np.array([325, 188])
        # pt3 = np.array([211, 32])
        # pt4 = np.array([332, 59])
        # # Calculate distances
        # y = np.linalg.norm(pt2 - pt1)
        # x = np.linalg.norm(pt4 - pt1)
        # z = 200
        # pts_src = np.array([[200, 168], [325, 188], [211, 32], [332, 59]], dtype=np.float32)
        # pts_dst = np.array([[z, z], [z+x, z], [z, z+y], [z+x,z+y]], dtype=np.float32)
        # homography, _ = cv2.findHomography(pts_src, pts_dst)
        # transformed_pts = cv2.perspectiveTransform(pts_src.reshape(-1, 1, 2), homography)
        # frame = cv2.warpPerspective(frame, homography, (frame.shape[1]+100, frame.shape[0]+100))
        # frame = cv2.flip(cv2.rotate(frame, cv2.ROTATE_180), 1)
        # ############################
        frame = Perspective_Transform.correct_image(frame, H, base_img)

        degC = frame.copy()
        height, width = frame.shape

        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        num_nodes = int(input("Enter the number of nodes: "))
    
        if num_nodes != '0':
            cv2.imshow('Image', frame)
            cv2.setMouseCallback('Image', lambda event, x, y, flags, param: click_event(event, x, y, flags, param, frame, points))

            while len(points) < int(num_nodes):
                cv2.waitKey(1)
            for point in points:
                intensity = degC[point[1], point[0]]
                cv2.putText(frame, f'{round(intensity, 2)}', (point[0]-20, point[1]+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) # Display intensity values at selected points
            cv2.putText(frame, "JRG", (0, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
            cv2.imshow('Image', frame)  # Show the updated image
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            pass

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if input_path.endswith(".hdf5"):
            with h5py.File(input_path, "r") as f:
                images = f["FLIR"].keys()
                images = sorted(f["FLIR"].keys(), key=natural_sort_key) 
                with alive_bar(len(images), bar="filling") as bar:
                        for image in images:
                            img =  f["FLIR"][image][:]
                            img = FrameHandler_BB.convert_to_C(img, calibration, env)
                            img = Perspective_Transform.correct_image(img, H, base_img, flag = 'FLIR')
                            degC = img

                            # Write the image to video
                            img[0,0] = 27
                            img[0,1] = 60
                            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                            for point in points:
                                intensity = degC[point[1], point[0]]
                                cv2.putText(img, f'{round(intensity, 2)}', (point[0]-20, point[1]+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            cv2.putText(img, f"{os.path.splitext(image)[0]}", (width - 170, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
                            cv2.putText(img, "JRG", (0, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
                            video.write(img)

                            bar()

        else:
            with alive_bar(len(images), bar="filling") as bar:
                for image in images:
                    img_path = os.path.join(input_path, image)
                    img = np.load(img_path)
                    img = FrameHandler_BB.convert_to_C(img, calibration, env)
                    img = Perspective_Transform.correct_image(img, H, base_img, flag = 'FLIR')
                    degC = img.copy()
                    # Write the image to video
                    img[0,0] = 27
                    img[0,1] = 60
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                    for point in points:
                        intensity = degC[point[1], point[0]]
                        cv2.putText(img, f'{round(intensity, 2)}', (point[0]-20, point[1]+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(img, f"{os.path.splitext(image)[0]}", (width - 170, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
                    cv2.putText(img, "JRG", (0, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
                    video.write(img)

                    bar()
        
        video.release()

        print("Success, Video created successfully!")

        return True

    def Video_NIR(file_path, output_video_path, calibration_image, fps=5):
        """
        Processes NIR images to create video.

        Parameters:
            input_path (str): Path to the NIR images.
            output_video_path (str): Path to save the output video.
            fps (int, optional): Frames per second for the output video. Defaults to 5.

        Returns:
            None
        """
        H, X, Y, base_img = Perspective_Transform.NIR.get_transform(calibration_image)
        if file_path.endswith(".hdf5"):
            with h5py.File(file_path, 'r') as hdf_file:
                print(f"Creating NIR Video...")
                for group_name in hdf_file.keys():
                    datasets = sorted(hdf_file[group_name].keys(), key=natural_sort_key) 
                    data_array = hdf_file[group_name][datasets[0]][:]
                    data_array = Perspective_Transform.correct_image(data_array, H, base_img, flag = 'NIR')

                    height, width = data_array.shape
                    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (2000, 2000))          
                    with alive_bar(len(datasets), bar="filling") as bar:
                        for dataset_name in datasets:
                            data_array = hdf_file[group_name][dataset_name][:]
                            data_array = cv2.normalize(data_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                            data_array = Perspective_Transform.correct_image(data_array, H, base_img, flag = 'NIR')
                            frame = cv2.applyColorMap(data_array, cv2.COLORMAP_JET)
                            cv2.putText(frame, "JRG", (0, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
                            video.write(frame)
                            bar()
                    break


        else:
            images = [img for img in os.listdir(file_path) if img.endswith(".tiff")]
            path = os.path.join(file_path, images[0])
            frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            frame = Perspective_Transform.correct_image(frame, H, base_img,flag = 'NIR' )
            height, width = frame.shape

            video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

            with alive_bar(len(images), bar="filling") as bar:
                for image in images:
                    path = os.path.join(file_path, image)
                    frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    frame = Perspective_Transform.correct_image(frame, H, base_img, flag = 'NIR')
                    frame[0,0] = 0
                    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                    cv2.putText(frame, "JRG", (0, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
                    video.write(frame)
                    bar()
        
        video.release()

class ImageProcessor:

    def __init__(self, image_folder):
        root = tk.Tk()
        self.root = root
        self.root.title("Image Processor")

        # CSV file for storing intensity values
        if os.path.isfile(image_folder) and image_folder.lower().endswith('.hdf5'):
            self.csv_file_path = os.path.join(os.path.dirname(image_folder), "intensity_values.csv")
        else:
            self.csv_file_path = os.path.join(image_folder, "intensity_values.csv")

        # Process images in the folder
        self.process_images(image_folder)

    def process_images(self, image_source):
        """
        Process images from a given source folder for NIR Layer Creation.

        Parameters:
            image_source (str): Path to the folder/HDF5 archive containing images.

        Returns:
            None
        """
        if os.path.isdir(image_source):
            image_files = [f for f in os.listdir(image_source) if f.lower().endswith(('.tiff'))]
            image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x.split('.')[-2]))))

            # Open CSV file for writing
            with open(self.csv_file_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["Image Name","Max Intensity"])
                rows_to_write = []
                with alive_bar(len(image_files), bar="filling") as bar:
                    for image_file in image_files:
                        
                        image_path = os.path.join(image_source, image_file)
                        try:
                            # Load the image
                            self.image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                            if self.image is None:
                                raise Exception(f"Error: Unable to read the image from {image_path}")

                            self.height, self.width = self.image.shape

                            # Calculate intensity values
                            image8 = self.image.astype('uint8')
                            max_intensity = self.calculate_max_intensity(image8)

                            rows_to_write.append([dataset_name, max_intensity])
                            bar()

                        except Exception as e:
                            print(f"Error processing image {image_file}: {e}")

                    csv_writer.writerows(rows_to_write)

        elif os.path.isfile(image_source) and image_source.lower().endswith('.hdf5'):
            with h5py.File(image_source, 'r') as hdf_file:
                for group_name in hdf_file.keys():                
                    datasets = sorted(hdf_file[group_name].keys(), key=natural_sort_key)           
                    with open(self.csv_file_path, 'w', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(["Image Name", "Max Intensity"])
                        rows_to_write = []
                        with alive_bar(len(datasets), bar="filling") as bar:
                            for dataset_name in datasets:
                                # Extract numpy array from the dataset
                                self.image = hdf_file[group_name][dataset_name][:]
                                self.height, self.width = self.image.shape
                                image8 = self.image.astype('uint8')
                                # Calculate intensity values
                                max_intensity = self.calculate_max_intensity(image8)

                                # Write to CSV file
                                rows_to_write.append([dataset_name, max_intensity])
                                bar()
                csv_writer.writerows(rows_to_write)
        print("Processing complete.")

    def calculate_average_intensity(self, image):
        # Convert image to grayscale
        image =  cv2.GaussianBlur(image, (7, 7), 0) 
        return cv2.mean(image)[0]

    def calculate_max_intensity(self, image):
        return np.max(cv2.GaussianBlur(image, (5, 5), 0))

    def calculate_min_intensity(self, image):
        image =  cv2.GaussianBlur(image, (5, 5), 0)
        return image.min()

class Layer_Creation:
    def extract_consecutive_groups(intensity_threshold, image_folder, calibration_image):
        """
        Extracts consecutive groups of layers from images.

        Parameters:
            intensity_threshold (int): Intensity threshold for layer extraction.
            image_folder (str): Path to the folder containing images.
            calibration_image (mat): Calibration Image
            FLIR_Variables_path (str): FLIR Variable path

        Returns:
            None
        """
        H, X, Y, base_img = Perspective_Transform.NIR.get_transform(calibration_image)
        
        if os.path.isdir(image_folder):
            csv_file_path = os.path.join(image_folder, "intensity_values.csv")
            
        elif os.path.isfile(image_folder) and image_folder.lower().endswith('.hdf5'):
            csv_file_path = os.path.join(os.path.dirname(image_folder), "intensity_values.csv")
           
        print('Finding Layers...')
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)
        df.at[df.index[-1], 'Max Intensity'] = 255

        # Sort DataFrame based on frame numbers
        df['Frame'] = df['Image Name'].apply(lambda x: int(''.join(filter(str.isdigit, x))))
        df.sort_values('Frame', inplace=True)


        # Filter DataFrame based on intensity threshold
        above_threshold_df = df[df['Max Intensity'] > intensity_threshold]

        extracted_frames = []

        for index, row in above_threshold_df.iterrows():
            frame_value = row['Frame']
            extracted_frames.append(frame_value)

        ref = []
        image_name = []
        ref.append(0)
        for i in range(1, len(extracted_frames)):
            if extracted_frames[i] != extracted_frames[i - 1] + 1:
                ref.append(i)
                ref.append(i+1)
        ref.pop()
        ref[0] = 1

        for i in range(0, len(extracted_frames)):
                corresponding_row = above_threshold_df[above_threshold_df['Frame'] == extracted_frames[i]].iloc[0]
                # print(i, extracted_frames[i])
                image_name.append(corresponding_row['Image Name'])
        

        layer = 0

        reflist = np.array(ref).reshape(-1,2).tolist()

        # print(reflist)
        layer_folder = os.path.join(image_folder, "Layer")
        video_name = os.path.join(layer_folder,'layer.avi') 
        fps = 5

        if os.path.isdir(image_folder):
            layer_folder = os.path.join(image_folder,"Layer")
            os.makedirs(layer_folder, exist_ok=True)
            video_name = os.path.join(layer_folder,'layer.avi') 
            fps = 5
            temp_img = Perspective_Transform.correct_image(cv2.imread(os.path.join(image_folder, f"{image_name[0]}"),cv2.IMREAD_UNCHANGED), H, base_img, flag = 'NIR')

            height, width = temp_img.shape
            video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (2*width, height))
            
            title_text_color = (0, 255, 0)  # Green text color

            try:
                for start, end in reflist:
                    layer += 1
                    temp_img =  cv2.normalize((cv2.imread(os.path.join(image_folder, f"{image_name[0]}"),cv2.IMREAD_UNCHANGED)), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    temp_img = Perspective_Transform.correct_image(temp_img, H, base_img, flag = 'NIR')
                    cum_image = np.uint8(np.zeros_like(temp_img))

                    print("On Layer: ", layer)
                    
                    for ii in range(start-1, end):    
                        image_path = f"{image_folder}/{image_name[ii]}"

                        image = (cv2.imread(image_path, cv2.IMREAD_UNCHANGED))
                        print(df.at[ii, 'Max Intensity'])
                        image = cv2.normalize(image, None, 0, df.at[ii, 'Max Intensity'], cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        
                        image = Perspective_Transform.correct_image(image, H, base_img, flag = 'NIR')
                        cum_image = np.maximum(cum_image, image)
                        cum_image_copy = cum_image.copy() 

                        image = cv2.applyColorMap(image,cv2.COLORMAP_JET)
                        cum_image_copy = cv2.applyColorMap(cum_image_copy,cv2.COLORMAP_JET)

                        layer_filepath = os.path.join(layer_folder, f"Layer{layer}.jpg")   
                        layer_cm_filepath = os.path.join(layer_folder, f"Layer_cm_{layer}.jpg")

                        calculated_layer = layer*0.03
                        text = f"Thickness: {round(calculated_layer,2)} mm"
                        
                        cv2.putText(cum_image_copy, f"Layer:  {layer}", (width - 330, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, title_text_color, 2, cv2.LINE_AA)
                        cv2.putText(cum_image_copy, text, (width-330, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, title_text_color, 2, cv2.LINE_AA)
                        cv2.putText(cum_image_copy, "JRG", (0, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
                        both = cv2.hconcat([image, cum_image_copy])
                        video.write(both)

                        
                    cv2.imwrite(layer_filepath, cum_image)
                    cv2.imwrite(layer_cm_filepath, cum_image_copy)

                print("Found ", layer, "Layers")
                        
            except IndexError:
                print("Complete")
        
        elif os.path.isfile(image_folder) and image_folder.lower().endswith('.hdf5'):
            layer_folder = os.path.join(os.path.dirname(image_folder),"Layer")
            os.makedirs(layer_folder, exist_ok=True)
            video_name = os.path.join(layer_folder,'layer.avi') 
            fps = 5
            with h5py.File(image_folder, 'r') as hdf_file:
                for group_name in hdf_file.keys():                
                    datasets = sorted(hdf_file[group_name].keys(), key=natural_sort_key)           
                    temp_img = hdf_file[group_name][datasets[0]][:]
                    temp_img = Perspective_Transform.correct_image(temp_img, H, base_img, flag = 'NIR')

                    height, width = temp_img.shape
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    video = cv2.VideoWriter(video_name, fourcc, fps, (2*width, height))
                        
                    title_text_color = (0, 255, 0)  # Green text color
                    try:
                        print(reflist)
                        for start, end in reflist:
                            layer += 1
                            temp_img = hdf_file[group_name][image_name[0]][:]
                            temp_img = Perspective_Transform.correct_image(temp_img, H, base_img, flag = 'NIR')
                            cum_image = np.uint8(np.zeros_like(temp_img))

                            print("On Layer: ", layer)
                            for ii in range(start-1, end):    
                                image = hdf_file[group_name][image_name[ii]][:]
                                # print(f"{ii}/{end}", end='\r')
                                number = re.search(r'\d+', image_name[ii]).group()

                                image = cv2.normalize(image, None, 0, df.at[int(number)-1, 'Max Intensity'], cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                                image = Perspective_Transform.correct_image(image, H, base_img, flag = 'NIR')
                                cum_image = np.maximum(cum_image, image)
                                cum_image_copy = cum_image.copy() 

                                image = cv2.applyColorMap(image,cv2.COLORMAP_JET)
                                cum_image_copy = cv2.applyColorMap(cum_image_copy,cv2.COLORMAP_JET)

                                layer_filepath = os.path.join(layer_folder, f"Layer{layer}.jpg")   
                                layer_cm_filepath = os.path.join(layer_folder, f"Layer_cm_{layer}.jpg")

                                calculated_layer = layer*0.03
                                text = f"Thickness: {round(calculated_layer,2)} mm"
                                
                                cv2.putText(cum_image_copy, f"Layer:  {layer}", (width - 330, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, title_text_color, 2, cv2.LINE_AA)
                                cv2.putText(cum_image_copy, text, (width-330, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, title_text_color, 2, cv2.LINE_AA)
                                cv2.putText(cum_image_copy, "JRG", (0, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
                                both = cv2.hconcat([image, cum_image_copy])
                                video.write(both)

                                
                            cv2.imwrite(layer_filepath, cum_image)
                            cv2.imwrite(layer_cm_filepath, cum_image_copy)

                        print("Found ", layer, "Layers")
                                
                    except IndexError:
                        print("Complete")

        
        return 

    def extract_consecutive_groups_3D(csv_file_path, intensity_threshold, image_folder):
        """
        Extracts consecutive groups of layers in 3D from images.

        Parameters:
            csv_file_path (str): Path to the CSV file containing image information.
            intensity_threshold (int): Intensity threshold for layer extraction.
            image_folder (str): Path to the folder containing images.

        Returns:
            pd.DataFrame: DataFrame containing information about the extracted layers.
        """
        print('Finding Layers...')
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)
        df.at[df.index[-1], 'Max Intensity'] = 255

        # Sort DataFrame based on frame numbers
        df['Frame'] = df['Image Name'].apply(lambda x: int(''.join(filter(str.isdigit, x))))
        df.sort_values('Frame', inplace=True)


        # Filter DataFrame based on intensity threshold
        above_threshold_df = df[df['Max Intensity'] > intensity_threshold]
        extracted_frames = []

        for index, row in above_threshold_df.iterrows():
            frame_value = row['Frame']
            extracted_frames.append(frame_value)

        ref = []
        image_name = []
        ref.append(0)
        for i in range(1, len(extracted_frames)):
            if extracted_frames[i] != extracted_frames[i - 1] + 1:
                ref.append(i)
                ref.append(i+1)
        ref.pop()
        ref[0] = 1
        for i in range(0, len(extracted_frames)):
                corresponding_row = above_threshold_df[above_threshold_df['Frame'] == extracted_frames[i]].iloc[0]
                image_name.append(corresponding_row['Image Name'])
        layer = 0

        reflist = np.array(ref).reshape(-1,2).tolist()
        if os.path.isdir(image_folder):
            layer_folder = os.path.join(image_folder,"Layer")
            os.makedirs(layer_folder, exist_ok=True)
            temp_img = cv2.imread(os.path.join(image_folder, f"{image_name[0]}"),cv2.IMREAD_UNCHANGED)
            height, width = temp_img.shape
            crop_x1, crop_y1, crop_x2, crop_y2 = 120, 247, 450, 1600
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            fig_cum = plt.figure()
            ax_cum = fig.add_subplot(111, projection= '3d')
            try:
                for start, end in reflist:
                    layer += 1
                    temp_img = cv2.imread(os.path.join(image_folder, f"{image_name[0]}"),cv2.IMREAD_UNCHANGED)
                    cum_image = np.zeros_like(temp_img)
                    print("On Layer: ", layer)
                    
                    for ii in range(start-1, end):    
                        image_path = f"{image_folder}/{image_name[ii]}"

                        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                        
                        cum_image = np.maximum(cum_image, image)
                        # Create 2D arrays for X and Y coordinates
                        x = np.arange(0, crop_x2 - crop_x1, 1)
                        y = np.arange(0, crop_y2 - crop_y1, 1)

                        # Create 2D grid from X and Y coordinates
                        x, y = np.meshgrid(x, y)

                        # Flatten the images to use as Z coordinates
                        z_r = image.flatten()
                        z_l = cum_image.flatten()

                        # Reshape the flattened arrays
                        # z_r = z_r.reshape((crop_y2 - crop_y1, crop_x2 - crop_x1))
                        # z_l = z_l.reshape((crop_y2 - crop_y1, crop_x2 - crop_x1))

                        surface_r = ax.plot_surface(x, y, z_r, cmap='viridis', alpha=0.5)
                        surface_l = ax.plot_surface(x, y, z_l, cmap='viridis', alpha=0.5) 

                        ax.set_xlabel('X Position')
                        ax.set_ylabel('Y Position')
                        ax.set_zlabel('Intensity')

                        plt.pause(0.01)

                print("Found ", layer, "Layers")
                
            except IndexError:
                print("Complete")

        elif os.path.isfile(image_folder) and image_folder.lower().endswith('.hdf5'):
            layer_folder = os.path.join(os.path.dirname(image_folder),"Layer")
            os.makedirs(layer_folder, exist_ok=True)
            with h5py.File(image_folder, 'r') as hdf_file:
                for group_name in hdf_file.keys():                
                    datasets = sorted(hdf_file[group_name].keys(), key=natural_sort_key)    
                    temp_img = hdf_file[group_name][datasets[0]][:]

                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    fig2 = plt.figure()
                    ax2 = fig2.add_subplot(111, projection= '3d')
                    fig3 = plt.figure()
                    ax3 = fig3.add_subplot(111, projection= '3d')

                    crop_x1, crop_y1, crop_x2, crop_y2 = 200, 500, 500, 1500

                    try:
                        for start, end in reflist:
                            layer += 1
                            cum_image = np.zeros_like(temp_img)
                            print("On Layer: ", layer)
                            max_val = []
                            for ii in range(start-1, end):   
                                image = hdf_file[group_name][image_name[ii]][:]
                                max_val.append(image.max())

                                cum_image = np.maximum(cum_image, image)
                                
                                # # Create 2D arrays for X and Y coordinates
                                x = np.arange(0, 2048, 1)
                                y = np.arange(0, 2048, 1)


                            z_l = cum_image.flatten()
                            z_l = z_l.reshape((2048, 2048))
                            crop_x1 = max(0, crop_x1)
                            crop_x2 = min(2048, crop_x2)
                            crop_y1 = max(0, crop_y1)
                            crop_y2 = min(2048, crop_y2)
                            z_l = z_l[crop_y1:crop_y2,crop_x1:crop_x2]
                            z_l8 = z_l
                            z_l = z_l/1024
                            x = np.arange(crop_x1, crop_x2)
                            x = x/2048
                            y = np.arange(crop_y1, crop_y2)
                            y = y/2048
                            x, y = np.meshgrid(x, y)
                            surface_l = ax.plot_surface(x, y, z_l, cmap='viridis')
                            ax.set_xlabel('X Position')
                            ax.set_ylabel('Y Position')
                            ax.set_zlabel('Intensity')
                            ax.set_title('10 Bit')
                            ax.axes.set_aspect('equal')
                            z_8_max = np.max(z_l) * 1.1  # Increase factor can be adjusted as needed
                            ax.set_zlim(0, z_8_max)
                            print("The max value in this layer was: ",np.array(max_val).max())

                            # 8 Bit
                            z_8 = z_l8/1024 * 255
                            z_8 = z_8.astype(int)/255
                            surface_8 = ax2.plot_surface(x, y, z_8, cmap='viridis')
                            ax2.set_xlabel('X Position')
                            ax2.set_ylabel('Y Position')
                            ax2.set_zlabel('Intensity')
                            ax2.set_title('8 Bit')
                            ax2.axes.set_aspect('equal')
                            ax2.set_zlim(0, z_8_max)

                            print("The max value in this layer was: ", z_8.max())
                            

                            # Differences
                            z_diff = z_l - z_8
                            surface_8 = ax3.plot_surface(x, y, z_diff, cmap='viridis')
                            ax3.set_xlabel('X Position')
                            ax3.set_ylabel('Y Position')
                            ax3.set_zlabel('Intensity')
                            ax3.set_title('Difference')
                            ax3.axes.set_aspect('equal')
                            cbar = plt.colorbar(surface_8, ax=ax3, orientation='horizontal',shrink=0.5)
                            cbar.set_label('Intensity')
                            print("The max value in this layer was: ", abs(z_diff).max())

                            plt.show()

                                
                        print("Found ", layer, "Layers")
                        
                    except IndexError:
                        print("Complete")
                        
        
        # Needs to open a np.zero_like of cropped image intially and then it needs to be take the higher image of each image within the range. After the for loop is complete, the new image needs to be saved as Layer#.jpg
        return above_threshold_df

class Model3DCreator:
    def __init__(self):
        '''
        processor = Model3DCreator()
        folder_path = filedialog.askdirectory(title="Select a folder of images")
        if not folder_path:
            print("No folder selected.")
        else:
            processor.load_images(folder_path)
            
            # Assuming isotropic pixel spacing of 1.0 in all directions
            
            # pixel_spacing = (1.0, 1.0, 0.02) #FLIR
            pixel_spacing = (1.0, 1.0, 0.3)  # NIR

            processor.create_3D_model(folder_path, pixel_spacing, flag='Optical')
            '''
        import vtk
        import vtkmodules.vtkInteractionStyle
        import vtkmodules.vtkRenderingOpenGL2
        from vtkmodules.vtkCommonColor import vtkNamedColors
        from vtkmodules.vtkCommonCore import vtkLookupTable
        from vtkmodules.vtkCommonDataModel import vtkPlane
        from vtkmodules.vtkFiltersCore import (
            vtkContourFilter,
            vtkCutter,
            vtkPolyDataNormals,
            vtkStripper,
            vtkStructuredGridOutlineFilter,
            vtkTubeFilter
        )
        from vtkmodules.vtkFiltersExtraction import vtkExtractGrid
        from vtkmodules.vtkIOParallel import vtkMultiBlockPLOT3DReader
        from vtkmodules.vtkRenderingCore import (
            vtkActor,
            vtkPolyDataMapper,
            vtkRenderWindow,
            vtkRenderWindowInteractor,
            vtkRenderer
        )
        self.image3D = None

    def load_images(self, folder_path):
        import vtk
        import vtkmodules.vtkInteractionStyle
        import vtkmodules.vtkRenderingOpenGL2
        from vtkmodules.vtkCommonColor import vtkNamedColors
        from vtkmodules.vtkCommonCore import vtkLookupTable
        from vtkmodules.vtkCommonDataModel import vtkPlane
        from vtkmodules.vtkFiltersCore import (
            vtkContourFilter,
            vtkCutter,
            vtkPolyDataNormals,
            vtkStripper,
            vtkStructuredGridOutlineFilter,
            vtkTubeFilter
        )
        from vtkmodules.vtkFiltersExtraction import vtkExtractGrid
        from vtkmodules.vtkIOParallel import vtkMultiBlockPLOT3DReader
        from vtkmodules.vtkRenderingCore import (
            vtkActor,
            vtkPolyDataMapper,
            vtkRenderWindow,
            vtkRenderWindowInteractor,
            vtkRenderer
        )
        print("Processing 2D Images to vtkImageData...")
        images = []
        reader = vtk.vtkJPEGReader()
        image3D = vtk.vtkImageAppend()
        image3D.SetAppendAxis(2)
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                f = os.path.join(folder_path, filename)
                reader.SetFileName(f)
                reader.Update()

                # Check if the image has multiple components
                if reader.GetOutput().GetNumberOfScalarComponents() > 1:
                    # Extract the first component
                    component_extractor = vtk.vtkImageExtractComponents()
                    component_extractor.SetInputConnection(reader.GetOutputPort())
                    component_extractor.SetComponents(0)  # Extract the first component
                    component_extractor.Update()

                    t_img = vtk.vtkImageData()
                    t_img.DeepCopy(component_extractor.GetOutput())
                    image3D.AddInputData(t_img)
                else:
                    # Single-component image, just copy
                    t_img = vtk.vtkImageData()
                    t_img.DeepCopy(reader.GetOutput())
                    image3D.AddInputData(t_img)

        image3D.Update()
        print("Complete")
        self.image3D = image3D.GetOutput()

    def create_3D_model(self, folder_path, pixel_spacing=(1.0, 1.0, 1.0), flag='FLIR'):
        import vtk
        import vtkmodules.vtkInteractionStyle
        import vtkmodules.vtkRenderingOpenGL2
        from vtkmodules.vtkCommonColor import vtkNamedColors
        from vtkmodules.vtkCommonCore import vtkLookupTable
        from vtkmodules.vtkCommonDataModel import vtkPlane
        from vtkmodules.vtkFiltersCore import (
            vtkContourFilter,
            vtkCutter,
            vtkPolyDataNormals,
            vtkStripper,
            vtkStructuredGridOutlineFilter,
            vtkTubeFilter
        )
        from vtkmodules.vtkFiltersExtraction import vtkExtractGrid
        from vtkmodules.vtkIOParallel import vtkMultiBlockPLOT3DReader
        from vtkmodules.vtkRenderingCore import (
            vtkActor,
            vtkPolyDataMapper,
            vtkRenderWindow,
            vtkRenderWindowInteractor,
            vtkRenderer
        )
        print("Creating 3D Model...")
        
        if flag == 'NIR':
            iso_value = 200  # NIR
        
        if flag == 'FLIR':
            iso_value = 250  # FLIR

        if flag == 'Optical':
            iso_value = 130

        colors = vtkNamedColors()
        
        volume = self.image3D

        surface = vtk.vtkMarchingCubes()
        surface.SetInputData(volume)
        surface.ComputeNormalsOn()
        surface.SetValue(10, iso_value)

        renderer = vtk.vtkRenderer()
        renderer.SetBackground(colors.GetColor3d('DarkSlateGray'))
        
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetWindowName('vtkMarchingCubes')
        render_window.SetMultiSamples(0)  # Disable anti-aliasing for better parallel rendering

        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(surface.GetOutputPort())
        mapper.ScalarVisibilityOff()
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d('MistyRose'))

        renderer.AddActor(actor)

        # Set the spacing information
        volume.SetSpacing(pixel_spacing)
        print("Starting Render...")
        render_window.Render()
        interactor.Start()

        print("Starting STL File Creation...")
        # Create STL writer
        stl_writer = vtk.vtkSTLWriter()

        # Decimate the mesh
        decimate = vtk.vtkQuadricDecimation()
        decimate.SetInputConnection(surface.GetOutputPort())
        decimate.SetTargetReduction(0.9)  # Adjust the target reduction factor as needed
        decimate.Update()

        stl_writer.SetInputConnection(decimate.GetOutputPort())

        # Set the output file name
        stl_writer.SetFileName(os.path.join(folder_path, "output.stl"))

        # Write the STL file
        stl_writer.Write()

        print("STL file exported successfully.")

class Delete_images:   
    def select_folder_and_delete():
        """
        Prompts user to select a folder and deletes all contents inside it.

        Returns:
            None
        """
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            delete_folder_contents(folder_selected)
            print("All contents in the folder have been deleted.")

class Perspective_Transform:
    class NIR:
        def get_transform(calibration_image):
            """
            Computes the transformation matrix for NIR images.

            Parameters:
                calibration_image (str): Path to the calibration image.

            Returns:
                tuple: Transformation matrix (H), X, Y coordinates, and base image.
            """
            BW_NIR = NIR_Generate_BaseImage(calibration_image)
            
            _, base_img = create_obj3d()
            H, X, Y = calibrate_NIR(BW_NIR, base_img)
            return H, X, Y, base_img
        
        def add_scale_NIR(image, ref_point, x_scale, y_scale, rec_height = 20):
            """
            Adds a scale to the NIR image.

            Parameters:
                image (numpy.ndarray): NIR image.
                ref_point (tuple): Reference point coordinates.
                x_scale (float): Scale in the x-direction.
                y_scale (float): Scale in the y-direction.
                rec_height (int, optional): Height of the rectangle. Defaults to 20.

            Returns:
                numpy.ndarray: Image with scale added.
            """
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            x_initial_point = ref_point
            y_initial_point = ref_point
            
            # X-dir
            for i in range(1,12):
                if i % 2 == 0:
                    color = (0,0,0)
                else:
                    color = (255,255,255)

                # x-dir
                x1 = x_initial_point[0]
                y1 = x_initial_point[1]
                x2 = int(x1 + x_scale)
                y2 = int(y1 + rec_height)
                x_initial_point = (x2, y1)
                cv2.rectangle(image, (x1, y1), (x2, y2),color, -1)  # -1 thickness fills the rectangle
                
                # Y-dir
                x1 = y_initial_point[0]
                y1 = y_initial_point[1]
                x2 = int(x1 - rec_height)
                y2 = int(y1-y_scale)
                y_initial_point = (x1, y2)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)  # -1 thickness fills the rectangle

            # font 
            font = cv2.FONT_HERSHEY_SIMPLEX 
            fontScale = 0.55
            color = (0, 0, 0) 
            thickness = 2
            refer = (ref_point[0]+1, ref_point[1]+16)
            image = cv2.putText(image, '20 mm', refer, font,fontScale, color, thickness, cv2.LINE_AA) 
            return image

    class FLIR:
        def get_transform(calibration_image):
            """
            Computes the transformation matrix for FLIR images.

            Parameters:
                calibration_image (str): Path to the calibration image.

            Returns:
                tuple: Transformation matrix (H), X, Y coordinates, and base image.
            """
            BW_FLIR = FLIR_Generate_BaseImage(calibration_image)
            
            _, base_img = create_obj3d_FLIR()
            H, X, Y = calibrate_NIR(BW_FLIR, base_img)
            return H, X, Y, base_img
        
        def add_scale_FLIR(image, ref_point, x_scale, y_scale, rec_height = 20):
            """
            Adds a scale to the FLIR image.

            Parameters:
                image (numpy.ndarray): FLIR image.
                ref_point (tuple): Reference point coordinates.
                x_scale (float): Scale in the x-direction.
                y_scale (float): Scale in the y-direction.
                rec_height (int, optional): Height of the rectangle. Defaults to 20.

            Returns:
                numpy.ndarray: Image with scale added.
            """
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            x_initial_point = ref_point
            y_initial_point = ref_point
            
            # X-dir
            for i in range(1,12):
                if i % 2 == 0:
                    color = (0,0,0)
                else:
                    color = (255,255,255)

                # x-dir
                x1 = x_initial_point[0]
                y1 = x_initial_point[1]
                x2 = int(x1 + x_scale)
                y2 = int(y1 + rec_height)
                x_initial_point = (x2, y1)
                cv2.rectangle(image, (x1, y1), (x2, y2),color, -1)  # -1 thickness fills the rectangle
                
                # Y-dir
                x1 = y_initial_point[0]
                y1 = y_initial_point[1]
                x2 = int(x1 - rec_height)
                y2 = int(y1-y_scale)
                y_initial_point = (x1, y2)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)  # -1 thickness fills the rectangle

            # font 
            font = cv2.FONT_HERSHEY_SIMPLEX 
            fontScale = 0.55
            color = (0, 0, 0) 
            thickness = 2
            refer = (ref_point[0]+1, ref_point[1]+16)
            image = cv2.putText(image, '20 mm', refer, font,fontScale, color, thickness, cv2.LINE_AA) 
            return image

    class OPTICAL:
        def get_transform(calibration_image):
            """
            Computes the transformation matrix for optical images.

            Parameters:
                calibration_image (str): Path to the calibration image.

            Returns:
                tuple: Transformation matrix (H), X, Y coordinates, and base image.
            """
            BW_Optical = Optical_Generate_BaseImage(calibration_image)
            _, base_img = create_obj3d()
            H, X, Y = calibrate_optical(BW_Optical, base_img)
            return H, X, Y, base_img
        
        def add_scale(image, ref_point, x_scale, y_scale, rec_height = 20):
            """
            Adds a scale to the optical image.

            Parameters:
                image (numpy.ndarray): Optical image.
                ref_point (tuple): Reference point coordinates.
                x_scale (float): Scale in the x-direction.
                y_scale (float): Scale in the y-direction.
                rec_height (int, optional): Height of the rectangle. Defaults to 20.

            Returns:
                numpy.ndarray: Image with scale added.
            """
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            x_initial_point = ref_point
            y_initial_point = ref_point
            for i in range(1,12):
                if i % 2 == 0:
                    color = (0,0,0)
                else:
                    color = (255,255,255)
                # x-dir
                x1 = x_initial_point[0]
                y1 = x_initial_point[1]
                x2 = int(x1 + x_scale)
                y2 = int(y1 + rec_height)
                x_initial_point = (x2, y1)
                cv2.rectangle(image, (x1, y1), (x2, y2),color, -1)  # -1 thickness fills the rectangle
                # Y-dir
                x1 = y_initial_point[0]
                y1 = y_initial_point[1]
                x2 = int(x1 - rec_height)
                y2 = int(y1-y_scale)
                y_initial_point = (x1, y2)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)  # -1 thickness fills the rectangle
                # font 
            font = cv2.FONT_HERSHEY_SIMPLEX 
            fontScale = 0.55
            color = (0, 0, 0) 
            thickness = 2
            refer = (ref_point[0]+1, ref_point[1]+16)
            image = cv2.putText(image, '20 mm', refer, font,fontScale, color, thickness, cv2.LINE_AA) 
            return image
        
    def correct_image(image, H, base_img, flag = None):
        """
        Corrects the perspective of the image.

        Parameters:
            image (numpy.ndarray): Input image.
            H (numpy.ndarray): Transformation matrix.
            base_img (numpy.ndarray): Base image.
            flag (str, optional): Type of image. Defaults to None.

        Returns:
            numpy.ndarray: Corrected image.
        """
        if flag == 'Optical':
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        elif flag == 'NIR':
            image = cv2.rotate(image, cv2.ROTATE_180)
            base_img = np.ones((2000,2000))
        
        elif flag == 'FLIR':
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        image_warp = cv2.warpPerspective(image, H, (base_img.shape[1], base_img.shape[0]))
        return image_warp

class Overlay_Images:
    class Frame:
        def NIR_FLIR(NIR, FLIR):
            pass
    class Layer:
        def Optical_FLIR(Optical, FLIR):
            pass
        def NIR_FLIR(NIR, FLIR):
            pass
        def Optical_NIR(Optical, NIR):
            pass
    

@staticmethod
def calibrate_NIR(image, base_img):
    """
        Calibrates near-infrared (NIR) images using a base image.

        Args:
            image: The NIR image to be calibrated.
            base_img: The base image used for calibration.

        Returns:
            H: Homography matrix.
            X: X scale.
            Y: Y scale.
        """
    ########################################Blob Detector##############################################
    blobParams = cv2.SimpleBlobDetector_Params()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    blobParams.minThreshold = 1
    blobParams.maxThreshold = 399
    blobParams.filterByArea = True
    blobParams.minArea = 10     # minArea may be adjusted to suit for your experiment
    blobParams.maxArea = 15000   # maxArea may be adjusted to suit for your experiment
    blobParams.filterByCircularity = True
    blobParams.minCircularity = 0.1
    blobParams.filterByConvexity = True
    blobParams.minConvexity = 0.87
    blobParams.filterByInertia = True
    blobParams.minInertiaRatio = 0.01
    Detector = cv2.SimpleBlobDetector_create(blobParams)
    ###################################################################################################
    img = image.copy()
    _ , pts_src = cv2.findCirclesGrid(image, (11,9), None, flags = cv2.CALIB_CB_SYMMETRIC_GRID)   # Find the circle grid
    _ , pts_dst = cv2.findCirclesGrid(base_img, (11,9), None, flags = cv2.CALIB_CB_SYMMETRIC_GRID)   # Find the circle grid
    H, _ = cv2.findHomography(pts_src, pts_dst)
    img1_warp = cv2.warpPerspective(img, H, (base_img.shape[1], base_img.shape[0]))
    keypoints = Detector.detect(img1_warp) # Detect blobs
    
    points = []
    
    for keyPoint in keypoints:
        x = keyPoint.pt[0]
        y = keyPoint.pt[1]
        points.append([x, y])
    points = np.array(points)

    point_ex = points[:10]
    sorted_pairs_1 = point_ex[point_ex[:, 0].argsort()]

    point_ex = points[11:21]
    sorted_pairs_2 = point_ex[point_ex[:, 0].argsort()]

    Right = sorted_pairs_1[1,:]
    Center = sorted_pairs_1[0,:]
    Bottom = sorted_pairs_2[0,:]
    Y_scale = Center[1] - Bottom[1]
    X_scale = Right[0] - Center[0]
    X = X_scale
    Y = Y_scale
    return H, X, Y

@staticmethod
def calibrate_optical(image, base_img):
    """
        Calibrates optical images using a base image and object points.

        Args:
            image: The optical image to be calibrated.
            objp: Object points used for calibration.
            base_img: The base image used for calibration.

        Returns:
            H: Homography matrix.
            X: X scale.
            Y: Y scale.
    """
    ########################################Blob Detector##############################################
    # Setup SimpleBlobDetector parameters.
    blobParams = cv2.SimpleBlobDetector_Params()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    blobParams.minThreshold = 1
    blobParams.maxThreshold = 399
    blobParams.filterByArea = True
    blobParams.minArea = 150     # minArea may be adjusted to suit for your experiment
    blobParams.maxArea = 15000   # maxArea may be adjusted to suit for your experiment
    blobParams.filterByCircularity = True
    blobParams.minCircularity = 0.1
    blobParams.filterByConvexity = True
    blobParams.minConvexity = 0.87
    blobParams.filterByInertia = True
    blobParams.minInertiaRatio = 0.01
    Detector = cv2.SimpleBlobDetector_create(blobParams)
    ###################################################################################################

    img = image.copy()
    _ , pts_src = cv2.findCirclesGrid(image, (11,9), None, flags = cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector=Detector)   # Find the circle grid
    _ , pts_dst = cv2.findCirclesGrid(base_img, (11,9), None, flags = cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector=Detector)   # Find the circle grid
    
    try:
        if len(pts_src) != len(pts_dst):
            print("Error, pts_src = ", len(pts_src), " pts_dst = ", len(pts_dst),". Adjust blobDetector")
    except Exception as e:
        print(f"Failed to find points: {e}")

    H, _ = cv2.findHomography(pts_src, pts_dst)
    img1_warp = cv2.warpPerspective(img, H, (base_img.shape[1], base_img.shape[0]))
    keypoints = Detector.detect(img1_warp) # Detect blobs
    points = []
    for keyPoint in keypoints:
        x = keyPoint.pt[0]
        y = keyPoint.pt[1]
        points.append([x, y])
    points = np.array(points)

    point_ex = points[:10]
    sorted_pairs_1 = point_ex[point_ex[:, 0].argsort()]

    point_ex = points[11:21]
    sorted_pairs_2 = point_ex[point_ex[:, 0].argsort()]

    Right = sorted_pairs_1[1,:]
    Center = sorted_pairs_1[0,:]
    Bottom = sorted_pairs_2[0,:]
    Y_scale = Center[1] - Bottom[1]
    X_scale = Right[0] - Center[0]
    X = X_scale
    Y = Y_scale
    return H, X, Y

@staticmethod
def Optical_Generate_BaseImage(img):
    """
        Generates a base image for optical image calibration.

        Args:
            img: The input optical image.

        Returns:
            The base image for calibration.
    """
    ########################################Blob Detector##############################################
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    blobParams = cv2.SimpleBlobDetector_Params()
    blobParams.minThreshold = 1
    blobParams.maxThreshold = 399
    blobParams.filterByArea = True
    blobParams.minArea = 2500     # minArea may be adjusted to suit for your experiment
    blobParams.maxArea = 16000   # maxArea may be adjusted to suit for your experiment
    blobParams.filterByCircularity = True
    blobParams.minCircularity = 0.1
    blobParams.filterByConvexity = True
    blobParams.minConvexity = 0.87
    blobParams.filterByInertia = True
    blobParams.minInertiaRatio = 0.01
    blobDetector = cv2.SimpleBlobDetector_create(blobParams)
    ########################################Blob Detector##############################################
    img = cv2.medianBlur(img,5)
    img = mask_image(img, [(3766, 3245), (3975, 796), (1180, 709),(1317, 3182)])
    keypoints = blobDetector.detect(img) # Detect blobs
    im_with_keypoints =cv2.drawKeypoints(np.zeros_like(img), keypoints, np.array([]), (255,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(im_with_keypoints, 100, 255, cv2.THRESH_BINARY)
    contours,_  = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im_with_keypoints, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    im_with_keypoints = cv2.bitwise_not(im_with_keypoints)

    return im_with_keypoints

@staticmethod
def FLIR_Generate_BaseImage(calibration_image):
    """
        Generates a base image for FLIR image calibration.

        Args:
            calibration_image: The calibration image.

        Returns:
            The base image for FLIR calibration.
    """
    ########################################Blob Detector##############################################
    blobParams = cv2.SimpleBlobDetector_Params()
    blobParams.minThreshold = 1
    blobParams.maxThreshold = 399
    blobParams.filterByArea = True
    blobParams.minArea = 120     # minArea may be adjusted to suit for your experiment
    blobParams.maxArea = 500   # maxArea may be adjusted to suit for your experiment
    blobParams.filterByCircularity = True
    blobParams.minCircularity = 0.1
    blobParams.filterByConvexity = True
    blobParams.minConvexity = 0.87
    blobParams.filterByInertia = True
    blobParams.minInertiaRatio = 0.01
    blobDetector = cv2.SimpleBlobDetector_create(blobParams)
    ########################################Blob Detector##############################################
    img = cv2.normalize(calibration_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.medianBlur(img,7)
    keypoints = blobDetector.detect(img) # Detect blobs.
    im_with_keypoints =cv2.drawKeypoints(np.zeros_like(img), keypoints, np.array([]), (255,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(im_with_keypoints, 80, 255, cv2.THRESH_BINARY)
    contours,_  = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im_with_keypoints, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    im_with_keypoints = cv2.bitwise_not(im_with_keypoints)
    im_with_keypoints = cv2.rotate(im_with_keypoints, cv2.ROTATE_90_CLOCKWISE)
    return im_with_keypoints

@staticmethod
def set_pixels_above_intensity_in_region_top(image, threshold, height_range):
    """
    Sets pixels above a certain intensity threshold in the top region of an image to a specific value.

    Args:
        image: The input image.
        threshold: The intensity threshold.
        height_range: A tuple representing the range of heights to consider for the top region.

    Returns:
        The modified image with pixels set above the intensity threshold in the specified region.
    """
    region_top = height_range[0]
    region_bottom = height_range[1]
    region = slice(region_top, region_bottom), slice(None)
    region_image  = image[region]

    mask = region_image  > threshold
    region_image[mask] = 170
    image[region] = region_image

    return image
      
@staticmethod
def NIR_Generate_BaseImage(img):
    """
        Generates a base image for near-infrared (NIR) image calibration.

        Args:
            img: The input NIR image.

        Returns:
            The base image for calibration.
    """
    ########################################Blob Detector##############################################
    # Setup SimpleBlobDetector parameters.
    blobParams = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    blobParams.minThreshold = 1
    blobParams.maxThreshold = 399
    # Filter by Area.
    blobParams.filterByArea = True
    blobParams.minArea = 700     # minArea may be adjusted to suit for your experiment
    blobParams.maxArea = 3000   # maxArea may be adjusted to suit for your experiment
    # Filter by Circularity
    blobParams.filterByCircularity = True
    blobParams.minCircularity = 0.5
    # Filter by Convexity
    blobParams.filterByConvexity = True
    blobParams.minConvexity = 0.87
    # Filter by Inertia
    blobParams.filterByInertia = True
    blobParams.minInertiaRatio = 0.01
    # Create a detector with the parameters
    blobDetector = cv2.SimpleBlobDetector_create(blobParams)
    ####################################################################################################
    img1 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img1 = mask_image(img1, [(261, 186),(1680, 150),(1711, 1886),(231, 1689)])
    img1 = cv2.rotate(img1, cv2.ROTATE_180)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(5,5))
    equalized_image = clahe.apply(img1)
    img1 = cv2.medianBlur(equalized_image, 31)
    keypoints = blobDetector.detect(img1) # Detect blobs.
    im_with_keypoints =cv2.drawKeypoints(np.zeros_like(img1), keypoints, np.array([]), (255,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(im_with_keypoints, 100, 255, cv2.THRESH_BINARY)
    contours,_  = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im_with_keypoints, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    
    im_with_keypoints = cv2.bitwise_not(im_with_keypoints)
    
    return im_with_keypoints

@staticmethod
def create_obj3d():
    """
    Creates a 3D object representing the real-world coordinates of a circular grid.

    Returns:
        obj3d: 3D array containing the coordinates of the circular grid.
        image: Image representing the circular grid.
    """
    # Real world coordinates of circular grid 
    obj3d = np.zeros((99, 3), np.float32)
    a = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    b = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    A, B = np.meshgrid(a, b)
    obj3d = np.column_stack((A.flatten(), B.flatten(), np.zeros_like(A.flatten())))
    image = np.ones((1000, 1200), dtype=np.uint8) * 255  # White background
    # Calculate a uniform scale for both dimensions
    uniform_scale = 1000 / max(max(a), max(b))

    for point in obj3d:
        x, y, _ = point + (20, 20, 0)  # Offset to prevent clipping at the edge
        x = int(x * uniform_scale)
        y = int(y * uniform_scale)
        radius = int(4 * uniform_scale)  # Radius based on the uniform scaling factor
        cv2.circle(image, (x, y), radius, 0, -1)  # Draw filled black circle

    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return obj3d, image

@staticmethod
def create_obj3d_FLIR():
    """
    Creates a 3D object representing the real-world coordinates of a circular grid for FLIR images.

    Returns:
        obj3d: 3D array containing the coordinates of the circular grid.
        image: Image representing the circular grid.
    """
    # Real world coordinates of circular grid 
    obj3d = np.zeros((88, 3), np.float32)
    a = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    b = [0, 20, 40, 60, 80, 100, 120, 140]
    A, B = np.meshgrid(a, b)
    obj3d = np.column_stack((A.flatten(), B.flatten(), np.zeros_like(A.flatten())))
    image = np.ones((1000, 1200), dtype=np.uint8) * 255  # White background
    # Calculate a uniform scale for both dimensions
    uniform_scale = 1000 / max(max(a), max(b))

    for point in obj3d:
        x, y, _ = point + (20, 20, 0)  # Offset to prevent clipping at the edge
        x = int(x * uniform_scale)
        y = int(y * uniform_scale)
        radius = int(4 * uniform_scale)  # Radius based on the uniform scaling factor
        cv2.circle(image, (x, y), radius, 0, -1)  # Draw filled black circle

    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return obj3d, image

@staticmethod
def mask_image(img, points):
    """
    Masks an image using the specified points.

    Args:
        img: Input image.
        points: List of points for creating the mask.

    Returns:
        Masked image.
    """
    mask = np.zeros_like(img)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], (255,255,255))
    result = cv2.bitwise_and(img, mask)

    return result

@staticmethod
def click_event(event, x, y, flags, param, frame, points):
    """
    Handles mouse click events.

    Args:
        event: Type of mouse event.
        x: X-coordinate of the mouse click.
        y: Y-coordinate of the mouse click.
        flags: Additional flags passed by OpenCV.
        param: Additional parameters passed to the function.
        frame: Frame on which the click event occurred.
        points: List to store the clicked points.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image', frame)

@staticmethod
def natural_sort_key(s):
    """
    Key function for natural sorting.

    Args:
        s: String to be sorted.

    Returns:
        Sorted string.
    """
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

@staticmethod
def count_files_and_folders(folder_path):
    """
    Counts the number of files and folders in the specified directory.

    Args:
        folder_path: Path to the directory.

    Returns:
        Tuple containing the count of folders and files.
    """
    file_count = 0
    folder_count = 0
    for root, dirs, files in os.walk(folder_path):
        folder_count += len(dirs)
        file_count += len(files)

    return folder_count, file_count

@staticmethod
def delete_folder_contents(folder_path):
    """
    Deletes all contents inside the specified folder.

    Parameters:
        folder_path (str): The path to the folder whose contents are to be deleted.

    Returns:
        None
    """
    total_folders, total_files = count_files_and_folders(folder_path)
    with alive_bar(total_folders + total_files) as bar:
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
                bar()
            for dirname in dirs:
                dir_path = os.path.join(root, dirname)
                try:
                    os.rmdir(dir_path)
                except Exception as e:
                    print(f"Failed to delete {dir_path}: {e}")
                bar()
