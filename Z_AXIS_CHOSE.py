#===============================================
# This programe used for chose false positive
# Written by Ji Hongchen
# 20180806
#===============================================

import pandas as pd
import numpy as np
import cv2
import os

class ZAXISCHOSE:
    def __init__(self, input_path, output_path = ''):
        self.input_path = input_path
        self.output_path = output_path
        self.patient_list = []
        self.origin_size = (88, 410, 512)
        self.blank_cube = ((88, 88, 88))
    
    def Generate_Dets_Dict(self, patten):
        """
        Read the images and lay out a dict:
        {patientID: [[instance_num, class, x_min,
                      y_min, x_max, y_max, 
                      coincidence],[label2], ...[labeli]]}
        """
        label_frame = pd.read_csv(self.input_path + '/' + patten + '.csv')
        label_list = np.array(label_frame)
        dets_dict = {}
        self.patient_list = []
        for i in label_list:
            patient_info = i[0]
            patient_info = patient_info.split('.')[0]
            patient_ID = patient_info.split('_')[0]
            instance_num = int(patient_info.split('_')[2])
            c_class = i[1]
            x_min = i[2]
            y_min = i[3]
            x_max = i[4]
            y_max = i[5]
            confidence = i[6]
            if patient_ID not in self.patient_list:
                self.patient_list.append(patient_ID)
                dets_dict[patient_ID] = [[instance_num,
                                         c_class,
                                         x_min, y_min, x_max, y_max,
                                         confidence]]
            else:
                dets_dict[patient_ID].append([instance_num,
                                              c_class,
                                              x_min, y_min, x_max, y_max,
                                              confidence])
            
        return dets_dict
    
    def Calculate_Max(self,
                      label_list,
                      origin_size = (88, 410, 512),
                      blank_cube = np.zeros((88, 88, 88))):# z, y, x
        """
        Generate a hot space cube,
        origin_size is the size of MR space
        blank_cube is the size of the hot cube 
        Generate a hot cube of one patient, one patten.
        """
        for labels in label_list:
            blank_cube_xmin = int(labels[2]/origin_size[2] * blank_cube.shape[2])
            blank_cube_ymin = int(labels[3]/origin_size[1] * blank_cube.shape[1])
            blank_cube_xmax = int(labels[4]/origin_size[2] * blank_cube.shape[2]) + 1
            blank_cube_ymax = int(labels[5]/origin_size[1] * blank_cube.shape[1]) + 1
            blank_cube_z = int(labels[0]/origin_size[0] * blank_cube.shape[0])
            # a effctive dets generate 3 layers in hot cube:
            #    Z-1, Z, Z+1
            if blank_cube_z <= 86:
                for z_Z in (blank_cube_z, blank_cube_z + 1):
                    for y_Y in range(blank_cube_ymin, blank_cube_ymax + 1):
                        for x_X in range(blank_cube_xmin, blank_cube_xmax + 1):
                            blank_cube[z_Z, y_Y, x_X] += labels[6]

        filled_cube_patten = np.array(blank_cube)
        return filled_cube_patten
    
    def Generate_Order_Cube(self, filled_cube):
        """
        Return the labels:
        ([[pt1(x, y, z)], [pt2(x, y, z)]])
        """
        ind_max = filled_cube.max()
        inds = np.where(filled_cube >= ind_max)
        
        z_max_coor = inds[0][int(0.5 * len(inds[0]))]
        y_max_coor = inds[1][int(0.5 * len(inds[1]))]
        x_max_coor = inds[2][int(0.5 * len(inds[2]))]
        
        z_min = z_max_coor
        z_max = z_max_coor
        
        cross_section = []
        
        #Z axis continous choise
        while filled_cube[z_min-1, y_max_coor, x_max_coor] != 0:
            z_min = z_min - 1
        while filled_cube[z_max+1, y_max_coor, x_max_coor] != 0:
            z_max = z_max + 1
        
        #Chose cross coor for each layer
        for z in range (z_min, z_max + 1):
            y_min = y_max_coor
            y_max = y_max_coor 
            x_min = x_max_coor
            x_max = x_max_coor            
            while filled_cube[z, y_min-1, x_max_coor] != 0:
                y_min = y_min - 1
            while filled_cube[z, y_max+1, x_max_coor] != 0:
                y_max = y_max + 1
            while filled_cube[z, y_max_coor, x_min-1] != 0:
                x_min = x_min - 1
            while filled_cube[z, y_max_coor, x_max+1] != 0:
                x_max = x_max + 1
            cross_section.append([z, x_min, y_min, x_max, y_max])
        
        #Calculate mean coor for each cross:
        cross_section = np.array(cross_section)
        x_min_final = int(np.percentile(cross_section[:,1], (50)))
        y_min_final = int(np.percentile(cross_section[:,2], (50)))
        x_max_final = int(np.percentile(cross_section[:,3], (50)))
        y_max_final = int(np.percentile(cross_section[:,4], (50)))
        
        x_min_final = round(x_min_final * 512 / 88)
        y_min_final = round(y_min_final * 410 / 88)
        x_max_final = round(x_max_final * 512 / 88)
        y_min_final = round(y_max_final * 410 / 88)   
        
        
        return([[x_min_final, y_min_final, z_min],
                [x_max_final, y_max_final, z_max]])            
            
    def Generate_Out_Label(self, image, pt1, pt2, class_name):
        """
        Label the images. image a patient, a patten for one time.
        pt1 is the point of xmin, ymin, zmin
        pt2 is the point of xmax, ymax, zmax
        class_name is 'HCC', 'ICC'.
        """
        im = cv2.imread(image, 0)
        image_name = image.split('/')[-1]
        image_name = image_name.split('.')[0]
        patientID = image_name.split('_')[0]
        instance_num = image_name.split('_')[2]
        if pt1[2] <= int(instance_num) <= pt2[2]:
            cv2.rectangle(img=im,
                          pt1=(int(pt1[0]), int(pt1[1])),
                          pt2=(int(pt2[0]), int(pt2[1])),
                          color=(255, 255, 255),
                          thickness=1)
            if (pt1[1] > 10):
                cv2.putText(img=im,
                            text=class_name,
                            org=(int(pt1[0]), int(pt1[1] - 6)),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=0.8,
                            color=(255, 255, 255))
            else:
                cv2.putText(img=im,
                            text=class_name,
                            org=(int(pt1[0]), int(pt1[1] + 15)),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=0.8,
                            color=(255, 255, 255)) 
        new_path = '/home/ji/Pictures/faster_rcnn_output/' + patientID
        if os.path.exists(new_path) == False:
            os.mkdir(new_path)
        cv2.imwrite(new_path + '/' + image_name + '.jpg', im)
        print (patientID + 'Finished.')
            
            
if __name__ == '__main__':
    zac = ZAXISCHOSE(input_path=r'/home/ji/Documents/tf-faster-rcnn_output/20180804-20180805_livercancer_MR_20000iters/dets',
                     output_path='')
    pre_dict = zac.Generate_Dets_Dict(patten='pre')
    A_1_dict = zac.Generate_Dets_Dict(patten='A_1')
    A_2_dict = zac.Generate_Dets_Dict(patten='A_2')
    V_1_dict = zac.Generate_Dets_Dict(patten='V_1')
    V_2_dict = zac.Generate_Dets_Dict(patten='V_2')
    post_dict = zac.Generate_Dets_Dict(patten='post')
    
    for patient_ID in zac.patient_list:
        try:
            cube_pre = pre_dict[str(patient_ID)]
        except KeyError:
            cube_pre = np.zeros((88, 88))
        try:
            cube_A_1 = A_1_dict[str(patient_ID)]
        except KeyError:
            cube_A_1 = np.zeros((88, 88))
        try:
            cube_A_2 = A_2_dict[str(patient_ID)]
        except KeyError:
            cube_A_2 = np.zeros((88, 88))
        try:
            cube_V_1 = V_1_dict[str(patient_ID)]
        except KeyError:
            cube_V_1 = np.zeros((88, 88))
        try:
            cube_V_2 = V_2_dict[str(patient_ID)]
        except KeyError:
            cube_V_2 = np.zeros((88, 88))
        try:
            cube_post = post_dict[str(patient_ID)]
        except KeyError:
            cube_post = np.zeros((88, 88))
        
        filled_cube_pre = zac.Calculate_Max(cube_pre)
        filled_cube_A_1 = zac.Calculate_Max(cube_A_1)
        filled_cube_A_2 = zac.Calculate_Max(cube_A_2)
        filled_cube_V_1 = zac.Calculate_Max(cube_V_1)
        filled_cube_V_2 = zac.Calculate_Max(cube_V_2)
        filled_cube_post = zac.Calculate_Max(cube_post)
        
        CUBE = np.zeros((88, 88, 88))
        for z in range(88):
            for y in range(88):
                for x in range(88):
                    CUBE[z, y, x] = filled_cube_pre[z, y, x] +\
                                    filled_cube_A_1[z, y, x] +\
                                    filled_cube_A_2[z, y, x] +\
                                    filled_cube_V_1[z, y, x] +\
                                    filled_cube_V_2[z, y, x] +\
                                    filled_cube_post[z, y, x]
                    
        cross_coor = zac.Generate_Order_Cube(filled_cube=CUBE)
        image_path = '/home/ji/tf-faster-rcnn/data/demo/'
        image_list = os.listdir(image_path)
        for imag in image_list:
            image_name = imag.split('.')[0]
            patient = image_name.split('_')[0]
            if patient == patient_ID:
                zac.Generate_Out_Label(image=image_path + imag,
                                       pt1=cross_coor[0],
                                       pt2=cross_coor[1],
                                       class_name='HCC')
