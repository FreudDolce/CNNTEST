#======================================================
#  Faster-RCNN data Generation
#  Writed by Liur Rong & Ji Hongchen
#  2018-07-31
#======================================================
import numpy as np
import pandas as pd
from PIL import Image
import pydicom
from xml.dom.minidom import Document
import SimpleITK as sitk
import os
from random import random

class Generate_VOC_data:
    """
    =======================================================
    This class focus on changing the origin data into 
    training data.
    It include 4 functions and 3 parameters
    Use like:
    gvd = Generate_VOC_data(input_image_path = r'iip')
    gvd.Read_Patient_Info()
    gvd.Generate_Train_Data()
    =======================================================    
    """
    def __init__(self, input_image_path, input_label_path, output_path):
        self.input_image_path = input_image_path
        self.input_label_path = input_label_path
        self.output_path = output_path
        self.patient_list = []
        self.sets_index = {}
        self.series_dict = {}
        
    def Read_Patient_Info(self):
        label_data_ori = pd.read_csv(self.input_label_path)
        label_data = np.array(label_data_ori)
        for label_info in label_data:
            self.sets_index[label_info[0]] = {}
            self.series_dict[label_info[0]] = []
            if label_info[0] not in self.patient_list:
                self.patient_list.append(label_info[0])
        for label_info in label_data:
            self.series_dict[label_info[0]].append(label_info[2])
            self.sets_index[label_info[0]][label_info[2]] = label_info[3: 14]
            
    def Generate_Xml(self,
                     filename_in,
                     size_in,
                     class_in,
                     xmin_in,
                     ymin_in,
                     xmax_in,
                     ymax_in,
                     save_path,
                     A_xmin_in,
                     A_ymin_in,
                     A_xmax_in,
                     A_ymax_in,
                     anno = True):
        
        doc = Document()
        annotation = doc.createElement('annotation')
        doc.appendChild(annotation)
        
        folder = doc.createElement('folder')
        fold_name = doc.createTextNode('VOC2007')
        folder.appendChild(fold_name)
        annotation.appendChild(folder)
        
        filename = doc.createElement('filename')
        filename_name = doc.createTextNode(filename_in)
        filename.appendChild(filename_name)
        annotation.appendChild(filename)
        
        source = doc.createElement('source')
        annotation.appendChild(source)
        
        database = doc.createElement('database')
        database.appendChild(doc.createTextNode('The VOC 2007 Database'))
        source.appendChild(database)
        
        annotation_s = doc.createElement('annotation')
        annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
        source.appendChild(annotation_s)
        
        image = doc.createElement('image')
        image.appendChild(doc.createTextNode('flickr'))
        source.appendChild(image)
        
        flickrid = doc.createElement('flickrid')
        flickrid.appendChild(doc.createTextNode('270214852'))
        source.appendChild(flickrid)
        
        owner = doc.createElement('owner')
        annotation.appendChild(owner)
        
        flickrid_o = doc.createElement('flickrid')
        flickrid_o.appendChild(doc.createTextNode('Livercancer'))
        owner.appendChild(flickrid_o)
        
        name_o = doc.createElement('name')
        name_o.appendChild(doc.createTextNode('JiHonchen'))
        owner.appendChild(name_o)
        
        size = doc.createElement('size')
        annotation.appendChild(size)
        
        width = doc.createElement('width')
        width.appendChild(doc.createTextNode(str(size_in[0])))
        height = doc.createElement('height')
        height.appendChild(doc.createTextNode(str(size_in[1])))
        depth = doc.createElement('depth')
        depth.appendChild(doc.createTextNode(str(3)))
        size.appendChild(width)
        size.appendChild(height)
        size.appendChild(depth)
        
        segmented = doc.createElement('segmented')
        segmented.appendChild(doc.createTextNode('1'))
        annotation.appendChild(segmented)
        
        objects = doc.createElement('object')
        annotation.appendChild(objects)
    
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode(str(3)))
        objects.appendChild(object_name)
    
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)
    
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('0'))
        objects.appendChild(truncated)
    
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
    
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)
   
        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(str(A_xmin_in)))
        bndbox.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(str(A_ymin_in)))
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(str(A_xmax_in)))
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(str(A_ymax_in)))
        bndbox.appendChild(ymax)
        
        if anno == True:
            objects = doc.createElement('object')
            annotation.appendChild(objects)
        
            object_name = doc.createElement('name')
            object_name.appendChild(doc.createTextNode(str(class_in)))
            objects.appendChild(object_name)
        
            pose = doc.createElement('pose')
            pose.appendChild(doc.createTextNode('Unspecified'))
            objects.appendChild(pose)
        
            truncated = doc.createElement('truncated')
            truncated.appendChild(doc.createTextNode('0'))
            objects.appendChild(truncated)
        
            difficult = doc.createElement('difficult')
            difficult.appendChild(doc.createTextNode('0'))
            objects.appendChild(difficult)
        
            bndbox = doc.createElement('bndbox')
            objects.appendChild(bndbox)
       
            xmin = doc.createElement('xmin')
            xmin.appendChild(doc.createTextNode(str(xmin_in)))
            bndbox.appendChild(xmin)
            ymin = doc.createElement('ymin')
            ymin.appendChild(doc.createTextNode(str(ymin_in)))
            bndbox.appendChild(ymin)
            xmax = doc.createElement('xmax')
            xmax.appendChild(doc.createTextNode(str(xmax_in)))
            bndbox.appendChild(xmax)
            ymax = doc.createElement('ymax')
            ymax.appendChild(doc.createTextNode(str(ymax_in)))
            bndbox.appendChild(ymax)
        
        f = open(save_path, 'w')
        f.write(doc.toprettyxml(indent = ''))
        f.close()
        
    def Generate_Jpg_Image(self, input_array, ins_num, path):
        jpg_array = input_array/input_array.max()*255
        jpg_array = jpg_array.round()
        jpg_image = Image.fromarray(jpg_array)
        jpg_ouput = jpg_image.convert('RGB')
        if ins_num <10:
            self.jpg_output_filename = path + '_00000' + str(ins_num)
        else:
            self.jpg_output_filename = path + '_0000' + str(ins_num)
        jpg_ouput.save(self.output_path + \
                       '/JPEGImages/' + \
                       self.jpg_output_filename + '.jpg')
        
    
    def Generate_Train_Data(self):
        train_t = open(self.output_path + '/ImageSets/Main/train.txt', 'w')
        val_t = open(self.output_path + '/ImageSets/Main/val.txt', 'w')
        test_t = open(self.output_path + '/ImageSets/Main/test.txt', 'w')
        trainval_t = open(self.output_path + '/ImageSets/Main/trainval.txt', 'w')
        for patientID in self.patient_list:
            patten_pool = self.series_dict[patientID]
            patten_list = os.listdir(self.input_image_path + '/' + patientID)
            for patten in patten_list:
                patten_path = self.input_image_path + '/' + patientID + '/' + patten
                image_list = os.listdir(patten_path)
                series_test_sample = pydicom.read_file(patten_path + '/' + image_list[3])
                if series_test_sample.SeriesNumber in patten_pool:
                    reader = sitk.ImageSeriesReader()
                    dcm_names = reader.GetGDCMSeriesFileNames(patten_path)
                    reader.SetFileNames(dcm_names)
                    image = reader.Execute()
                    image_array = sitk.GetArrayFromImage(image)
                    cutoff_one = self.sets_index[patientID][series_test_sample.SeriesNumber][5] - 3
                    cutoff_two = self.sets_index[patientID][series_test_sample.SeriesNumber][5]
                    cutoff_three = self.sets_index[patientID][series_test_sample.SeriesNumber][2]
                    cutoff_four = self.sets_index[patientID][series_test_sample.SeriesNumber][2] + 3
                    for i in range(len(image_array)):
                        if i in range (0, cutoff_one):
                            self.Generate_Jpg_Image(input_array=image_array[i],
                                                    ins_num=i,
                                                    path=patientID + '_' + str(series_test_sample.SeriesNumber))
                            self.Generate_Xml(filename_in=self.jpg_output_filename + '.jpg',
                                              size_in=[image_array[i].shape[1],
                                                     image_array[i].shape[0]],
                                              class_in=self.sets_index[patientID][series_test_sample.SeriesNumber][6],
                                              xmin_in=self.sets_index[patientID][series_test_sample.SeriesNumber][0],
                                              xmax_in=self.sets_index[patientID][series_test_sample.SeriesNumber][3],
                                              ymin_in=self.sets_index[patientID][series_test_sample.SeriesNumber][1],
                                              ymax_in=self.sets_index[patientID][series_test_sample.SeriesNumber][4],
                                              A_xmin_in=self.sets_index[patientID][series_test_sample.SeriesNumber][7],
                                              A_ymin_in=self.sets_index[patientID][series_test_sample.SeriesNumber][8],
                                              A_xmax_in=self.sets_index[patientID][series_test_sample.SeriesNumber][9],
                                              A_ymax_in=self.sets_index[patientID][series_test_sample.SeriesNumber][10],
                                              save_path=self.output_path + '/Annotations/' + self.jpg_output_filename + '.xml',
                                              anno=False)
                            if 0.5<random()<=0.75:
                                print (self.jpg_output_filename, file=val_t)
                                print (self.jpg_output_filename, file=trainval_t)
                            elif 0.75<random()<=1:
                                print (self.jpg_output_filename, file = test_t)
                            else:
                                print (self.jpg_output_filename, file = train_t)
                                print (self.jpg_output_filename, file = trainval_t)
                                
                        elif i in range (cutoff_two, (cutoff_three + 1)):
                            self.Generate_Jpg_Image(input_array=image_array[i],
                                                    ins_num=i,
                                                    path=patientID + '_' + str(series_test_sample.SeriesNumber))
                            self.Generate_Xml(filename_in=self.jpg_output_filename + '.jpg',
                                              size_in=[image_array[i].shape[1],
                                                     image_array[i].shape[0]],
                                              class_in=self.sets_index[patientID][series_test_sample.SeriesNumber][6],
                                              xmin_in=self.sets_index[patientID][series_test_sample.SeriesNumber][0],
                                              xmax_in=self.sets_index[patientID][series_test_sample.SeriesNumber][3],
                                              ymin_in=self.sets_index[patientID][series_test_sample.SeriesNumber][1],
                                              ymax_in=self.sets_index[patientID][series_test_sample.SeriesNumber][4],
                                              A_xmin_in=self.sets_index[patientID][series_test_sample.SeriesNumber][7],
                                              A_ymin_in=self.sets_index[patientID][series_test_sample.SeriesNumber][8],
                                              A_xmax_in=self.sets_index[patientID][series_test_sample.SeriesNumber][9],
                                              A_ymax_in=self.sets_index[patientID][series_test_sample.SeriesNumber][10],                                              
                                              save_path=self.output_path + '/Annotations/' + self.jpg_output_filename + '.xml',
                                              anno=True)
                            if 0.5<random()<=0.75:
                                print (self.jpg_output_filename, file=val_t)
                                print (self.jpg_output_filename, file=trainval_t)
                            elif 0.75<random()<=1:
                                print (self.jpg_output_filename, file = test_t)
                            else:
                                print (self.jpg_output_filename, file = train_t)
                                print (self.jpg_output_filename, file=trainval_t)
                                
                        elif i in range (cutoff_four, len(image_array)):
                            self.Generate_Jpg_Image(input_array=image_array[i],
                                                    ins_num=i,
                                                    path=patientID + '_' + str(series_test_sample.SeriesNumber))
                            self.Generate_Xml(filename_in=self.jpg_output_filename + '.jpg',
                                              size_in=[image_array[i].shape[1],
                                                     image_array[i].shape[0]],
                                              class_in=self.sets_index[patientID][series_test_sample.SeriesNumber][6],
                                              xmin_in=self.sets_index[patientID][series_test_sample.SeriesNumber][0],
                                              xmax_in=self.sets_index[patientID][series_test_sample.SeriesNumber][3],
                                              ymin_in=self.sets_index[patientID][series_test_sample.SeriesNumber][1],
                                              ymax_in=self.sets_index[patientID][series_test_sample.SeriesNumber][4],
                                              A_xmin_in=self.sets_index[patientID][series_test_sample.SeriesNumber][7],
                                              A_ymin_in=self.sets_index[patientID][series_test_sample.SeriesNumber][8],
                                              A_xmax_in=self.sets_index[patientID][series_test_sample.SeriesNumber][9],
                                              A_ymax_in=self.sets_index[patientID][series_test_sample.SeriesNumber][10],                                              
                                              save_path=self.output_path + '/Annotations/' + self.jpg_output_filename + '.xml',
                                              anno=False)
                            if 0.5<random()<=0.75:
                                print (self.jpg_output_filename, file=val_t)
                                print (self.jpg_output_filename, file=trainval_t)
                            elif 0.75<random()<=1:
                                print (self.jpg_output_filename, file = test_t)
                            else:
                                print (self.jpg_output_filename, file = train_t) 
                                print (self.jpg_output_filename, file=trainval_t)
                            
                            
                            



    
    
if __name__ == '__main__':
    gvd = Generate_VOC_data(input_label_path=r'/home/ji/Documents/cnn_test/label.csv',
                            input_image_path=r'/media/ji/data/choseen',
                            output_path=r'/home/ji/tf-faster-rcnn/data/VOCdevkit2007/VOC2007')
    gvd.Read_Patient_Info()
    gvd.Generate_Train_Data()