#===========================================
#Devide dicom files into different folders
#Written by Ji Hongchen
#20180828
#===========================================

import os
import pandas as pd
import numpy as np
import pydicom
import shutil

def Devide_Folder(input_path, output_path):
    """
    The new folder name is id+modality+serdescription+instanceid
    """
    patient_list = os.listdir(input_path)
    for patient in patient_list:
        print(patient, ' Start!')
        image_list = os.listdir(input_path + patient)
        for image in image_list:
            image_info = pydicom.read_file(input_path + patient + '/' + image)
            try:
                ExaminDate = image_info.SeriesDate
            except AttributeError:
                ExaminDate = '00000000'
            try:
                patientID = image_info.PatientID
            except AttributeError:
                patientID = '00000000'
            try:
                Modality = image_info.Modality
            except AttributeError:
                Modality = 'MR'
            try:
                Seriesdescription = image_info.SeriesDescription
            except AttributeError:
                Seriesdescription = 'SeriesDescriptionMissing'
            try:
                InstanceUID = image_info.SeriesInstanceUID
            except AttributeError:
                InstanceUID = 'InstanceUIDMissing'
            
            new_path = output_path + patientID + \
                '/' + patientID + \
                '-' + ExaminDate + \
                '-' + Modality + \
                '-' + Seriesdescription + \
                '-' + InstanceUID
            
            if os.path.exists(new_path) == False:
                os.makedirs(new_path)
            
            shutil.copy(input_path + patient + '/' + image,
                        new_path)    
        print(patient, 'Finished')
            
if __name__ == '__main__':
    Devide_Folder(input_path=r'/media/freud/livercancer/liver_cancer_pre/',
                  output_path=r'/media/freud/Data/choseen_3/')
