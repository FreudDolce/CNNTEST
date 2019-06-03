#coding=utf-8
#=========================================================
# label the tumor space by 'hot space'
# Written by Liu Rong & Ji Hongchen
# 20180810
#=========================================================
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

class HOTSPACELABEL:
    def __init__(self, input_path, output_path = ''):
        self.input_path = input_path
        self.output_path = output_path
        self.patient_list = []
        self.dets_dict = {}
        
    def Read_Dets(self, patten):
        dest_file = pd.read_csv(self.input_path + patten + '.csv')
        dest_list = np.array(dest_file)
        for dets in dest_list:
            file_name = dets[0].split('.')[0]
            patient_ID = file_name.split('_')[0]
            ins_num = int(file_name.split('_')[2])
            class_name = int(dets[1])
            x_min = int(dets[2])
            y_min = int(dets[3])
            x_max = int(dets[4])
            y_max = int(dets[5])
            coins = dets[6]
            if patient_ID not in self.patient_list:
                self.patient_list.append(patient_ID)
                self.dets_dict[patient_ID] = [[patient_ID,
                                               ins_num,
                                               class_name,
                                               x_min,
                                               y_min,
                                               x_max,
                                               y_max,
                                               coins]]
            else:
                self.dets_dict[patient_ID].append([patient_ID,
                                                   ins_num,
                                                   class_name,
                                                   x_min,
                                                   y_min,
                                                   x_max,
                                                   y_max,
                                                   coins])                
                
        return self.dets_dict
    
    def Generate_Zaix_Continue(self, dets_array):
        for i in range (len(dets_array)):
            #patient_ID = dets_array[i][0]
            #ins_num = dets_array[i][1]
            #class_name = det_array[i][2]
            #x_min = dets_array[i][3]
            #y_min = dets_array[i][4]
            #x_max = dets_array[i][5]
            #y_max = dets_array[i][6]
            #coins = dets_array[i][7]
            if ((dets_array[i][1] <= 14)
                or(dets_array[i][1] >= len(dets_array) -14)):
                dets_array[i][2] = 0
                dets_array[i][3] = 0
                dets_array[i][4] = 0
                dets_array[i][5] = 0
                dets_array[i][6] = 0
        print (len(dets_array))
        targ = 0
        for j in range (len(dets_array)-3):
            if targ == 1:
                targ -= 1
                continue
            elif dets_array[j][6] != 0:
                if dets_array[j-1][6] == 0:
                    for k in range(3, 8):
                        dets_array[j-1][k] = dets_array[j][k]
                if dets_array[j+1][6] == 0:
                    if dets_array[j+2][6] != 0:
                        for k in range(3, 7):
                            dets_array[j+1][k] = round(0.5 * (dets_array[j][k] +
                                                              dets_array[j+2][k]))
                            dets_array[j+1][7] = 0.5 * (dets_array[j][7] +
                                                        dets_array[j+2][7])
                    elif dets_array[j+2][6] == 0:
                        for k in range(3, 8):
                            dets_array[j+1][k] = dets_array[j][k]
                        targ += 1
                       
        return dets_array
    
    def Calculate_HWL_Ratio(self, dets_array):
        dets_array = np.array(dets_array)
        y_max_choseen = np.where(dets_array[:,6] != 0)[0]
        label_list = []
        for i in y_max_choseen:
            pass
        
    def Generate_Hot_Space(self, dets_array):
        pass
        
if __name__ == '__main__':
    hsl = HOTSPACELABEL(input_path=r'/home/freud/Documents/dets/')
    pre_dict = hsl.Read_Dets(patten='pre')
    output_array = []
    for patient in hsl.patient_list:
        pre_out_array = hsl.Generate_Zaix_Continue(dets_array=hsl.dets_dict[patient])
        output_array.extend(pre_out_array)
    output_array = np.array(output_array)
    output = pd.DataFrame(output_array)
    output.to_csv(r'/home/freud/output.csv')