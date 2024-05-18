import torch
import numpy as np
import cv2,os
import matplotlib.pyplot as plt
from IPython.display import display, Image
import PySimpleGUI as sg
from PIL import Image
import glob
import time

def detect_DAT(model1):
    img = "C:/Users/CCSX009/Documents/yolov5/camera1/*.jpg"
    for image in glob.glob(img):
        if len(image) == 0:
            print('no image')
        else: 
            t1 = time.time()
            results = model1(image,608,0.15)
            print('model1.names:',model1.names)
            area_remove = []
            table1 = results.pandas().xyxy[0]        
            for item in range(len(table1.index)):
                name_label = table1['name'][item]
                confidence = (table1['confidence'][item])*100
                for str_key, data_value in model1.names.items():
                    if values[f'{data_value}_1'] == True:
                        if data_value == name_label:
                            if confidence < int(values[f'{data_value}_Conf_1']): 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove.append(item)
            name = list(table1['name'])
            print(name)
            show1 = np.squeeze(results.render(area_remove))
            show1 = cv2.resize(show1, (1000,800), interpolation = cv2.INTER_AREA)
            show1 = cv2.cvtColor(show1, cv2.COLOR_BGR2RGB)
            imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
            window['image1'].update(data= imgbytes1)
            os.remove(image)
            t2 = time.time() - t1
            time_processing = str(int(t2*1000)) + 'ms'
            window['time_cam1'].update(value= time_processing, text_color='black') 

def detect_ChatGPT(model1):
    img_folder = "C:/Users/CCSX009/Documents/yolov5/camera1/"
    for image_path in glob.glob(img_folder + "*.jpg"):
        t1 = time.time()
        results = model1(image_path, 608, 0.15)
        area_remove = []
        table1 = results.pandas().xyxy[0]        
        for item, (name_label, confidence) in enumerate(zip(table1['name'], table1['confidence'] * 100)):
            for str_key, data_value in model1.names.items():
                if values[f'{data_value}_1'] and data_value == name_label and confidence < int(values[f'{data_value}_Conf_1']):
                    table1.drop(item, axis=0, inplace=True)
                    area_remove.append(item)
                    break 
        name = list(table1['name'])
        print(name)   
        show1 = np.squeeze(results.render(area_remove))
        show1 = cv2.resize(show1, (1000,800), interpolation=cv2.INTER_AREA)
        show1 = cv2.cvtColor(show1, cv2.COLOR_BGR2RGB)
        imgbytes1 = cv2.imencode('.png', show1)[1].tobytes()
        window['image1'].update(data=imgbytes1)
        os.remove(image_path)
        t2 = time.time() - t1
        time_processing = str(int(t2*1000)) + 'ms'
        window['time_cam1'].update(value=time_processing, text_color='black')

def make_window():
    layout = [
        [sg.Image(filename='', size=(1000,800), key='image1', background_color='black')],
        [sg.Text(' ')],
        [sg.Text('Time Processing : ',justification='center' ,font= ('Helvetica',30),text_color='red', expand_y=True),sg.Text('0 ms',font=('Helvetica',40), key='time_cam1', expand_x=True)],
        [sg.Text('Confidence',size=(12,1),font=('Helvetica',15), text_color='red'), sg.Slider(range=(1,100),default_value=30,orientation='h',size=(60,20),font=('Helvetica',11), key= 'conf_thres1'),],
        [sg.Frame('',[
                [
                    sg.Text(f'{model1.names[i1]}_1',size=(12,1),font=('Helvetica',15), text_color='yellow'), 
                    sg.Checkbox('',size=(3,1),default=True,font=('Helvetica',15),  key=f'{model1.names[i1]}_1'), 
                    sg.Radio('',group_id=f'Cam1 {i1}',size=(3,1),default=False,font=('Helvetica',15),  key=f'{model1.names[i1]}_OK_1'), 
                    sg.Input('1',size=(2,1),font=('Helvetica',15),key= f'{model1.names[i1]}_Num_1',text_color='navy'), 
                    sg.Text('',size=(2,1),font=('Helvetica',15), text_color='red'), 
                    sg.Radio('',group_id=f'Cam1 {i1}',size=(2,1),default=False,font=('Helvetica',15),  key=f'{model1.names[i1]}_NG_1'), 
                    sg.Input('0',size=(7,1),font=('Helvetica',15),key= f'{model1.names[i1]}_Wn_1',text_color='navy'), 
                    sg.Text('',size=(1,1),font=('Helvetica',15), text_color='red'), 
                    sg.Input('1600',size=(7,1),font=('Helvetica',15),key= f'{model1.names[i1]}_Wx_1',text_color='navy'), 
                    sg.Text('',size=(1,1),font=('Helvetica',15), text_color='red'), 
                    sg.Input('0',size=(7,1),font=('Helvetica',15),key= f'{model1.names[i1]}_Hn_1',text_color='navy'), 
                    sg.Text('',size=(1,1),font=('Helvetica',15), text_color='red'), 
                    sg.Input('1200',size=(7,1),font=('Helvetica',15),key= f'{model1.names[i1]}_Hx_1',text_color='navy'), 
                    sg.Text('',size=(2,1),font=('Helvetica',15), text_color='red'), 
                    sg.Input('0',size=(7,1),font=('Helvetica',15),key= f'{model1.names[i1]}_PLC_1',text_color='navy'),
                    sg.Slider(range=(1,100),default_value=30,orientation='h',size=(28,9),font=('Helvetica',10), key= f'{model1.names[i1]}_Conf_1'),
                ] for i1 in range(len(model1.names))
            ], relief=sg.RELIEF_FLAT)],
        [sg.Button('Open Image'), sg.Button('Exit')]
    ]
    layout_option1 = [[sg.Column(layout, scrollable = True)]]
    layout = [[sg.TabGroup([[  sg.Tab('Main', layout_option1)]])
               ]]
    window = sg.Window('HuynhLeVu', layout, location=(0,0),resizable=True).Finalize()
    # window.Maximize()
    return window

model1 = torch.hub.load('C:/Users/CCSX009/Documents/yolov5','custom', path="C:/Users/CCSX009/Documents/yolov5/file_train/X75_M100_DEN_CAM1_2024-03-11.pt", source='local',force_reload =False)
window = make_window()
while True:
    event, values = window.read(timeout=20)
    if event == sg.WINDOW_CLOSED or event == 'Exit':
        break

    detect_ChatGPT(model1)
       
window.close()