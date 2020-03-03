# Face detection demo using MTCNN and OpenCV
# Author: Juan-Pablo Ramirez-Paredes <jpi.ramirez@ugto.mx>
# Course: Artificial Intelligence
# University of Guanajuato
#
# This small demonstration needs Tensorflow 1.14, Keras, OpenCV 4 and MTCNN
# A quick way to install these on Win10 and Anaconda (as admin):
#
# conda create -n facerec pip python=3.7
# conda activate facerec
# pip install tensorflow==1.14 opencv-contrib-python==4.1.0.25
# pip install keras
# pip install mtcnn
#pip install Pillow
#pip install matplotlib
#pip install sklearn
#https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
# pip install mtcnn
#https://drive.google.com/drive/folders/12aMYASGCKvDdkygSv1yQq8ns03AStDO_


from mtcnn.mtcnn import MTCNN
import cv2
from PIL import Image #For image processing
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

def SacarVector(model,cara):
    #Preparar imagen para el modelo
    #Escalar pixeles
    cara = cara.astype('float32')
    #Estandarizar los valores en los canales
    m, std = cara.mean(),cara.std()
    cara=(cara - m )/std
    #Convetir a una muestra
    muestra = np.expand_dims(cara, axis=0)
    #Vector de características
    res= model.predict(muestra)
    #Normalizar vector
    res=Normalizer(norm='l2').transform(res)
    return res[0]


def ExtraerRostro():
    x1, y1 = abs(bbox[0]), abs(bbox[1])
    x2, y2 = x1 + bbox[2] ,y1 + bbox[3]
    cara=imgsml[y1:y2 , x1:x2]
    #Image preprocessing
    rostromono=Image.fromarray(cara)
    rostrores=rostromono.resize((160,160))
    rostro_array=np.asarray(rostrores)
    return rostro_array
        
modelos2={}
model = load_model('facenet_keras.h5')
w=640
h=480
color=(0,0,0)
analizando=False
identificando=False
nombre=''
#TODO: Borrar esta linea
#nombre='dara'

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    mirror = True
    detector = MTCNN()
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        imgsml = cv2.resize(img, (w, h))
        #TODO: Hacer que solo detecte una cara
        faces = detector.detect_faces(imgsml) 
        if len(faces) > 0:
            for k in range(len(faces)): 
                bbox = faces[k]['box']
                keypoints = faces[k]['keypoints']
                imgsml = cv2.rectangle(imgsml, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,255,0))
                cv2.circle(imgsml,(keypoints['left_eye']), 2, (0,155,255), 2)
                cv2.circle(imgsml,(keypoints['right_eye']), 2, (0,155,255), 2)
                cv2.circle(imgsml,(keypoints['nose']), 2, (0,155,255), 2)
                cv2.circle(imgsml,(keypoints['mouth_left']), 2, (0,155,255), 2)
                cv2.circle(imgsml,(keypoints['mouth_right']), 2, (0,155,255), 2)
                
        cv2.circle(imgsml,(w-20,h-20), 10, color, -1)     
        #TODO: Checar si no hay un bug (valores negativos en el bounding box)
        
                                  
        cv2.imshow('Webcam View', imgsml)
        k=cv2.waitKey(1)
        modelos2=paraPrueba()
        
        if k == 27:
            break  # esc to exit
 
# =============================================================================
#         #Guardar un rostro
# =============================================================================
        if k == 32: #Espacio
            rostro=list()
            modelo=list()
            if len(faces)==1: #Solo un rostro puede ser extraído a la vez
                print('Guardando rostro')
                if len(nombre)==0:
                    nombre=input('Cuál es tu nombre?')
                        
                #------------Extraer rostro            
                rostro_out=ExtraerRostro()
                #------------Agregar
                rostro.append(rostro_out)
                np.asarray(rostro)
                #-----------Vector
                modelo=SacarVector(model,rostro[0])
                #modelo.append(res)
                
                if nombre in modelos2:
                    modelos2[nombre].append(modelo)
                    
                else:
                    modelos2[nombre]=[modelo]
                        
                print('Rostro guardado') 
                agregar =input('Quieres agregar más imágenes? s para si \t')
                if agregar != 's':
                    nombre=''
               
            rostro.clear()
            #modelo.clear()
            k=-1
            continue
# =============================================================================
#         #Analizar rostro
# =============================================================================
        if k==9:#Tab
            if not analizando:
                analizando=True
            else:
                analizando=False
            k=-1
        
        if analizando:
            if len(modelos2)>0:  
                if len(faces)==1: #Solo un rostro puede ser extraído a la vez
                    #------------Extraer rostro
                    rostro_out=ExtraerRostro()
                    #-----------Vector 
                    res=SacarVector(model,rostro_out)
                    for nom in modelos2:
                        for modelo in modelos2[nom]:
                            dist = np.linalg.norm(res-modelo)
                            print(dist)
                            if dist<=0.5:
                                print('Rostro de ',nom)
                                #cv2.putText(imgsml, nombre, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)   
                                color=(0,255,0)
                            else:
                                color=(0,0,0)
# =============================================================================
# Identificar un rostro                  
# =============================================================================
        if k==43: #+
            if not identificando:
                identificando=True
            else:
                identificando=False
            k=-1
        
        if identificando:
            trainX=np.empty((len(modelos2),128))
            trainY=np.empty((len(modelos2),))
            if len(modelos2)>1: #Debe haber mínimo 2 clases 
                if len(faces)==1: #Solo un rostro puede ser extraído a la vez
                    #------------Extraer rostro
                    rostro_out=ExtraerRostro()
                    #-----------Vector 
                    res=SacarVector(model,rostro_out)
                    for nom in modelos2:
                            for modelo in modelos2[nom]:
                                temp=np.append(trainX,[modelo],axis=0)
                                trainX=np.copy(temp)
                                temp2=np.append(trainY,[nom],axis=0)
                                trainY=np.copy(temp2)   
                    model2 = SVC(kernel='linear', probability=True)
                    model2.fit(trainX, trainY)
                    pred=model2.predict([res])
                    print(pred)
                    
          
    
    cv2.destroyAllWindows()
