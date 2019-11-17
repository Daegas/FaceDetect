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


from mtcnn.mtcnn import MTCNN
import cv2
from PIL import Image #For image processing
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import Normalizer



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
    return res
def ExtraerRostro(bbox,imgsml):
    x1, y1 = abs(bbox[0]), abs(bbox[1])
    x2, y2 = x1 + bbox[2] ,y1 + bbox[3]
    rostro=imgsml[y1:y2 , x1:x2]
    #Image preprocessing
    rostromono=Image.fromarray(rostro)
    rostrores=rostromono.resize((160,160))
    rostro_array=np.asarray(rostrores)
    return rostro_array

rostros=list()
modelos=list()
rostroanalizar=np.array
model = load_model('facenet_keras.h5')

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    mirror = True
    detector = MTCNN()
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        imgsml = cv2.resize(img, (640, 480))
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
                
        #TODO: Checar si no hay un bug (valores negativos en el bounding box)
        
                                  
        cv2.imshow('Webcam View', imgsml)
        k=cv2.waitKey(1)
         
        
        if k == 27:
            break  # esc to exit
 
        #Guardar un rostro
        if k == 32: #Espacio
            print('Guardando rostro')
            if len(faces)==1: #Solo un rostro puede ser extraído a la vez
                #------------Extraer rostro
                rostro_out=ExtraerRostro(bbox,imgsml)
                #------------Agregar
                rostros.append(rostro_out)
                np.asarray(rostros)
                #TODO: Agregar más fotos de la misma persona
                plt.axis('off')
                plt.imshow(rostros[0])
                #-----------Vector
                for cara in rostros: 
                    res=SacarVector(model,cara)
                    modelos.append(res[0])
                print('Rostro guardado')             
            k=-1
            continue
        #Analizar rostro
        if len(modelos)>0:  
            if len(faces)==1: #Solo un rostro puede ser extraído a la vez
                #------------Extraer rostro
                rostro_out=ExtraerRostro(bbox,imgsml)
                #-----------Vector 
                res=SacarVector(model,rostro_out)
                for modelo in modelos:
                    dist = np.linalg.norm(res-modelo)
                    print(dist)
                    if dist<0.3:
                        print('Rostro conocido')
                
            k=-1
                    
          
    
    cv2.destroyAllWindows()
