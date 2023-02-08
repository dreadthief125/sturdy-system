import cv2
import numpy as np
import tensorflow as tenflow
model= tf.keras.models.load_model('keras_model.h5')
video = cv2.VideoCapture(0)
  
while(True):
    ret, frame = vid.read()
    image=cv2.resize(frame,(224,224))

    rockpaperscizorsimg=np.array(image,dtype=np.float32)
    testrockpaperscizorsimg_image=np.expand_dims(rockpaperscizorsimg,axis=0)
    rockpaperscizorsimage=rockpaperscizorsimg/255
    predictMessage=model.predict(rockpaperscizorsimage)
    print(predictMessage)
    cv2.imshow('frame', frame)      
    # NOTE: press space to quit
    spaceKey = cv2.waitKey(1)
    
    if spaceKey == 32:
        break
  
vid.release()
cv2.destroyAllWindows()