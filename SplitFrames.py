import cv2
import numpy as np
 
 
cap=cv2.VideoCapture('C:\\Users\\sayandip_sarkar\\Desktop\\pos_vid\\pos.mp4')
print(type(cap))
folder="C:\\Users\\sayandip_sarkar\\Desktop\\pos_vid\\vid"

index=0

while True:
        ok , img = cap.read()
        file_name=folder+str(index)+".jpg"    
        cv2.imwrite( file_name, img )
        index+=1
        print(index ,"done") 
# 
# Fix issue for breaking loop
     

cap.release()

