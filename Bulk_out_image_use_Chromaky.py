import cv2
import numpy as np
import os
import glob

def fileRead(file_dir):
    data = []
    img_names = []
    for file in glob.glob(file_dir):
        data.append( cv2.imread(file, 1) )
        img_names.append(file)
    return data,img_names

def create_dataset(img, back_imgs,file_name, param,num):
    names =["_A","_B","_C","_D","_E"]
    for i in range(num):
        with open("labels/" + file_name + names[i] +".txt","w")as f:
            f.write(param)
        result_img = chromaky(img,back_imgs[i])
        cv2.imwrite("labels/"+file_name + names[i] + ".png",result_img)

def chromaky(img, back):
    kernel = np.ones((10,10),np.uint8)
    print("Generate image")
    h,w,_ = img.shape[:3]
    cpimg = img.copy()
    f_img = cv2.GaussianBlur(cpimg,(5,5),1)
    rimg = cv2.resize(back,(w,h))
    mask = np.zeros((h,w))
    # Convert BGR to HSV
    hsv = cv2.cvtColor(f_img, cv2.COLOR_BGR2HSV)

    # define range of green color in HSV
    lower_green = np.array([70,50,50])
    upper_green = np.array([85,255,255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(cpimg,cpimg, mask= mask)
    bc = cv2.bitwise_and(rimg,rimg, mask = mask)
    cpimg -=res
    cpimg +=bc

    return cpimg


if __name__ == '__main__':
    int number = 2
    
    img_dir = 'images/*.png'   #traing_image dir
    back_img_dir ='back_img/*' #back_ground_image_dir
    data,img_names = fileRead(img_dir)
    back_imgs,_ =fileRead(back_img_dir)
    for img,name in zip(data,img_names):
        with open(name[:-3]+"txt","r")as f:
            param = f.read()       
        create_dataset(img, back_imgs,name.replace("images/","")[:-4], param, number)
