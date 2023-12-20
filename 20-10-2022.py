import cv2
import numpy as np
import os
import xlwt
from xlwt import Workbook
import os
from os import listdir
i=0  
wb = Workbook()   
sheet1 = wb.add_sheet('Sheet1')                       #Finding the Orientation Of the Cotton Boll Detected
def Fun_orientation_finding(img):
    image= img.copy() 
    cv2.imshow('original image',image)
             # Copy of the Image
    hsv1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
    #cv2.imshow('image hsv1',hsv1)  # Converting into 2 copy of Hsv  Image  
    hsv2 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #cv2.imshow('image hsv2',hsv2)    # two separate images : one for carpel (Sepal) and another for cotton 
    # define range of brown color in HSV
    # change it according to your need !

    #https://github.com/hariangr/HsvRangeTool    HSV Range tool for getting range for required color 

    #Creating mask for sepal/carpel detection (brown / redish / blackish) 
    lower_sepal = np.array([0,21,59], dtype=np.uint8)       # Sepla Mask Range Sepal/carpel color mask range (brown color)  Minimum range 
    upper_sepal = np.array([76,162,255], dtype=np.uint8)    # color mask range (brown color)  maximum  range

    # define range of white color in HSV
    # change it according to your need !
    # Creating mask for cotton (white) 
    lower_white = np.array([0,0,158], dtype=np.uint8)      # Cotton Mask Range color mask range (white color)  Minimum range 
    upper_white = np.array([179,35,255], dtype=np.uint8)   # Cotton Mask Range color mask range (white color)  Maximum range  
    # Threshold the HSV image to get only brown colors
    mask1 = cv2.inRange(hsv1, lower_sepal, upper_sepal)  
    #cv2.imshow('image mask1',mask1) 
    # apply mask (thresholding) for sepal /carpel for first image 
    # Threshold the HSV image to get only white colors
    mask2 = cv2.inRange(hsv2, lower_white, upper_white)     # apply mask (thresholding) for white boll for second image 
    # Bitwise-AND mask and original image
    #cv2.imshow('image mask2',mask2) 
   
    kernel = np.ones((1,1),np.uint8)                                    #parameters setting for dilate function 
    img_dilation = cv2.dilate(mask1 , kernel, iterations=1)             #opencv Dialation of the Image , for dilating image using OPENCV function   TBD (Dilation) 
    notofcotton=cv2.bitwise_not(mask2)  
    #cv2.imshow('image not of cotton',notofcotton)                                # appling Not Function To the image  inverting white cotton mask image (black color) 
    closing = cv2.morphologyEx(img_dilation, cv2.MORPH_CLOSE, kernel)   #opencv Dialation of the Image morphology function of OPENCV 
    bitwiseAnd= cv2.bitwise_and(notofcotton,closing)  
    #cv2.imshow('imageofcotton',bitwiseAnd)                  #to get perfect carpel 
    height, width, _ = image.shape                                      # height and width of the image (bounding box image) 
    x1=width*1/2                                                        # get center point of bounding box 
    y1=height*1/2
    firstqudrant= bitwiseAnd[0:int(x1),0:int(y1)]         # Quadrant 1    -----|-----
    #cv2.imshow('firstqudrant',firstqudrant)
    secondqudrant= bitwiseAnd[0:int(y1),int(x1):width]
    #cv2.imshow('secondqudrant',secondqudrant)   #  Quadrant 2    - Q1 | Q2 -  
    thirdqudrant= bitwiseAnd[int(y1):height,0:int(x1)]    # Quadrant 3    -----------
    #cv2.imshow('thirdqudrant',thirdqudrant) 
    fourthqudrant= bitwiseAnd[int(y1):height,int(x1):width] #Quadrant 4   - Q3 | Q4 -
    #cv2.imshow('fourthqudrant',fourthqudrant) 
    #calculating white pixels in every quadrant 
    n1_white_pix = np.sum(firstqudrant == 0)           
    n1_dark_pix = np.sum(firstqudrant == 255)    #Calculating number of white pixels in Q1 
    #print('Number of white pixels inQ1:', n1_white_pix)
    #print()
    n2_white_pix = np.sum(secondqudrant == 0)
    n2_dark_pix = np.sum(secondqudrant== 255)               #Calculating number of white pixels in Q2 
    #print()
    n3_white_pix = np.sum(thirdqudrant == 0)
    n3_dark_pix = np.sum(thirdqudrant == 255)                #Calculating number of white pixels in Q3
    #print('Number of white pixels inQ3:', n3_white_pix)
    #print()
    n4_white_pix = np.sum(fourthqudrant == 0)  
    n4_dark_pix = np.sum(fourthqudrant == 255)           #Calculating number of white pixels in Q4  
    print('NoofWhiPixQuadrant: Q1: {} , Q2: {} , Q3: {} ,Q4:{}'.format( n1_white_pix,n2_white_pix,n3_white_pix,n4_white_pix))
    print('NoofDrkPixQuadrant: Q1: {} , Q2: {} , Q3: {} ,Q4:{}'.format( n1_dark_pix,n2_dark_pix,n3_dark_pix,n4_dark_pix))
    number_of_white_pix_carpal = np.sum(closing ==0 )     #number of white pixels in carpel mask image (equivalent to carpel pixels after masking)
    number_of_white_pix_cotton = np.sum(bitwiseAnd)       #number of white pixels in cotton boll mask image (equivalent to white cotton boll pixels after masking)
    #print()
    #print('Number of carpal pixels:', number_of_white_pix_carpal )
    #print('Number of cotton pixels:', number_of_white_pix_cotton)
    #print()
    
    quotient1 = number_of_white_pix_carpal / number_of_white_pix_cotton             #ratio of carpel pixels to white cotton boll pixels 
    percent =abs(quotient1 * 100)
    print(percent)
    
    ArrayQuadrantwhitePixel=[[n1_white_pix,1],[n2_white_pix,2],[n3_white_pix,3],[n4_white_pix,4]]     #number of white pixels stored in array for Q1,Q2,Q3,Q4
    ArrayQuadrantdarkPixel=[[n1_dark_pix,1],[n2_dark_pix,2],[n3_dark_pix,3],[n4_dark_pix,4]] 
    #sorting data in array for number of white pixels in each quadrant in desending order (highest firt) 
    ArrayQuadrantwhitePixel=sorted(ArrayQuadrantwhitePixel, key=lambda ArrayQuadrantPixel: ArrayQuadrantPixel[0] ,reverse=True)
    print()
    print(ArrayQuadrantwhitePixel)
    ArrayQuadrantdarkPixel=sorted(ArrayQuadrantdarkPixel, key=lambda ArrayQuadrantPixel: ArrayQuadrantPixel[0] ,reverse=True)
    print()
    print(ArrayQuadrantdarkPixel)
    #result=np.count_nonzero(ArrayQuadrantPixel)
    #print(result)
    #print(count)
    facing_ratio = ((ArrayQuadrantwhitePixel[0][1])+(ArrayQuadrantwhitePixel[1][1])/(ArrayQuadrantdarkPixel[0][1])+(ArrayQuadrantdarkPixel[1][1]))
    print(facing_ratio)
    sheet1.write(i, 7,percent) 
    sheet1.write(i, 8,facing_ratio ) 
    down_ratio = (n3_white_pix+n4_white_pix)/(n1_white_pix+n2_white_pix)
    print(down_ratio)
    sheet1.write(i, 9,down_ratio) 
    up_ratio = (n1_white_pix+n2_white_pix)/(n3_white_pix+n4_white_pix)
    print(up_ratio)
    sheet1.write(i, 10,up_ratio) 
    left_ratio = (n3_white_pix+n1_white_pix)/(n2_white_pix+n4_white_pix)
    print(left_ratio)
    sheet1.write(i, 11,left_ratio) 
    right_ratio = (n4_white_pix+n2_white_pix)/(n1_white_pix+n3_white_pix)
    print(right_ratio)
    sheet1.write(i, 12,right_ratio) 
#-------------------- end of function ----------------------------------
#if (number_of_white_pix_carpal<15000) :                  #for branch , NOT working 15000 is magic number TBD 
    ArrayOrientationResult =[ArrayQuadrantwhitePixel[0][1],ArrayQuadrantwhitePixel[1][1]]   #highest quadrant data received and printed 
    print()
    print(ArrayOrientationResult) 
    if (facing_ratio>4):
        if (((ArrayOrientationResult == [4,3])|(ArrayOrientationResult == [3,4])|(down_ratio>1))):
            print("Downfacing")  #debugging by printing percentage and carpel pixel 
            return ('D')
        elif ((ArrayOrientationResult == [3,1])|(ArrayOrientationResult == [1,3])|(left_ratio>1)):
            print("LeftFacing")   
            return ('L')        
        elif ((ArrayOrientationResult == [4,2])|(ArrayOrientationResult == [2,4])|(right_ratio>1)):
            print("Rightfacing")  
            return ('R')   
        elif ((ArrayOrientationResult == [2,1])|(ArrayOrientationResult == [1,2])|(up_ratio>1)):
            print("Upfacing")
            return ('U')
 
    else:
      print("Frontfacing")


for filename in os.listdir("C:/Users/JyoSH PC/Desktop/JYOSH_Orientation/modifiedorientation/frontfacing/"):
  
    filen=os.path.splitext(os.path.basename(filename))[0]
    img = cv2.imread("C:/Users/JyoSH PC/Desktop/JYOSH_Orientation/modifiedorientation/frontfacing/"+filen + ".png")
    height, width, _ = img.shape  
    Boll_orientation=Fun_orientation_finding(img)  
    #Cotton_boll_orientation = Fun_Cotton_Orientation(img)  
    sheet1.write(i, 2,filen) 
    sheet1.write(i, 3,Boll_orientation) 
    #sheet1.write(i,2,Cotton_boll_orientation)
    i +=1
    wb.save('xlwt_updatanewfront.xls')
cv2.waitKey(0)    

           
   

   