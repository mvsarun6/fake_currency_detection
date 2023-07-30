
#Author : Sarun Kumar

import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
import math
from pytesseract import pytesseract #https://stackoverflow.com/questions/55582511/receiving-pytesseract-not-in-your-path-error-on-the-exact-same-code-that-used
import time

filename = "FakeNoteInput_Success.jpg"
# filename = "NotFakeNote_InputNotSuccess.jpeg"
# filename = "NotFakeNote_InputNotSuccess_2.jpeg"


# pre-defined areas of interest where visually inspectable features present on a 1090x500 20 Dollar bill :
# serial_number, poppies, film_production, transparent
film_production = {} 
film_production['filename'] = "nil"
film_production['x'] = 530
film_production['y'] = 465
film_production['w'] = 220
film_production['h'] = 30
film_production['type'] = "no black pixel"
film_production['matches'] = 0

transparent = {} 
transparent['filename'] = "nil"
transparent['x'] = 290
transparent['y'] = 290
transparent['w'] = 70
transparent['h'] = 50
transparent['type'] = "no white pixels"
transparent['matches'] = 0

serial_number = {} 
serial_number['filename'] = "nil"
serial_number['x'] = 740
serial_number['y'] = 425
serial_number['w'] = 300
serial_number['h'] = 50
serial_number['type'] = "ocr"
serial_number['matches'] = "DB66688803"

poppies = {} 
poppies['filename'] = "poppies_870-146_220x177.png"
poppies['x'] = 870
poppies['y'] = 146
poppies['w'] = 220
poppies['h'] = 177
poppies['type'] = "hu"
poppies['tolerance'] = 10


def find_top_left(img, intensity_cut):
   height = img.shape[0]
   width = img.shape[1]
   top_left={}
   for now in range(0,height,1):
      for k in range(0,now):
         if(img[k][abs(now-k)]>intensity_cut):
            top_left['x'] = abs(now-k)
            top_left['y'] = k
            return top_left
def find_bottom_left(img, intensity_cut):
   height = img.shape[0]
   width = img.shape[1]
   bottom_left={}
   for now in range(0,height,1):
      for k in range(0,now):
         if(img[height-1-k][abs(k-now)]>intensity_cut):
            bottom_left['x'] = abs(k-now)
            bottom_left['y'] = height-1-k
            return bottom_left
def find_top_right(img, intensity_cut):
   height = img.shape[0]
   width = img.shape[1]
   top_right={}
   for now in range(0,height,1):
      for k in range(0,now):
         # if(now < 5):
            # print("TOP RIGHT=",k,",",width-now+k)
         if(img[k][width-now+k]>intensity_cut):
            top_right['x'] = width-now+k
            top_right['y'] = k
            return top_right

def find_bottom_right(img, intensity_cut):
   height = img.shape[0]
   width = img.shape[1]
   bottom_right={}
   for now in range(0,height,1):
      for k in range(0,now):
         if(img[height-1-now][abs(width-1-now-k)]>intensity_cut):
            bottom_right['x'] = abs(width-1-now-k)
            bottom_right['y'] = height-1-now
            return bottom_right

def GetLineAlignAngle(a, b):
   if(a['y']==b['y']):
      return 0.0
   elif (a['y']>b['y']):
      c ={}
      c['x']=b['x']
      c['y']=a['y']
      ang = math.degrees(math.atan2(c['y']-a['y'], c['x']-a['x']) - math.atan2(a['y']-b['y'], a['x']-b['x']))
      ang = 180+ang
      return ang
   elif (a['y']<b['y']):
      # print("THIS PART NOT TESTED IN GetLineAlignAngle")
      c ={}
      c['x']=b['x']
      c['y']=a['y']
      ang = math.degrees(math.atan2(c['y']-a['y'], c['x']-a['x']) - math.atan2(a['y']-b['y'], a['x']-b['x']))
      return ang


def GetTotCoordinate(point, angle):
   new_point = {}
   RotMatr = cv2.getRotationMatrix2D((0, 0), -angle, 1.0)
   PointMatr = np.array([point['x'], point['y'], 1])
   new_point = np.matmul(RotMatr,PointMatr)
   return new_point

is_it_fake = False
print("\nInput image  = ",filename)
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

height = img.shape[0]
width = img.shape[1]
# cv2.imshow('Original gray',img)
cv2.imwrite("output/Original"+"_"+filename, img)

img = cv2.GaussianBlur(img,(5,5),2.1,2.1,cv2.BORDER_REPLICATE)	
# cv2.imshow('blur Image',img)
cv2.imwrite("output/"+"blurred"+"_"+filename, img)

ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

denoise_image = img

denoise_image = np.float32(denoise_image)

start_time = time.time()
top_left= find_top_left(img,128)
bottom_left= find_bottom_left(img,128)
top_right= find_top_right(img,128)
# print("top left =",top_left['x'], " and ", top_left['y'])
# print("bottom left =",bottom_left['x'], " and ", bottom_left['y'])
# print("top right =",top_right['x'], " and ", top_right['y'])
specimen_width = round(math.sqrt((top_right['x']-top_left['x'])**2 + (top_right['y']-top_left['y'])**2))
specimen_height = round(math.sqrt((bottom_left['x']-top_left['x'])**2 + (bottom_left['y']-top_left['y'])**2))

angle = GetLineAlignAngle(top_left,top_right)

if(angle>90):
   angle = (180-angle)
end_time = time.time()
# print("EXECUTION TIME=",end_time-start_time )
# print("angle = ",angle)

if (top_left['y']>top_right['y']):
   angle = -angle
# else:
   # print("**THIS PART NOT TESTED")

M = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
# def warpAffine(src: Mat, M, dsize: typing.Tuple[int, int], dst: Mat = ..., flags: int = ..., borderMode=..., borderValue=...) -> typing.Any:
rotated = cv2.warpAffine(img, M, (width, height), cv2.INTER_NEAREST, cv2. BORDER_CONSTANT,0)
top_left=find_top_left(rotated,128)

cv2.imwrite("output/Aligned_Rotated"+"_"+filename, rotated)
# cv2.imshow('Aligned',rotated)

new_top_left = GetTotCoordinate(top_left,-angle)
# print("new top left",new_top_left[0])

# print("new y pos=",top_left['y'])
crop_img = rotated[top_left['y']:top_left['y']+specimen_height, top_left['x']:top_left['x']+specimen_width]

# cv2.imshow('Final',crop_img)
cv2.imwrite("output/cropped image"+"_"+filename, crop_img)
# #cv2.imshow('Rotated',rotated)

#get crop co-ordinates
# print("Height = ",crop_img.shape[0])
cy = 500/crop_img.shape[0];
cx = 1090/crop_img.shape[1];

dim =(1090,500)
resize_img_500_1090 = cv2.resize(crop_img, dim,0.0,0.0, interpolation = cv2.INTER_AREA)
cv2.imshow('resize_img_500_1090',resize_img_500_1090)

# resize_img_500_1090 = cv2.blur(resize_img_500_1090,(2,2))
# cv2.imshow('blur resize_img_500_1090',resize_img_500_1090)

#---------------------------------------------
#Check:1----------------poppies---------------
#---------------------------------------------
reference_img = cv2.imread(poppies['filename'])
reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)

img_segment = resize_img_500_1090[poppies['y']:poppies['y']+poppies['h'], poppies['x']:poppies['x']+poppies['w']]
cv2.imshow('poppies',img_segment)

#operation on reference image
reference_img = cv2.imread(poppies['filename'])
reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)

# Threshold image 
_,im = cv2.threshold(reference_img, 128, 255, cv2.THRESH_BINARY)
# Calculate Moments 
ref_moments = cv2.moments(im, True) 
# Calculate Hu Moments 
refhuMoments = cv2.HuMoments(ref_moments)
# print("Reference poppies Image, Hu's moments : ",refhuMoments)

#operation on segmented image
# Threshold image 
_,im = cv2.threshold(img_segment, 128, 255, cv2.THRESH_BINARY)
cv2.imshow('poppies_segment_thresh',im)
# Calculate Moments 
moments = cv2.moments(im,True) 
# Calculate Hu Moments 
huMoments = cv2.HuMoments(moments)
# print("Segmented poppies Image, Hu's moments : ",huMoments)

def is_within_threshold(parm1, param2):
   limH = parm1 + ((parm1*poppies["tolerance"])/100)
   limL = parm1 - ((parm1*poppies["tolerance"])/100)
   if(limH<param2 and limL>param2):
      return True
   else:
      return False
   
if (is_within_threshold(refhuMoments[0],huMoments[0])):
   print("check poppies TEST_FAIL")
   is_it_fake = True
else:
   print("check poppies TEST_PASS")

#-------------------------------------------------
#Check:2----------------serial_number-------------
#-------------------------------------------------
img_segment = resize_img_500_1090[serial_number['y']:serial_number['y']+serial_number['h']-1, serial_number['x']:serial_number['x']+serial_number['w']-1]
cv2.imshow('img_segment',img_segment)
ret, serial_number_segment_thresh = cv2.threshold(img_segment, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('serial_number_segment_thresh',serial_number_segment_thresh)
serial_number_segment_thresh_neg = abs(255-serial_number_segment_thresh)
cv2.imshow('serial_number thresh negative',serial_number_segment_thresh_neg)
image_to_text = pytesseract.image_to_string(serial_number_segment_thresh_neg)
# print(image_to_text)
if (image_to_text==serial_number['matches']):
   print("serial_number TEST_FAIL = ", image_to_text)
   is_it_fake = True
else:
   print("check serial_number TEST_PASS")

#-------------------------------------------------
#Check:3----------------film_production-----------
#-------------------------------------------------
img_segment = resize_img_500_1090[film_production['y']:film_production['y']+film_production['h']-1, film_production['x']:film_production['x']+film_production['w']-1]
img_segment = cv2.blur(img_segment,(2,2))
ret,film_production_thresh = cv2.threshold(img_segment,90,255,cv2.THRESH_BINARY)
cv2.imshow('film_production_thresh',film_production_thresh)
black_pixel_count = 0
for y in range(0,film_production_thresh.shape[0]):
   for x in range(0,film_production_thresh.shape[1]):
      if(film_production_thresh[y][x]<10):
         black_pixel_count = black_pixel_count+1
if(black_pixel_count>100):
   print("check film_production TEST_FAIL, black pixel count= ",black_pixel_count)
   is_it_fake = True
else:
   print("check film_production TEST_PASS")
#-------------------------------------------------
#Check:4----------------transparent---------------
#-------------------------------------------------
img_segment = resize_img_500_1090[transparent['y']:transparent['y']+transparent['h'], transparent['x']:transparent['x']+transparent['w']]
img_segment = cv2.blur(img_segment,(2,2))
ret,transparent_thresh = cv2.threshold(img_segment,127,255,cv2.THRESH_BINARY)
cv2.imshow('transparent_thresh',transparent_thresh)
black_pixel_count = 0
white_pixel_count =0
# print("Trans width and hight",transparent_thresh.shape[0],transparent_thresh.shape[1])
for y in range(0,transparent_thresh.shape[0]):
   for x in range(0,transparent_thresh.shape[1]):
      if(transparent_thresh[y][x]<10):
         black_pixel_count = black_pixel_count+1
      else:
         white_pixel_count = white_pixel_count+1
# print("transparent_thresh black_pixel_count = ",black_pixel_count)
# print("transparent_thresh white_pixel_count = ",white_pixel_count)
if(white_pixel_count>100):
   print("check transparent area TEST_FAIL, white_pixel_count = ",white_pixel_count)
   is_it_fake = True
else:
   print("check transparent area TEST_PASS")
#-------------------------------------------------
#-------------------------------------------------
if (is_it_fake==True):
   print("ITS A FAKE NOTE")
else:
   print("ITS NOT A FAKE NOTE")

cv2.waitKey(0)
cv2.destroyAllWindows()
