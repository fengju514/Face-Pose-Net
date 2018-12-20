import sys
import os
import numpy as np
import cv2
import math
from math import cos, sin, atan2, asin
import fileinput

## Index to remap landmarks in case we flip an image                                                                                                                                                     
repLand = [ 17,16,15,14,13,12,11,10, 9,8,7,6,5,4,3,2,1,27,26,25, \
            24,23,22,21,20,19,18,28,29,30,31,36,35,34,33,32,46,45,44,43, \
            48,47,40,39,38,37,42,41,55,54,53,52,51,50,49,60,59,58,57,56, \
            65,64,63,62,61,68,67,66 ]


def increaseBbox(bbox, factor):
    tlx = bbox[0] 
    tly = bbox[1] 
    brx = bbox[2] 
    bry = bbox[3] 
    dx = factor
    dy = factor
    dw = 1 + factor
    dh = 1 + factor
    #Getting bbox height and width
    w = brx-tlx;
    h = bry-tly;
    tlx2 = tlx - w * dx
    tly2 = tly - h * dy
    brx2 = tlx + w * dw
    bry2 = tly + h * dh
    nbbox = np.zeros( (4,1), dtype=np.float32 )
    nbbox[0] = tlx2
    nbbox[1] = tly2
    nbbox[2] = brx2
    nbbox[3] = bry2 
    return nbbox


def increaseBbox_rescaleCASIA(bbox, factor):
        tlx = bbox[0] 
        tly = bbox[1] 
        brx = bbox[2] 
        bry = bbox[3] 
    
        ww = brx - tlx; 
        hh = bry - tly; 
        cx = tlx + ww/2;
        cy = tly + hh/2;
        tsize = max(ww,hh)/2;
    
        bl = cx - factor[0]*tsize;
        bt = cy - factor[1]*tsize;
        br = cx + factor[2]*tsize;
        bb = cy + factor[3]*tsize;
    
        nbbox = np.zeros( (4,1), dtype=np.float32 )
        nbbox[0] = bl;
        nbbox[1] = bt;
        nbbox[2] = br;
        nbbox[3] = bb;

    
        return nbbox


def increaseBbox_rescaleYOLO(bbox, im):

    rescaleFrontal = [1.4421, 2.2853, 1.4421, 1.4286];
    rescaleCS2 = [0.9775, 1.5074, 0.9563, 0.9436];


    l = bbox[0]
    t = bbox[1]
    ww = bbox[2]
    hh = bbox[3]

    # Approximate LM tight BB
    h = im.shape[0];
    w = im.shape[1];
    
    cx = l + ww/2;
    cy = t + hh/2;
    tsize = max(ww,hh)/2;
    l = cx - tsize;
    t = cy - tsize;
    cx = l + (2*tsize)/(rescaleCS2[0]+rescaleCS2[2]) * rescaleCS2[0];
    cy = t + (2*tsize)/(rescaleCS2[1]+rescaleCS2[3]) * rescaleCS2[1];
    tsize = 2*tsize/(rescaleCS2[0]+rescaleCS2[2]);

    
    """
    # Approximate inplane align (frontal)
    nbbox = np.zeros( (4,1), dtype=np.float32 )
    nbbox[0] = cx - rescaleFrontal[0]*tsize;
    nbbox[1] = cy - rescaleFrontal[1]*tsize;
    nbbox[2] = cx + rescaleFrontal[2]*tsize;
    nbbox[3] = cy + rescaleFrontal[3]*tsize;
    """
    
    nbbox = np.zeros( (4,1), dtype=np.float32 )
    nbbox[0] = cx - tsize;
    nbbox[1] = cy - tsize;
    nbbox[2] = cx + tsize;
    nbbox[3] = cy + tsize;
    
    return nbbox




def image_bbox_processing_v2(img, bbox, landmarks=None):
    img_h, img_w, img_c = img.shape
    lt_x = bbox[0]
    lt_y = bbox[1]
    rb_x = bbox[2]
    rb_y = bbox[3]

    fillings = np.zeros( (4,1), dtype=np.int32)
    if lt_x < 0: ## 0 for python
        fillings[0] = math.ceil(-lt_x)
    if lt_y < 0:
        fillings[1] = math.ceil(-lt_y)
    if rb_x > img_w-1:
        fillings[2] = math.ceil(rb_x - img_w + 1)
    if rb_y > img_h-1:
        fillings[3] = math.ceil(rb_y - img_h + 1)
    new_bbox = np.zeros( (4,1), dtype=np.float32 )
    # img = [zeros(size(img,1),fillings(1),img_c), img]
    # img = [zeros(fillings(2), size(img,2),img_c); img]
    # img = [img, zeros(size(img,1), fillings(3),img_c)]

    # new_img = [img; zeros(fillings(4), size(img,2),img_c)]
    imgc = img.copy()
    if fillings[0] > 0:
        img_h, img_w, img_c = imgc.shape
        imgc = np.hstack( [np.zeros( (img_h, fillings[0][0], img_c), dtype=np.uint8 ), imgc] )    
    if fillings[1] > 0:

        img_h, img_w, img_c = imgc.shape
        imgc = np.vstack( [np.zeros( (fillings[1][0], img_w, img_c), dtype=np.uint8 ), imgc] )
    if fillings[2] > 0:


        img_h, img_w, img_c = imgc.shape
        imgc = np.hstack( [ imgc, np.zeros( (img_h, fillings[2][0], img_c), dtype=np.uint8 ) ] )    
    if fillings[3] > 0:
        img_h, img_w, img_c = imgc.shape
        imgc = np.vstack( [ imgc, np.zeros( (fillings[3][0], img_w, img_c), dtype=np.uint8) ] )


    new_bbox[0] = lt_x + fillings[0]
    new_bbox[1] = lt_y + fillings[1]
    new_bbox[2] = rb_x + fillings[0]
    new_bbox[3] = rb_y + fillings[1]


    if len(landmarks) == 0: #len(landmarks) == 0: #landmarks == None:
        return imgc, new_bbox
    else:
        landmarks_new = np.zeros([landmarks.shape[0], landmarks.shape[1]])
        #print "landmarks_new's shape: \n"                                                                                                                                                               
        #print landmarks_new.shape                                                                                                                                                                       
        landmarks_new[:,0] = landmarks[:,0] + fillings[0]
        landmarks_new[:,1] = landmarks[:,1] + fillings[1]
        return imgc, new_bbox, landmarks_new

    #return imgc, new_bbox


def image_bbox_processing_v3(img, bbox):
    img_h, img_w, img_c = img.shape
    lt_x = bbox[0]
    lt_y = bbox[1]
    rb_x = bbox[2]
    rb_y = bbox[3]

    fillings = np.zeros( (4,1), dtype=np.int32)
    if lt_x < 0: ## 0 for python
        fillings[0] = math.ceil(-lt_x)
    if lt_y < 0:
        fillings[1] = math.ceil(-lt_y)
    if rb_x > img_w-1:
        fillings[2] = math.ceil(rb_x - img_w + 1)
    if rb_y > img_h-1:
        fillings[3] = math.ceil(rb_y - img_h + 1)
    new_bbox = np.zeros( (4,1), dtype=np.float32 )
    # img = [zeros(size(img,1),fillings(1),img_c), img]
    # img = [zeros(fillings(2), size(img,2),img_c); img]
    # img = [img, zeros(size(img,1), fillings(3),img_c)]

    # new_img = [img; zeros(fillings(4), size(img,2),img_c)]
    imgc = img.copy()
    if fillings[0] > 0:
        img_h, img_w, img_c = imgc.shape
        imgc = np.hstack( [np.zeros( (img_h, fillings[0][0], img_c), dtype=np.uint8 ), imgc] )    
    if fillings[1] > 0:

        img_h, img_w, img_c = imgc.shape
        imgc = np.vstack( [np.zeros( (fillings[1][0], img_w, img_c), dtype=np.uint8 ), imgc] )
    if fillings[2] > 0:


        img_h, img_w, img_c = imgc.shape
        imgc = np.hstack( [ imgc, np.zeros( (img_h, fillings[2][0], img_c), dtype=np.uint8 ) ] )    
    if fillings[3] > 0:
        img_h, img_w, img_c = imgc.shape
        imgc = np.vstack( [ imgc, np.zeros( (fillings[3][0], img_w, img_c), dtype=np.uint8) ] )


    new_bbox[0] = lt_x + fillings[0]
    new_bbox[1] = lt_y + fillings[1]
    new_bbox[2] = rb_x + fillings[0]
    new_bbox[3] = rb_y + fillings[1]


    
    return imgc, new_bbox
    

def preProcessImage(im, lmks, bbox, factor, _alexNetSize, flipped):

        sys.stdout.flush()
        

        if flipped == 1: # flip landmarks and indices if it's flipped imag
            lmks = flip_lmk_idx(im, lmks)
            
        lmks_flip = lmks


        lt_x = bbox[0]
        lt_y = bbox[1]
        rb_x = lt_x + bbox[2]
        rb_y = lt_y + bbox[3]
        w = bbox[2]
        h = bbox[3]
        center = ( (lt_x+rb_x)/2, (lt_y+rb_y)/2 )
        side_length = max(w,h);
        
        # make the bbox be square
        bbox = np.zeros( (4,1), dtype=np.float32 )
        bbox[0] = center[0] - side_length/2
        bbox[1] = center[1] - side_length/2
        bbox[2] = center[0] + side_length/2
        bbox[3] = center[1] + side_length/2
        img_2, bbox_green = image_bbox_processing_v2(im, bbox)
        
        #%% Get the expanded square bbox
        bbox_red = increaseBbox(bbox_green, factor)
        bbox_red2 = increaseBbox(bbox, factor)
        bbox_red2[2] = bbox_red2[2] - bbox_red2[0]
        bbox_red2[3] = bbox_red2[3] - bbox_red2[1]
        bbox_red2 = np.reshape(bbox_red2, [4])

        img_3, bbox_new, lmks = image_bbox_processing_v2(img_2, bbox_red, lmks)
    
        #%% Crop and resized
        bbox_new =  np.ceil( bbox_new )
        side_length = max( bbox_new[2] - bbox_new[0], bbox_new[3] - bbox_new[1] )
        bbox_new[2:4] = bbox_new[0:2] + side_length
        bbox_new = bbox_new.astype(int)

        crop_img = img_3[bbox_new[1][0]:bbox_new[3][0], bbox_new[0][0]:bbox_new[2][0], :];
        lmks_new = np.zeros([lmks.shape[0],2])
        lmks_new[:,0] = lmks[:,0] - bbox_new[0][0]
        lmks_new[:,1] = lmks[:,1] - bbox_new[1][0]

        resized_crop_img = cv2.resize(crop_img, ( _alexNetSize, _alexNetSize ), interpolation = cv2.INTER_CUBIC)
        old_h, old_w, channels = crop_img.shape
        lmks_new2 = np.zeros([lmks.shape[0],2])
        lmks_new2[:,0] = lmks_new[:,0] * _alexNetSize / old_w
        lmks_new2[:,1] = lmks_new[:,1] * _alexNetSize / old_h
        #print _alexNetSize, old_w, old_h
       

        return  resized_crop_img, lmks_new2, bbox_red2, lmks_flip, side_length, center


def resize_crop_rescaleCASIA(im, bbox, lmks, factor):

    lt_x = bbox[0]
    lt_y = bbox[1]
    rb_x = lt_x + bbox[2]
    rb_y = lt_y + bbox[3]
    bbox = np.reshape([lt_x, lt_y, rb_x, rb_y], [-1])

    # Get the expanded square bbox
    bbox_red = increaseBbox_rescaleCASIA(bbox, factor)


    img_3, bbox_new, lmks = image_bbox_processing_v2(im, bbox_red, lmks);
    lmks_filling = lmks.copy()


    #%% Crop and resized
    bbox_new =  np.ceil( bbox_new )
    side_length = max( bbox_new[2] - bbox_new[0], bbox_new[3] - bbox_new[1] )
    bbox_new[2:4] = bbox_new[0:2] + side_length

    #bbox_new[0] = max(0, bbox_new[0]) 
    #bbox_new[1] = max(0, bbox_new[1])
    #bbox_new[2] = min(img_3.shape[1]-1, bbox_new[2])
    #bbox_new[3] = min(img_3.shape[0]-1, bbox_new[3])
    bbox_new = bbox_new.astype(int)


    crop_img = img_3[bbox_new[1][0]:bbox_new[3][0], bbox_new[0][0]:bbox_new[2][0], :];
    lmks_new = np.zeros([lmks.shape[0],2])
    lmks_new[:,0] = lmks[:,0] - bbox_new[0][0]
    lmks_new[:,1] = lmks[:,1] - bbox_new[1][0]
    old_h, old_w, channels = crop_img.shape


    resized_crop_img = cv2.resize(crop_img, ( 224, 224 ), interpolation = cv2.INTER_CUBIC)
    lmks_new2 = np.zeros([lmks.shape[0],2])
    lmks_new2[:,0] = lmks_new[:,0] * 224 / old_w
    lmks_new2[:,1] = lmks_new[:,1] * 224 / old_h



    return resized_crop_img, bbox_new, lmks_new2, lmks_filling, old_h, old_w, img_3



def resize_crop_rescaleCASIA_v2(im, bbox, lmks, factor, bbox_type):

    

    # Get the expanded square bbox
    if bbox_type == "casia":

        lt_x = bbox[0]
        lt_y = bbox[1]
        rb_x = lt_x + bbox[2]
        rb_y = lt_y + bbox[3]
        bbox = np.reshape([lt_x, lt_y, rb_x, rb_y], [-1])

        bbox_red = increaseBbox_rescaleCASIA(bbox, factor)

    elif bbox_type == "yolo":

        lt_x = bbox[0]
        lt_y = bbox[1]
        rb_x = lt_x + bbox[2]
        rb_y = lt_y + bbox[3]
        w = bbox[2]
        h = bbox[3]
        center = ( (lt_x+rb_x)/2, (lt_y+rb_y)/2 )
        side_length = max(w,h);
        
        # make the bbox be square
        bbox = np.zeros( (4,1), dtype=np.float32 )
        bbox[0] = center[0] - side_length/2
        bbox[1] = center[1] - side_length/2
        bbox[2] = center[0] + side_length/2
        bbox[3] = center[1] + side_length/2
        img_2, bbox_green = image_bbox_processing_v3(im, bbox)
        
        #%% Get the expanded square bbox
        bbox_red = increaseBbox(bbox_green, factor)
                
        



    img_3, bbox_new, lmks = image_bbox_processing_v2(im, bbox_red, lmks);
    lmks_filling = lmks.copy()


    #%% Crop and resized
    bbox_new =  np.ceil( bbox_new )
    side_length = max( bbox_new[2] - bbox_new[0], bbox_new[3] - bbox_new[1] )
    bbox_new[2:4] = bbox_new[0:2] + side_length

    #bbox_new[0] = max(0, bbox_new[0]) 
    #bbox_new[1] = max(0, bbox_new[1])
    #bbox_new[2] = min(img_3.shape[1]-1, bbox_new[2])
    #bbox_new[3] = min(img_3.shape[0]-1, bbox_new[3])
    bbox_new = bbox_new.astype(int)


    crop_img = img_3[bbox_new[1][0]:bbox_new[3][0], bbox_new[0][0]:bbox_new[2][0], :];
    lmks_new = np.zeros([lmks.shape[0],2])
    lmks_new[:,0] = lmks[:,0] - bbox_new[0][0]
    lmks_new[:,1] = lmks[:,1] - bbox_new[1][0]
    old_h, old_w, channels = crop_img.shape


    resized_crop_img = cv2.resize(crop_img, ( 224, 224 ), interpolation = cv2.INTER_CUBIC)
    lmks_new2 = np.zeros([lmks.shape[0],2])
    lmks_new2[:,0] = lmks_new[:,0] * 224 / old_w
    lmks_new2[:,1] = lmks_new[:,1] * 224 / old_h



    return resized_crop_img, bbox_new, lmks_new2, lmks_filling, old_h, old_w, img_3


def resize_crop_AFLW(im, bbox, lmks):

    lt_x = bbox[0]
    lt_y = bbox[1]
    rb_x = lt_x + bbox[2]
    rb_y = lt_y + bbox[3]
    bbox = np.reshape([lt_x, lt_y, rb_x, rb_y], [-1])


    crop_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :];
    lmks_new = np.zeros([lmks.shape[0],2])
    lmks_new[:,0] = lmks[:,0] - bbox[0]
    lmks_new[:,1] = lmks[:,1] - bbox[1]
    old_h, old_w, channels = crop_img.shape


    resized_crop_img = cv2.resize(crop_img, ( 224, 224 ), interpolation = cv2.INTER_CUBIC)
    lmks_new2 = np.zeros([lmks.shape[0],2])
    lmks_new2[:,0] = lmks_new[:,0] * 224 / old_w
    lmks_new2[:,1] = lmks_new[:,1] * 224 / old_h


    bbox_new = np.zeros([4])
    bbox_new[0] = bbox[0] * 224 / old_w
    bbox_new[1] = bbox[1] * 224 / old_h
    bbox_new[2] = bbox[2] * 224 / old_w
    bbox_new[3] = bbox[3] * 224 / old_h

    bbox_new[2] = bbox_new[2] - bbox_new[0] # box width
    bbox_new[3] = bbox_new[3] - bbox_new[1] # box height


    return resized_crop_img, bbox_new, lmks_new2





def preProcessImage_v2(im, bbox, factor, _resNetSize, if_cropbyLmks_rescaleCASIA):

        sys.stdout.flush()
        

        if if_cropbyLmks_rescaleCASIA == 0:
                lt_x = bbox[0]
                lt_y = bbox[1]
                rb_x = lt_x + bbox[2]
                rb_y = lt_y + bbox[3]
                w = bbox[2]
                h = bbox[3]
                center = ( (lt_x+rb_x)/2, (lt_y+rb_y)/2 )
                side_length = max(w,h);
                
                # make the bbox be square
                bbox = np.zeros( (4,1), dtype=np.float32 )
                bbox[0] = center[0] - side_length/2
                bbox[1] = center[1] - side_length/2
                bbox[2] = center[0] + side_length/2
                bbox[3] = center[1] + side_length/2
                img_2, bbox_green = image_bbox_processing_v2(im, bbox)
                
                #%% Get the expanded square bbox
                bbox_red = increaseBbox(bbox_green, factor)
                img_3, bbox_new = image_bbox_processing_v2(img_2, bbox_red)

        elif if_cropbyLmks_rescaleCASIA == 1:
               
                bbox[2] = bbox[0] + bbox[2]
                bbox[3] = bbox[1] + bbox[3]

                bbox_red = increaseBbox_rescaleCASIA(bbox, factor)
                #print bbox_red
                img_3, bbox_new = image_bbox_processing_v3(im, bbox_red)

        else:
            
                bbox2 = increaseBbox_rescaleYOLO(bbox, im)
                bbox_red = increaseBbox_rescaleCASIA(bbox2, factor)

                img_3, bbox_new = image_bbox_processing_v2(im, bbox_red)



        #bbox_red2 = increaseBbox(bbox, factor)
        #bbox_red2[2] = bbox_red2[2] - bbox_red2[0]
        #bbox_red2[3] = bbox_red2[3] - bbox_red2[1]
        #bbox_red2 = np.reshape(bbox_red2, [4])
    
        #%% Crop and resized
        bbox_new =  np.ceil( bbox_new )
        side_length = max( bbox_new[2] - bbox_new[0], bbox_new[3] - bbox_new[1] )
        bbox_new[2:4] = bbox_new[0:2] + side_length
        bbox_new = bbox_new.astype(int)

        crop_img = img_3[bbox_new[1][0]:bbox_new[3][0], bbox_new[0][0]:bbox_new[2][0], :];
        #print crop_img.shape

        resized_crop_img = cv2.resize(crop_img, ( _resNetSize, _resNetSize ), interpolation = cv2.INTER_CUBIC)
        
       
        return  resized_crop_img



def preProcessImage_useGTBBox(im, lmks, bbox, factor, _alexNetSize, flipped, to_train_scale, yolo_bbox):

        sys.stdout.flush()
        #print bbox, yolo_bbox, to_train_scale

        if flipped == 1: # flip landmarks and indices if it's flipped imag
            lmks = flip_lmk_idx(im, lmks)
            
        lmks_flip = lmks


        lt_x = bbox[0]
        lt_y = bbox[1]
        rb_x = lt_x + bbox[2]
        rb_y = lt_y + bbox[3]
        w = bbox[2]
        h = bbox[3]
        center = ( (lt_x+rb_x)/2, (lt_y+rb_y)/2 )
        side_length = max(w,h);
        
        # make the bbox be square
        bbox = np.zeros( (4,1), dtype=np.float32 )
        #print bbox
        bbox_red = np.zeros( (4,1), dtype=np.float32 )

        if to_train_scale == 1:
                _, _, _, _, side_length2, center2 = preProcessImage(im, lmks, yolo_bbox, factor, _alexNetSize, flipped)
            
                center3 = ( (center[0]+center2[0])/2, (center[1]+center2[1])/2 )
                bbox[0] = center3[0] - side_length2/2
                bbox[1] = center3[1] - side_length2/2
                bbox[2] = center3[0] + side_length2/2
                bbox[3] = center3[1] + side_length2/2

                bbox_red[0] = center3[0] - side_length2/2
                bbox_red[1] = center3[1] - side_length2/2
                bbox_red[2] = side_length2
                bbox_red[3] = side_length2

        else:

                bbox[0] = center[0] - side_length/2
                bbox[1] = center[1] - side_length/2
                bbox[2] = center[0] + side_length/2
                bbox[3] = center[1] + side_length/2
                #print center, side_length, bbox[0], bbox[1], bbox[2], bbox[3]

                
                bbox_red[0] = center[0] - side_length/2
                bbox_red[1] = center[1] - side_length/2
                bbox_red[2] = side_length
                bbox_red[3] = side_length

        bbox_red = np.reshape(bbox_red, [4])

        #print bbox, bbox_red

        img_2, bbox_green = image_bbox_processing_v2(im, bbox) 
        #print img_2.shape, bbox_green

        #%% Crop and resized
        bbox_new =  np.ceil( bbox_green )
        side_length = max( bbox_new[2] - bbox_new[0], bbox_new[3] - bbox_new[1] )
        bbox_new[2:4] = bbox_new[0:2] + side_length
        bbox_new = bbox_new.astype(int)

        #print bbox_new
        crop_img = img_2[bbox_new[1][0]:bbox_new[3][0], bbox_new[0][0]:bbox_new[2][0], :];
        lmks_new = np.zeros([68,2])
        lmks_new[:,0] = lmks[:,0] - bbox_new[0][0]
        lmks_new[:,1] = lmks[:,1] - bbox_new[1][0]

        #print crop_img.shape

        resized_crop_img = cv2.resize(crop_img, ( _alexNetSize, _alexNetSize ), interpolation = cv2.INTER_CUBIC)
        old_h, old_w, channels = crop_img.shape
        lmks_new2 = np.zeros([68,2])
        lmks_new2[:,0] = lmks_new[:,0] * _alexNetSize / old_w
        lmks_new2[:,1] = lmks_new[:,1] * _alexNetSize / old_h
        #print _alexNetSize, old_w, old_h
       

        return  resized_crop_img, lmks_new2, bbox_red, lmks_flip



def replaceInFile(filep, before, after):
    for line in fileinput.input(filep, inplace=True):
        print line.replace(before,after),



def flip_lmk_idx(img, lmarks):

    # Flipping X values for landmarks                                                                                                                    \                                               
    lmarks[:,0] = img.shape[1] - lmarks[:,0]

    # Creating flipped landmarks with new indexing                                                                                                                                                       
    lmarks_flip =  np.zeros((68,2))
    for i in range(len(repLand)):
        lmarks_flip[i,:] = lmarks[repLand[i]-1,:]


    return lmarks_flip




def pose_to_LMs(pose_Rt):

        pose_Rt = np.reshape(pose_Rt, [6])
        ref_lm = np.loadtxt('./lm_m10.txt', delimiter=',')
        ref_lm_t = np.transpose(ref_lm)
        numLM = ref_lm_t.shape[1] 
        #PI = np.array([[  4.22519775e+03,0.00000000e+00,1.15000000e+02], [0.00000000e+00, 4.22519775e+03, 1.15000000e+02], [0, 0, 1]]);
        PI = np.array([[  2.88000000e+03, 0.00000000e+00, 1.12000000e+02], [0.00000000e+00, 2.88000000e+03, 1.12000000e+02], [0, 0, 1]]);


        rvecs = pose_Rt[0:3]
        tvec = np.reshape(pose_Rt[3:6], [3,1])
        tsum = np.repeat(tvec,numLM,1)
        rmat, jacobian = cv2.Rodrigues(rvecs, None)
        transformed_lms = np.matmul(rmat, ref_lm_t) + tsum
        transformed_lms = np.matmul(PI, transformed_lms)
        transformed_lms[0,:] = transformed_lms[0,:]/transformed_lms[2,:]
        transformed_lms[1,:] = transformed_lms[1,:]/transformed_lms[2,:]
        lms = np.transpose(transformed_lms[:2,:])


        return lms



def RotationMatrix(angle_x, angle_y, angle_z):
        # get rotation matrix by rotate angle

        phi = angle_x; # pitch
        gamma = angle_y; # yaw
        theta = angle_z; # roll

        R_x = np.array([ [1, 0, 0] , [0, np.cos(phi), np.sin(phi)] , [0, -np.sin(phi), np.cos(phi)] ]);
        R_y = np.array([ [np.cos(gamma), 0, -np.sin(gamma)] , [0, 1, 0] , [np.sin(gamma), 0, np.cos(gamma)] ]);
        R_z = np.array([ [np.cos(theta), np.sin(theta), 0] , [-np.sin(theta), np.cos(theta), 0] , [0, 0, 1] ]);

        R = np.matmul( R_x , np.matmul(R_y , R_z) );


        return R



def matrix2angle(R):
    ''' compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: yaw
        y: pitch
        z: roll
    '''
    # assert(isRotationMatrix(R))

    if R[2,0] !=1 or R[2,0] != -1:
        #x = asin(R[2,0])
        #y = atan2(R[2,1]/cos(x), R[2,2]/cos(x))
        #z = atan2(R[1,0]/cos(x), R[0,0]/cos(x))

        x = -asin(R[2,0])
        #x = np.pi - x
        y = atan2(R[2,1]/cos(x), R[2,2]/cos(x))
        z = atan2(R[1,0]/cos(x), R[0,0]/cos(x))

        
    else:# Gimbal lock
        z = 0 #can be anything
        if R[2,0] == -1:
            x = np.pi/2
            y = z + atan2(R[0,1], R[0,2])
        else:
            x = -np.pi/2
            y = -z + atan2(-R[0,1], -R[0,2])

    return x, y, z


def P2sRt(P):
    ''' decompositing camera matrix P. 
    Args: 
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation. 
        t3d: (3,). 3d translation.
    '''
    #t2d = P[:2, 3]
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2.0
    r1 = R1/np.linalg.norm(R1)
    r2 = R2/np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d