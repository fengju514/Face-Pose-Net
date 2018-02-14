import csv
import lmdb
import sys
import numpy as np
import cv2
import os
this_path = os.path.dirname(os.path.abspath(__file__))
render_path = this_path+'/face_renderer/'
sys.path.append(render_path)
try:
    import myutil
except ImportError as ie:
    print '****************************************************************'
    print '**** Have you forgotten to "git clone --recursive"?          ****'
    print '**** You have to do that to also download the face renderer ****'
    print '****************************************************************'
    print ie.message
    exit(0)
import config
opts = config.parse()
import camera_calibration as calib
import ThreeD_Model
import renderer as renderer_core
import get_Rts as getRts
#pose_models = ['model3D_aug_-00_00','model3D_aug_-22_00','model3D_aug_-40_00','model3D_aug_-55_00','model3D_aug_-75_00']
newModels = opts.getboolean('renderer', 'newRenderedViews')
if opts.getboolean('renderer', 'newRenderedViews'):
    pose_models_folder = '/models3d_new/'
    pose_models = ['model3D_aug_-00_00','model3D_aug_-22_00','model3D_aug_-40_00','model3D_aug_-55_00','model3D_aug_-75_00']
else:
    pose_models_folder = '/models3d/'
    pose_models = ['model3D_aug_-00','model3D_aug_-40','model3D_aug_-75',]
nSub = 10
allModels = myutil.preload(render_path,pose_models_folder,pose_models,nSub)

def render_fpn(inputFile, output_pose_db, outputFolder):
    ## Opening FPN pose db
    pose_env = lmdb.open( output_pose_db, readonly=True )
    pose_cnn_lmdb = pose_env.begin()
    ## looping over images
    with open(inputFile, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        lines = csvfile.readlines()
        for lin in lines:
            ### key1, image_path_key_1
            image_key = lin.split(',')[0]
            if 'flip' in image_key:
                continue

            image_path = lin.split(',')[-1].rstrip('\n')
            img = cv2.imread(image_path, 1)
            pose_Rt_raw = pose_cnn_lmdb.get( image_key )
            pose_Rt_flip_raw = pose_cnn_lmdb.get(image_key + '_flip')

            if pose_Rt_raw is not None:
                pose_Rt = np.frombuffer( pose_Rt_raw, np.float32 )
                pose_Rt_flip = np.frombuffer( pose_Rt_flip_raw, np.float32 )
                
                yaw = myutil.decideSide_from_db(img, pose_Rt, allModels)
                

                if yaw < 0: # Flip image and get the corresponsidng pose
                    img = cv2.flip(img,1)
                    pose_Rt = pose_Rt_flip
                
                

                listPose = myutil.decidePose(yaw, opts, newModels)
                ## Looping over the poses
                for poseId in listPose:
                    posee = pose_models[poseId]
                    ## Looping over the subjects
                    for subj in [10]:
                        pose =   posee + '_' + str(subj).zfill(2) +'.mat'
                        print '> Looking at file: ' + image_path + ' with ' + pose
                        # load detections performed by dlib library on 3D model and Reference Image
                        print "> Using pose model in " + pose
                        ## Indexing the right model instead of loading it each time from memory.
                        model3D = allModels[pose]
                        eyemask = model3D.eyemask
                        # perform camera calibration according to the first face detected
                        proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, pose_Rt, pose_db_on=True)
                        ## We use eyemask only for frontal
                        if not myutil.isFrontal(pose):
                            eyemask = None
                        ##### Main part of the code: doing the rendering #############
                        rendered_raw, rendered_sym, face_proj, background_proj, temp_proj2_out_2, sym_weight = renderer_core.render(img, proj_matrix,\
                                                                                                 model3D.ref_U, eyemask, model3D.facemask, opts)
                        ########################################################

                        if myutil.isFrontal(pose):
                            rendered_raw = rendered_sym
                        ## Cropping if required by crop_models
                        #rendered_raw = myutil.cropFunc(pose,rendered_raw,crop_models[poseId])
                        ## Resizing if required
                        #if resizeCNN:
                        #    rendered_raw = cv2.resize(rendered_raw, ( cnnSize, cnnSize ), interpolation=cv2.INTER_CUBIC )
                        ## Saving if required
                        if opts.getboolean('general', 'saveON'):
                            subjFolder = outputFolder + '/'+ image_key.split('_')[0]
                            myutil.mymkdir(subjFolder)
                            savingString = subjFolder +  '/' + image_key +'_rendered_'+ pose[8:-7]+'_'+str(subj).zfill(2)+'.jpg'
                            cv2.imwrite(savingString,rendered_raw)
