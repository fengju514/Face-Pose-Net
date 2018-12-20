import sys
import numpy as np
import tensorflow as tf
import cv2
import scipy.io as sio
sys.path.append('./utils')
import pose_utils as pu
import os
import os.path
from glob import glob
import time
import pickle

sys.path.append('./kaffe')
sys.path.append('./ResNet')
from ThreeDMM_shape import ResNet_101 as resnet101_shape



# Global parameters
factor = 0.25
_resNetSize = 224
n_hidden1 = 2048
n_hidden2 = 4096
ifdropout = 0


gpuID = int(sys.argv[1])
input_sample_list_path = str(sys.argv[2]) #'./input_list.txt' # You can change to your own image list


tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('image_size', 224, 'Image side length.')

output_path = './output_6DoF' 
tf.app.flags.DEFINE_string('save_output_path', output_path, 'Directory to keep the checkpoints')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_integer('batch_size', 1, 'Batch Size') # 60




if not os.path.exists(FLAGS.save_output_path):
        os.makedirs(FLAGS.save_output_path)





def extract_3dmm_pose():


        ########################################
        # Load train image mean, train label mean and std
        ########################################
        

        

        # labels stats on 300W-LP
        train_label_mean = np.load('./train_stats/train_label_mean_300WLP.npy')
        train_label_std = np.load('./train_stats/train_label_std_300WLP.npy')

        Pose_label_mean = train_label_mean[:6]
        Pose_label_std = train_label_std[:6]

        #ShapeExpr_label_mean_300WLP = train_label_mean[6:]
        #ShapeExpr_label_std_300WLP = train_label_std[6:]

       
        # Get training image mean from Anh's ShapeNet (CVPR2017)
        mean_image_shape = np.load('./train_stats/3DMM_shape_mean.npy') # 3 x 224 x 224 
        train_image_mean = np.transpose(mean_image_shape, [1,2,0]) # 224 x 224 x 3, [0,255]

        

       

        ########################################
        # Build CNN graph
        ########################################

        # placeholders for the batches                                                                                                                                      
        x_img = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3])
       
    

        # Resize Image
        x2 = tf.image.resize_bilinear(x_img, tf.constant([224,224], dtype=tf.int32))
        x2 = tf.cast(x2, 'float32')
        x2 = tf.reshape(x2, [-1, 224, 224, 3])
        
        # Image normalization
        mean = tf.reshape(train_image_mean, [1, 224, 224, 3])
        mean = tf.cast(mean, 'float32')
        x2 = x2 - mean
       


        ########################################
        # New-FPN with ResNet structure
        ########################################

        with tf.variable_scope('shapeCNN'):
                net_shape = resnet101_shape({'input': x2}, trainable=True) # False: Freeze the ResNet Layers
                pool5 = net_shape.layers['pool5']
                pool5 = tf.squeeze(pool5)
                pool5 = tf.reshape(pool5, [1, 2048])
                print pool5.get_shape() # batch_size x 2048

           
        with tf.variable_scope('Pose'):   

                with tf.variable_scope('fc1'):
                       
                        fc1W = tf.Variable(tf.random_normal(tf.stack([pool5.get_shape()[1].value, n_hidden1]), mean=0.0, stddev=0.01), trainable=True, name='W')
                        fc1b = tf.Variable(tf.zeros([n_hidden1]), trainable=True, name='baises')
               
                        fc1 = tf.nn.relu_layer(tf.reshape(pool5, [-1, int(np.prod(pool5.get_shape()[1:]))]), fc1W, fc1b, name='fc1')
                        print "\nfc1 shape:"
                        print fc1.get_shape(), fc1W.get_shape(), fc1b.get_shape() # (batch_size, 4096) (2048, 4096) (4096,)
                        
                        if ifdropout == 1:
                                fc1 = tf.nn.dropout(fc1, prob, name='fc1_dropout')

                with tf.variable_scope('fc2'):

                        fc2W = tf.Variable(tf.random_normal([n_hidden1, n_hidden2], mean=0.0, stddev=0.01), trainable=True, name='W')
                        fc2b = tf.Variable(tf.zeros([n_hidden2]), trainable=True, name='baises')

                        fc2 = tf.nn.relu_layer(fc1, fc2W, fc2b, name='fc2')
                        print fc2.get_shape(), fc2W.get_shape(), fc2b.get_shape() # (batch_size, 29 (2048, 2048) (2048,)

                        if ifdropout == 1:
                                fc2 = tf.nn.dropout(fc2, prob, name='fc2_dropout')

                with tf.variable_scope('fc3'):
               
                        # Move everything into depth so we can perform a single matrix multiplication.                            
                        fc2 = tf.reshape(fc2, [FLAGS.batch_size, -1])
                
                        dim = fc2.get_shape()[1].value
                        print "\nfc2 dim:"
                        print fc2.get_shape(), dim
                
                        fc3W = tf.Variable(tf.random_normal(tf.stack([dim,6]), mean=0.0, stddev=0.01), trainable=True, name='W')
                        fc3b = tf.Variable(tf.zeros([6]), trainable=True, name='baises')
                        #print "*** label shape: " + str(len(train_label_mean))
                        Pose_params_ZNorm = tf.nn.xw_plus_b(fc2, fc3W, fc3b)  
                        print "\nfc3 shape:"
                        print Pose_params_ZNorm.get_shape(), fc3W.get_shape(), fc3b.get_shape() 



                        Pose_label_mean = tf.cast(tf.reshape(Pose_label_mean, [1, -1]), 'float32')
                        Pose_label_std = tf.cast(tf.reshape(Pose_label_std, [1, -1]), 'float32')
                        Pose_params = Pose_params_ZNorm * (Pose_label_std + 0.000000000000000001) + Pose_label_mean

             


        ########################################
        # Start extracting 3dmm pose
        ########################################        
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        saver_ini_shape_net = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shapeCNN'))
        saver_shapeCNN = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='shapeCNN'))
        saver_Pose = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Pose'))
       

        config = tf.ConfigProto(allow_soft_placement=True) #, log_device_placement=True)
        #config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
               

                sess.run(init_op)
                start_time = time.time()
                
                
                # For non-trainable parameters such as the parameters for batch normalization
                load_path = "./models/ini_shapeNet_model_L7L_trainable.ckpt"
                saver_ini_shape_net.restore(sess, load_path)
                
                # For other trainable parameters
                load_path = "./models/model_0.0001_1_18_0.0_2048_4096.ckpt"
                saver_shapeCNN.restore(sess, load_path)
                saver_Pose.restore(sess, load_path)


                load_model_time = time.time() - start_time                
                print("Model restored: " + str(load_model_time))


                with open(input_sample_list_path, 'r') as fin:

                        for line in fin:

                                curr_line = line.strip().split(',')
                                image_path = curr_line[0]
                                bbox = np.array([float(curr_line[1]), float(curr_line[2]), float(curr_line[3]), float(curr_line[4])]) # [lt_x, lt_y, w, h]
                                image_key = image_path.split('/')[-1][:-4]


                                image = cv2.imread(image_path,1) # BGR                                                                            
                                image = np.asarray(image)
                        

                                # Fix the grey image                                                                                                                       
                                if len(image.shape) < 3:
                                        image_r = np.reshape(image, (image.shape[0], image.shape[1], 1))
                                        image = np.append(image_r, image_r, axis=2)
                                        image = np.append(image, image_r, axis=2)



                                # Crop and expand (25%) the image based on the tight bbox (from the face detector or detected lmks)
                                factor = [1.9255, 2.2591, 1.9423, 1.6087];
                                img_new = pu.preProcessImage_v2(image.copy(), bbox.copy(), factor, _resNetSize, 1)
                                image_array = np.reshape(img_new, [1, _resNetSize, _resNetSize, 3])



                                (params_pose, pool5_feats) = sess.run([Pose_params, pool5], feed_dict={x_img: image_array}) # [scale, pitch, yaw, roll, translation_x, translation_y]
                                params_pose = params_pose[0]
                                print params_pose #, pool5_feats       


                                # save the predicted pose
                                with open(FLAGS.save_output_path + '/' + image_key + '.txt', 'w') as fout:

                                        for pp in params_pose:
                                                fout.write(str(pp) + '\n')
                                

                                
                                # Convert the 6DoF predicted pose to 3x4 projection matrix (weak-perspective projection)
                                # Load BFM model 
                                shape_mat = sio.loadmat('./BFM/Model_Shape.mat')
                                mu_shape = shape_mat['mu_shape'].astype('float32')
                                
                                expr_mat = sio.loadmat('./BFM/Model_Exp.mat')
                                mu_exp = expr_mat['mu_exp'].astype('float32')
                                
                                mu = mu_shape + mu_exp
                                len_mu = len(mu) 
                                mu = np.reshape(mu, [-1,1])

                                keypoints = np.reshape(shape_mat['keypoints'], [-1]) - 1 # -1 for python index         
                                keypoints = keypoints.astype('int32')
                                
                                
                        
                                vertex = np.reshape(mu, [len_mu/3, 3]) # # of vertices x 3
                                # mean shape
                                mesh = vertex.T # 3 x # of vertices
                                mesh_1 = np.concatenate([mesh, np.ones([1,len_mu/3])], axis=0) # 4 x # of vertices


                                # Get projection matrix from 6DoF pose
                                scale, pitch, yaw, roll, tx, ty = params_pose
                                R = pu.RotationMatrix(pitch, yaw, roll)
                                ProjMat = np.zeros([3,4])
                                ProjMat[:,:3] = scale * R
                                ProjMat[:,3] = np.array([tx,ty,0])
                                

                                # Get predicted shape
                                #print ProjMat, ProjMat.shape
                                #print mesh_1, mesh_1.shape
                                pred_shape = np.matmul(ProjMat, mesh_1) # 3 x # of vertices
                                pred_shape = pred_shape.T # # of vertices x 3
                                

                                pred_shape_x = np.reshape(pred_shape[:,0], [len_mu/3, 1])
                                pred_shape_z = np.reshape(pred_shape[:,2], [len_mu/3, 1])
                                pred_shape_y = 224 + 1 - pred_shape[:,1]
                                pred_shape_y = np.reshape(pred_shape_y, [len_mu/3, 1])
                                pred_shape = np.concatenate([pred_shape_x, pred_shape_y, pred_shape_z], 1)
                



                                # Convert shape and lmks back to the original image scale

                                _, bbox_new, _, lmks_filling, old_h, old_w, img_new = pu.resize_crop_rescaleCASIA(image.copy(), bbox.copy(), pred_shape.copy(), factor)
                                #print lmks_filling
                                pred_shape[:,0] = pred_shape[:,0] * old_w / 224.
                                pred_shape[:,1] = pred_shape[:,1] * old_h / 224.
                                pred_shape[:,0] = pred_shape[:,0] + bbox_new[0]
                                pred_shape[:,1] = pred_shape[:,1] + bbox_new[1]

                                # Get predicted lmks
                                pred_lmks = pred_shape[keypoints]



                                sio.savemat(FLAGS.save_output_path + '/' + image_key + '.mat', {'shape_3D': pred_shape, 'lmks_3D': pred_lmks})
                                #cv2.imwrite(FLAGS.save_output_path + '/' + image_key + '.jpg', img_new)

                        


             


               
               
              


def main(_):
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpuID)
        if FLAGS.num_gpus == 0:
                dev = '/cpu:0'
        elif FLAGS.num_gpus == 1:
                dev = '/gpu:0'
        else:
                raise ValueError('Only support 0 or 1 gpu.')

        
        print dev
        with tf.device(dev):
	       extract_3dmm_pose()



if __name__ == '__main__':
        tf.app.run()
