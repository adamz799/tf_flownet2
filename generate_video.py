import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)


import numpy as np
from scipy.misc import imread
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
#FlowNet2
from FlowNet2_src import FlowNet2, LONG_SCHEDULE
from FlowNet2_src import flow_to_image



if __name__ == '__main__':
  output_mode = 'bk'#or 'color'
  src_video = cv2.VideoCapture('testlrv.mp4')
  #right_eye = cv2.VideoCapture('testlrv-right-eye.mp4')
  
  width = int(src_video.get(3)/2)
  height = int(src_video.get(4))
  total_frame = int(src_video.get(7))

  if output_mode == 'bk':
    output_name = 'output-B&W.avi'
  else:
    output_name = 'output-COLOR.avi'

  output_video = cv2.VideoWriter(
    output_name, 
    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
    src_video.get(cv2.CAP_PROP_FPS), 
    (width, height)
    )

  # Graph construction
  im1_pl = tf.placeholder(tf.float32, [1, height, width, 3])
  im2_pl = tf.placeholder(tf.float32, [1, height, width, 3])

  flownet2 = FlowNet2()
  inputs = {'input_a': im1_pl, 'input_b': im2_pl}
  flow_dict = flownet2.model(inputs, LONG_SCHEDULE, trainable=False)
  pred_flow = flow_dict['flow']

  # imgl = tf.placeholder(tf.uint8, shape=[height, width, 3], name='imgl')
  # imgl_f = tf.image.convert_image_dtype(imgl, dtype=tf.float32)
  # imgr = tf.placeholder(tf.uint8, shape=[height, width, 3], name='imgr')
  # imgr_f = tf.image.convert_image_dtype(imgr, dtype=tf.float32)
  
  ckpt_file = 'FlowNet2_src/checkpoints/FlowNet2/flownet-2.ckpt-0'
  saver = tf.train.Saver()

  flag = True
  num = 1
  with tf.Session() as sess:
    saver.restore(sess, ckpt_file)

    while flag and num<73:
      flag, frame = src_video.read()
      if flag == False:
        print 'Error occurs at frame {}!'.format(num) 
        break

      #fSize = frame.shape
      framel = frame[:, 0:width,[2, 1, 0]].copy()
      #cv2.imshow('left',framel)
      #cv2.waitKey(0)
      #framel = cv2.resize(framel, (width, height))
      framer = frame[:, width:,[2, 1, 0]].copy()
      #cv2.imshow('right',framer)
      #cv2.waitKey(0)
      #framer = cv2.resize(framer, (width, height))

      im1 = np.array([framel]).astype(np.float32)
      im2 = np.array([framer]).astype(np.float32)
      if im1.max()>1.0:
        im1 = im1/255.
      if im2.max()>1.0:
        im2 = im2/255.
      feed_dict = {im1_pl: im1, im2_pl: im2}
      pred_flow_val = sess.run(pred_flow, feed_dict=feed_dict)
      # Visualization


      if output_mode == 'color':
        flow_im = flow_to_image(pred_flow_val[0])     
      else:
        u = pred_flow_val[0][:,:,0]
        u = -u
        u = np.where(u>0.,u,0.)
        u = np.where(u<255.,u,255.)
        eps = 0.001
        m, M = u.min(), u.max()
        norm = (u - m) / (M - m) * 128.
        te = 128-eps
        norm = np.where(norm>te,te,norm)
        norm = np.where(norm<eps,eps,norm)          
        
        t = np.uint8(norm)
        #flow_im = cv2.merge([t,t,t])
        gray = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        flow_im = np.concatenate((frame[:,0:width,:],gray),axis = 1)

      cv2.imshow('result',flow_im)
      cv2.waitKey(0)
      output_video.write(flow_im)
      
      print 'Generating %.2f%%' %(num*100./total_frame)
      num+=1
      if num > total_frame:
        print 'Finish!'
        break

  src_video.release()
  output_video.release()

  
