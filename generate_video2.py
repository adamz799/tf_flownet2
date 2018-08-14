import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

import argparse
import numpy as np
import tensorflow as tf
import cv2

#FlowNet2
from FlowNet2_src import FlowNet2, LONG_SCHEDULE
from FlowNet2_src import flow_to_image



if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--fullname',
    type=str,
    required=True,
    help='File name with type'
  )
  
  FLAGS = parser.parse_args()

  src_name = FLAGS.fullname

  src_video = cv2.VideoCapture('src/'+src_name)

  width = int(src_video.get(3)/2)
  height = int(src_video.get(4))
  total_frame = int(src_video.get(7))

  src_name = src_name.split('.')[0]
  
  output_name = src_name+'-output.avi'
  output_video = cv2.VideoWriter(
    'output/'+output_name,
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

  ckpt_file = 'FlowNet2_src/checkpoints/FlowNet2/flownet-2.ckpt-0'
  saver = tf.train.Saver()

  flag = True
  num = 1
  with tf.Session() as sess:
    saver.restore(sess, ckpt_file)
    print 'Loading weights...'

    while flag:# and num<73
      flag, frame = src_video.read()
      if flag == False:
        print 'Error occurs at frame {}!'.format(num)
        break

      framel = frame[:, 0:width,[2, 1, 0]].copy()
      framer = frame[:, width:,[2, 1, 0]].copy()
      
      im1 = np.array([framel]).astype(np.float32)
      im2 = np.array([framer]).astype(np.float32)
      if im1.max()>1.0:
        im1 = im1/255.
      if im2.max()>1.0:
        im2 = im2/255.
      feed_dict = {im1_pl: im1, im2_pl: im2}
      pred_flow_val = sess.run(pred_flow, feed_dict=feed_dict)

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
      gray = cv2.merge([t,t,t])

      #cv2.imshow('gray',flow_to_image(pred_flow_val[0]))
      #cv2.waitKey(0)

      flow_im = np.concatenate((frame[:, 0:width,:].copy(),gray),axis = 1)     
      cv2.imshow('result',flow_im)
      cv2.waitKey(0)
      output_video.write(flow_im)
      
      print 'Generating %.2f%%' % (num*100./total_frame) 
      num+=1
      if num > total_frame:
        print 'Finish!'
        break

  src_video.release()
  output_video.release()


