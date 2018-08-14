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
  
  width = src_video.get(3)/2
  heigth = src_video.get(4)
  total_frame = src_video.get(7)

  if output_mode == 'bk':
    output_name = 'output-B&W.avi'
  else:
    output_name = 'output-COLOR.avi'

  output_video = cv2.VideoWriter(
    output_name, 
    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
    left_eye.get(cv2.CAP_PROP_FPS), 
    (width, heigth)
    )

  # Graph construction
  im1_pl = tf.placeholder(tf.float32, [1, heigth, width, 3])
  im2_pl = tf.placeholder(tf.float32, [1, heigth, width, 3])

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

    while flag:
      flag, frame = src_video.read()
      if flag == False:
        print 'Error occurs at frame {}!'.format(num) 
        break

      #fSize = frame.shape
      framel = frame[:, 0:width, :]
      framel = cv2.resize(framel, (width, height))
      framer = frame[:, width:, :]
      framer = cv2.resize(framer, (width, height))

      im1 = np.array([framel]).astype(np.float32)
      im2 = np.array([framer]).astype(np.float32)
    
      feed_dict = {im1_pl: im1, im2_pl: im2}
      pred_flow_val = sess.run(pred_flow, feed_dict=feed_dict)
      # Visualization


      if output_mode == 'color':
        flow_im = flow_to_image(pred_flow_val[0])     
      else:
        u, v = cv2.split(pred_flow_val)
        #print(u.min(), "  ", u.max()) 
        u[u < 0] = 0
        u[u > 255] = 0
        eps = 0.001
        (m, M) = (u.min(), u.max())
        norm = 1.0 * (u - m) / (M - m) * 128
        norm[norm > (125 - eps)] = 125 - eps
        norm[norm < eps] = eps          
        #norm = np.uint8(u)
        flow_im = np.uint8(norm)

      output_video.write(flow_im)
      #plt.imshow(flow_im)
      #plt.show()
      print 'Output frame {}'.format(num)
      num+=1
      if num > total_frame:
        print 'Finish!'
        break

  left_eye.release()
  right_eye.release()
  output_video.release()

  
