## FlowNet2 (TensorFlow)

This repo includes FlowNetC, S, CS, CSS, CSS-ft-sd, SD, and 2 for TensorFlow. Most part are from this [repo](https://github.com/sampepose/flownet2-tf), and we have made some modifications:
* It can deal with arbitrary size of input now.
* After installation, just copy the whole folder `FlowNet2_src` to your codebase to use. See `demo.py` for details.

### Generate video
This fork is for generating optical flow videos.

Run the commond: 
```
python generate_video1.py --fullname src_video.mp4 -- mode 'color'
```

Here, 'src_video.mp4' is the full name(include the format) of the source video. 
And 'mode' is the way how to generate video: 'color' means generating colorful results which contain both horizontal and vertical optical flow; 'bk' means the results only contain horizontal optical flow which can be treated as horizontal parallax in some cases.

### Environment

This code has been tested with Python2.7 and TensorFlow1.2.1 based on CUDA8.0 and CuDNN5.1, with GV100. The system is Ubuntu 16.04.

### Installation

You must have CUDA installed: `make all`

**Note:** you might need to modify [this line](https://github.com/vt-vl-lab/tf_flownet2/blob/master/Makefile#L13), according to the GPU you use.

Add by adamz799:

I change the above line to suits different GPUs [like this](https://github.com/adamz799/tf_flownet2/Makefile#13). Some GPUs like GV100 whose compute capability is 7.0 are not fully supported by CUDA8(6.1). To run the code on these GPUs, setting 'gpucflag' to default is feasible.

On the other hand, this fork use OpenCV to deal with videos, so OpenCV Python version is needed.
```
pip install opencv-python
```

### Download weights
To download the weights for all models (4.4GB), run the `download.sh` script in the `FlowNet2_src/checkpoints` directory. All test scripts rely on these checkpoints to work properly.


### Inference mode

```
python demo.py
```

If installation is successful, you should see the following:
![FlowNet2 Sample Prediction](/FlowNet2_src/example/0flow-pred-flownet2.png?raw=true)

Notice that the model itself will handle the RGB to BGR operation for you. And please be care about your input scale and datatype.


### Performance (w/o fine-tuning)

| Model       | KITTI2012 Train EPE     |  KITTI2015 Train EPE     |  KITTI2015 Train F1     |  Sintel Final Train EPE |
|-------------|--------|--------|--------|--------|
| FlowNetS       | 7.2457     |  14.0753     |  0.5096     |  3.9140     |
| FlowNetC       | 5.9793    |  11.8957     |  0.4509     |  3.1001     |
| FlowNet2       | 4.3167     |  10.9869     |  0.3241     |  2.1592     |

FlowNetS and FlowNetC are better than paper, but FlowNet2 is slightly worse.

### TODO
* Add fine-tune mode
* Remove the `training_schedule` variable from inference mode.


### Reference
[1] E. Ilg, N. Mayer, T. Saikia, M. Keuper, A. Dosovitskiy, T. Brox
FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks,
IEEE Conference in Computer Vision and Pattern Recognition (CVPR), 2017.

### Acknowledgments
As noted in the beginning, most part are from [sampepose/flownet2-tf](https://github.com/sampepose/flownet2-tf)
