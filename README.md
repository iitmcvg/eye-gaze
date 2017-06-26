# eye-gaze
## Repositiory for Eye Gaze Detection and Tracking

We have implemented an Eye Gaze tracking system ( currently in Beta version ) using a series of algorithms. It computes the following :

* Pupil detection

* Facial Normal

* Gaze direction

To have a look into how we did it, just clone the repository and checkout **v1.0**.

```
$ cd eye-gaze
$ git checkout tags/v1.0
$ make
```
Or you can download **eye-gaze v1.0** directly from releases.

For a trial,

```
$ cd eye-gaze
$ ./bin/oic
```

## Dependencies

* OpenCV ( used 2.4.9 )

## Sample outputs

<img src = "https://raw.githubusercontent.com/iitmcvg/eye-gaze/master/res/demo/sample_output_1.png" width = "40%" />
<br><br>
<img src = "https://raw.githubusercontent.com/iitmcvg/eye-gaze/master/res/demo/sample_output_2.png" width = "40%" />
<br><br>

## References

### **Head-pose estimation**

1.Michael Sapienza and Kenneth P. Camilleri - “Fasthpe: A recipe for quick head pose estimation”

2.Michael Sapienza - “Head Motion Tracking and Pose Estimation in the Six Degrees of Freedom”

3.Roberto Valenti, Nicu Sebe , and Theo Gevers  - “Combining Head Pose and Eye Location Information for Gaze Estimation”

4.Xuehan Xiong ,Fernando De la Torre - “Supervised Descent Method and its Applications to Face Alignment”

5.Hiyam Hatem , Zou Beiji  , Raed Majeed  , Jumana Waleed Mohammed Lutf - “Head Pose Estimation Based On Detecting Facial Features “

6.Oliver Jesorsky, Klaus J. Kirchberg, and Robert W. Frischholz - “Robust Face Detection”

### **Pupil localisation**

1.Fabian Timm and Erhardt Barth  - “ACCURATE EYE CENTRE LOCALISATION BY MEANS OF GRADIENTS”

2.Tom Heyman  , Vincent Spruyt  , Alessandro Ledda - 3D Face “Tracking and Gaze Estimation Using a Monocular Camera”

3.Luke Allen and Adam Jensen  - “Webcam-based Gaze Estimation”

4.Jian-Gang Wang , Eric Sung , Ronda Venkateswarlu - ”Eye Gaze Estimation from a Single Image of One Eye”
