# Pytorch_OpenPose_Reproduce Notes
## Network structure
### Same as Keras?
1. Not check using tensorboy code have this weight decay problem or not<br>
2. Using no extra layers<br>
3. forward function different<br>
4. init same, already check<br>
5. weight decay also in bias in Pytorch, only in weight in Keras<br>
6. using average or not have some difference? change it middle occur bugs<br>
7. large leanring rate can't be used when add weight decay? result show both can't work<br>
8. check the loss function<br>
9. seems not data problem, any kinds of data can't work.
10. change keras opt and 

`PAF_detax_19channels`<br>
`PAF_datay_19channels`<br>
`Related_length_19channels`<br>
### Groundtruth 
Range of PAF:<br>
[-1,+1]<br>
Range of centerx and centery and length:<br>
[0,1]<br>
    <br>
## Network
This model use the openpose network and change the input channels.<br>
The PAF is in branch 1.<br>
Total channel `38`.<br>
The Related_length is in brance 2.<br>
Total channel `19`.<br>
## Training results
Total time: `5`days in GTX 1080.<br>
Epoch:`109`.<br>
The details is in training/`training.csv`<br>
    <br>
Loss Graph:<br>
![](https://github.com/HaiyangLiu1997/Model_V2_BC_0422_2019/raw/master/result_images/loss.png)<br>
    <br>
Results:<br>
Wrong condition:<br>
<img src="https://github.com/HaiyangLiu1997/Model_V2_BC_0422_2019/raw/master/result_images/result_1.png" width="300" height="300" alt="related_length_wrong"/><br>
Correct condition:<br>
<img src="https://github.com/HaiyangLiu1997/Model_V2_BC_0422_2019/raw/master/result_images/result_2.png" width="300" height="300" alt="related_length_correct"/><br>
Final result:<br>
<img src="https://github.com/HaiyangLiu1997/Model_V2_BC_0422_2019/raw/master/result_images/result_0.png" width="300" height="300" alt="final_result"/><br>
## Results analysis
### Why give up
The related length's outputs are not good, the final range is in 0~0.8, It's can't get the right real length directly and need some extra fixed unit, the fixed unit don't has the robustness.<br>
I can't estimation all keypoints use the real length which I calculated, because the accuracy will be lower than the openpose, the openpose doesn't need "gauss". totally, for the visible condition using keypoints heatmap directly is better.<br>
### Accuracy
Haven't test.<br>
## How to use this model
For training:<br> 
Open the training file, change the dataset path in:<br>
`train_pose.py`<br> 
`dataset.py`<br>
If restart training, run `train_pose.py`.<br>
If continue training, get `weight.best.h5` first.<br>
https://1drv.ms/u/s!AmvJMVSfZX21kE8NrHC9UGIlIeyg<br>
For test result:<br>
Get the `weight.best.h5` first.<br>
Open the `demo.ipynb` with jupyter notebook or colab.<br>
Run the ipynb code step by step.<br>