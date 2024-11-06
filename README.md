# AFJPDA: Appearance Feature-aided JPDA
***
## Version Information

* 1st Mar 2023 Beta 0.0: First commit
* 30th Mar 2023 Beta 1.0: First full

--- Older versions are not available on github ---
* 19th Jul 2024 Release 1.0: First public 
***
## 1. Software Introduction

 This is a multi-class multi-object tracking (MCMOT) algorithm using joint probabilistic data association (JPDA) filter based on FairMOT and MCMOT on following links:
 
[FairMOT](https://github.com/ifzhang/FairMOT)
[MCMOT](https://github.com/CaptainEven/MCMOT)

Detail of the algorithm can be found from following paper:
[AFJPDA](https://arc.aiaa.org/doi/full/10.2514/1.I011301)

**Developed by Sukkeun Kim**
* Email: <s.kim.aero@gmail.com>


## 2. Running the Demo

1. Setup the environment following the FairMOT repository.
2. Run by following commands:
	>$ conda activate AFJPDA
	>$ cd src
	>$ python demo.py --load_model ../Your_pretrained_model.pth --input-video ../Your_test_video.mp4 --id_weight 2 --conf_thres 0.4
	
  * Examples:
	>$ python demo.py --load_model ../exp/models/mcmot_last_track_dla_34_carla_64000.pth --input-video ../exp/videos/Test.mp4 --id_weight 2 --conf_thres 0.4
	>$ python demo.py --load_model ../exp/models/mcmot_last_track_dla_18_visdrone.pth --input-video ../exp/videos/Test_visdrone.mp4 --id_weight 2
* Note: id_weight 0 for detection only, 1 for MCMOT by Even, 2 for JPDA, and 3 for AFJPDA


## 3. Using Own Dataset 

* Need to check below two files for using other dataset:
  * opts.py file in src/lib: Number of classes is defined here
  * gen_dataset_yourdataset: Class IDs are defined here (Check multitracker.py file)

* Training
	>$ python train.py
* Training data label: [ClassID, ReID, X, Y, W, H]
