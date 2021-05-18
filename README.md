# **urban-governance**
In the process of urban governance, there appearing such scenes occasionally: 
- whether **trash can** on the road and in the community is placed correctly.<sup>1</sup>
- whether **slag car** is clean and run in the right road.<sup>2</sup>
- whether **flotage** on the river is slaving.<sup>3</sup>
- whether **trash & fallen leaves** on the road is swept.<sup>4</sup>
- whether **blot** that scatter from the truck et. clotting on the road.<sup>5</sup> [TODO]
 
The taskes and API mapping as follow:  
- For 1, API is **predict_can & predict_ctrashc**
- For 2, API is **predict**
- For 3, API is **predict_smpfw**
- For 4, API is **predict_trash**
- For 5, API is **predict_blot** [TODO]

## Models [TODO]

## Installation
Nvidia cuda compiler driver: 10.2
```bash
cd urban-governance
conda create -n cszz python=3.6 -y
source activate cszz
pip install torch==1.6.0 torchvision==0.7.0
pip install opencv-python
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.6/index.html
```

## Train


## Test

