# MSST-RT: Multi-Stream Spatial-Temporal Relative Transformer for Skeleton-Based Action Recognition (MSST-RT)

## Abstract

Skeleton-based human action recognition has made great progress, especially with the development of a graph convolution network (GCN). The most important work is ST-GCN, which automatically learns both spatial and temporal patterns from skeleton sequences. However, this method still has some imperfections: only short-range correlations are appreciated, due to the limited receptive field of graph convolution. However, long-range dependence is essential for recognizing human action. In this work, we propose the use of a spatial–temporal relative transformer (ST-RT) to overcome these defects. Through introducing relay nodes, ST-RT avoids the transformer architecture, breaking the inherent skeleton topology in spatial and the order of skeleton sequence in temporal dimensions. Furthermore, we mine the dynamic information contained in motion at different scales. Finally, four ST-RTs, which extract spatial–temporal features from four kinds of skeleton sequence, are fused to form the final model, multi-stream spatial-temporal relative transformer (MSST-RT), to enhance performance. Extensive experiments evaluate the proposed methods on three benchmarks for skeleton-based action recognition: NTU RGB+D, NTU RGB+D 120 and UAV-Human. The results demonstrate that MSST-RT is on par with SOTA in terms of performance. 

![image](https://github.com/YIXIN-85/MSST-RT/blob/master/images/figure1.png)

Figure 1: Illustration of the overall architecture of the proposed MSST-RT. The sum of all scores from four ST-RTs is treated as the final prediction.



## Framework
![image](https://github.com/YIXIN-85/MSST-RT/blob/master/images/figure2.png)

Figure 2: Illustration of the spatial–temporal relative transformer (SR-RT). The skeleton data are processed by three modules and then fed into the fully connected layer to predict the score for each action class.

## The details of three modules
### Dynamics Representation(DR)
![image](https://github.com/YIXIN-85/MSST-RT/blob/master/images/figure3.png)
Figure 3: Illustrationofdynamicsrepresentation(DR).Therearefourstreamsofskeletoninformation embedded into a higher dimension by the embedding block and then concatenated as an input of the spatial relative transformer. Each block consists of two convolution layers and two activation layers.

### Spatial Relative Transformer (SRT)
![image](https://github.com/YIXIN-85/MSST-RT/blob/master/images/figure4.png)
Figure 4: Illustration of the update blocks in a spatial relative transformer (SRT). The graph structure in SRT is described in (a). Updating operates on each joint node by obtaining local information from adjacent joint nodes and non-local information from the spatial-relay node in (b). Spatial-relay nodes are updated by scoring the contribution of each node, including spatial joint nodes and the spatial-relay node in (c).

### Dynamics Representation(TRT)
![image](https://github.com/YIXIN-85/MSST-RT/blob/master/images/figure5.png)
Figure 5: Illustration of the update blocks in temporal relative transformer (TRT). The same joint nodes in all sampled skeleton are connected in order and the joint nodes in the first and last frame are also connected in (d). This approach constitutes a ring-shaped structure, as shown in (a). Furthermore, each joint node and the temporal-relay node are updated by TJU in (b) and TRU in (c), respectively, similar to the methods in SRT.

## Data Preparation

- Extract and process the NTU60, NTU120 and UAV_Huamn datasets respectively.
```bash
 python get_raw_skes_data.py
 python get_raw_denoised_data.py
 python seq_transformation.py
```


## Training
UAV
```bash
python  main.py --train 1
```
NTU60 or NTU120
```bash
# For the CS setting
python  main.py --train 1 --case 0
# For the CV setting
python  main.py --train 1 --case 1
```

## Testing
UAV
```bash
python  main.py --train 0
```
NTU60 or NTU120
```bash
# For the CS setting
python  main.py --train 0 --case 0
# For the CV setting
python  main.py --train 0 --case 1
```

## combining four stream
```bash
python ensemble.py
```
