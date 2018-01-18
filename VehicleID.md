## Priv_VehicleID-Benchmark
ReID Performance on VehicleID(http://www.pkuml.org/resources/pku-vehicleid.html)

### VehicleID:
* Contains data captured during daytime by multiple real-world surveillance cameras distributed in a small city in China.
* 221567 images of 26328 vehicles(8.42 images/vehicle in average) in the entire dataset. 
* Train set has 13164 vehicles(113346 images). In order to verify the classification performance, we pick out a picture from each category to form a val set 
* Test set has 13164 vehicles(118221 images). The test set contains several subtest sets.

Number of images|Small|Medium|Large|
:---:|:---:|:---:|:---:
Gallery size|800|1600|2400|
Query size|5693|11777|17377|


### Classification and ReID performance

* **Environment:** [**`pytorch-priv`**](https://github.com/soeaver/pytorch-priv), `cuda: 8.0`, `cudnn: 6.0`, `GPU: titan xp`
* **Network:** `lr-decay: cosine`, `fix_bn: False`
* **Training:** `batch-size: 256`, `epochs: 100`, `crop-size: 224`
* **validation:** `batch-size: 200`, `crop-type: center-crop`, `base-size: 256`, `crop-size: 224`
* **Testing:** `crop-type: center-crop`, `base-size: 256`, `crop-size: 224`

#### 1 Performance of different architectures
* **`train-crop-type: randomresized-crop`**

 Network|base lr|weight decay|train top1|val top1|top1@S|top5@S|top10@S|
 :---:|:---:|:---:|:---:|:---:|:---:|:---:|:---: 
 resnet18|0.1|0.0001|94.55|65.08|54.90|66.42|70.73
 resnet18|0.1|0.0003|94.01|71.92|61.81|72.45|77.12
 resnet18|0.1|0.0005|92.52|74.13|64.69|74.51|79.32
 resnet18|0.1|0.0007|91.29|74.35|64.22|75.64|81.73
 resnet18|0.1|0.0009|89.58|73.47|--|--|--
 resnet18|0.2|0.0005|92.73|76.08|65|73.67|78.65
 resnet18|0.4|0.0005|90.52|76.74|64.9|73.5|78.29
 air50_1x32d|0.4|0.0001|95.30|74.46|61.94|69.04|72.40
 
 * **`train-crop-type: center-crop`**, `base-size: 256`
 
 Network|base lr|weight decay|train top1|val top1|top1@S|top5@S|top10@S|
 :---:|:---:|:---:|:---:|:---:|:---:|:---:|:---: 
 resnet18|0.4|0.0005|99.99|78.32|68.66|80.05|85.39
 resnet18-1x64d|0.2|0.0005|99.99|79.18|68.37|79.96|85.49
 resnet18-1x64d|0.4|0.0005|99.99|79.41|69.40|81.16|86.63
 resnet18-1x64d|0.4|0.0005|99.99|79.22|69.07|80.41|85.70
 resnet18-1x64d|0.4|0.0005|99.99|79.84|69.05|79.67|85.38
 air50_1x32d|0.4|0.0001|99.99|75.79|63.69|72.58|77.22
 air50_1x32d|0.4|0.0005|99.99|80.26|69.5|81.44|86.42
 resnet18-1x96d|0.4|0.0005|99.99|79.72|--|--|--
 
 * **`mixup-train`**, `epochs: 200`
 
 Network|base lr|weight decay|train crop type|val top1|base val top1| 
 :---:|:---:|:---:|:---:|:---:|:---:
 resnet18_1x64d|0.2|0.0005|randomresized-crop|77.12|--
 resnet18_1x64d|0.2|0.0001|center-crop|74.92|--
 **resnet18_1x64d**|0.2|0.0005|center-crop|79.20|79.18
 resnet18_1x64d|0.4|0.0005|center-crop|78.68|79.41
 air50_1x32d|0.4|0.0001|center-crop|77.78|75.79
 air50_1x32d|0.4|0.0005|center-crop|78.05|80.26
 
 * **`mixup-train-fineturn`**, `Network: resnet18_1x64d`, `epochs: 20`, `weight-decay: 0.0005`, `train-crop-type: center-crop`
 
 base lr|epochs|lr decay|val top1|mixup val top1|base val top1| 
 :---:|:---:|:---:|:---:|:---:|:---:
 0.0002|20|--|79.37|79.20|79.18
 0.002|20|[10] * 0.1|80.13|79.20|79.18
 0.2|20|cosine 220|80.14|79.20|79.18
 0.04|100|cosine|80.80|79.20|79.18
 
 
 * **`data_augmentation`**, `epochs: 100`, `base-lr: 0.4`, `weight-decay: 0.0005`, `train-crop-type: center-crop`
 
 Network|rotation|pixel jitter|val top1|base val top1|
 :---:|:---:|:---:|:---:|:---:
 resnet18_1x64d|[-10, 10]|--|80.96|79.41|
 resnet18_1x64d|[-20, 20]|--|80.60|79.41|
 resnet18_1x64d|--|[-10, 10]|79.79|79.41|
 resnet18_1x64d|--|[-20, 20]|79.79|79.41|
 resnet18_1x64d|[-10, 10]|[-10, 10]|80.33|79.41|
 
 
 
 
 
 

 

 


 

 

