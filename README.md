# TdDS
We have designed a top-down deep supervision (TdDS) method. TdDS can guide the model to learn the segmentation process from coarse to fine, which is similar to the process of human tumor identification (first quickly locate in the entire image, then confirm carefully). Experiments on colorectal cancer have shown that TdDS improves the performance of the model in delineating fuzzy boundaries.

We have implemented it on the U-Net framework. Good results were obtained on colorectal cancer dataset.
![](https://github.com/odindis/TdDS/blob/main/model.png)


