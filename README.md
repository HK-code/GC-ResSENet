# Graph-Constrained Residual Self-Expressive Network Learning for Hyperspectral Images
Kun Huang, Xin Li, Yingdong Pi, Hao Cheng, Guowei Xu

## Abstract
> Hyperspectral images are gradually being used in various industries because of their rich spectral information. Meanwhile, the difficult acquisition of data labels makes unsupervised classification attracts attention.
	Subspace clustering as an unsupervised classification method is widely used for hyperspectral image analysis because of its excellent performance and robustness. However, conventional subspace clustering does not consider the nonlinear structure of hyperspectral data, and deep subspace clustering tends to ignore the intrinsic structure of hyperspectral data. To address these problems, we developed a self-expressive learning network, ResSENet, for hyperspectral data; we then proposed the application of ResSENet under graph constraints (GC-ResSENet), considering the intrinsic graph structure of the data. Unlike conventional deep subspace clustering, our model discards the self-expressive layer; self-expressive coefficients between datasets are directly solved by the data using network parameters. Hyperparameters are used in the joint loss to balance the self-expressive properties of the data and the graph constraint terms. We evaluated GC-ResSENet by applying it to four well-known datasets, and our network achieved optimal performance. Additionally, because of its abandonment of the self-expressive layer, ResSENet is theoretically capable of clustering with large datasets; thus, we evaluated it using two large datasets. The source code is accessible at [https://github.com/HK-code/GC-ResSENet](https://github.com/HK-code/GC-ResSENet).

## Dataset
SalinasA, Indian Pines and pavia university datasets you can download from [here](https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_and_University), and Houston2013 dataset you can download from [here](https://hyperspectral.ee.uh.edu/?page_id=459).

## Training & Evaluation

If you have any questions and needs, you can contact me, my email is: 2021202130068@whu.edu.cn.
