.txt文件里面保存了ppl和各项指标，训练用的指导数据集为C4
其中，only开头的是去掉了绝对值，只保留相对值W_metric = W_metric  + gamma * (ss - ss_not_change)
reg开头的是保留绝对值的情况下加上相对值作为正则化W_metric = W_metric  + ss*rho + gamma * (ss - ss_not_change)
后面的数字表示gamma的值
norm是原本论文中压缩的效果
origin是普通llama7b的效果