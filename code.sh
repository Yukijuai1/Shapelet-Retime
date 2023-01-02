#修改dataset为想要测试的数据集
#KNN
python3 KNN/knn.py --dataset=Car

#PPSN
#python3 PPSN/ppsn.py --datatser=Car

#LearningShapelet
python3 LearningShapelet/example.py --dataset=Car

#ShapeNet
python3 ShapeNet/shapenet_classify.py --dataset Car --path .. --save_path model/ --hyper ShapeNet/default_parameters.json --cuda

#Retime
python3 Retime/main.py --dataset=Car