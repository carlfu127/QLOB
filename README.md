# myCode
# Question-Led Object Attention for Visual Question Answering

This is the code that we wrote to train our VQA models for COCOQA and VQA datasets. We use python with Theano library to train the
model in our experiments.

In order to train the model:
 - Data Preprocessing
    You can use the code provided by https://github.com/jiasenlu/HieCoAttenVQA for data preprocessing. Here, we have published the 
	COCOQA dataset that we extracted from original data https://drive.google.com/open?id=18qZ7-5czVX2DyLTfN3skmB6VG_Woa9q4. See directory "cocoqa_datasets". And for image preprocessing, you can use the pretrained features provided by https://github.com/peteanderson80/bottom-up-attention

 - Training
    After adding corresponding file path, run the file 'sgd_main.py' to train the model. You can change parameters according to your own needs.
 
 - Evaluation
   COCOQA dataset has published the answers, you can directly test the accuracy. VQA dataset do not provide the answers, and the evaluation refers to http://www.visualqa.org/evaluation.html
  
