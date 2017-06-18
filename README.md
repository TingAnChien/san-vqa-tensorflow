# Tensorflow Implementation of Stacked Attention Networks for Image Question Answering

![]()

Provide tensorflow edition for [SAN](https://arxiv.org/pdf/1511.02274.pdf), stacked attention network for image question answering model. The LSTM and CNN based question models are provided, and they both using two attention layers.
This code is modified from a tensorflow edition for deeper LSTM and normalized CNN VQA ([VQA-tensorflow](https://github.com/JamesChuanggg/VQA-tensorflow)).

### Requirements

The code is written in Python and requires [Tensorflow](https://www.tensorflow.org)(>r1.0). The preprocssinng code is in Python.</br>
(I also provide an old version(r0.10) for tensorflow model in branch r0.10)

### Prepare Data (modified from [VQA-tensorflow](https://github.com/JamesChuanggg/VQA-tensorflow))
(Some texts are copied from the original readme.md)
The first thing you need to do is to download the data and do some preprocessing. Head over to the `data/` folder and run

#### Download and Preprocess Dataset
for VQA 1.0:
```
$ python vqa_preprocessing.py --download True --split 1
```
We modify a version for VQA 2.0:
```
$ python vqa_preprocessing_v2.py --download True --split 1
```

`--download Ture` means you choose to download the VQA data from the [VQA website](http://www.visualqa.org/) and `--split 1` means you use COCO train set to train and validation set to evaluation. `--split 2 ` means you use COCO train+val set to train and test set to evaluate. After this step, it will generate two files under the `data` folder. `vqa_raw_train.json` and `vqa_raw_test.json`

#### Preprocess Texts
Once you have these, we are ready to get the question and image features. Back to the main folder, run

```
$ python prepro.py --input_train_json data/vqa_raw_train.json --input_test_json data/vqa_raw_test.json --num_ans 1000
```

If you want to use *model_VQA_w2v.py* ([VQA-tensorflow](https://github.com/JamesChuanggg/VQA-tensorflow) + word2vec) , run  

```
$ python prepro_w2v.py --input_train_json data/vqa_raw_train.json --input_test_json data/vqa_raw_test.json --num_ans 1000
```

to get the question features. `--num_ans` specifiy how many top answers you want to use during training. You will also see some question and answer statistics in the terminal output. This will generate two files in your main folder, `data_prepro.h5` and `data_prepro.json`.

#### Extract image features
To get the image features, run

```
$ python prepro_img.py
```

Here we use caffe to extract the `pool5` feature map instead of `fc7` from VGG_ILSVRC_19_layers [model](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77). The path of the caffe model and the output file is designated in the script. After this step, you can get the image feature `data_img.h5`. We have prepared everything and ready to launch training.


### Training and Testing
The `san_lstm_att.py` is for the LSTM based question model, and `san_cnn_att.py` is for the CNN based question model.</br>
To train on the prepared dataset, comment out `test()`.
Take LSTM basaed question model for example, we simply run the program with python.

```
$ python san_lstm_att.py
```

with the default parameter, this will take several hours and will generate the model under `model/san_lstm_att`.</br>
To test, comment out `train()` and run the same program, this will generate `san_lstm_att.json`.</br>
Modify the json file name in `s2i.py`, then run the program to correct the generated json files.

```
$ python s2i.py
```

This will generate the result `OpenEnded_mscoco_lstm_results.json`. To evaluate the accuracy of generate result, you need to download the [VQA evaluation tools](https://github.com/VT-vision-lab/VQA).

### Demo Website
We also provide a demo website project for this code, please see `demo/`.
Here are some results:
![](http://i.imgur.com/Q0YJeP1.png)

![](http://i.imgur.com/21hhMqo.png)

![](http://i.imgur.com/HqH381v.png)

![](http://i.imgur.com/4LgShex.png)
