
import numpy as np
import json

f = json.load(open('../cocoqa_datasets/cocoqa_data_prepro.json'))
vocabulary = f['ix_to_word']
answers = f['ix_to_ans']
train = np.load('../cocoqa_datasets/trainset.npz')
valid = np.load('../cocoqa_datasets/validset.npz')
test = np.load('../cocoqa_datasets/testset.npz')

imdir='../features/image/trainval/coco_trainval_%s.npy'
trainset_x,trainset_mask,trainset_y,trainset_img = train['trainset_x'],train['trainset_mask'],train['trainset_y'],train['trainset_img']
testset_x,testset_mask,testset_y,testset_img = test['testset_x'],test['testset_mask'],test['testset_y'],test['testset_img']
validset_x,validset_mask,validset_y,validset_img = valid['validset_x'],valid['validset_mask'],valid['validset_y'],valid['validset_img']

trainset_x = np.concatenate((trainset_x,validset_x))
trainset_mask = np.concatenate((trainset_mask,validset_mask))
trainset_y = np.concatenate((trainset_y,validset_y))
trainset_img = np.concatenate((trainset_img,validset_img))
print 'loading data!!!'

def process_batch_train(batch_idx,batch_size):
    subtype = 'train2014'
    input_idx = trainset_x[batch_idx*batch_size:(batch_idx+1)*batch_size]
    input_mask = trainset_mask[batch_idx*batch_size:(batch_idx+1)*batch_size]
    input_label = trainset_y[batch_idx*batch_size:(batch_idx+1)*batch_size]
    input_img = trainset_img[batch_idx*batch_size:(batch_idx+1)*batch_size]
    image_feat = []

    for i in range(batch_size):
        id = int(input_img[i][-10:])
        feat = np.load(imdir % str(id))
        feat_g = np.load("../features/resnet101/train2014/%s.npy" % input_img[i])
        feat = np.concatenate([feat, feat_g[None, : ]], axis=0)
        image_feat.append(feat)
    image_feat = np.asarray(image_feat)
    return input_idx, input_mask, input_label, image_feat

def process_batch_valid(batch_idx, batch_size):
    subtype = 'val2014'
    input_idx = trainset_x[batch_idx*batch_size:(batch_idx+1)*batch_size]
    input_mask = trainset_mask[batch_idx*batch_size:(batch_idx+1)*batch_size]
    input_label = trainset_y[batch_idx*batch_size:(batch_idx+1)*batch_size]
    input_img = trainset_img[batch_idx*batch_size:(batch_idx+1)*batch_size]
    image_feat = []
    for i in range(batch_size):
        image_path = imdir % (subtype, int(input_img[i]))
        feat = np.load('../features/resnet_feature/val2014/%s.npy' % input_img[i]).reshape((2048, -1))
        image_feat.append(np.transpose(feat))
    image_feat = np.asarray(image_feat)
    return input_idx, input_mask, input_label, image_feat

def process_batch_test(batch_idx,batch_size):
    subtype = 'val2014'
    input_idx = testset_x[batch_idx*batch_size:(batch_idx+1)*batch_size]
    input_mask = testset_mask[batch_idx*batch_size:(batch_idx+1)*batch_size]
    input_label = testset_y[batch_idx*batch_size:(batch_idx+1)*batch_size]
    input_img = testset_img[batch_idx*batch_size:(batch_idx+1)*batch_size]
    image_feat = []
    for i in range(batch_size):
        id = int(input_img[i][-10:])
        feat = np.load(imdir % str(id))
        feat_g = np.load("../features/resnet101/val2014/%s.npy" % input_img[i])
        feat = np.concatenate([feat, feat_g[None, :]], axis=0)
        image_feat.append(feat)
    image_feat = np.asarray(image_feat)
    return input_idx, input_mask, input_label, image_feat