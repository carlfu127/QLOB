from sgd_main import *

def test_model(shared_params, options):
    trng = RandomStreams(1234)
    drop_ratio = options['drop_ratio']
    # batch_size = options['batch_size']
    # n_dim = options['n_dim']
    #
    # w_emb = shared_params['w_emb']

    dropout = theano.shared(numpy.float32(0.))
    image_feat = T.ftensor3('image_feat')
    # batch_size x T
    input_idx = T.imatrix('input_idx')
    input_mask = T.matrix('input_mask')
    # label is the TRUE label
    # label = T.ivector('label')

    empty_word = theano.shared(value=np.zeros((1, options['n_emb']),
                                              dtype='float32'),
                               name='empty_word')
    w_emb_extend = T.concatenate([empty_word, shared_params['w_emb']],
                                 axis=0)
    input_emb = w_emb_extend[input_idx]
    # input_emb = w_emb[input_idx]
    # a trick here, set the maxpool_h/w to be large
    # maxpool_shape = (options['maxpool_h'], options['maxpool_w'])

    # turn those appending words into zeros
    # batch_size x T x n_emb
    input_emb = input_emb * input_mask[:, :, None]
    if options['sent_drop']:
        input_emb = dropout_layer(input_emb, dropout, trng, drop_ratio)

    if options['use_unigram_conv']:
        unigram_conv_feat = fflayer(shared_params, input_emb, options,
                                    prefix='conv_unigram',
                                    act_func=options.get('sent_conv_act', 'tanh'))
        unigram_pool_feat = unigram_conv_feat.max(axis=1)
    if options['use_bigram_conv']:
        idx = T.concatenate([T.arange(input_emb.shape[1])[:-1],
                             T.arange(input_emb.shape[1])[1:]]).reshape((2, input_emb.shape[1] - 1)).transpose().flatten()
        bigram_emb = T.reshape(input_emb[:, idx, :], (input_emb.shape[0],
                                                      input_emb.shape[1] - 1,
                                                      2 * input_emb.shape[2]))
        bigram_conv_feat = fflayer(shared_params, bigram_emb,
                                   options, prefix='conv_bigram',
                                   act_func=options.get('sent_conv_act', 'tanh'))
        bigram_pool_feat = bigram_conv_feat.max(axis=1)
    if options['use_trigram_conv']:
        idx = T.concatenate([T.arange(input_emb.shape[1])[:-2],
                             T.arange(input_emb.shape[1])[1:-1],
                             T.arange(input_emb.shape[1])[2:]]).reshape((3, input_emb.shape[1] - 2)).transpose().flatten()
        trigram_emb = T.reshape(input_emb[:, idx, :], (input_emb.shape[0],
                                                      input_emb.shape[1] - 2,
                                                      3 * input_emb.shape[2]))
        trigram_conv_feat = fflayer(shared_params, trigram_emb,
                                    options, prefix='conv_trigram',
                                    act_func=options.get('sent_conv_act', 'tanh'))
        trigram_pool_feat = trigram_conv_feat.max(axis=1)  #

    pool_feat = T.concatenate([unigram_pool_feat,
                               bigram_pool_feat,
                               trigram_pool_feat], axis=1)

    image_feat_down = fflayer(shared_params, image_feat, options,
                              prefix='image_mlp',
                              act_func=options.get('image_mlp_act',
                                                   'tanh'))
    image_feat_down_gate = fflayer(shared_params, image_feat, options,
                              prefix='image_mlp_gate',
                              act_func='sigmoid')
    image_feat_down = image_feat_down * image_feat_down_gate

    if options.get('use_before_attention_drop', False):
        image_feat_down = dropout_layer(image_feat_down, dropout, trng, drop_ratio)
        pool_feat = dropout_layer(pool_feat, dropout, trng, drop_ratio)

    image_ques = image_feat_down + pool_feat[:, None, :]
    # attention model begins here
    # first layer attention model
    image_feat_attention_1 = fflayer(shared_params, image_ques, options,
                                     prefix='image_att_mlp_1',
                                     act_func=options.get('image_att_mlp_act',
                                                          'tanh'))
    image_feat_attention_1_gate = fflayer(shared_params, image_ques, options,
                                     prefix='image_att_mlp_1_gate',
                                     act_func='sigmoid')
    image_feat_attention_1 = image_feat_attention_1 * image_feat_attention_1_gate

    pool_feat_attention_1 = fflayer(shared_params, pool_feat, options,
                                    prefix='sent_att_mlp_1',
                                    act_func=options.get('sent_att_mlp_act',
                                                         'tanh'))
    pool_feat_attention_1_gate = fflayer(shared_params, pool_feat, options,
                                    prefix='sent_att_mlp_1_gate',
                                    act_func='sigmoid')
    pool_feat_attention_1 = pool_feat_attention_1 * pool_feat_attention_1_gate

    combined_feat_attention_1 = image_feat_attention_1 + \
                                pool_feat_attention_1[:, None, :]
    if options['use_attention_drop']:
        combined_feat_attention_1 = dropout_layer(combined_feat_attention_1,
                                                  dropout, trng, drop_ratio)

    combined_feat_attention_1 = fflayer(shared_params, combined_feat_attention_1,
                                        options, prefix='combined_att_mlp_1', act_func='tanh')
    prob_attention_1 = T.nnet.softmax(combined_feat_attention_1[:, :, 0])

    image_feat_ave_1 = (prob_attention_1[:, :, None] * image_ques).sum(axis=1)

    # combined_hidden_1 = image_feat_ave_1 + pool_feat
    """
    image_ques_2 = image_feat_down + image_feat_ave_1[:, None, :]
    # second layer attention model

    image_feat_attention_2 = fflayer(shared_params, image_ques_2, options,
                                     prefix='image_att_mlp_2',
                                     act_func=options.get('image_att_mlp_act',
                                                          'tanh'))
    image_feat_attention_2_gate = fflayer(shared_params, image_ques_2, options,
                                     prefix='image_att_mlp_2_gate',
                                     act_func='sigmoid')
    image_feat_attention_2 = image_feat_attention_2 * image_feat_attention_2_gate

    pool_feat_attention_2 = fflayer(shared_params, image_feat_ave_1, options,
                                    prefix='sent_att_mlp_2',
                                    act_func=options.get('sent_att_mlp_act',
                                                         'tanh'))
    pool_feat_attention_2_gate = fflayer(shared_params, image_feat_ave_1, options,
                                    prefix='sent_att_mlp_2_gate',
                                    act_func='sigmoid')
    pool_feat_attention_2 = pool_feat_attention_2 * pool_feat_attention_2_gate
    combined_feat_attention_2 = image_feat_attention_2 + \
                                pool_feat_attention_2[:, None, :]
    if options['use_attention_drop']:
        combined_feat_attention_2 = dropout_layer(combined_feat_attention_2,
                                                  dropout, trng, drop_ratio)

    combined_feat_attention_2 = fflayer(shared_params,
                                        combined_feat_attention_2, options,
                                        prefix='combined_att_mlp_2',
                                        act_func=options.get(
                                            'combined_att_mlp_act', 'tanh'))
    prob_attention_2 = T.nnet.softmax(combined_feat_attention_2[:, :, 0])

    image_feat_ave_2 = (prob_attention_2[:, :, None] * image_ques_2).sum(axis=1)

    if options.get('use_final_image_feat_only', False):
        combined_hidden = image_feat_ave_2 + pool_feat
    else:
        combined_hidden = image_feat_ave_2 #+ combined_hidden_1
    """
    combined_hidden = image_feat_ave_1
    for i in range(options['combined_num_mlp']):
        if options.get('combined_mlp_drop_%d'%(i), False):
            combined_hidden = dropout_layer(combined_hidden, dropout, trng,
                                            drop_ratio)
        if i == options['combined_num_mlp'] - 1:
            combined_hidden = fflayer(shared_params, combined_hidden, options,
                                      prefix='combined_mlp_%d'%(i),
                                      act_func='linear')
        else:
            combined_hidden = fflayer(shared_params, combined_hidden, options,
                                      prefix='combined_mlp_%d'%(i),
                                      act_func=options.get('combined_mlp_act_%d'%(i),
                                                           'tanh'))

    # drop the image output
    # semantic_loss = T.sqrt(T.sum((image_feat_ave_2 - pool_feat) ** 2))
    prob = T.nnet.softmax(combined_hidden)
    #prob_y = prob[T.arange(prob.shape[0]), label]
    pred_label = T.argmax(prob, axis=1)
    # sum or mean?
    #cost = -T.mean(T.log(prob_y)) #+ 0.001*semantic_loss
    # cost = T.mean(T.nnet.categorical_crossentropy(prob, label))
    # accu = T.mean(T.eq(pred_label, label))
    # return image_feat, input_idx, input_mask, \
        # label, dropout, cost, accu
    return image_feat, input_idx, input_mask, dropout, pred_label

def test(options):
    # batch_size = options['batch_size']
    #
    #
    # max_epochs = options['max_epochs']

    ###############
    # build model #
    ###############
    # f = h5py.File('../datasets/vqa_data_prepro.h5')
    # question_id_test = f['ques_id_test'].value
    params = np.load('model_stack.npy')
    shared_params = params[np.newaxis][0]

    image_feat, input_idx, input_mask, \
    label, dropout, cost, accu, pred_label, \
        = build_model(shared_params, options)
    f_val = theano.function(inputs=[image_feat, input_idx, input_mask, label],
                            outputs=[cost, accu, pred_label],
                            on_unused_input='warn')

    # image_feat, input_idx, input_mask, dropout, pred_label = \
    #     test_model(shared_params, options)
    #
    #
    #
    # f_predict = theano.function(inputs=[image_feat, input_idx, input_mask],
    #                             outputs=pred_label,
    #                             on_unused_input='warn'
    #                             )
    test_batch_size = 107 * 4
    # train_batch = options['traindata'] // batch_size
    test_batch = options['testdata'] // test_batch_size
    dropout.set_value(numpy.float32(0.))
    test_cost_list = []
    test_accu_list = []
    results = []
    for index in range(test_batch):
        input_idx, input_mask, input_label, image_feat = \
            process_batch_test(index, test_batch_size)
        input_idx = input_idx.astype('int32')
        input_mask = input_mask.astype('float32')
        [cost, accu, pred_label] = f_val(image_feat, input_idx, input_mask,
                             input_label.astype('int32').flatten())
        for i in range(pred_label.shape[0]):
            answer = answers['%s' % (str(pred_label[i] + 1))]
            grth = answers['%s' % (str(input_label[i] + 1))]
            print '  ' + answer,
            print '  ' + grth
            results.append({'answer': str(answer), 'groundtruth': str(grth)})
        test_cost_list.append(cost)
        test_accu_list.append(accu)
    ave_test_cost = sum(test_cost_list) / float(test_batch)
    ave_test_accu = sum(test_accu_list) / float(test_batch)
    print ('testing cost: %f accu: %f' % (ave_test_cost, ave_test_accu))
    json.dump(results, open('cocoqa_test_ASC_results.json', 'w'))
    """
    dropout.set_value(numpy.float32(0.))
    results = []
    for i in range(options['testdata']):
        input_idx, input_mask, input_label, image_feat = \
            process_batch_test(i, 1)
        input_idx = input_idx.astype('int32')
        input_mask = input_mask.astype('float32')
        pred_label = f_predict(image_feat, input_idx, input_mask)
        # print pred_label
        # print pred_label.shape
        print i,
        for j in range(input_idx.shape[1]):
            if (input_idx[-1][j]) == 0:
                break
            print vocabulary['%s' % str(input_idx[-1][j])],
        answer = answers['%s' % (str(pred_label[0] + 1))]
        grth = answers['%s' % (str(input_label[0] + 1))]
        print '  ' + answer,
        print '  ' + grth
        results.append({'question_id': i, 'answer': str(answer), 'groundtruth':str(grth)})
    json.dump(results, open('cocoqa_test_ASC_results.json', 'w'))
    """
if __name__ == '__main__':
    test(options)
