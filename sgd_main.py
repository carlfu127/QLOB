import numpy
import logging
import math
from cnn_att import *
from read_pool5 import *

options = OrderedDict()

# dimensions
options['traindata'] = trainset_x.shape[0]
options['validdata'] = validset_x.shape[0]
options['testdata'] = testset_x.shape[0]
options['n_emb'] = 512
options['n_dim'] = 512
options['n_common_feat'] = 512
options['n_image_feat'] = 2048
options['n_attention'] = 512
options['n_words'] = len(vocabulary)
options['n_output'] = len(answers)

options['num_filter_unigram'] = 512
options['num_filter_bigram'] = 512
options['num_filter_trigram'] = 512


# structure options
options['combined_num_mlp'] = 1
options['combined_mlp_drop_0'] = True
options['combined_mlp_act_0'] = 'linear'
options['sent_drop'] = False
options['use_tanh'] = False

options['use_unigram_conv'] = True
options['use_bigram_conv'] = True
options['use_trigram_conv'] = True
options['use_attention_drop'] = False
options['use_before_attention_drop'] = False

# initialization
options['init_type'] = 'gaussian'
options['range'] = 0.01
options['std'] = 0.01
options['init_lstm_svd'] = True
options['use_tanh'] = True

options['forget_bias'] = numpy.float32(1.0)
# learning parameters
options['optimization'] = 'sgd' # choices
options['batch_size'] = 100
options['lr'] = numpy.float32(0.05)
options['w_emb_lr'] = numpy.float32(80)
options['momentum'] = numpy.float32(0.9)
options['gamma'] = 1
options['step'] = 10
options['step_start'] = 100
options['max_epochs'] = 30
options['weight_decay'] = 0.0005
options['decay_rate'] = numpy.float32(0.999)
options['drop_ratio'] = numpy.float32(0.5)
options['smooth'] = numpy.float32(1e-8)
options['grad_clip'] = numpy.float32(0.1)

# log params
options['disp_interval'] = 100
options['eval_interval'] = 500
options['save_interval'] = 1000
def get_lr(options, curr_epoch):
    if options['optimization'] == 'sgd':
        power = max((curr_epoch - options['step_start']) / options['step'], 0)
        power = math.ceil(power)
        return options['lr'] * (options['gamma'] ** power)  #
    else:
        return options['lr']

def train(options):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='image_attention.log')
    logger = logging.getLogger('root')
    logger.info(options)
    logger.info('start training')

    batch_size = options['batch_size']
    max_epochs = options['max_epochs']
    ###############
    # build model #
    ###############
    params = init_params(options)
    shared_params = init_shared_params(params)
    # model = np.load('model_stack.npy')
    # shared_params = model[np.newaxis][0]

    image_feat, input_idx, input_mask, \
    label, dropout, cost, accu, pred_label, \
        = build_model(shared_params, options)
    logger.info('finished building model')

    ####################
    # add weight decay #
    ####################
    weight_decay = theano.shared(numpy.float32(options['weight_decay']), name='weight_decay')
    reg_cost = 0

    for k in shared_params.iterkeys():
        if k != 'w_emb':
            reg_cost += (shared_params[k] ** 2).sum()

    reg_cost *= weight_decay
    reg_cost = cost + reg_cost

    ###############
    # # gradients #
    ###############
    grads = T.grad(reg_cost, wrt=shared_params.values())
    grad_buf = [theano.shared(p.get_value() * 0, name='%s_grad_buf' % k)
                for k, p in shared_params.iteritems()]
    # accumulate the gradients within one batch
    update_grad = [(g_b, g) for g_b, g in zip(grad_buf, grads)]
    # need to declare a share variable ??
    grad_clip = options['grad_clip']
    grad_norm = [T.sqrt(T.sum(g_b ** 2)) for g_b in grad_buf]
    update_clip = [(g_b, T.switch(T.gt(g_norm, grad_clip),
                                  g_b * grad_clip / g_norm, g_b))
                   for (g_norm, g_b) in zip(grad_norm, grad_buf)]

    # corresponding update function
    f_grad_clip = theano.function(inputs=[],
                                  updates=update_clip)
    f_output_grad_norm = theano.function(inputs=[],
                                         outputs=grad_norm)
    f_train = theano.function(inputs=[image_feat, input_idx, input_mask, label],
                              outputs=[cost, accu],
                              updates=update_grad,
                              on_unused_input='warn')
    # validation function no gradient updates
    f_val = theano.function(inputs=[image_feat, input_idx, input_mask, label],
                            outputs=[cost, accu],
                            on_unused_input='warn')

    f_grad_cache_update, f_param_update \
        = eval(options['optimization'])(shared_params, grad_buf, options)
    logger.info('finished building function')

    # calculate how many iterations we need
    test_batch_size = 107*4
    train_batch = options['traindata'] // batch_size
    test_batch = options['testdata'] // test_batch_size
    # max_iters = max_epochs * num_iters_one_epoch
    # eval_interval_in_iters = options['eval_interval']
    # disp_interval = options['disp_interval']

    best_val_accu = 0.0
    epoch = 0
    last_cost = np.inf

    while (epoch < max_epochs):
        epoch += 1
        current_cost = 0
        dropout.set_value(numpy.float32(1.))
        for index in range(train_batch + 1):
            if index == train_batch:
                input_idx, input_mask, input_label, image_feat = \
                    process_batch_train(index, options['traindata'] % batch_size)
            else:
                input_idx, input_mask, input_label, image_feat = \
                    process_batch_train(index, batch_size)
            input_idx = input_idx.astype('int32')
            input_mask = input_mask.astype('float32')
            [cost, accu] = f_train(image_feat, input_idx, input_mask,
                                   input_label.astype('int32').flatten())
            current_cost += cost
            f_grad_clip()
            f_grad_cache_update()
            lr_t = get_lr(options, epoch / float(train_batch))
            f_param_update(np.float32(lr_t))
            print ('epoch %d/%d batch : cost %f accu %f' % (epoch, index, cost, accu))
            # print prob[-1]
        print('last_cost : %f, current_cost : %f' % (last_cost, current_cost))
        logging.info('epoch:%d last_cost : %f, current_cost : %f' % (epoch, last_cost, current_cost))
        last_cost = current_cost

        dropout.set_value(numpy.float32(0.))
        test_cost_list = []
        test_accu_list = []
        for index in range(test_batch):
            input_idx, input_mask, input_label, image_feat = \
                process_batch_test(index, test_batch_size)
            input_idx = input_idx.astype('int32')
            input_mask = input_mask.astype('float32')
            [cost, accu] = f_val(image_feat, input_idx, input_mask,
                                 input_label.astype('int32').flatten())
            test_cost_list.append(cost)
            test_accu_list.append(accu)
        ave_test_cost = sum(test_cost_list) / float(test_batch)
        ave_test_accu = sum(test_accu_list) / float(test_batch)
        print ('testing cost: %f accu: %f' % (ave_test_cost, ave_test_accu))
        logging.info('epoch:%d testing cost: %f accu: %f' % (epoch, ave_test_cost, ave_test_accu))
        if ave_test_accu > best_val_accu:
            best_val_accu = ave_test_accu
            np.save('model_stack', shared_params)
    return best_val_accu
if __name__ == '__main__':
    print train(options)
