#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, random
import numpy as np
import scipy

import tensorflow as tf
import tensorlayer as tl
from model import SRGAN_g, SRGAN_d, Vgg19_simple_api
from utils import crop_sub_imgs_fn, downsample_fn

## This code is adapted from source code of SRGAN from https://github.com/tensorlayer/srgan
## Thanks for the idea from  "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" by Christian Ledig et al.
## Thanks for the source code from https://github.com/zsdonghao

def train(train_lr_path, train_hr_path, save_path, save_every_epoch=2, validation=True, ratio=0.9,
    batch_size=16, lr_init=1e-4, beta1=0.9, n_epoch_init=10,
    n_epoch=20, lr_decay=0.1):
    '''
    Parameters:
    data:
        train_lr_path/train_hr_path: path of data
        save_path: the parent folder to save model result
        validation: whether to split data into train set and validation set
        save_every_epoch: how frequent to save the checkpoints and sample images
    Adam: 
        batch_size
        lr_init
        beta1
    Generator Initialization
        n_epoch_init
    Adversarial Net
        n_epoch
        lr_decay
    '''

    ## Folders to save results
    save_dir_ginit = os.path.join(save_path, 'srgan_ginit')
    save_dir_gan = os.path.join(save_path, 'srgan_gan')
    checkpoint_dir = os.path.join(save_path, 'checkpoint')
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###======LOAD DATA======###
    train_lr_img_list = sorted(tl.files.load_file_list(path=train_lr_path, regx='.*.jpg', printable=False))
    train_hr_img_list = sorted(tl.files.load_file_list(path=train_hr_path, regx='.*.jpg', printable=False))
    
    if validation:
        idx = np.random.choice(len(train_lr_img_list), int(len(train_lr_img_list)*ratio), replace=False)
        valid_lr_img_list = sorted([x for i,x in enumerate(train_lr_img_list) if i not in idx])
        valid_hr_img_list = sorted([x for i,x in enumerate(train_hr_img_list) if i not in idx])
        train_lr_img_list = sorted([x for i,x in enumerate(train_lr_img_list) if i in idx])
        train_hr_img_list = sorted([x for i,x in enumerate(train_hr_img_list) if i in idx])

        valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=train_lr_path, n_threads=32)
        valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=train_hr_path, n_threads=32)

    train_lr_imgs = tl.vis.read_images(train_lr_img_list, path=train_lr_path, n_threads=32)
    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=train_hr_path, n_threads=32)


    
    ###======DEFINE MODEL======###
    ## train inference
    lr_image = tf.placeholder('float32', [None, 96, 96, 3], name='lr_image')
    hr_image = tf.placeholder('float32', [None, 192, 192, 3], name='hr_image')

    net_g = SRGAN_g(lr_image, is_train=True, reuse=False)
    net_d, logits_real = SRGAN_d(hr_image, is_train=True, reuse=False)
    _, logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

    # net_g.print_params(False)
    # net_g.print_layers()
    # net_d.print_params(False)
    # net_d.print_layers()

    ## resize original hr images for VGG19
    hr_image_224 = tf.image.resize_images(
        hr_image, size=[224, 224], method=0, # BICUBIC
        align_corners=False)

    ## generated hr image for VGG19
    generated_image_224 = tf.image.resize_images(
        net_g.outputs, size=[224, 224], method=0, #BICUBIC
        align_corners=False)
    
    ## scale image to [0,1] and get conv characteristics
    net_vgg, vgg_target_emb = Vgg19_simple_api((hr_image_224 + 1) / 2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((generated_image_224 + 1) / 2, reuse=True)

    ## test inference
    net_g_test = SRGAN_g(lr_image, is_train=False, reuse=True)

    ###======DEFINE TRAIN PROCESS======###
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    d_loss = d_loss1 + d_loss2

    prediction1 = tf.greater(logits_real, tf.fill(tf.shape(logits_real),0.5))
    acc_metric1 = tf.reduce_mean(tf.cast(prediction1, tf.float32))
    prediction2 = tf.less(logits_fake, tf.fill(tf.shape(logits_fake), 0.5))
    acc_metric2 = tf.reduce_mean(tf.cast(prediction2, tf.float32))
    acc_metric = acc_metric1 + acc_metric2


    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    mse_loss = tl.cost.mean_squared_error(net_g.outputs, hr_image, is_mean=True)
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    g_loss = mse_loss + g_gan_loss + vgg_loss

    psnr_metric = tf.image.psnr(net_g.outputs, hr_image, max_val=2.0, name='psnr')

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)

    ## Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)

    ## SRGAN
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    if tl.files.file_exists(os.path.join(checkpoint_dir, 'g_srgan.npz')):
        tl.files.load_and_assign_npz(sess=sess, name=os.path.join(checkpoint_dir, 'g_srgan.npz'), network=net_g)
    else:
        tl.files.load_and_assign_npz(sess=sess, name=os.path.join(checkpoint_dir, 'g_srgan_init.npz'), network=net_g)
    
    tl.files.load_and_assign_npz(sess=sess, name=os.path.join(checkpoint_dir, 'd_srgan.npz'), network=net_d)

    ###======LOAD VGG======###
    vgg19_npy_path = '../lib/SRGAN/vgg19.npy'
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # net_vgg.print_params(False)
    # net_vgg.print_layers()

    ###======TRAINING======###
    ## use train set to have a quick test during training
    ni = 4
    num_sample = ni*ni
    idx = np.random.choice(len(train_lr_imgs), num_sample, replace=False)
    sample_imgs_lr = tl.prepro.threading_data([img for i, img in enumerate(train_lr_imgs) if i in idx],
        fn=crop_sub_imgs_fn, size=(96, 96) ,is_random=False)
    sample_imgs_hr = tl.prepro.threading_data([img for i, img in enumerate(train_hr_imgs) if i in idx],
        fn=crop_sub_imgs_fn, size=(192, 192) ,is_random=False)

    print('sample LR sub-image:', sample_imgs_lr.shape, sample_imgs_lr.min(), sample_imgs_lr.max())
    print('sample HR sub-image:', sample_imgs_hr.shape, sample_imgs_hr.min(), sample_imgs_hr.max())
    
    ## save the images
    tl.vis.save_images(sample_imgs_lr, [ni, ni], os.path.join(save_dir_ginit, '_train_sample_96.jpg'))
    tl.vis.save_images(sample_imgs_hr, [ni, ni], os.path.join(save_dir_ginit, '_train_sample_192.jpg'))
    tl.vis.save_images(sample_imgs_lr, [ni, ni], os.path.join(save_dir_gan, '_train_sample_96.jpg'))
    tl.vis.save_images(sample_imgs_hr, [ni, ni], os.path.join(save_dir_gan, '_train_sample_192.jpg'))
    print('finish saving sample images')

    ###====== initialize G ======###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    for epoch in range(0, n_epoch_init + 1):
        epoch_time = time.time()
        total_mse_loss, total_psnr, n_iter = 0, 0, 0

        # random shuffle the train set for each epoch
        random.shuffle(train_hr_imgs)

        for idx in range(0, len(train_lr_imgs), batch_size):
            step_time = time.time()
            b_imgs_192 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, 
                                                    size=(192,192), is_random=True)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_192, fn=downsample_fn, size=(96,96))
            ## update G
            errM, metricP, _ = sess.run([mse_loss, psnr_metric, g_optim_init], {lr_image: b_imgs_96, hr_image: b_imgs_192})
            print("Epoch [%2d/%2d] %4d time: %4.2fs, mse: %.4f, psnr: %.4f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM, metricP.mean()))
            total_mse_loss += errM
            total_psnr += metricP.mean()
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.2fs, mse: %.4f, psnr: %.4f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter, total_psnr / n_iter)
        print(log)
        if validation:
            b_imgs_192_V = tl.prepro.threading_data(valid_hr_imgs, fn=crop_sub_imgs_fn, 
                                                    size=(192,192), is_random=True)
            b_imgs_96_V = tl.prepro.threading_data(b_imgs_192_V, fn=downsample_fn, size=(96,96))
            errM_V, metricP_V, _ = sess.run([mse_loss, psnr_metric, g_optim_init], {lr_image: b_imgs_96_V, hr_image: b_imgs_192_V})
            print("Validation | mse: %.4f, psnr: %.4f" % (errM_V, metricP_V.mean()))

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % save_every_epoch == 0):
            out = sess.run(net_g_test.outputs, {lr_image: sample_imgs_lr})
            print("[*] save sample images")
            tl.vis.save_images(out, [ni, ni], os.path.join(save_dir_ginit, 'train_{}.jpg'.format(epoch)))

        ## save model
        if (epoch != 0) and (epoch % save_every_epoch == 0):
            tl.files.save_npz(net_g.all_params, name=os.path.join(checkpoint_dir, 'g_srgan_init.npz'), sess=sess)

    ###========================= train GAN (SRGAN) =========================###
    ## Learning rate decay
    decay_every = int(n_epoch / 2) if int(n_epoch / 2) > 0 else 1

    for epoch in range(0, n_epoch + 1):

        # random shuffle the train set for each epoch
        random.shuffle(train_hr_imgs)

        ## update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, total_mse_loss, total_psnr, total_acc, n_iter = 0, 0, 0, 0, 0, 0

        for idx in range(0, len(train_lr_imgs), batch_size):
            step_time = time.time()
            b_imgs_192 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, 
                                                    size=(192,192), is_random=True)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_192, fn=downsample_fn, size=(96,96))
            ## update D
            errD, metricA, _ = sess.run([d_loss, acc_metric, d_optim], {lr_image: b_imgs_96, hr_image: b_imgs_192})
            ## update G
            errG, errM, metricP, _ = sess.run([g_loss, mse_loss, psnr_metric, g_optim], {lr_image: b_imgs_96, hr_image: b_imgs_192})
            print("Epoch [%2d/%2d] %4d time: %4.2fs, d_loss: %.4f g_loss: %.4f (mse: %.4f, psnr: %.4f, accuracy: %.4f)" %
                  (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, metricP.mean(), metricA / 2))
            total_d_loss += errD
            total_g_loss += errG
            total_mse_loss += errM
            total_psnr += metricP.mean()
            total_acc += metricA / 2
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.2fs, d_loss: %.4f g_loss: %.4f (mse: %4f, psnr: %.4f, accuracy: %.4f)" % (epoch, n_epoch, 
                            time.time() - epoch_time, total_d_loss / n_iter, total_g_loss / n_iter,
                            total_mse_loss / n_iter, total_psnr / n_iter, total_acc / n_iter)
        print(log)

        if validation:
            b_imgs_192_V = tl.prepro.threading_data(valid_hr_imgs, fn=crop_sub_imgs_fn, 
                                                    size=(192,192), is_random=True)
            b_imgs_96_V = tl.prepro.threading_data(b_imgs_192_V, fn=downsample_fn, size=(96,96))
            errM_V, metricP_V, _ = sess.run([mse_loss, psnr_metric, g_optim], {lr_image: b_imgs_96_V, hr_image: b_imgs_192_V})
            print("Validation | mse: %.4f, psnr: %.4f" % (errM_V, metricP_V.mean()))


        ## quick evaluation on train set
        if (epoch != 0) and (epoch % save_every_epoch == 0):
            out = sess.run(net_g_test.outputs, {lr_image: sample_imgs_lr})
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], os.path.join(save_dir_gan, 'train_{}.jpg'.format(epoch)))

        ## save model
        if (epoch != 0) and (epoch % save_every_epoch == 0):
            tl.files.save_npz(net_g.all_params, name=os.path.join(checkpoint_dir, 'g_srgan.npz'), sess=sess)
            tl.files.save_npz(net_d.all_params, name=os.path.join(checkpoint_dir, 'd_srgan.npz'), sess=sess)
    


def predict(test_lr_path, checkpoint_path, save_path):
    '''
    Parameters:
    data:
        test_lr_path: path of test data
        checkpoint_path: where to fetch weights
        save_path: where to save output
    '''
    ## create folders to save result images
    save_dir = os.path.join(save_path, 'test_gen')
    tl.files.exists_or_mkdir(save_dir)

    ###======PRE-LOAD DATA======###
    test_lr_img_list = sorted(tl.files.load_file_list(path=test_lr_path, regx='.*.jpg', printable=False))
    
    test_lr_imgs = tl.vis.read_images(test_lr_img_list, path=test_lr_path, n_threads=32)

    ###======DEFINE MODEL======###

    test_lr_imgs = [(img / 127.5)-1 for img in test_lr_imgs] # rescale to ［－1, 1]

    test_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

    net_g = SRGAN_g(test_image, is_train=False, reuse=False)

    ###======RESTORE G======###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    tl.files.load_and_assign_npz(sess=sess, name=os.path.join(checkpoint_path, 'g_srgan.npz'), network=net_g)

    ###======EVALUATION======###
    start_time = time.time()
    for i in range(len(test_lr_img_list)):
        img = test_lr_imgs[i]
        out = sess.run(net_g.outputs, {test_image: [img]})
        out = (out[0]+1)*127.5
        tl.vis.save_image(out.astype(np.uint8), os.path.join(save_dir, '{}'.format(test_lr_img_list[i])))
        if (i != 0) and (i % 10 == 0):
            print('saving %d images, ok' % i)

    print('take: %4.2fs' % (time.time() - start_time))


if __name__ == '__main__':
    pass