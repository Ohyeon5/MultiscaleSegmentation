import argparse
import random
import tensorflow as tf
import os

from util import loader as ld
from util import model
from util import repoter as rp


def load_dataset(train_rate):
    loader = ld.Loader(dir_original="../../data/VOCdevkit/VOC2012/JPEGImages",
                       dir_segmented="../../data/VOCdevkit/VOC2012/SegmentationClass")
    return loader.load_train_test(train_rate=train_rate, shuffle=False)


def train(parser):
    # Load train and test datas
    print("Start training")
    train, test = load_dataset(train_rate=parser.trainrate)
    valid = train.perm(0, 30)
    test = test.perm(0, 150)

    # Create Reporter Object
    reporter = rp.Reporter(parser=parser)
    #accuracy_fig = reporter.create_figure("Accuracy", ("epoch", "accuracy"), ["train", "test"])
    #loss_fig = reporter.create_figure("Loss", ("epoch", "loss"), ["train", "test"])

    # Whether or not using a GPU
    gpu = parser.gpu

    # Create a model
    model_unet = model.UNet(l2_reg=parser.l2reg).model

    # Set a loss function and an optimizer
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_unet.teacher,
                                                                           logits=model_unet.outputs))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(model_unet.outputs, 3), tf.argmax(model_unet.teacher, 3))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Calculate mean iou
    labels = tf.reshape(model_unet.teacher, [tf.shape(model_unet.teacher)[0],-1])
    preds = tf.reshape(tf.clip_by_value(model_unet.outputs,0,10000), [tf.shape(model_unet.outputs)[0],-1])
    weights = tf.cast(tf.less_equal(preds, len(ld.DataSet.CATEGORY)-1),tf.int32) # Ignoring all labels greater than or equal to n_classes.
    #miou, update_op_miou = tf.metrics.mean_iou(labels = labels,
    #                                           predictions = preds,
    #                                           num_classes = len(ld.DataSet.CATEGORY),
    #                                           weights=weights)

    # Initialize session
    saver = tf.train.Saver()
    gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7), device_count={'GPU': 1},
                                log_device_placement=False, allow_soft_placement=True)
    sess = tf.InteractiveSession(config=gpu_config) if gpu else tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    # Train the model
    epochs = parser.epoch
    batch_size = parser.batchsize
    is_augment = parser.augmentation
    train_dict = {model_unet.inputs: valid.images_original, model_unet.teacher: valid.images_segmented,
                  model_unet.is_training: False}
    test_dict = {model_unet.inputs: test.images_original, model_unet.teacher: test.images_segmented,
                 model_unet.is_training: False}

    for epoch in range(epochs):
        for batch in train(batch_size=batch_size, augment=is_augment):
            # input images
            inputs = batch.images_original
            teacher = batch.images_segmented
            # Training
            sess.run(train_step, feed_dict={model_unet.inputs: inputs, model_unet.teacher: teacher,
                                            model_unet.is_training: True})

        # Evaluation
        if epoch % 1 == 0:
            loss_train = sess.run(cross_entropy, feed_dict=train_dict)
            loss_test = sess.run(cross_entropy, feed_dict=test_dict)
            accuracy_train = sess.run(accuracy, feed_dict=train_dict)
            accuracy_test = sess.run(accuracy, feed_dict=test_dict)
            #sess.run(update_op_miou, feed_dict=train_dict)
            #sess.run(update_op_miou, feed_dict=test_dict)
            #miou_train = sess.run(miou, feed_dict=train_dict)
            #miou_test = sess.run(miou, feed_dict=test_dict)
            print("Epoch:", epoch)
            print("[Train] Loss:", loss_train, " Accuracy:", accuracy_train)
            print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)
            #accuracy_fig.add([accuracy_train, accuracy_test], is_update=True)
            #loss_fig.add([loss_train, loss_test], is_update=True)
            if epoch % 3 == 0:
                saver.save(sess,os.path.join(reporter._result_dir,'model'))
                idx_train = random.randrange(10)
                idx_test = random.randrange(100)
                outputs_train = sess.run(model_unet.outputs,
                                         feed_dict={model_unet.inputs: [train.images_original[idx_train]],
                                                    model_unet.is_training: False})
                outputs_test = sess.run(model_unet.outputs,
                                        feed_dict={model_unet.inputs: [test.images_original[idx_test]],
                                                   model_unet.is_training: False})
                train_set = [train.images_original[idx_train], outputs_train[0], train.images_segmented[idx_train]]
                test_set = [test.images_original[idx_test], outputs_test[0], test.images_segmented[idx_test]]
                reporter.save_image_from_ndarray(train_set, test_set, train.palette, epoch,
                                                 index_void=len(ld.DataSet.CATEGORY)-1)

    # Test the trained model
    loss_test = sess.run(cross_entropy, feed_dict=test_dict)
    accuracy_test = sess.run(accuracy, feed_dict=test_dict)
    # sess.run(update_op_miou, feed_dict=test_dict)
    # miou_test = sess.run(miou, feed_dict=test_dict)
    print("Result")
    print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)
    save_path = saver.save(sess,os.path.join(reporter._result_dir,'model'))
    print("Model saved in file: ", save_path)
    for ii in range(100):
        outputs_test = sess.run(model_unet.outputs,
                                feed_dict={model_unet.inputs: [test.images_original[ii]],
                                          model_unet.is_training: False})
        test_set = [test.images_original[ii], outputs_test[0], test.images_segmented[ii], test.filenames[ii]]
        reporter.save_image_from_ndarray([], test_set, test.palette, 1000000,
                                         index_void=len(ld.DataSet.CATEGORY)-1,fnames=test.filenames[ii])



def test(parser):
    # load test dataset
    _, test = load_dataset(train_rate=parser.trainrate)
    test = test.perm(0, 150)

    # Create Reporter Object
    reporter = rp.Reporter(parser=parser)

    # Whether or not using a GPU
    gpu = parser.gpu

    # Create a model
    model_unet = model.UNet(l2_reg=parser.l2reg).model

    # Import the graph from the file
    saver = tf.train.import_meta_graph(os.path.join(parser.saverpath,'model.meta'))

    # Set a loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_unet.teacher,
                                                                           logits=model_unet.outputs))

    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(model_unet.outputs, 3), tf.argmax(model_unet.teacher, 3))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Calculate mean iou
    labels = tf.reshape(model_unet.teacher, [tf.shape(model_unet.teacher)[0],-1])
    preds = tf.reshape(tf.clip_by_value(model_unet.outputs,0,10000), [tf.shape(model_unet.outputs)[0],-1])
    weights = tf.cast(tf.less_equal(preds, len(ld.DataSet.CATEGORY)-1),tf.int32) # Ignoring all labels greater than or equal to n_classes.
    # miou, update_op_miou = tf.metrics.mean_iou(labels = labels,
    #                                            predictions = preds,
    #                                            num_classes = len(ld.DataSet.CATEGORY),
    #                                            weights=weights)

    # Initialize session
    gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7), device_count={'GPU': 1},
                                log_device_placement=False, allow_soft_placement=True)
    sess = tf.InteractiveSession(config=gpu_config) if gpu else tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    saver.restore(sess,tf.train.latest_checkpoint(parser.saverpath))

    # Set up the test dataset
    test_dict = {model_unet.inputs: test.images_original, model_unet.teacher: test.images_segmented,
                 model_unet.is_training: False}

    # Test the trained model
    loss_test = sess.run(cross_entropy, feed_dict=test_dict)
    accuracy_test = sess.run(accuracy, feed_dict=test_dict)
    # sess.run(update_op_miou, feed_dict=test_dict)
    # miou_test = sess.run(miou, feed_dict=test_dict)
    print("TEST Result")
    print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)

    for ii in range(100):
        outputs_test = sess.run(model_unet.outputs,
                                feed_dict={model_unet.inputs: [test.images_original[ii]],
                                          model_unet.is_training: False})
        test_set = [test.images_original[ii], outputs_test[0], test.images_segmented[ii], test.filenames[ii]]
        reporter.save_image_from_ndarray([], test_set, test.palette, 1000000,
                                         index_void=len(ld.DataSet.CATEGORY)-1,fnames=test.filenames[ii])


def debug(parser):
    # load test dataset
    _, test = load_dataset(train_rate=parser.trainrate)
    test = test.perm(0, 150)

    saver = tf.train.import_meta_graph(os.path.join(parser.saverpath,'model.meta'))

    # Create Reporter Object
    reporter = rp.Reporter(result_dir=parser.saverpath,parser=parser)
    # Whether or not using a GPU
    gpu = parser.gpu

    # Create a model
    model_unet = model.UNet(l2_reg=parser.l2reg).model

    # Set a loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_unet.teacher,
                                                                           logits=model_unet.outputs))

    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(model_unet.outputs, 3), tf.argmax(model_unet.teacher, 3))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Calculate mean iou
    labels = tf.reshape(model_unet.teacher, [tf.shape(model_unet.teacher)[0],-1])
    preds = tf.reshape(tf.clip_by_value(model_unet.outputs,0,10000), [tf.shape(model_unet.outputs)[0],-1])
    weights = tf.cast(tf.less_equal(preds, len(ld.DataSet.CATEGORY)-1),tf.int32) # Ignoring all labels greater than or equal to n_classes.
    miou, update_op_miou = tf.metrics.mean_iou(labels = labels,
                                               predictions = preds,
                                               num_classes = len(ld.DataSet.CATEGORY),
                                               weights=weights)
    # shape of each
    shape_teacher = tf.shape(model_unet.teacher)
    shape_output = tf.shape(model_unet.outputs)

    # Initialize session

    gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7), device_count={'GPU': 1},
                                log_device_placement=False, allow_soft_placement=True)
    sess = tf.InteractiveSession(config=gpu_config) if gpu else tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    saver.restore(sess,tf.train.latest_checkpoint(parser.saverpath))
    print("Saver restore model variables from ", tf.train.latest_checkpoint(parser.saverpath) )

    # Set up the test dataset
    test_dict = {model_unet.inputs: test.images_original[0:2], model_unet.teacher: test.images_segmented[0:2],
                 model_unet.is_training: False}

    # Test the trained model
    loss_test = sess.run(cross_entropy, feed_dict=test_dict)
    accuracy_test = sess.run(accuracy, feed_dict=test_dict)
    sess.run(update_op_miou, feed_dict=test_dict)
    miou_test = sess.run(miou, feed_dict=test_dict)
    shapes_test1 = sess.run(shape_teacher, feed_dict=test_dict)
    shapes_test2 = sess.run(shape_output, feed_dict=test_dict)
    print("TEST Result")
    print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test," Mean IOU:", miou_test, " Shape", shapes_test1, shapes_test2, len(ld.DataSet.CATEGORY))

def test_saver(parser):
    # delete the current graph
    tf.reset_default_graph()

    # import the graph from the file
    saver = tf.train.import_meta_graph(os.path.join(parser.saverpath,'model.meta'))
    print("Recall from ", parser.saverpath )

    graph = tf.get_default_graph()
    # model_unet = graph.get_tensor_by_name("model_unet:0")

    # list all the tensors in the graph
    for tensor in graph.get_operations():
        print(tensor.name)



def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python main.py',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )

    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPUs')
    parser.add_argument('-e', '--epoch', type=int, default=250, help='Number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=32, help='Batch size')
    parser.add_argument('-t', '--trainrate', type=float, default=0.85, help='Training rate')
    parser.add_argument('-a', '--augmentation', action='store_true', help='Number of epochs')
    parser.add_argument('-r', '--l2reg', type=float, default=0.0001, help='L2 regularization')
    # find latest saved results in ./result folder
    latest_result_dir = get_latestResultDir()
    parser.add_argument('-p', '--saverpath', type=str, default=latest_result_dir, help='initialize saved model path')

    return parser

def get_latestResultDir():
    from glob import glob
    # get the latest folder from ./result
    all_result_dirs = glob("./result/*/")
    all_result_dirs.sort()
    return all_result_dirs[-1]

if __name__ == '__main__':
    parser = get_parser().parse_args()
    train(parser)
    test(parser)
    # debug(parser)
