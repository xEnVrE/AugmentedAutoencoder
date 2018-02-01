# -*- coding: utf-8 -*-


from dataset import Dataset
from queue import Queue
from ae import AE
from encoder import Encoder
from decoder import Decoder
from codebook import Codebook

def build_dataset(dataset_path, args):
    dataset_args = { k:v for k,v in 
        args.items('Dataset') + 
        args.items('Paths') + 
        args.items('Augmentation')+ 
        args.items('Queue') +
        args.items('Embedding')}
    dataset = Dataset(dataset_path, **dataset_args)
    return dataset

def build_queue(dataset, args):
    NUM_THREADS = args.getint('Queue', 'NUM_THREADS')
    QUEUE_SIZE = args.getint('Queue', 'QUEUE_SIZE')
    BATCH_SIZE = args.getint('Training', 'BATCH_SIZE')
    queue = Queue(
        dataset, 
        NUM_THREADS, 
        QUEUE_SIZE, 
        BATCH_SIZE
    )
    return queue

def build_encoder(x, args):
    LATENT_SPACE_SIZE = args.getint('Network', 'LATENT_SPACE_SIZE')
    NUM_FILTER = eval(args.get('Network', 'NUM_FILTER'))
    KERNEL_SIZE_ENCODER = args.getint('Network', 'KERNEL_SIZE_ENCODER')
    STRIDES = eval(args.get('Network', 'STRIDES'))
    encoder = Encoder(
        x,
        LATENT_SPACE_SIZE, 
        NUM_FILTER, 
        KERNEL_SIZE_ENCODER, 
        STRIDES
    )
    return encoder

def build_decoder(reconstruction_target, encoder, args, is_training=True):
    NUM_FILTER = eval(args.get('Network', 'NUM_FILTER'))
    KERNEL_SIZE_DECODER = args.getint('Network', 'KERNEL_SIZE_DECODER')
    STRIDES = eval(args.get('Network', 'STRIDES'))
    LOSS = args.get('Network', 'LOSS')
    BOOTSTRAP_RATIO = args.getint('Network', 'BOOTSTRAP_RATIO')
    VARIATIONAL = args.getfloat('Network', 'VARIATIONAL') if is_training else False
    decoder = Decoder(
        reconstruction_target,
        encoder.sampled_z if VARIATIONAL else encoder.z,
        list( reversed(NUM_FILTER) ),
        KERNEL_SIZE_DECODER,
        list( reversed(STRIDES) ),
        LOSS,
        BOOTSTRAP_RATIO
    )
    return decoder

def build_ae(encoder, decoder, args):
    NORM_REGULARIZE = args.getfloat('Network', 'NORM_REGULARIZE')
    VARIATIONAL = args.getfloat('Network', 'VARIATIONAL')
    ae = AE(encoder, decoder, NORM_REGULARIZE, VARIATIONAL)
    return ae

def build_optimizer(ae, args):
    LEARNING_RATE = args.getfloat('Training', 'LEARNING_RATE')
    OPTIMIZER_NAME = args.get('Training', 'OPTIMIZER')
    import tensorflow
    optimizer = eval('tensorflow.train.{}Optimizer'.format(OPTIMIZER_NAME))
    optim = optimizer(LEARNING_RATE).minimize(
        ae.loss,
        global_step=ae.global_step
    )
    return optim

def build_codebook(encoder, dataset, args):
    embed_bb = args.getboolean('Embedding', 'EMBED_BB')
    codebook = Codebook(encoder, dataset, embed_bb)
    return codebook

def build_codebook_from_name(experiment_name, experiment_group='', return_dataset=False, return_decoder = False):
    import os
    import ConfigParser
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')

    if workspace_path == None:
        print 'Please define a workspace path:\n'
        print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
        exit(-1)

    import utils as u
    import tensorflow as tf

    log_dir = u.get_log_dir(workspace_path, experiment_name, experiment_group)
    checkpoint_file = u.get_checkpoint_basefilename(log_dir)
    cfg_file_path = u.get_config_file_path(workspace_path, experiment_name, experiment_group)
    dataset_path = u.get_dataset_path(workspace_path)

    if os.path.exists(cfg_file_path):
        args = ConfigParser.ConfigParser()
        args.read(cfg_file_path)
    else:
        print 'ERROR: Config File not found: ', cfg_file_path
        exit()

    with tf.variable_scope(experiment_name):
        dataset = build_dataset(dataset_path, args)
        x = tf.placeholder(tf.float32, [None,] + list(dataset.shape))
        encoder = build_encoder(x, args)
        codebook = build_codebook(encoder, dataset, args)
        if return_decoder:
            reconst_target = tf.placeholder(tf.float32, [None,] + list(dataset.shape))
            decoder = build_decoder(reconst_target, encoder, args, is_training=True)


    if return_dataset:
        if return_decoder:
            return codebook, dataset, decoder
        else:
            return codebook, dataset
    else:
        return codebook




def restore_checkpoint(session, saver, ckpt_dir):

    import tensorflow as tf

    chkpt = tf.train.get_checkpoint_state(ckpt_dir)

    if chkpt and chkpt.model_checkpoint_path:
        saver.restore(session, chkpt.model_checkpoint_path)
        