#!/usr/bin/env python

import time
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torch.autograd import Variable
from robot import Robot
from trainer import Trainer
from logger import Logger
import utils

def main(args):

    # --------------- Setup options ---------------
    random_seed = args.random_seed
    force_cpu = args.force_cpu

    # ------------- Algorithm options -------------
    method = args.method # 'reactive' (supervised learning) or 'reinforcement' (reinforcement learning ie Q-learning)
    push_rewards = args.push_rewards if method == 'reinforcement' else None  # Use immediate rewards (from change detection) for pushing?
    future_reward_discount = args.future_reward_discount
    experience_replay = args.experience_replay # Use prioritized experience replay?

    # -------------- Testing options --------------
    is_testing = args.is_testing

    # ------ Pre-loading and logging options ------
    load_snapshot = args.load_snapshot # Load pre-trained snapshot of model?
    snapshot_file = os.path.abspath(args.snapshot_file)  if load_snapshot else None
    continue_logging = args.continue_logging # Continue logging from previous session
    logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('logs')

    # Set random seed
    np.random.seed(random_seed)

    # Initialize trainer
    trainer = Trainer(method, push_rewards, future_reward_discount,
                      is_testing, load_snapshot, snapshot_file, force_cpu)

    # Initialize data logger
    logger = Logger(continue_logging, logging_directory)

    # Find last executed iteration of pre-loaded log, and load execution info and RL variables
    if continue_logging:
        trainer.preload(logger.transitions_directory)

    sample_primitive_action_id = 1 # 'grasp'
    # Get samples of the same primitive but with different results
    sample_ind_failure = np.argwhere(np.logical_and(np.asarray(trainer.reward_value_log)[1:trainer.iteration,0] == 0, np.asarray(trainer.executed_action_log)[1:trainer.iteration,0] == sample_primitive_action_id))
    sample_ind_success = np.argwhere(np.logical_and(np.asarray(trainer.reward_value_log)[1:trainer.iteration,0] == 1, np.asarray(trainer.executed_action_log)[1:trainer.iteration,0] == sample_primitive_action_id))

    sample_ind_failure = np.squeeze(sample_ind_failure)
    sample_ind_success = np.squeeze(sample_ind_success)

    # Start main training/testing loop
    training_iteration = 0
    while True:
        print('\n training iteration: %d' % (training_iteration))
        iteration_time_0 = time.time()

        # Do sampling for experience replay
        batch_size = 200
        random_batch_failure = np.random.choice(sample_ind_failure, batch_size, replace=False)
        random_batch_success = np.random.choice(sample_ind_success, batch_size, replace=False)
        trainer.backprop_batch(logger, trainer, random_batch_failure, random_batch_success)

        # Visualize executed primitive, and affordances
        if args.save_visualizations:
            # test on a random sample sample
            sample_iteration = np.random.randint(trainer.iteration)
            color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration)))
            color_heightmap = cv2.cvtColor(color_heightmap, cv2.COLOR_BGR2RGB)
            depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration)), -1)
            depth_heightmap = depth_heightmap.astype(np.float32)/100000
            best_pix_ind = (np.asarray(trainer.executed_action_log)[sample_iteration,1:4]).astype(int)
            object_mass = trainer.objectmass_log[sample_iteration][0]
            print('Quick test on the mass value: %f' % (object_mass))

            # get visualization and print it to disk
            push_predictions, grasp_predictions, state_feat = trainer.forward(color_heightmap, depth_heightmap, is_volatile=True, object_mass = object_mass)
            grasp_pred_vis = trainer.get_prediction_vis(grasp_predictions, color_heightmap, best_pix_ind)
            # logger.save_visualizations(trainer.iteration, grasp_pred_vis, 'grasp')
            cv2.imwrite(('visualization.grasp.%d.png' %(training_iteration)), grasp_pred_vis)
            time.sleep(1)

        # Save model snapshot
        # if not is_testing:
        #     if training_iteration % 500 == 0:
        #         logger.save_model(training_iteration, trainer.model, method)
        #         if trainer.use_cuda:
        #             trainer.model = trainer.model.cuda()

        iteration_time_1 = time.time()
        training_iteration = training_iteration + 1
        print('Time elapsed: %f' % (iteration_time_1-iteration_time_0))


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=False,                                    help='run in simulation?')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/blocks',                  help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=10,                                help='number of objects to add to simulation')
    parser.add_argument('--tcp_host_ip', dest='tcp_host_ip', action='store', default='100.127.7.223',                     help='IP address to robot arm as TCP client (UR5)')
    parser.add_argument('--tcp_port', dest='tcp_port', type=int, action='store', default=30002,                           help='port to robot arm as TCP client (UR5)')
    parser.add_argument('--rtc_host_ip', dest='rtc_host_ip', action='store', default='100.127.7.223',                     help='IP address to robot arm as real-time client (UR5)')
    parser.add_argument('--rtc_port', dest='rtc_port', type=int, action='store', default=30003,                           help='port to robot arm as real-time client (UR5)')
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store', default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234,                      help='random seed for simulation and neural net initialization')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,                                    help='force code to run in CPU mode')

    # ------------- Algorithm options -------------
    parser.add_argument('--method', dest='method', action='store', default='reinforcement',                               help='set to \'reactive\' (supervised learning) or \'reinforcement\' (reinforcement learning ie Q-learning)')
    parser.add_argument('--push_rewards', dest='push_rewards', action='store_true', default=False,                        help='use immediate rewards (from change detection) for pushing?')
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store', default=0.5)
    parser.add_argument('--experience_replay', dest='experience_replay', action='store_true', default=False,              help='use prioritized experience replay?')
    parser.add_argument('--heuristic_bootstrap', dest='heuristic_bootstrap', action='store_true', default=False,          help='use handcrafted grasping algorithm when grasping fails too many times in a row during training?')
    parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=False)
    parser.add_argument('--grasp_only', dest='grasp_only', action='store_true', default=False)

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=30,                help='maximum number of test runs per case/scenario')
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default='test-10-obj-01.txt')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False,                      help='load pre-trained snapshot of model?')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,                help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=False,          help='save visualizations of FCN predictions?')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
