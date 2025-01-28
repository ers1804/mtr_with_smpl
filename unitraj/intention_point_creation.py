import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed, find_latest_checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
import hydra
from omegaconf import OmegaConf
import os
from sklearn.cluster import KMeans
import math
import pickle
from enum import Enum#
import numpy as np

class VRUType(Enum):
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3

def dataloader(args, cfg):
    dist_train = False
    args.without_sync_bn = True
    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs
    if args.fix_random_seed:
        common_utils.set_random_seed(666)
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    # log to file
    logger.info('**********************Start logging**********************')
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        add_worker_init_fn=args.add_worker_init_fn,
    )
    return train_set, train_loader, train_sampler
# Define batch processing function
def process_batch(data, last_valid_positions, only_pred=True, extra_tag=''):
    # Define category mappings
    CATEGORY_MAPPING = {
        (1, 0, 0): '1', #VEHICLE
        (0, 1, 0): '2', #PEDESTRIAN
        (0, 0, 1): '3' #CYCLIST
    }
    
    dataloader_iter = iter(data)
    total_batches = len(data)
    
    print('Total number of batches:', total_batches)
    
    # Set batch save interval
    batch_save_interval = 100  # Save every 100 batches
    intermediate_file = os.path.join('/home/erik/ssd2', f'intermediate_results_{extra_tag}.pkl')
    # Remove the old intermediate file if it exists
    if os.path.exists(intermediate_file):
        os.remove(intermediate_file)
    # Process the data in batches
    for i in range(total_batches):
        data = next(dataloader_iter)
        if only_pred:
            # for only predicted objects
            pred = data['input_dict']['center_gt_trajs']
            last_position_index = data['input_dict']['center_gt_final_valid_idx']
            type = data['input_dict']['center_objects_type']
            
            accepted_indices = [59]
            
            for center_obj_idx in range(pred.shape[0]):
                last_index = int(last_position_index[center_obj_idx])
                if last_index in accepted_indices:
                # if last_index >= 0:
                    category = type[center_obj_idx]
                    pos = pred[center_obj_idx, last_index, :2].cpu().numpy()
                    if np.isnan(pos).any():
                        continue
                    else:
                        last_valid_positions[VRUType(category).name].append(pos)
                    
        elif not only_pred:
            # for all objects in the batch
            traj = data['input_dict']['obj_trajs_future_state']
            mask = data['input_dict']['obj_trajs_future_mask']
            obj_traj = data['input_dict']['obj_trajs']
            valid_last_indices = mask.sum(dim=-1) - 1
            valid_last_indices = valid_last_indices.long()
            category_indices = obj_traj[:, :, 0, 6:9]
            for center_obj_idx in range(traj.shape[0]):
                for obj_idx in range(traj.shape[1]):
                    last_index = valid_last_indices[center_obj_idx, obj_idx]
                    if last_index >= 0:
                        category_one_hot = tuple(category_indices[center_obj_idx, obj_idx].cpu().numpy())
                        category = CATEGORY_MAPPING.get(category_one_hot, 'unknown')
                        if category in last_valid_positions:
                            last_valid_positions[category].append(traj[center_obj_idx, obj_idx, last_index, :2].cpu().numpy())
        if i % batch_save_interval == 0 and i > 0:
            print(f'Processing batch {i}/{total_batches}')
            # Save intermediate results
            save_intermediate_results(last_valid_positions, intermediate_file)
            # Clear the dictionary and free memory
            del last_valid_positions
            last_valid_positions = {'VEHICLE': [], 'PEDESTRIAN': [], 'CYCLIST': []}
    # Save the final batch of results
    save_intermediate_results(last_valid_positions, intermediate_file)
    
    return last_valid_positions, intermediate_file
# Define a function to save intermediate results
def save_intermediate_results(data, filename):
    with open(filename, 'ab') as f:
        pickle.dump(data, f)
        
# Load all intermediate results
def load_intermediate_results(filename):
    data = {'VEHICLE': [], 'PEDESTRIAN': [], 'CYCLIST': []}
    with open(filename, 'rb') as f:
        while True:
            try:
                batch_data = pickle.load(f)
                for key in data:
                    data[key].extend(batch_data[key])
            except EOFError:
                break
    return data
# Define the K-means clustering function
def perform_kmeans_clustering(states, num_centroids):
    if len(states) == 0:
        return []
    kmeans = KMeans(n_clusters=num_centroids, random_state=0).fit(states)
    return kmeans.cluster_centers_
# Define the all clustering function
def perform_clustering(last_valid_positions):
    centroids = {'VEHICLE': [], 'PEDESTRIAN': [], 'CYCLIST': []}
    for category, positions in last_valid_positions.items():
        if len(positions) > 0:
            positions_array = torch.tensor(positions).numpy()  # Convert to numpy array for K-means
            centroids[category] = perform_kmeans_clustering(positions_array, 64)
        
    return centroids
# Get degree from x-axis
def get_degree_from_x_axis(x, y):
    return math.degrees(math.atan2(y, x))
# Define constrains for the K-means clustering
def filter_states(last_valid_states):
    # Use the last valid positions to create the centroids but differentiate additionally if the object is left, right or straight compared to the ego vehicle
    # left: x < 0, right: x > 0, straight: +/- 25° from the x-axis
    
    filtered_state_left = {'TYPE_VEHICLE': [], 'TYPE_PEDESTRIAN': [], 'TYPE_CYCLIST': []}
    filtered_state_right = {'TYPE_VEHICLE': [], 'TYPE_PEDESTRIAN': [], 'TYPE_CYCLIST': []}
    filtered_state_straight = {'TYPE_VEHICLE': [], 'TYPE_PEDESTRIAN': [], 'TYPE_CYCLIST': []}
    
    for category, position in last_valid_states.items():
        if category == 'TYPE_VEHICLE':
            for i, pos in enumerate(position):
                if pos[1] < 0:
                    filtered_state_left[category].append(pos)
                if pos[1] > 0:
                    filtered_state_right[category].append(pos)
                if abs(get_degree_from_x_axis(pos[0], pos[1])) < 25: # and not np.array_equal(pos, [0, 0]):
                    filtered_state_straight[category].append(pos)
        # add other categories without filtering          
        elif category == 'TYPE_PEDESTRIAN' or category == 'TYPE_CYCLIST':
            filtered_state_left[category] = position
            filtered_state_right[category] = position
            filtered_state_straight[category] = position
                    
    return filtered_state_left, filtered_state_right, filtered_state_straight

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    train_set = build_dataset(cfg)

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices) // train_set.data_chunk_size, 1)

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,
        collate_fn=train_set.collate_fn)

    extra_tag = ''
    
    save_path = f'/home/erik/gitprojects/UniTraj/unitraj/models/mtr/last_valid_pos_ecp.pkl'
    save_path_centroids = f'/home/erik/gitprojects/UniTraj/unitraj/models/mtr/cluster_64_center_dict_ecp.pkl'
    
    # Create an empty dictionary to store the last valid states categorized by type
    last_valid_positions = {
        'VEHICLE': [], #VEHICLE
        'PEDESTRIAN': [], #PEDESTRIAN
        'CYCLIST': [] #CYCLIST
    }
    
    ############################################################################
    
    # Process the data in batches
    last_valid_positions, intermediate_file = process_batch(train_loader, last_valid_positions, True, extra_tag)
    # Load all intermediate results
    last_valid_positions = load_intermediate_results(intermediate_file)
    # Print the number of objects in each category
    for category, positions in last_valid_positions.items():
        print(f'{category}: {len(positions)}')
    # Save the last valid positions to a file
    with open(save_path, 'wb') as f:
        pickle.dump(last_valid_positions, f)
    print(f'Last valid positions saved to {save_path}')
    
    
    # Cluster the last valid positions to be used as initial centroids for K-means clustering
    centroids = perform_clustering(last_valid_positions)
    
    
    ############################################################################
    
    # Save the centroids to a file
    with open(save_path_centroids, 'wb') as f:
        pickle.dump(centroids, f)
    print(f'Centroids saved to {save_path_centroids}')
    ############################################################################
    
    
    print('Done!')
if __name__ == '__main__':
    main()