import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

def read_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def write_pkl(data, pkl_path, new_name):
    original_name = os.path.basename(pkl_path)
    pkl_path = pkl_path.replace(original_name, new_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

def plot_data(data, img_name= 'data.png'):
    # depending on the agent type, plot the data in a different color
    agent_type = data.keys()
    
    # color depending on Agent type
    color = {
        'VEHICLE': 'blue',
        'PEDESTRIAN': 'red',
        'CYCLIST': 'green'
    }
    
    for agent in agent_type:
        plt.plot([], [], color[agent], label=agent)
        for i in range(len(data[agent])):
            x = data[agent][i, 0]
            y = data[agent][i, 1]
            plt.plot(x, y, color[agent], marker='o', markersize=3)
    
    plt.grid()
    plt.legend()
    plt.savefig(img_name)
    plt.close()
    
def plot_compare(data1, data2, img_name= 'data.png', extra_tag=''):
    # depending on the agent type, plot the data in a different color
    agent_type = data1.keys()
    agent2_type = data2.keys()
    
    # color depending on Agent type
    color = {
        'VEHICLE': 'blue',
        'PEDESTRIAN': 'red',
        'CYCLIST': 'green'
    }
    # slightly brigther colors
    color2 = {
        'TYPE_VEHICLE': 'skyblue',
        'TYPE_PEDESTRIAN': 'salmon',
        'TYPE_CYCLIST': 'lightgreen'
    }
    
    for agent1, agent2 in zip(agent_type, agent2_type):
        plt.plot([], [], color[agent1], label=agent1)
        for i in range(len(data1[agent1])):
            x = data1[agent1][i, 0]
            y = data1[agent1][i, 1]
            plt.plot(x, y, color[agent1], marker='o', markersize=3)
        
        for i in range(len(data2[agent2])):
            x = data2[agent2][i, 0]
            y = data2[agent2][i, 1]
            plt.plot(x, y, color2[agent2], marker='x', markersize=3)
        
        # plot per agent type
        if not '_' in img_name:
            img_name = img_name.replace('.png', f'_{agent1}_{extra_tag}.png')
        else:
            img_name = img_name.split('_')[0] + f'_{agent1}_{extra_tag}.png'
        
        img_path = os.path.join(img_name)
        plt.grid()
        plt.legend()
        plt.savefig(img_path)
        plt.close()
        

    
def rotate_points(data, angle):
    # Ensure points are numpy arrays with numerical data types
    angle_rad = np.radians(angle)
    rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                           [np.sin(angle_rad), np.cos(angle_rad)]])
    
    rotated_data = {}
    
    for agent_type, points in data.items():
        # Convert points to a numpy array with numerical data types
        points = np.array(points, dtype=np.float64)
        
        # Transpose points if necessary to align dimensions for dot product
        if points.shape[0] != 2 and points.shape[1] == 2:
            points = points.T
        
        # Perform matrix multiplication
        rotated_points = np.dot(rot_matrix, points)
        
        # Ensure the points are in the original shape (transpose if necessary)
        if rotated_points.shape[0] == 2 and rotated_points.shape[1] != 2:
            rotated_points = rotated_points.T
        
        rotated_data[agent_type] =  np.array(rotated_points)
    
    return rotated_data

# safe the data in the right format
def write_pkl(data, pkl_path):
    original_name = os.path.basename(pkl_path)
    pkl_path = pkl_path.replace(original_name, 'test.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

def check_data_type(data, pkl_path):
    # if data values are not numpy arrays, convert them
    for agent_type, points in data.items():
        if not isinstance(points, np.ndarray):
            data[agent_type] = np.array(points)
            write_pkl(data, pkl_path)

        

def main():
    # read the pkl file
    args = argparse.ArgumentParser()
    args.add_argument('--pkl_path', type=str, default=None)
    args.add_argument('--pkl_path2', type=str, default=None)
    args.add_argument('--extra_tag', type=str, default='')
    args = args.parse_args()
    pkl_path = args.pkl_path
    pkl_path2 = args.pkl_path2
    extra_tag = args.extra_tag
    data = read_pkl(pkl_path)
    data2 = read_pkl(pkl_path2)
    
    # check if data is in the right format
    check_data_type(data, pkl_path)
    check_data_type(data2, pkl_path2)
    
    
    
    plot_compare(data, data2, 'compare.png', extra_tag)
    
    # data_left = rotate_points(data, 45)
    # data_right = rotate_points(data, -45)
    # plot_data(data_left, 'left.png')
    # plot_data(data_right, 'right.png') 
    # plot_data(data, 'original.png')
    # write_pkl(data_left, pkl_path, 'cluster_64_left_dict.pkl')
    # write_pkl(data_right, pkl_path, 'cluster_64_right_dict.pkl')
    
    print('Done!')
    
if __name__ == '__main__':
    main()