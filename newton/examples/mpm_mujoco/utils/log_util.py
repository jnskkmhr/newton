# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import List, Dict
import numpy as np
import pickle

class DictBenchmarkLogger:
    def __init__(self, 
                 log_dir:str, 
                 tag:str, 
                 num_envs:int, 
                 max_trials:int, 
                 max_episode_length:int, 
                 log_item: List[str]):
        
        self.log_dir = log_dir
        self.tag = tag
        self.num_envs = num_envs
        self.max_trials = max_trials
        self.max_episode_length = max_episode_length
        self.log_item = log_item
        
        self.item_log_dir = {}
        self.item_log_cache = {}
        self.item_log_buffer = {}
        
        for item in log_item:
            self.item_log_dir[item] = os.path.join(log_dir, tag, item)
            os.makedirs(self.item_log_dir[item], exist_ok=True)
            
            # temporary log cache
            self.item_log_cache[item] = [np.array([])]*num_envs
            
            # buffer that is saved as pkl
            # 2d list: (max_trials, num_envs)
            self.item_log_buffer[item] = [[np.array([]) for _ in range(self.num_envs)] for _ in range(max_trials)]
        
        # episode length 
        self.episode_length_log_dir = os.path.join(log_dir, tag, "episode")
        os.makedirs(self.episode_length_log_dir, exist_ok=True)
        self.episode_length_buffer = [[0 for _ in range(self.num_envs)] for _ in range(max_trials)]
        
    """
    Operations.
    """
    
    def log(self, item_dict: Dict[str, np.ndarray|None]):
        """
        item_dict: Dictionary containing the items to log.
        Each key should be one of the log items specified during initialization.
        The value should be a numpy array or None.
        """
        for idx in range(self.num_envs):
            for item, data in item_dict.items():
                if data is not None:
                    if len(data.shape) == 1:
                        data = data[:, None] # (num_envs, 1)
                        
                    if len(self.item_log_cache[item][idx]) == 0:
                        self.item_log_cache[item][idx] = np.expand_dims(data[idx], axis=0)
                    else:
                        self.item_log_cache[item][idx] = np.concatenate([self.item_log_cache[item][idx], np.expand_dims(data[idx], axis=0)], axis=0)
                        
    def save_to_buffer(self, trial_id:int=0, env_idx:int=0):
        for item in self.log_item:
            self.item_log_buffer[item][trial_id][env_idx] = self.fill_missing_data(self.item_log_cache[item][env_idx])
        self.reset_buffer(env_idx)
    
    def save_episode_length_to_buffer(self, trial_id:int=0, env_idx:int=0, episode_length:int=0):
        self.episode_length_buffer[trial_id][env_idx] = episode_length
        
    def save(self):
        for item in self.log_item:
            with open(os.path.join(self.item_log_dir[item], f"{item}.pkl"), "wb") as f:
                pickle.dump(self.item_log_buffer[item], f)
        
        # episode length
        with open(os.path.join(self.episode_length_log_dir, "episode_length.pkl"), "wb") as f:
            pickle.dump(self.episode_length_buffer, f)
            
    """ 
    Utilities.
    """
    
    def reset_buffer(self, env_idx:int=0):
        for item in self.log_item:
            self.item_log_cache[item][env_idx] = np.array([])
    
    def fill_missing_data(self, data):
        num_dim = len(data.shape)
        if num_dim==1:
            return np.concatenate([data, np.zeros(self.max_episode_length - len(data))], axis=0)
        elif num_dim==2:
            return np.concatenate([data, np.tile(np.zeros_like(data[0:1]), (self.max_episode_length - len(data), 1))], axis=0)
        elif num_dim==3:
            return np.concatenate([data, np.tile(np.zeros_like(data[0:1]), (self.max_episode_length - len(data), 1, 1))], axis=0)
        else:
            raise ValueError("Invalid data shape")