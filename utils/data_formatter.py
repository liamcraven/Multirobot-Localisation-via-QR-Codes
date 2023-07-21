import json
import numpy as np

from typing import List
from utils.bit_stream import BitBuffer

from base64 import b64encode, b64decode

class DataFormatter:
    
    def __init__(self, partition_method = "list_splitting", partition_count=3, window_size=3):
        self.partition_method = partition_method
        self.partition_count = partition_count
        self.window_size = window_size
        self.data = []
        
    def set_data(self, data):
        self.data = data
        
    def get_data_len(self):
        return len(self.data)
    
    # Function to serialize the data into a list of floats, whilst also returning the shape of the data
    def serialize_data(self):
        serialized_data, shape = traverse_data(self.data)
        self.data = flatten(serialized_data)
        return self.data
    
    # Function to split the data according to the partition method
    def split_data(self):
        if self.partition_method == "list_splitting":
            return split_list(self.data, self.partition_count)
        return split_floats(self.data)
    
    # Function converts each data partition into a bit stream
    def convert_to_bit_stream(self, datas) -> List[BitBuffer]:
        bit_streams = []
        for data in datas:
            bit_stream = BitBuffer()
            for d in data:
                if self.partition_method == "list_splitting":
                    bit_stream.append_float(d)
                else:
                    bit_stream.append_int(d)
            bit_streams.append(bit_stream)
        return bit_streams
    
    # Function converts each bit stream into a list of ints/floats
    def convert_from_bit_stream(self, bit_streams):
        datas = []
        for bit_stream in bit_streams:
            data = []
            while bit_stream.get_length() > 0:
                if self.partition_method == "list_splitting":
                    data.append(float(bit_stream.take_float()))
                else:
                    data.append(int(bit_stream.take_int()))
            datas.append(data)
        return datas
    
    # Function to recombine the data according to the partition method
    def recombine_data(self, data):
        if self.partition_method == "list_splitting":
            return recombine_lists(data)
        return recombine_floats(data[0], data[1], data[2])
    
    # Function to deserialize the data into a dictionary
    def deserialize_data(self, serialized_data):
        it = iter(serialized_data)
        
        def take(n):
            # Return next n items from the iterator
            return [next(it) for _ in range(n)]
        
        #Create nested dictionary
        data = {
            "linearization_points":[take(3) for _ in range(self.window_size)],
            "factor_to_variable_messages": {
                "message means": [[take(3), take(3)] for _ in range(self.window_size)],
                "message precisions": [[take(9), take(9)] for _ in range(self.window_size)]
            },
            "variable_to_factor_messages": {
                "message means": [[take(3), take(3)] for _ in range(self.window_size)],
                "message precisions": [[take(9), take(9)] for _ in range(self.window_size)]
            }
        }
        
        return data
         
""" 
    Helper functions
"""
# Recombines the lists into a single list
def recombine_lists(lsts):
    return [item for lst in lsts for item in lst]

# Recombines the floats into a single list
def recombine_floats(ints: List[int], msds: List[int], lsds: List[int], precision: int = 6) -> List[float]:
    data = []
    for int_part, msd, lsd in zip(ints, msds, lsds):
        msd_places = precision//2
        lsd_places = precision - msd_places
        
        # Reconstruction of the fraction part
        frac_part = msd / 10**msd_places + lsd / 10**(msd_places + lsd_places)
        
        f = int_part + frac_part
        
        data.append(f) 
    return data

#Splits the list into n parts
def split_list(lst, n):
    # First we split the list into n parts
    split_lst = np.array_split(lst, n)
    # Then we convert each part into a list
    split_lst = [list(x) for x in split_lst]
    return split_lst
    
        
# Function to split floats into integer, most significant decimals, and least significant decimals
def split_floats(data: List[float], precision: int = 6) -> List[List[int]]:
    """
    Args:
        data (list): List of floats to be split
    
    Returns:
        split_data (2D list): List of integers, most significant decimals, and least significant decimals
    """
    ints = []
    msds = []
    lsds = []
    for f in data:
        # Split float into integer and fractions parts
        int_part, frac_part = divmod(f, 1)
        
        # Convert fraction part to string with given precision
        frac_str = "{:.{}f}".format(frac_part, precision)
        
        # Split fraction part into most and least significant decimals
        msd = int(frac_str[2:(precision//2)+2])
        lsd = int(frac_str[(precision//2)+2:])
        
        # Append integer, most significant decimals, and least significant decimals to lists
        ints.append(int(int_part))
        msds.append(msd)
        lsds.append(lsd)
        
    # Return list of integers, most significant decimals, and least significant decimals
    return [ints, msds, lsds]
        
# Function to insert the shape of the data into the shape dictionary
def insert_shape(shape, path, index):
    sub_shape = shape
    for key in path[:-1]:
        sub_shape = sub_shape.setdefault(key, {})
    shape[path[-1]] = index

# Traverses the dictionary and returns data with its shape
def traverse_data(data, path=None, output=None, shape=None):
    if output is None:
        output = []
    if shape is None:
        shape = {}
    if path is None:
        path = []
        
    if isinstance(data, dict):
        for key, value in data.items():
            traverse_data(value, path + [key], output, shape)
    elif isinstance(data, list):
        shape_ = shape.copy()
        for i, v in enumerate(data):
            shape_ = shape_.setdefault(i, {})
            traverse_data(v, path + [i], output, shape_)
        else:
            output.append(data)
            insert_shape(shape, path, len(output) - 1)
    return output, shape

# Flattens output of serialize_data function
def flatten(lst):
    flat_lst = []
    for item in lst:
        if isinstance(item, list):
            flat_lst.extend(flatten(item))
        else:
            flat_lst.append(item)
    return flat_lst