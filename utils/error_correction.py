from utils.constants import BlockFormat, ErrorCorrectionLevel
from utils.bit_stream import BitBuffer

import reedsolo


"""
    This module contains the functions and classes for error correction.
"""


class RSBlock:
    """
    Class for Reed-Solomon blocks.
    """
    def __init__(self, format: BlockFormat, ec_lvl: ErrorCorrectionLevel):
        self.format = format
        self.data = BitBuffer()
        
        self.data.append_bits(ec_lvl.get_format_bits(), 2)
    
    def add_data(self, data: BitBuffer):
        self.data = data
    
    # Adds the terminator and ensures that the block is separable into bytes
    def terminate(self):
        self.data.append_bits(0, min(4, (8 - self.data.get_length() % 8)))
        if self.data.get_length() % 8 != 0:
            self.data.append_bits(0, 8 - (self.data.get_length() % 8))
        
    # Generates the error correction codewords
    def generate_error_correction(self):
        rs = reedsolo.RSCodec(self.format.get_num_error_codewords())
        
        data_and_ec = rs.encode(self.data.get_bytes())
        new_data = BitBuffer()
        for byte in data_and_ec:
            new_data.append_bits(byte, 8)
        self.data = new_data
    
    # Adds alternating pad bytes until the block is full
    def pad(self):
        pad_bytes = [0xEC, 0x11]
        while len(self.data.get_bytes()) < self.format.get_total_codewords():
            self.data.append_bits(pad_bytes[len(self.data.get_bytes()) % 2], 8)
        
    
    def get_remaining_capacity(self):
        #Return the remaining capacity in bits
        return (self.format.get_num_data_codewords() * 8)- (self.data.get_length())
    
    def as_codewords(self):
        return self.data.get_codewords()