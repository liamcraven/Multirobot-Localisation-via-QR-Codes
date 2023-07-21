from utils.constants import QR_VERSION_CAPACITY, EC_LEVELS
import lzma

from typing import List

"""
    This file contains utility functions for the QR code generator.
"""
    
# Bit buffer class for storing data in bits
class BitBuffer:
    # Constructor
    def __init__(self):
        self.buffer = []
        self.length = 0
        
    # Returns the length of the buffer in bytes
    def get_length(self):
        return self.length
    
    # Returns the buffer as a string
    def as_string(self):
        return ''.join(str(bit) for bit in self.buffer)
    
    # Appends a bit to the buffer
    def append_bit(self, bit):
        i = self.length // 8
        if len(self.buffer) <= i:
            self.buffer.append(0)
        if bit:
            self.buffer[i] |= (0x80 >> (self.length % 8))
        self.length += 1
        
    # Gets byte
    def get_byte(self, index: int):
        return self.buffer[index]
    
    # Appends a specified number of bits to the buffer
    def append_bits(self, num, length: int):
        for i in range(length):
            self.append_bit(((num >> (length - i - 1)) & 1) == 1)
            
    # Appends a 16-bit integer to the buffer
    def append_int(self, num: int):
        self.append_bits(num, 16)
    
    # Appends a 32-bit float to the buffer
    def append_float(self, num: float, decimal_places: int = 6):
        #We need to convert this to a fixed point number
        num = int(num * (10 ** decimal_places))
        self.append_bits(num, 32)
        
    
    
    # This function returns the index-th bit in the buffer   TODO: Check if this is correct 
    def get_bit(self, index: int):
        return ((self.buffer[index // 8] << (index % 8)) & 0x80) != 0
        
    # Removes and returns the first n bits from the buffer
    def take_bits(self, n: int):
        if n > self.length:
            raise ValueError("Cannot take more bits than there are in the buffer")
        result = 0
        for i in range(n):
            result = (result << 1) | self.get_bit(i)
        new_buffer = BitBuffer()
        new_buffer.append_bits(result, n)
        self.buffer = self.buffer[n // 8:]
        self.length -= n
        return new_buffer
    
    # Takes integer from the buffer
    def take_int(self):
        bytes = self.take_bits(16).get_bytes()
        return int.from_bytes(bytes, byteorder="big")
    
    # Takes float from the buffer
    def take_float(self, decimal_places: int = 6):
        bytes = self.take_bits(32).get_bytes()
        # Convert from fixed point to float
        return int.from_bytes(bytes, byteorder="big") / (10 ** decimal_places)
        
    
    # Returns the contents of the buffer as a byte array
    def get_bytes(self):
        #if self.length % 8 != 0: TODO: Check if this is correct
        #    raise ValueError(f"Buffer length must be a multiple of 8 but is {self.length%8}")
        return bytes(self.buffer)
    
    def get_codewords(self):
        return self.buffer
    
    #Clear the buffer
    def clear_buffer(self):
        self.buffer = []
        self.length = 0
        
    def set_format_info(self, num_blocks: int, data_codewords: int):
        # Ensure that the block format is a single byte
        if data_codewords > 255:
            raise Exception(f"Block format has more than 255 codewords, cannot encode into a single byte.")
        if num_blocks > 255:
            raise Exception(f"Block count has more than 255 blocks, cannot encode into a single byte.")
        self.buffer[0] = num_blocks & 0xFF
        self.buffer[1] = data_codewords & 0xFF
    
    # Append a byte to the front of the buffer
    def append_byte_to_front(self, byte: int):
        self.buffer.insert(0, byte)
        
    def export_for_qrcode(self):
        return bytes(self.buffer)
        
        
        
    def print_buffer(self):
        print(f"Buffer: {self.buffer}")
        
        
def get_min_version(data_lens: List[int]):
        """
            This function returns the minimum version required to encode the data
        
        print(f"Data length: {data_len}")
        for version in range(1, 41):
            if data_len <= (get_version_data_capacity(version, ecl) * 8):
                return version
        raise ValueError("Data is too large to be encoded in a QR code")
        """
        #Calculate total codewords needed
        total_bits = 0
        for i in range(len(data_lens)):
            EC_LEVEL = EC_LEVELS[i]
            recovery_capacity = EC_LEVEL.get_recovery_capacity()
            total_bits += data_lens[i] + int(recovery_capacity * data_lens[i])
        total_bytes = total_bits // 8
        for version in range(1, 41):
            if total_bytes <= QR_VERSION_CAPACITY[version] // 8:
                return version
        raise ValueError("Data is too large to be encoded in a QR code")
    
        
def get_version_data_capacity(version: int):
        """
            This function returns the maximum number of data codewords for a given version and error correction level
        """
        return QR_VERSION_CAPACITY[version]