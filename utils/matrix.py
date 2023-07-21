import numpy as np
import re
from utils.constants import FINDER_PATTERN, ALIGNMENT_PATTERN, ALIGNMENT_PATTERN_TABLE, MASK_PATTERNS
from utils.bit_stream import BitBuffer
from PIL import Image
"""
    This file contains the matrix class, responsible for storing the QR code data in matrix form.
"""

class Matrix:
    
    def __init__(self, version):
        self.version = version
        self.size = 21 + (version - 1) * 4
        self.matrix = np.full((self.size, self.size), None)
        self.is_restricted_module = np.full((self.size, self.size), False)
        self.mask = None
        self.penalty_score = None
        
        
        self.add_finder_patterns()
        self.add_separators()
        self.add_reserved_areas()
        self.add_timing_patterns()
        self.add_dark_module()
        self.add_alignment_patterns()
        
    # We need to add the finder patterns to the matrix
    def add_finder_patterns(self):
        patterns = [
            (0, 0), # Top left
            (0, self.size - 7), # Top right
            (self.size - 7, 0) # Bottom left
        ]
        for x, y in patterns:
            self.add_pattern((x, x + 7), (y, y + 7), FINDER_PATTERN)
    
    # We need to add sepatators to the matrix          
    def add_separators(self):
        for i in range(8):
            self.matrix[i, 7] = self.matrix[7, i] = self.matrix[i, self.size - 8] = self.matrix[self.size - 8, i] = 0
            self.is_restricted_module[i, 7] = self.is_restricted_module[7, i] = self.is_restricted_module[i, self.size - 8] = self.is_restricted_module[self.size - 8, i] = True
            self.matrix[self.size - i - 1, 7] = self.matrix[7, self.size - i - 1] = self.matrix[self.size - i - 1, self.size - 8] = self.matrix[self.size - 8, self.size - i - 1] = 0
            self.is_restricted_module[self.size - i - 1, 7] = self.is_restricted_module[7, self.size - i - 1] = self.is_restricted_module[self.size - i - 1, self.size - 8] = self.is_restricted_module[self.size - 8, self.size - i - 1] = True
            
    # We need to reserve the areas for the format and version information  
    def add_reserved_areas(self):
        self.matrix[8, :6] = self.matrix[8, self.size - 8:self.size] = 0
        self.is_restricted_module[8, :6] = self.is_restricted_module[8, self.size - 8:self.size] = True
        self.matrix[:6, 8] = self.matrix[self.size - 8:self.size, 8] = 0
        self.is_restricted_module[:6, 8] = self.is_restricted_module[self.size - 8:self.size, 8] = True
        if self.version >= 7:
            self.matrix[self.size - 11:self.size - 8, 0:6] = self.matrix[0:6, self.size - 11:self.size - 8] = 0
            self.is_restricted_module[self.size - 11:self.size - 8, 0:6] = self.is_restricted_module[0:6, self.size - 11:self.size - 8] = True

            
    # We need to add the timing patterns to the matrix
    def add_timing_patterns(self):
        for i in range(8, self.size - 8):
            self.matrix[6, i] = self.matrix[i, 6] = i % 2 == 0
            self.is_restricted_module[6, i] = self.is_restricted_module[i, 6] = True
            
                
    # We need to add the dark module to the matrix
    def add_dark_module(self):
        self.matrix[self.size - 8][8] = 1
        self.is_restricted_module[self.size - 8][8] = True
            
    # We need to add the alignment patterns to the matrix
    def add_alignment_patterns(self):
        pattern_locations = self.get_alignment_pattern_locations()
        for x, y in pattern_locations:
            self.add_pattern((x - 2, x + 3), (y - 2, y + 3), ALIGNMENT_PATTERN)
            
    def get_alignment_pattern_locations(self):
        if self.version == 1:
            return []
        else:
            coords = [(x,y) for x,y in zip(ALIGNMENT_PATTERN_TABLE[self.version], ALIGNMENT_PATTERN_TABLE[self.version])]
            return coords
        
        
    # Function to add the format and version information to the matrix
    def add_format_and_version_info(self, format_info, version_info):
        self.add_format_info(format_info)
        self.add_version_info(version_info)
            
    def add_version_info(self, version_info):
        if self.version < 7:
            return
        field = iter(version_info[::-1])
        
        start = self.size - 11
        
        for i in range(6):
            for j in range(start, start + 3):
                bit = next(field)
                self.matrix[i][j] = bit
                self.matrix[j][i] = bit
                
    def add_format_info(self, format_info):
        field = iter(format_info)
        for i in range(7):
            bit = int(next(field))
            # Skip timing patterns
            if i < 6:
                self.matrix[8][i] = bit
            else:
                self.matrix[8][i + 1] = bit
            
            self.matrix[self.size-i-1][8] = bit
                
        for i in range(8):
            bit = int(next(field))
            self.matrix[8][self.size-8+i] = bit
            if i < 2:
                self.matrix[8-i][8] = bit
            else:
                self.matrix[8-i-1][8] = bit
        
                
            
    def add_pattern(self, x_range, y_range, pattern):
        # We first want to check if any of the space is occupied
        if not self.matrix[x_range[0]:x_range[1], y_range[0]:y_range[1]].any():
            self.matrix[x_range[0]:x_range[1], y_range[0]:y_range[1]] = pattern
            self.is_restricted_module[x_range[0]:x_range[1], y_range[0]:y_range[1]] = True
            return True
        else:
            return False
        
    def get_total_free_space(self):
        freespace = 0
        for x in range(self.size):
            for y in range(self.size):
                if self.matrix[x][y] == None:
                    freespace += 1
        return freespace
    
    def print_(self):
        for row in self.matrix:
            for module in row:
                if module is None:
                    print(" ", end=" ")
                elif module:
                    print("X", end=" ")
                else:
                    print(".", end=" ")
            print()
            
            
    def add_data(self, data: BitBuffer):
        direction = -1 # -1 for up, 1 for down
        row = self.size - 1
        bit_idx = 7
        byte_idx = 0
        
        data_length = data.get_length() // 8 # Get the length of the data in bytes
        
        if data_length > self.get_total_free_space():
            raise Exception("Data does not fit in the matrix")
        elif data_length < self.get_total_free_space():
            # We add pad codewords
            pad_codewords = [236, 17]
            for i in range(self.get_total_free_space() - data_length):
                data.append_bits(pad_codewords[i % 2], 8)
            data_length = data.get_length() // 8
        
        for col in range(self.size - 1, 0, -2):
            if col <= 6:
                col -= 1
            
            col_range = (col, col - 1)
            
            while True:
                for c in col_range:
                    if self.matrix[row][c] is None:
                        if byte_idx < data_length:
                            self.matrix[row][c] = data.get_bit(byte_idx*8 + bit_idx)
                            
                        bit_idx -= 1
                        if bit_idx == -1:
                            bit_idx = 7
                            byte_idx += 1
                row += direction
                
                if row < 0 or self.size <= row:
                    row -= direction
                    direction = -direction
                    break
                
                        
                
    def get_image(self, border, colour, module_size = 10):
        
        # Calculate the final size of the image
        final_image_size = (self.size + 2*border) * module_size
        
        # Create a new blank image with the specified size
        image = Image.new("RGB", (final_image_size, final_image_size), color="white")
        pixels = image.load()
        
        colour_val = (255, 255, 255)
        if colour == "red":
            colour_val = (255, 0, 0)
        elif colour == "green":
            colour_val = (0, 255, 0)
        elif colour == "blue":
            colour_val = (0, 0, 255)
            
        
        # Loop through the matrix and draw the modules
        for i in range(self.size):
            for j in range(self.size):
                # Calculate the position of the module
                x = (border + j) * module_size
                y = (border + i) * module_size
                
                # Determine the colour of the module
                color = (0, 0, 0) if self.matrix[i][j] else colour_val
                
                # Set the colour of the pixels in the image for the module
                for dx in range(module_size):
                    for dy in range(module_size):
                        pixels[x + dx, y + dy] = color
        
        return image
    
    # Function to apply the most suitable mask to the matrix
    def perform_masking(self):
        # We first need to find the best mask
        best_pattern = None
        min_penalty = self.get_penalty_score(self.matrix)
        for i in range(0, 8):
            masked_matrix = self.toggle_mask(i, False)
            
            penalty = self.get_penalty_score(masked_matrix)
            
            if i == 0 or penalty < min_penalty:
                min_penalty = penalty
                best_pattern = i
                
        # Apply the best mask
        self.toggle_mask(best_pattern, True)
        self.mask = best_pattern
        
    
    # Apply mask pattern
    def toggle_mask(self, id, set):
        matrix_copy = self.matrix.copy()
        for i in range(self.size):
            for j in range(self.size):
                if not self.is_restricted_module[i][j] and self.matrix[i][j] is not None:
                    if MASK_PATTERNS[id](i, j):
                        matrix_copy[i][j] = not self.matrix[i][j]
        if set:
            self.matrix = matrix_copy
        return matrix_copy
    
    # Get matrix penalty score
    def get_penalty_score(self, matrix):
        def rule_1_pentalty():
            penalty = 0
            for row in matrix:
                row_str = "".join([str(x) for x in row])
                # Split the row into sequences of consecutive modules of the same colour
                occurrences = re.findall(r"0+|1+", row_str)
                # Calculate the penalty for each sequence
                for occurrence in occurrences:
                    if len(occurrence) >= 5:
                        penalty += len(occurrence) - 2
                        
            for col in matrix.T:
                col_str = "".join([str(x) for x in col])
                # Split the column into sequences of consecutive modules of the same colour
                occurrences = re.findall(r"0+|1+", col_str)
                # Calculate the penalty for each sequence
                for occurrence in occurrences:
                    if len(occurrence) >= 5:
                        penalty += len(occurrence) - 2
            return penalty
        
        def rule_2_penalty():
            penalty = 0
            for i in range(self.size - 1):
                for j in range(self.size - 1):
                    if (
                        matrix[i][j] == matrix[i][j + 1] == matrix[i + 1][j] == matrix[i + 1][j + 1] == 1 or
                        matrix[i][j] == matrix[i][j + 1] == matrix[i + 1][j] == matrix[i + 1][j + 1] == 0
                    ):
                        penalty += 3
            return penalty
        
        def rule_3_penalty():
            dark_modules = sum([sum(row) for row in matrix])
            total_modules = self.size ** 2
            dark_ratio = dark_modules / total_modules
            deviation = abs(dark_ratio - 0.5) / 0.05
            penalty = int(deviation) * 10
            
            return penalty
        
        def rule_4_pentalty():
            dark_modules = sum([sum(row) for row in matrix])
            total_modules = self.size ** 2
            light_modules = total_modules - dark_modules
            percentage_dark = dark_modules / total_modules * 100
            percentage_light = light_modules / total_modules * 100
            
            penalty_dark = int(abs(percentage_dark - 50) / 5)
            penalty_light = int(abs(percentage_light - 50) / 5)
            
            penalty =  (penalty_dark + penalty_light) * 10
            
            return penalty
        
        return rule_1_pentalty() + rule_2_penalty() + rule_3_penalty() + rule_4_pentalty()
            
        

    def set_matrix(self, matrix):
        self.matrix = matrix
                    
        # Locate format and version information
        
    
    """
        Data extraction functions 
        
    """
    def get_format_info(self):
        format_info_1 = []
        format_info_2 = []
        for i in range(7):
            # Skip timing patterns
            if i < 6:
                format_info_1.append(self.matrix[8][i])
            else:
                format_info_1.append(self.matrix[8][i + 1])

            format_info_2.append(self.matrix[self.size-i-1][8])
        for i in range(8):
            format_info_2.append(self.matrix[8][self.size-8+i])
            if i < 2:
                format_info_1.append(self.matrix[8-i][8])
            else:
                format_info_1.append(self.matrix[8-i-1][8])
            
        
        # Check both copies of the format info match
        if format_info_1 != format_info_2:
            raise Exception("Format info does not match")
        
        # Reverse the list
        
        # Translate to a binary string
        format_info = "".join([str(x) for x in format_info_1])
        
        return format_info
            
        
            
    
    def get_version_info(self):
        if self.version < 7:
            return None
        
        version_info_1 = [None]*18
        version_info_2 = [None]*18
        start = self.size - 11
        
        idx = 0
        for i in range(6):
            for j in range(start, start+3):
                version_info_1[idx] = self.matrix[i][j]
                version_info_2[idx] = self.matrix[j][i]
                idx += 1
                
        # Reverse the lists
        version_info_1 = version_info_1[::-1]
        version_info_2 = version_info_2[::-1]
                
        # Check both copies of the version info match
        if version_info_1 != version_info_2:
            raise Exception("Version info does not match")
        
        # We need to convert the array to a binary string
        version_info_1 = "".join([str(x) for x in version_info_1])
        
        return version_info_1
    
    def read_data(self) -> BitBuffer:
        direction = -1 # -1 for up, 1 for down
        row = self.size - 1
        bit_idx = 0
        byte_idx = 0
        
        
        data = BitBuffer()
        byte = 0
  
        for col in range(self.size - 1, 0, -2):
            if col <= 6:
                col -= 1
            
            col_range = (col, col - 1)
            
            while True:
                for c in col_range:
                    if not self.is_restricted_module[row][c]:
                        bit = 1 if self.matrix[row][c] else 0
                        byte |= (bit << bit_idx)
                            
                        bit_idx += 1
                        if bit_idx == 8:
                            data.append_bits(byte, 8)
                            bit_idx = 0
                            byte = 0
                row += direction
                
                if row < 0 or self.size <= row:
                    row -= direction
                    direction = -direction
                    break
                
        return data