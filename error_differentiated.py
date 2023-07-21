import math
import numpy as np
import cv2
import lzma
from base64 import b64encode, b64decode
from typing import List

from utils.data_formatter import DataFormatter
from utils.bit_stream import BitBuffer, get_min_version
from utils.matrix import Matrix
from utils.constants import BlockFormat, QR_VERSION_CAPACITY, EC_LEVELS, BLOCK_COUNTS, FORMAT_INFO, VERSION_INFO
from utils.error_correction import RSBlock



class ErrorDifferentiatedQR:
    
    def __init__(self, data, split_method="float_splitting", border = 4, colour = "black", window_size=3):
        self.data_formatter = DataFormatter(split_method, partition_count=3, window_size=window_size)
        self.data_formatter.set_data(data)
        self.data_formatter.serialize_data()
        split_data = self.data_formatter.split_data()
        self.bbs = self.data_formatter.convert_to_bit_stream(split_data)
        
        self.border = border
        self.colour = colour
        
        # We want to allocate two bytes for format information
        for bb in self.bbs:
            bb.append_byte_to_front(0)
            bb.append_byte_to_front(0)
            
        # Compress the data
        for bb in self.bbs:
            compressed = lzma.compress(bb.get_bytes())
            compressed = b64encode(compressed)
            bb.clear_buffer()
            for byte in compressed:
                bb.append_bits(byte, 8)
            
            
        self.data_lens = [bb.get_length() for bb in self.bbs]
        self.total_data_len = sum(self.data_lens)
        
        self.version = get_min_version(self.data_lens)
        
        self.matrix = Matrix(self.version)
        
        
    """
        Generation functions
    """
    def generate(self):
        # We first need to split the data into data blocks
        data_blocks = self.rs_blocks(self.bbs)
        
        # We now need to split the data blocks into codewords
        codewords = [i.as_codewords() for i in data_blocks]
        
        # We now need to interleave the codewords
        interleaved_codewords = self.interleave(codewords)
        
        # Convert the interleaved codewords into a bit stream
        bb = BitBuffer()
        for codeword in interleaved_codewords:
            bb.append_bits(codeword, 8)
            
        # Place data into matrix
        self.matrix.add_data(bb)
        
        # Apply mask
        self.matrix.perform_masking()
        
        # Add format and version information
        if self.matrix.mask is not None:
            self.matrix.add_format_and_version_info(self.generate_format_information(self.matrix.mask), self.generate_version_information(self.version))
            
        # Export image
        img = self.matrix.get_image(self.border, self.colour)
        
        return img, self.data_formatter
            
    def rs_blocks(self, data: List[BitBuffer]):
        # Determine the number of blocks and their formats
        block_counts, block_formats = self.determine_block_count_and_format()
        for i in range(len(data)):
            data[i].set_format_info(block_counts[i], block_formats[i].get_total_codewords())
        
        # We now need to split the data into the appropriate number of blocks, with the appropriate
        # error correction level set for each block
        blocks = []        
        
        # Iterates over each data partition
        for i in range(len(data)):

            # WHile there are still bits left in the data partition
            while data[i].get_length() > 0:
                # If there are still main blocks left
                block = None
                if block_counts[i] > 0:
                    block_format = block_formats[i]
                    block = RSBlock(block_format, EC_LEVELS[i])
                    remaining_capacity = block.get_remaining_capacity()
                    block.add_data(data[i].take_bits(min(remaining_capacity, data[i].get_length())))
                    block.terminate()
                    block.generate_error_correction()
                    block.pad()
                    blocks.append(block)
                    block_counts[i] -= 1
                else:
                    raise Exception(f"No blocks left to assign but there is still {sum([x.get_length() for x in data])} bits left to assign. ") 
            if block_counts[i] > 0:
                # Fill with empty block
                block = RSBlock(block_formats[i], EC_LEVELS[i])
                block.pad()
                blocks.append(block)      
        if sum(block_counts) != 0:
            for i, block_count in enumerate(block_counts):
                print(f"Block count {i}: {block_count}")

        return blocks

    
    def determine_block_count_and_format(self): #TODO: Currently splits into equal codeword size blocks, need to factor in error correction level
        # Given we have three data partitions with different error correction levels, we need to split the free capacity into three based on the error correction level of each data partition
        # We calculate the proportion of the total free capacity that each data partition requires
        required_capacities = [x.get_length() + (y.get_recovery_capacity() * x.get_length()) for x, y in zip(self.bbs, EC_LEVELS)] # +16 for format information
        # Append format bytes to first data partition
        proportions = [x / sum(required_capacities) for x in required_capacities]
        # Now we have the proportions, we need to determine the amount of free space available to each data partition
        free_space = [(QR_VERSION_CAPACITY[self.version]//8) * x for x in proportions] # bytes
        # We not need to determine the number of blocks for each data partition based on the free space available
        blocks_per_partition = [round(x / BLOCK_COUNTS[self.version]) for x in free_space]
        # We now need to determine the block format for each data partition
        # We need to determine the total number of codewords per block for each data partition
        codewords_per_block = [math.ceil(x / y) for x, y in zip(free_space, blocks_per_partition)] # We use floor as we need to round down
        # We now need to determine the number of data codewords per block for each data partition
        data_per_block = [round(x / (1 + y.get_recovery_capacity())) for x, y in zip(codewords_per_block, EC_LEVELS)]
        # We now need to determine the number of error correction codewords per block for each data partition
        ec_per_block = [x- y for x, y in zip(codewords_per_block, data_per_block)]
        # We now need to create the block formats for each data partition
        block_formats = [BlockFormat(x, y, z) for x, y, z in zip(codewords_per_block, data_per_block, ec_per_block)]
        return blocks_per_partition, block_formats
        
    
    def interleave(self, codeword_blocks: List[List[int]]):
        # We want to have easy access to the format info so we do not interleave that
        interleaved_codewords = []
        block_num = 0
        for i in range(3):
            num_blocks = codeword_blocks[block_num].pop(0)
            total_codewords = codeword_blocks[block_num].pop(0)
            interleaved_codewords.append(num_blocks)
            interleaved_codewords.append(total_codewords)
            block_num += num_blocks
            
        print(f"Interleaved codewords: {interleaved_codewords}")
            
        # We now want to interleave the codewords
        max_codewords = max([len(i) for i in codeword_blocks])
        for i in range(max_codewords):
            for j in range(len(codeword_blocks)):
                if i < len(codeword_blocks[j]):
                    interleaved_codewords.append(codeword_blocks[j][i])
        return interleaved_codewords
    
    # This function generates the version information
    def generate_version_information(self, version):
        version_info = VERSION_INFO[version - 7]
        as_array = np.array([int(x) for x in str(version_info)])
        return as_array
    
    # This function generates the format information
    def generate_format_information(self, mask, error_correction_level='M'):
        format_info = FORMAT_INFO[error_correction_level][mask]
        as_array = np.array([int(x) for x in str(format_info)])
        return as_array
    
"""
    Scanning 
"""
    
class ErrorDifferentiatedScanner:
    
    def __init__(self):
        pass
    
    def scan(self, img, data_formatter: DataFormatter, deserialize=True):
        image = cv2.imread(img)
        # Apply preprocessing to the image
        preprocessed_img = self.preprocess_image(image)
        
        # Get the cropped QR codes
        cropped_qrs = self.get_cropped_qr_codes(preprocessed_img)
        
        # Decode each QR into a matrix
        matrices = []
        for qr in cropped_qrs:
            finder_patterns = self.get_finder_patterns(qr)
            module_width = self.find_module_width(qr, finder_patterns)
            version = self.determine_version(qr, module_width)
            
            size = self.calculate_matrix_size(version)
            
            matrix = self.decode_to_matrix(qr, size, module_width)
            new_matrix = Matrix(version)
            new_matrix.set_matrix(matrix)
            matrices.append(new_matrix)
            
        # We now check the version information matches
        version_infos = [matrix.get_version_info() for matrix in matrices]
        versions = []
        for version_info in version_infos:
            i = 1
            while i <= 40:
                if version_info == VERSION_INFO[i]:
                    versions.append(i)
                    break
                i += 1
            if i > 40:
                raise Exception("QR version not matched.")
            
        # We now need to decode the format information
        format_infos = [matrix.get_format_info() for matrix in matrices]
        masks = [self.decode_format_info(format_info) for format_info in format_infos]
        
        # We now want to unmask the matrices
        for matrix, mask in zip(matrices, masks):
            matrix.toggle_mask(mask, True)
            
        # We now want to extract the data from the matrices
        datas = [matrix.read_data() for matrix in matrices]
        
        # Decompress the data
        for data in datas:
            data.decompress_bytes()
        
        # Split the data back into blocks
        blockss = []
        for data, version in zip(datas, versions):
            blocks = get_blocks(datas, version) #TODO: Implement this function
            blockss.append(blocks)
        
        # Perform error correction
        block_datas = []
        for blocks in blockss:
            block_data = []
            for block in blocks:
                valid, d = block.correct_errors()
                if not valid:
                    raise Exception("Error correction failed.")
                block_data.append(d)
            block_datas.append(block_data)
            
        # Split the data back into partitions
        data_partitionss= []
        for block_data in block_datas:
            partitions = get_partitions(block_data) #TODO: Implement this function
            data_partitions = []
            for partition in data_partitions:
                bb = BitBuffer()
                for i in partition:
                    bb.append_bits(i, 8)
                data = data_formatter.convert_from_bit_stream(bb)
                data_partitions.append(data)
            data_partitionss.append(data_partitions)
        
        # Reconstruct the data
        final_data = []
        for data_partitions in data_partitionss:
            data = data_formatter.recombine_data(data_partitions)
            if deserialize:
                data = data_formatter.deserialize_data(data)
            final_data.append(data)
        # Return the data
        return final_data
        
    def decode_format_info(self, format_info):
        print("Format info: ", format_info)
        for ecl, masks in FORMAT_INFO.items():
            print("ECL: ", ecl)
            for m, info in masks.items():
                print(f"Mask: {m}")
                print(f"Info: {info}")
                if format_info == info:
                    return m
        raise Exception("Format info not matched.")
        
    def preprocess_image(self, image, binarization_threshold = 128):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Binarize the image
        binarized = cv2.threshold(gray, binarization_threshold, 255, cv2.THRESH_BINARY)[1]
        return binarized
    
    def get_cropped_qr_codes(self, image):
        # Get the qr detector
        qr_detector = cv2.QRCodeDetector()
    
        # Use detector to get the bounding box for each QR code
        qrs = qr_detector.detectAndDecodeMulti(image)
        (_, _,code_locations, _) = qrs
        
        
        transformed_codes = []
        for corners in code_locations:
            # We want to extend the corners out slightlys

            widthA = np.sqrt((corners[0][0] - corners[1][0])**2 + (corners[0][1] - corners[1][1])**2)
            widthB = np.sqrt((corners[2][0] - corners[3][0])**2 + (corners[2][1] - corners[3][1])**2)
            width = max(int(widthA), int(widthB))
            
            heightA = np.sqrt((corners[0][0]-1 - corners[3][0]+1)**2 + (corners[0][1]-1 - corners[3][1]+1)**2)
            heightB = np.sqrt((corners[1][0]-1 - corners[2][0]+1)**2 + (corners[1][1]-1 - corners[2][1]+1)**2)
            height = max(int(heightA), int(heightB))
            
            dst = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-1]], dtype = "float32")
            
            transformation_matrix = cv2.getPerspectiveTransform(corners, dst)
            
            warped = cv2.warpPerspective(image, transformation_matrix, (width, height))
           
            transformed_codes.append(warped)
           
        return transformed_codes
    
    
    def get_finder_patterns(self, image):
        # We first find the contours in the image
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # We first filter out contours that are too small
        contours = [contour for contour in contours if cv2.contourArea(contour) > 100 and cv2.contourArea(contour) < image.shape[0] * image.shape[1] / 2]
        
        finder_patterns = []        
        for contour in contours:
            perimeter  = cv2.arcLength(contour, True)
            
            # We now want to approximate the contour
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # We now want to check if the contour is a square (4 sides)
            if len(approx) <= 5 and len(approx) >= 3:
                # We check if the ratio is correct
                x,y,w,h = cv2.boundingRect(approx)
                center_x, center_y = (x + w // 2, y + h // 2)
                ratio = w / float(h)
                max_dist = image.shape[1] // 7
                if not(ratio > 0.9 and ratio < 1.1):
                    pass
                # Check if contour is near a corner
                if not((center_x < max_dist and center_y < max_dist) or (image.shape[1] - center_x < max_dist and center_y < max_dist) or (center_x < max_dist and image.shape[0] - center_y < max_dist) or (image.shape[1] - center_x < max_dist and image.shape[0] - center_y < max_dist)):
                    pass
                # We now want to check if the contour fits the ratio 1:1:3:1:1
                # TODO: Maybe check the ratio of the run length array
                finder_patterns.append((x,y,w,h))
                
        # We want to sort the finder patterns by their area from largest to smallest
        sorted_finder_patterns = sorted(finder_patterns, key = lambda x: x[2] * x[3], reverse = True)
                
        return sorted_finder_patterns[:3]
                
        
    def find_module_width(self, img, finder_patterns):
        module_widths = []
        for x, y, w, h in finder_patterns:
            pixel_array = []
            center_x = x + w // 2
            for i in range(y, y+h):
                if img[i, center_x] < 128:
                    pixel_array.append(0)
                else:
                    pixel_array.append(1)
            # convert pixel array into run length array
            run_length_array = []
            curr = 0
            for i in range(0, len(pixel_array)):
                if i == 0:
                    run_length_array.append(1)
                    continue
                elif pixel_array[i] != pixel_array[i-1] or i == 0:
                    run_length_array.append(1)
                    curr += 1
                else:
                    run_length_array[curr] += 1
            # We find the module with from the run length array
            array_len = len(run_length_array)
            module_width = 0
            if array_len == 3:
                module_width = sum([run_len for run_len in run_length_array]) / 5
                module_width = round(module_width)
                print("Module width:s {}".format(module_width))
                module_widths.append(module_width)
        return sum(module_widths) // len(module_widths)


    def determine_version(self, image, module_width):
        size_pixels = image.shape[0]
        
        # We can use the size of the image to determine the version
        size_modules = size_pixels // module_width
        
        # The version is the size of the image minus 17, divided by 4
        version = (size_modules - 17) // 4
        
        return version
    
    def determine_version2(self, image, module_width):
        # Initialize transition counter
        transition_counters = []
        # Use several rows and columns to determine the version
        rows = np.linspace(module_width//2, image.shape[0] - module_width//2, num=10, dtype=int)
        for row in rows:
            transition_counter = 0
            for col in range(module_width//2, image.shape[1] - module_width//2, module_width):
                # Check if there is a transition
                if image[row, col] != image[row, col + module_width]:
                    transition_counter += 1
            transition_counters.append(transition_counter)
        
        # Get the version based on the transition counters
        modules = np.median(transition_counters) // (2 * len(rows))
        
        version = (modules - 17) // 4
        
        return version
    
    def calculate_matrix_size(self, version):
        # The matrix size is 21 + 4 * version
        return (version-1) * 4 + 21
    
    
    
    def build_jump_matrix(self, image, matrix_size, module_width):
        # We want to identify the minimum module width for each row and column
        # Initialize the matrix with none values
        matrix = np.full((matrix_size, matrix_size), None)
        
    
    def decode_to_matrix2(self, image, matrix_size):
        
        _, binarized = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        row_transitions = [self.count_transitions(row) for row in image]
        col_transitions = [self.count_transitions(col) for col in image.T]
        
        # We want to combine all the row transitions and column transitions into one list and remove all duplicates
        # We also want to ignore transitions that are within 1-2 pixels of one another(delete non majority)
        
        #Check the proposed grid is a suitable fit
        # If so, for each cell, take the majority color.

        
    def count_transitions(self, row):
        transition_pnts = []
        prev_col = -1
        for pixel in row:
            colour = 0
            if pixel > 128:
                colour = 1
            if prev_col != -1 and prev_col != colour:
                transition_pnts.append(pixel)
            prev_col = colour
        return transition_pnts
                    
    def decode_to_matrix(self, image, matrix_size, module_width):
        print("matrix size: {}".format(matrix_size))
        # Apply adaptive thresholding to binarize the image
        _, binarized = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binararized_cpy= binarized.copy()
        binararized_cpy = cv2.cvtColor(binararized_cpy, cv2.COLOR_GRAY2BGR)
        
        
        # Initialize the matrix with none values
        matrix = np.full((matrix_size, matrix_size), None)
        
        curr_row = module_width//2
        placement_row = 0
        while curr_row < binarized.shape[0]:
            placement_col = 0
            curr_col = module_width//2
            #print(f"Row before: {curr_row}")
            sample_col = binarized[curr_row-module_width:curr_row, curr_col]
            sample_col = sample_col[::-1]
            #print(sample_col)
            for i in range(0, len(sample_col)):
                    if sample_col[i] != sample_col[i+1]:
                        curr_row = curr_row - i + module_width // 2
                        break
            #print(f"Row after: {curr_row}")
            while curr_col < binarized.shape[1]:
                #print(f"Col before: {curr_col}")
                sample_row = binarized[curr_row, curr_col-module_width:curr_col-1]
                sample_row = sample_row[::-1]
                
            # TODO: FIX THIS!!!! IF WE CANT GET THIS TO WORK WE CAN JUST MANUALLY SET THE VERSION NUMBER
                

                for i in range(0, len(sample_row)):
                    if sample_row[i] != sample_row[i+1]:
                        curr_col = curr_col - i + module_width // 2
                        break          
                
                if placement_col == matrix_size-1:
                    curr_col = binarized.shape[1]-1
                    
                if placement_row == matrix_size-1:
                    curr_row = binarized.shape[0]-1
                    
                    
                # Get the value of the module
                module_value = binarized[curr_row, curr_col]
                
                    
                
                cv2.circle(binararized_cpy, (curr_col, curr_row), 1, (0, 0, 255), 1)

                # Set the value of the matrix
                if placement_col >= matrix_size:
                    break
                matrix[placement_row, placement_col] = 0 if module_value > 128 else 1
                
                # Update the index's
                curr_col += module_width
                placement_col += 1
            curr_row += module_width
            placement_row += 1
            
        
        matrix[matrix_size-1, matrix_size-1] = 0 if binarized[binarized.shape[0]-1, binarized.shape[1]-1] > 128 else 1
                
        return matrix
    
    def deinterleave(self, data):
        # We first extract the format info
        format_info = []
        for i in range(6):
            format_info.append(data.pop(0))
        block_counts = [format_info[0], format_info[2], format_info[4]]
        codeword_counts = [format_info[1], format_info[3], format_info[5]]
        deinterleaved_data = {}
        total_blocks = sum(block_counts)
        for i in range(total_blocks):
            deinterleaved_data[i] = []
            
        block_num = 0
        for i in range(3):
            block_count = format_info[i*2]
            num_codewords = format_info[i*2+1]
            deinterleaved_data[block_num].append(block_count)
            deinterleaved_data[block_num].append(num_codewords)
            block_num += block_count
    
        