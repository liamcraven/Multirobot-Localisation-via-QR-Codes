from utils.data_formatter import DataFormatter
from standard import StandardQR, StandardScanner
from error_differentiated import ErrorDifferentiatedScanner, ErrorDifferentiatedQR
from utils.bit_stream import BitBuffer

from typing import List
from base64 import b64encode, b64decode
from IPython.display import display
import lzma

import cv2
import numpy as np
import time
import random
import pyzbar.pyzbar as decode
import qrcode


# We have:
# - SequentialQR: Generates a sequence of QR codes (This should not include the start and end QR codes)
# - SequentialTransmitter: Transmits a sequence of QR codes (Adds start and end QR codes) and displays them
# - SequentialReceiver: Receives a sequence of QR codes, when sequence is complete, returns QR images. (This should not include the start and end QR codes)
# - SequentialScanner: Scans a sequence of QR codes (This should not include the start and end QR codes) and returns the data

class SequentialQR:
    
    def __init__(self, data, partition_method, partition_count, window_size=3, error_correction = "L", border = 4, colour = "black"):
        self.data = data
        self.border = border
        self.colour = colour
        self.error_correction = qrcode.ERROR_CORRECT_L
        if error_correction == "M":
            self.error_correction = qrcode.ERROR_CORRECT_M
        elif error_correction == "Q":
            self.error_correction = qrcode.ERROR_CORRECT_Q
        elif error_correction == "H":
            self.error_correction = qrcode.ERROR_CORRECT_H
        self.partition_method = partition_method
        self.partition_count = partition_count
        self.window_size = window_size

        self.formatter = DataFormatter(partition_method = self.partition_method, partition_count = self.partition_count, window_size=self.window_size)
        self.formatter.set_data(self.data)
        
    def generate(self, qrs = None):
        self.formatter.serialize_data()
        
        split_data = self.formatter.split_data()
        bb = BitBuffer()
        byte_datas = []
        random_id = random.randint(0, 1000000)
        for i, splitting in enumerate(split_data):
            new_data = f"{random_id} {i} {splitting}"
            # We want to load the unicode data into the bit buffer
            as_bytes = bytearray(new_data, 'utf-8')
            compressed = lzma.compress(as_bytes)
            as_bytes = b64encode(compressed)
            bb = BitBuffer()
            for byte in as_bytes:
                bb.append_bits(byte, 8)
            byte_datas.append(bb)
        qrs = []
        for i in range(len(byte_datas)):
            qr = qrcode.QRCode(
                version=None,
                error_correction= self.error_correction,
                box_size=10,
                border=self.border,
            )
            qr.add_data(byte_datas[i].get_bytes(), optimize=0)
            qr.make(fit=True)
            qrs.append(qr.make_image(fill_color=self.colour, back_color="white"))
        return qrs, self.formatter, random_id
    
class SequentialTransmitter:
    
    def __init__(self, qrs, sequence_id, time_interval=0.5, border = 4, colour = "black"):
        self.qrs = qrs
        self.time_interval = time_interval
        self.sequence_id = sequence_id
        self.border = border
        self.colour = colour
        
    def generate_start_stop_code(self):
        # We generate the stop code
        data_start = f"START {self.sequence_id} {len(self.qrs)}"
        data_stop = f"STOP {self.sequence_id} {len(self.qrs)}"
        # We want to load the unicode data into the bit buffer
        start_as_bytes = bytearray(data_start, 'utf-8')
        stop_as_bytes = bytearray(data_stop, 'utf-8')
        
        start_compressed = lzma.compress(start_as_bytes)
        start_out = b64encode(start_compressed)
        stop_compressed = lzma.compress(stop_as_bytes)
        stop_out = b64encode(stop_compressed)
        
        # We then generate the QR codes
        start_qr = qrcode.QRCode(
            version=None,
            error_correction= qrcode.ERROR_CORRECT_H,
            box_size=10,
            border=self.border,
        )
        stop_qr = qrcode.QRCode(
            version=None,
            error_correction= qrcode.ERROR_CORRECT_H,
            box_size=10,
            border=self.border,
        )
        start_qr.add_data(start_out, optimize=0)
        start_qr.make(fit=True)
        
        stop_qr.add_data(stop_out, optimize=0)
        stop_qr.make(fit=True)
        
        start_img = start_qr.make_image(fill_color=self.colour, back_color="white")
        stop_img = stop_qr.make_image(fill_color=self.colour, back_color="white")
        
        return start_img, stop_img
    
        
    def transmit(self):
        start, stop = self.generate_start_stop_code()
        # Add start and stop codes to start and end of the sequence respectively
        sequence = [start] + self.qrs + [stop]
        for i in range(2):
            for qr in sequence:
                # Convert to format displayable by cv2
                display(qr)
                time.sleep(self.time_interval)
        return sequence
    
                
class SequentialReceiver:
    
    def __init__(self, formatter: DataFormatter):
        self.sequential_formatter = formatter
        
        
    def receive(self):
        # Setup camera feed
        vid = cv2.VideoCapture(0)
        # Setup cv2 QR Detector
        decode = cv2.QRCodeDetector()
        qrs = []
        start_time = time.time()
        # For each incoming frame, we scan it
        while(True):
            curr_time = time.time()
            if curr_time - start_time > 20:
                break
            # Capture the frame
            ret, frame = vid.read()
            # We need to first find the start code
            retval, decoded_info, _, _ = decode.detectAndDecodeMulti(frame)
            data = []
            if len(decoded_info) > 0:
                decoded_info = decoded_info[0]
                # Decode and decompress
                start_in = b64decode(decoded_info)
                start_decompressed = str(lzma.decompress(start_in), 'utf-8')
                
                # Check if a start code has been found
                if "START" in data:
                    # We want to extract the sequence ID and the number of QR codes in the sequence
                    data = start_decompressed.split(" ")
                    sequence_id = int(data[1])
                    num_qrs = int(data[2])
                    
                    # We then scan the rest of the QR codes
                    found = [False] * num_qrs
                    found_count = 0
                    while(True):
                        # Capture the frame
                        ret, frame = vid.read()
                        # We then scan the frame
                        retval, decoded_info, _, straight_qr = decode.detectAndDecodeMulti(frame)
                        # Check if a QR code has been detected (but not decoded as this would mean it is an error corrected QR code)
                        # Check if error corrected QR code:
                        if retval and len(decoded_info) < 1:
                            # Error corrected QR code
                            qrs.append(straight_qr)
                            
                        elif len(decoded_info) > 0:
                            #Standard QR code, check if it is the correct sequence
                            # Decode and decompress
                            code_in = b64decode(decoded_info[0])
                            code_decompressed = str(lzma.decompress(code_in), 'utf-8')
                            # We extract sequence ID, QR ID and data
                            data = code_decompressed.split(" ")
                            if len(data) == 3:
                                sequence_id = int(data[0])
                                qr_id = int(data[1])
                                data = data[2]
                                # We check if the sequence ID is correct
                                if sequence_id == sequence_id:
                                    # We check if the QR ID is within the bounds
                                    if 0 <= qr_id < num_qrs:
                                        # We check if the QR code has already been found
                                        if not found[qr_id]:
                                            # We mark the QR code as found
                                            found[qr_id] = True
                                            # We add the data to the list of QR codes
                                            qrs.append(straight_qr)
                                            # We check if all QR codes have been found
                                            if all(found):
                                                break
                                else:
                                    # New sequence ID, go back to start
                                    break
                        else:
                            # No QR code detected
                            continue
                    # We have found all QR codes, we can now return the data
                    break
        # End the video capture
        vid.release()
        cv2.destroyAllWindows()
        
        # We then return the QR codes
        return qrs
    
class SequentialScanner:
    
    def __init__(self, formatter: DataFormatter):
        self.sequential_formatter = formatter
    
    def scan(self, qrs):
        datas = []
        for qr in qrs:
            # We first scan the QR code
            scanner = StandardScanner()
            data = scanner.scan(qr, self.sequential_formatter, deserialize=False)
            datas.append(data)
        # We then recombine the data
        data = self.sequential_formatter.recombine_data(datas)
        # We then deserialize the data
        data = self.sequential_formatter.deserialize_data(data)
        return data
            
                
                            
                        
                
   
                        
        

