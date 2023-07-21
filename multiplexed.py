from utils.data_formatter import DataFormatter
import numpy as np
import cv2
import qrcode
import lzma
from standard import StandardQR, StandardScanner
from base64 import b64encode, b64decode
from pyzbar.pyzbar import decode

from utils.bit_stream import BitBuffer

class MultiplexedQR:
    
    def __init__(self, data = None, error_correction = "L", border = 4, splitting_method="list_splitting", window_size=3):
        self.data = data
        self.border = border
        self.error_correction = qrcode.ERROR_CORRECT_L
        if error_correction == "M":
            self.error_correction = qrcode.ERROR_CORRECT_M
        elif error_correction == "Q":
            self.error_correction = qrcode.ERROR_CORRECT_Q
        elif error_correction == "H":
            self.error_correction = qrcode.ERROR_CORRECT_H
        self.splitting_method = splitting_method
        self.partition_count = 3
        self.window_size = window_size
        
        self.formatter = DataFormatter(partition_method = self.splitting_method, partition_count = self.partition_count, window_size=self.window_size)
        self.formatter.set_data(self.data)
        
    def generate(self):
        self.formatter.serialize_data()
        split_data = self.formatter.split_data()
        
        colours = ["red", "green", "blue"]
        
        datas = []
        for data in split_data:
            bb = BitBuffer()
            for val in data:
                if self.splitting_method == "list_splitting":
                    bb.append_float(val)
                else:
                    bb.append_int(val)
            data_compressed = lzma.compress(bb.get_bytes())
            data_compressed = b64encode(data_compressed)
            datas.append(data_compressed)
        qr = qrcode.QRCode(
            version=None,
            error_correction= self.error_correction,
            box_size=1,
            border=self.border,
        )
        min_version = self.get_min_version(datas)
        imgs = [self.generate_qr(data, colour, min_version) for data, colour in zip(datas, colours)]
        multiplexed = multiplex(imgs[0], imgs[1], imgs[2])
            
        return multiplexed, self.formatter
    
    def get_min_version(self, datas):
        versions = []
        for data in datas:
            qr = qrcode.QRCode(
                version=None,
                error_correction= self.error_correction,
                box_size=1,
                border=self.border,
            )
            qr.add_data(data)
            qr.make(fit=True)
            versions.append(qr.version)
            qr.clear()
        return max(versions)

    def generate_qr(self, data, colour, version):
        qr = qrcode.QRCode(
            version=version,
            error_correction= self.error_correction,
            box_size=10,
            border=self.border,
        )
        qr.add_data(data)
        qr.make(fit=False)
        img = qr.make_image(fill_color=colour, back_color="white")
        return img
    
    
class MultiplexedScanner:
    
    def __init__(self, formatter: DataFormatter, deserialize = True):
        self.formatter = formatter
        self.deserialize = deserialize
        
    def scan(self, img):
        red_qr, green_qr, blue_qr = demultiplex(img)
        
        # We then scan each of the 3 channels
        red_data = decode.decode(red_qr)[0].data
        red_data = b64decode(red_data)
        red_data = lzma.decompress(red_data)
        
        green_data = decode.decode(green_qr)[0].data
        green_data = b64decode(green_data)
        green_data = lzma.decompress(green_data)
        
        blue_data = decode.decode(blue_qr)[0].data
        blue_data = b64decode(blue_data)
        blue_data = lzma.decompress(blue_data)
        
        # We then reconstruct the data
        data = [red_data, green_data, blue_data]
        data = self.formatter.recombine_data(data)
        if self.deserialize:
            data = self.formatter.deserialize_data(data)
        return data
    
    
def demultiplex(img):
    # We first split the image into the 3 colour channels
    return cv2.split(img)
            

def multiplex(red_qr, green_qr, blue_qr):
    """
    This function multiplexes the 3 colour channels into a single image.
    """
    # Use the opencv module to combine the 3 channels into a single image

    #We want to convert red_qr to numpy array
    red_qr = np.array(red_qr)
    green_qr = np.array(green_qr)
    blue_qr = np.array(blue_qr)

    red_qr = cv2.cvtColor(red_qr, cv2.COLOR_RGB2GRAY)
    green_qr = cv2.cvtColor(green_qr, cv2.COLOR_RGB2GRAY)
    blue_qr = cv2.cvtColor(blue_qr, cv2.COLOR_RGB2GRAY)

    multiplexed_qr_code = cv2.merge([red_qr, green_qr, blue_qr])
    
    

    return multiplexed_qr_code
            
        
        
        
    