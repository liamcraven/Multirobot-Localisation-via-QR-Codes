import qrcode
import pyzbar.pyzbar as decode
import cv2
import lzma

from utils.bit_stream import BitBuffer
from utils.data_formatter import DataFormatter

from base64 import b64encode, b64decode
"""
    Standard transmission method: we simply take the data and encode it into a QR code
"""

class StandardQR:
    
    def __init__(self, data, error_correction = "L", border = 4, colour = "black", splitting_method="list_splitting"):
        self.data = data
        self.partition_method = splitting_method
        self.border = border
        self.colour = colour
        
        self.ec_level = qrcode.ERROR_CORRECT_L
        if error_correction == "M":
            self.ec_level = qrcode.ERROR_CORRECT_M
        elif error_correction == "Q":
            self.ec_level = qrcode.ERROR_CORRECT_Q
        elif error_correction == "H":
            self.ec_level = qrcode.ERROR_CORRECT_H
            
        self.qr = qrcode.QRCode(
            version=None,
            error_correction= self.ec_level,
            box_size=10,
            border=self.border,
        )
        
        self.data_formatter = DataFormatter(partition_method = self.partition_method)
        self.data_formatter.set_data(self.data)
        
        
    def generate(self, serialize = True):
        # Generates a QR code with the given data, error correction level and border size, resizes it to the given size and returns it
        # We first need to assign the correct error correction level
        
        # Load the data into a bit buffer
        data = self.data
        if serialize:
            data = self.data_formatter.serialize_data()
        bb = BitBuffer()
        for i in data:
            if self.partition_method == "list_splitting":
                bb.append_float(i)
            else:
                bb.append_int(i)
        
        out_bytes = bb.get_bytes()
        
        out_bytes= lzma.compress(out_bytes)
        out_bytes = b64encode(out_bytes)
        
        self.qr.add_data(out_bytes, optimize=0)
        self.qr.make(fit=True)

        img = self.qr.make_image(fill_color=self.colour, back_color="white")
        
        return img, self.data_formatter
    
class StandardScanner:
    
    def __init__(self):
        pass
    
    def scan(self, img, formatter: DataFormatter, deserialize = True):
        # Uses pyzbar to scan the QR code and return the data
        data = decode.decode(img)[0].data
        data = b64decode(data)
        
        lzma_data = lzma.decompress(data)
        
        # We load the data into a bit buffer
        bb = BitBuffer()
        for i in data:
            bb.append_bits(i, 8)
        data = formatter.convert_from_bit_stream([bb])[0]
        if deserialize:
            data = formatter.deserialize_data(data)
            
        print(data)
        return data
        
        
        
    