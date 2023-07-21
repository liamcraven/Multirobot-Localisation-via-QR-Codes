import argparse
import json
import time
import os
import cv2
import random
import sys

from pyzbar.pyzbar import decode
from standard import StandardQR, StandardScanner
from sequential import SequentialQR, SequentialTransmitter, SequentialReceiver, SequentialScanner
from multiplexed import MultiplexedQR, MultiplexedScanner, multiplex, demultiplex
from error_differentiated import ErrorDifferentiatedQR, ErrorDifferentiatedScanner
from utils.data_formatter import DataFormatter

#Loads the config.json file and returns it as a dictionary.
def load_config():
    with open('config.json') as f:
        config = json.load(f)
    return config

def load_mode():
    with open('mode.json') as f:
        mode = json.load(f)
    return mode

# Creates the parser for the command line arguments.
def create_parser(config):
    parser = argparse.ArgumentParser(description='QR Code Transmission Test Runner')
    
    # Adds methods arguments to the parser
    # Argument for selecting whether we scan or generate
    parser.add_argument('-g', '--generate', action='store_true', help='Generate QR codes.')
    parser.add_argument('-s', '--scan', action='store_true', help='Scan QR codes.')
    
    parser.add_argument('-m', '--method', choices=config['transmission_methods'], nargs='+', default=config['transmission_methods'][0], help='Select the transmission method(s) to use.')
    
    parser.add_argument('-d', '--data', default=config['data_path'], help='Select the data to transmit.')
    parser.add_argument('-i', '--img-path', default=config['img_path'], help='Select the image to scan.')
    
    # Add data output paths
    parser.add_argument('-o', '--output', default=config['data_output_path'], help='Select the output path for the data.')
    # Add image output path
    parser.add_argument('-p', '--img-output', default=config['img_output_path'], help='Select the output path for the image.')
    
    # Add optional sequence id argument
    parser.add_argument('-q', '--sequence-id', default=random.randint(0, 1000000), help='Select the sequence id for the sequence.')
    
    
    for level in config["qr_code_generation"]["error_correction_levels"]:
        parser.add_argument(f"--ec-level-{level}", action="store_true", default="L", help=f"Use error correction level {level}")

    for seq_len in config["qr_code_generation"]["sequence_lengths"]:
        parser.add_argument(f"--sequence-length-{seq_len}", action="store_true", help=f"Use {seq_len} QR codes per sequence")
        
    for display_time in config["display"]["display_times"]:
        parser.add_argument(f"--display-time-{display_time}", action="store_true", help=f"Use {display_time} seconds per QR code")
        
    for window_size in config["window_sizes"]:
        parser.add_argument(f"--window-size-{window_size}", action="store_true", help=f"Use a window size of {window_size}")
        
    for partition_method in config["partitioning_methods"]:
        parser.add_argument(f"--partition-method-{partition_method}", action="store_true", help=f"Use {partition_method} partitioning")
    
    return parser


# Main function
def main():
    # Loads the config file
    config = load_config()
    
    # Loads default mode
    mode = load_mode()
    run_mode = mode["mode"]
    window_size = mode["window_size"]
    method = mode["transmission_methods"]
    data_path = mode["data_path"]
    img_path = mode["img_path"]
    data_out = mode["data_output_path"]
    img_out = mode["img_output_path"]
    ec_level = mode["error_correction_level"]
    sequence_length = mode["sequence_length"]
    display_time = mode["display_time"]
    partition_method = mode["partitioning_method"]
    sequence_id = mode["sequence_id"]
    # Creates the parser
    parser = create_parser(config)
    # Parses the arguments
    args = parser.parse_args()
    
    
    # If no arguments are given, print the help message
    if len(sys.argv) > 1:
        parser.print_help(sys.stderr)
        print("Running with default mode...")
    
        # Access individual parameters
        method = args.method
        data_path = args.data
        img_path = args.img_path
        data_out = args.output
        img_out = args.img_output
        ec_level = next(level for level in config["qr_code_generation"]["error_correction_levels"] if getattr(args, f"ec_level_{level}"))
        sequence_length = next(seq_len for seq_len in config["qr_code_generation"]["sequence_lengths"] if getattr(args, f"sequence_length_{seq_len}"))
        display_time = next(time for time in config["display"]["display_times"] if getattr(args, f"display_time_{time}"))
        window_size = next(size for size in config["window_sizes"] if getattr(args, f"window_size_{size}"))
        partition_method = next(method for method in config["partitioning_methods"] if getattr(args, f"partition_method_{method}"))
        sequence_id = args.sequence_id
    
    if args.generate or run_mode == "gen":
        # Load data from json
        data = []
        with open(data_path) as f:
            data = json.load(f)
        # We first need to check if the user has selected any methods
        if len(method) == 0:
            print("Please select at least one method to use.")
            return
        # We now check if the user has selected a single method
        if len(method) == 1:
            # We now check which method the user has selected
            if method[0] == "standard":
                # Track generation time
                start = time.time()
                # Generate the QR code
                generator = StandardQR(data, error_correction = ec_level, border = 4, colour = "black")
                img, formatter = generator.generate()
                end = time.time()
                print(f"Generation time: {end - start}")
                # Save the image
                img.save(f"{img_out}/standard_qr.png")
            elif method[0] == "sequential":
                #Track generation time
                start = time.time()
                # Generate the QR codes
                generator = SequentialQR(data, partition_method, sequence_length, window_size=window_size, error_correction = ec_level, border = 4, colour = "black")
                imgs, formatter, sequence_id = generator.generate()
                # Stores each of the images in its own folder
                os.mkdir(f"{img_out}/sequence_{sequence_id}")
                for i, img in enumerate(imgs):
                    # Save the image
                    img.save(f"{img_out}/sequence_{sequence_id}/{i}.png")
                    
                # Transmits the sequence
                transmitter = SequentialTransmitter(imgs, sequence_id, display_time)
                sequence_imgs = transmitter.transmit()
                end = time.time()
                print(f"Transmission time: {end - start}")
            elif method[0] == "multiplexed":
                #Track generation time
                start = time.time()
                # Generate the QR code
                generator = MultiplexedQR(data, error_correction = ec_level, border = 4, splitting_method=partition_method, window_size=window_size)
                img, formatter = generator.generate()
                end = time.time()
                print(f"Generation time: {end - start}")
                # Stores the image
                cv2.imwrite(f"{img_out}/multiplexed_qr.png", img)
            elif method[0] == "error_differentiated":
                #Track generation time
                start = time.time()
                # Generate the QR code
                generator = ErrorDifferentiatedQR(data, border = 4, window_size=window_size)
                img, formatter = generator.generate()
                end = time.time()
                print(f"Generation time: {end - start}")
                # Stores the image
                img.save(f"{img_out}/error_differentiated_qr.png")
            else:
                print("Invalid method selected.")
                return
        else:
            # The user has selected multiple methods so we need to check all combinations
            if "sequential" in method and "multiplexed" in method and "error_differentiated" in method:
                # Time generation
                start = time.time()
                # Split data for sequential formatting
                sequential_formatter = DataFormatter(partition_method = "list_splitting", window_size=window_size, partition_count = sequence_length)
                sequential_formatter.set_data(data)
                sequential_formatter.serialize_data()
                split_data = sequential_formatter.split_data()
                multiplexed_imgs = []
                for s in split_data:
                    # Multiplex the data
                    multiplexed_formatter = DataFormatter(partition_method = "list_splitting", window_size=window_size, partition_count = 3)
                    multiplexed_formatter.set_data(s)
                    multiplexed_split = multiplexed_formatter.split_data()
                    colours = ["red", "green", "blue"]
                    colour_imgs = []
                    for m, colour in zip(multiplexed_split, colours):
                        # Generate error differentiated QR code
                        generator = ErrorDifferentiatedQR(m, border = 4, window_size=window_size, colour=colour)
                        img, formatter = generator.generate()
                        colour_imgs.append(img)
                    multiplexed_img = multiplex(colour_imgs[0], colour_imgs[1], colour_imgs[2])
                    multiplexed_imgs.append(multiplexed_img)
                # Generate and save sequence
                sequence_id = random.randint(0, 1000000)
                transmitter = SequentialTransmitter(multiplexed_imgs, sequence_id, display_time)
                sequence_imgs = transmitter.transmit()
                end = time.time()
                print(f"Generation time: {end - start}")
                for i, img in enumerate(sequence_imgs):
                    img.save(f"{img_out}/sme_sequence_{sequence_id}/{i}.png")
            if "sequential" in method and "multiplexed" in method:
                # Time generation
                start = time.time()
                # Split data for sequential formatting
                sequential_formatter = DataFormatter(partition_method = "list_splitting", window_size=window_size, partition_count = sequence_length)
                sequential_formatter.set_data(data)
                sequential_formatter.serialize_data()
                split_data = sequential_formatter.split_data()
                # Generate multiplexed QR codes
                multiplexed_imgs = []
                for s in split_data:
                    generator = MultiplexedQR(s, error_correction = ec_level, border = 4, splitting_method=partition_method, window_size=window_size)
                    img, formatter = generator.generate()
                    multiplexed_imgs.append(img)
                # Generate and save sequence
                sequence_id = random.randint(0, 1000000)
                transmitter = SequentialTransmitter(multiplexed_imgs, sequence_id, display_time)
                sequence_imgs = transmitter.transmit()
                end = time.time()
                print(f"Generation time: {end - start}")
                for i, img in enumerate(sequence_imgs):
                    img.save(f"{img_out}/sequence_{sequence_id}/{i}.png")
            if "sequential" in method and "error_differentiated" in method:
                # Time generation
                start = time.time()
                # Split data for sequential formatting
                sequential_formatter = DataFormatter(partition_method = "list_splitting", window_size=window_size, partition_count = sequence_length)
                sequential_formatter.set_data(data)
                sequential_formatter.serialize_data()
                split_data = sequential_formatter.split_data()
                # Generate error differentiated QR codes
                error_differentiated_imgs = []
                for s in split_data:
                    generator = ErrorDifferentiatedQR(s, border = 4, window_size=window_size)
                    img, formatter = generator.generate()
                    error_differentiated_imgs.append(img)
                # Generate and save sequence
                sequence_id = random.randint(0, 1000000)
                transmitter = SequentialTransmitter(error_differentiated_imgs,sequence_id, display_time)
                sequence_imgs = transmitter.transmit()
                end = time.time()
                print(f"Generation time: {end - start}")
                for i, img in enumerate(sequence_imgs):
                    img.save(f"{img_out}/sequence_{sequence_id}/{i}.png")
            if "multiplexed" in method and "error_differentiated" in method:
                # Time generation
                start = time.time()
                # Generate multiplexed QR code
                multiplexed_formatter = DataFormatter(partition_method = "list_splitting", window_size=window_size, partition_count = 3)
                multiplexed_formatter.set_data(data)
                multiplexed_formatter.serialize_data()
                split_data = multiplexed_formatter.split_data()
                colours = ["red", "green", "blue"]
                colour_imgs = []
                for m, colour in zip(split_data, colours):
                    generator = ErrorDifferentiatedQR(m, border = 4, window_size=window_size, colour=colour)
                    img, formatter = generator.generate()
                    colour_imgs.append(img)
                multiplexed_img = multiplex(colour_imgs[0], colour_imgs[1], colour_imgs[2])
                
                end = time.time()
                print(f"Generation time: {end - start}")
                
                # Save the image
                multiplexed_img.save(f"{img_out}/multiplexed_error_diff_qr.png")
    elif args.scan or run_mode == "scan":
        # We first need to check if the user has a single method
        if len(method) == 1:
            # We now check which method the user has selected
            if method[0] == "standard":
                # Track scan time
                start = time.time()
                # Scan the QR code
                scanner = StandardScanner()
                formatter = DataFormatter(partition_method = partition_method, window_size=window_size, partition_count = 1)
                # Load image
                img = cv2.imread(f"{img_path}/standard_qr.png")
                data = scanner.scan(img, formatter)
                end = time.time()
                print(f"Scan time: {end - start}")
                # Save the data
                with open(f"{data_out}/output.json", 'w') as f:
                    json.dump(data, f)
            elif method[0] == "sequential":
                # Track scan time
                start = time.time()
                # Scan the QR codes
                formatter = DataFormatter(partition_method = partition_method, window_size=window_size, partition_count = sequence_length)
                reciever = SequentialReceiver(formatter=formatter)
                qrs = reciever.receive()
                
                # Decode the QR codes
                scanner = SequentialScanner(formatter=formatter)
                data = scanner.scan(qrs)
                end = time.time()
                print(f"Scan time: {end - start}")
                # Save the data
                with open(f"{data_out}/output.json", 'w') as f:
                    json.dump(data, f)
            elif method[0] == "multiplexed":
                # Track scan time
                start = time.time()
                # Scan the QR code
                formatter = DataFormatter(partition_method = partition_method, window_size=window_size, partition_count = 3)
                scanner = MultiplexedScanner(formatter=formatter)
                data = scanner.scan(f"{img_path}/multiplexed_qr.png")
                end = time.time()
                print(f"Scan time: {end - start}")
                # Save the data
                with open(f"{data_out}/output.json", 'w') as f:
                    json.dump(data, f)
            elif method[0] == "error_differentiated":
                # Track scan time
                start = time.time()
                # Scan the QR code
                formatter = DataFormatter(partition_method = partition_method, window_size=window_size, partition_count = 3)
                scanner = ErrorDifferentiatedScanner()
                data = scanner.scan(f"{img_path}/error_differentiated_qr.png", formatter)
                end = time.time()
                print(f"Scan time: {end - start}")
                # Save the data
                with open(f"{data_out}/output.json", 'w') as f:
                    json.dump(data, f)
        else:
            # The user has selected multiple methods so we need to check all combinations
            if "sequential" in method and "multiplexed" in method and "error_differentiated" in method:
                # Track scan time
                start = time.time()
                sequential_formatter = DataFormatter(partition_method = "list_splitting", window_size=window_size, partition_count = sequence_length)
                multiplexed_formatter = DataFormatter(partition_method = "list_splitting", window_size=window_size, partition_count = 3)
                error_differentiated_formatter = DataFormatter(partition_method = "float_splitting", window_size=window_size, partition_count = 3)
                sequence_datas = []
                # Iterate over each image in the sequence
                for i in range(sequence_length):
                    # Load the image
                    img = cv2.imread(f"{img_path}/{i}.png")
                    # Split the image into its 3 channels
                    demultiplexed_imgs = demultiplex(img)
                    # Iterate over each channel
                    multiplexed_datas = []
                    for j, demultiplexed_img in enumerate(demultiplexed_imgs):
                        # Scan the image
                        scanner = ErrorDifferentiatedScanner()
                        colour_data = scanner.scan(demultiplexed_img, error_differentiated_formatter, deserialize=False)
                        multiplexed_datas.append(colour_data)
                    # Reconstruct the data
                    symbol_data = multiplexed_formatter.recombine_data(multiplexed_datas)
                    sequence_datas.append(symbol_data)
                # Reconstruct the data
                sequential_formatter.set_data(sequence_datas)
                data = sequential_formatter.recombine_data(sequence_datas)
                data = sequential_formatter.deserialize_data(data)
                
                end = time.time()
                print(f"Scan time: {end - start}")
                # Save the data
                with open(f"{data_out}/output.json", 'w') as f:
                    json.dump(data, f)
            if "sequential" in method and "multiplexed" in method:
                # Track scan time
                start = time.time()
                # Get formatters
                sequential_formatter = DataFormatter(partition_method = "list_splitting", window_size=window_size, partition_count = sequence_length)
                multiplexed_formatter = DataFormatter(partition_method = partition_method, window_size=window_size, partition_count = 3)
                sequence_datas = []
                # Iterate over each image in the sequence
                for i in range(sequence_length):
                    # Load the image
                    img = cv2.imread(f"{img_path}/{i}.png")
                    # Scan the image
                    scanner = MultiplexedScanner(formatter=multiplexed_formatter, deserialize=False)
                    symbol_data = scanner.scan(img)
                    sequence_datas.append(symbol_data)
                # Reconstruct the data
                sequential_formatter.set_data(sequence_datas)
                data = sequential_formatter.recombine_data(sequence_datas)
                data = sequential_formatter.deserialize_data(data)
                
                end = time.time()
                print(f"Scan time: {end - start}")
                # Save the data
                with open(f"{data_out}/output.json", 'w') as f:
                    json.dump(data, f)
            if "sequential" in method and "error_differentiated" in method:
                # Track scan time
                start = time.time()
                # Get formatters
                sequential_formatter = DataFormatter(partition_method = "list_splitting", window_size=window_size, partition_count = sequence_length)
                error_differentiated_formatter = DataFormatter(partition_method = "float_splitting", window_size=window_size, partition_count = 3)
                sequence_datas = []
                # Iterate over each image in the sequence
                for i in range(sequence_length):
                    # Load the image
                    img = cv2.imread(f"{img_path}/{i}.png")
                    # Scan the image
                    scanner = ErrorDifferentiatedScanner()
                    symbol_data = scanner.scan(img, error_differentiated_formatter, deserialize=False)
                    sequence_datas.append(symbol_data)
                # Reconstruct the data
                sequential_formatter.set_data(sequence_datas)
                data = sequential_formatter.recombine_data(sequence_datas)
                data = sequential_formatter.deserialize_data(data)
                
                end = time.time()
                print(f"Scan time: {end - start}")
                # Save the data
                with open(f"{data_out}/output.json", 'w') as f:
                    json.dump(data, f)
                    
            if "multiplexed" in method and "error_differentiated" in method:
                # Track scan time
                start = time.time()
                # Get formatters
                multiplexed_formatter = DataFormatter(partition_method = partition_method, window_size=window_size, partition_count = 3)
                error_differentiated_formatter = DataFormatter(partition_method = "float_splitting", window_size=window_size, partition_count = 3)
                # Load the image
                with open(img_path) as f:
                    img = json.load(f)
                # Demultiplex the image
                demultiplexed_img = demultiplex(img)
                colours = ["red", "green", "blue"]
                multiplexed_datas = []
                # Iterate over each channel
                for i, colour in enumerate(colours):
                    # Scan the image
                    scanner = ErrorDifferentiatedScanner()
                    colour_data = scanner.scan(demultiplexed_img[i], error_differentiated_formatter, deserialize=False)
                    multiplexed_datas.append(colour_data)
                # Reconstruct the data
                multiplexed_formatter.set_data(multiplexed_datas)
                data = multiplexed_formatter.recombine_data(multiplexed_datas)
                data = multiplexed_formatter.deserialize_data(data)
                
                end = time.time()
                print(f"Scan time: {end - start}")
                # Save the data
                with open(f"{data_out}/output.json", 'w') as f:
                    json.dump(data, f)
    else:
        print("Please select either the generate or scan option.")
        return
    

if __name__ == "__main__":
    main()
    
        
        
            
    
    