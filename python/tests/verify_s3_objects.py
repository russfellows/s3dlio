import boto3
import h5py
import numpy as np
import tensorflow as tf
import zipfile
import io
import json
import yaml
import struct
import tempfile
import os
import zlib
import sys
from termcolor import colored

SUCCESS_ICON = colored('✅', 'green')
ERROR_ICON = colored('❌', 'red')
INFO_ICON = colored('📂', 'blue')



def load_config(config_path):
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        elif config_path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError('Config file must be .yaml, .yml, or .json')


def list_objects(bucket_name, prefix, s3_client):
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if 'Contents' in response:
        return [obj['Key'] for obj in response['Contents']]
    return []


def verify_npz_object(s3_client, bucket_name, key):
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    with zipfile.ZipFile(io.BytesIO(response['Body'].read())) as z:
        npz_content = {}
        for name in z.namelist():
            with z.open(name) as file:
                try:
                    # Attempt to load the file within the NPZ archive
                    npz_content[name] = np.load(file, allow_pickle=False)
                except ValueError as e:
                    print(f"❌ Failed to read '{name}' from '{key}': {e}")
                    print("Attempting to inspect header directly...")

                    # Read the first few bytes to inspect the header
                    file.seek(0)
                    header = file.read(256)
                    print(f"Header snippet (first 256 bytes):\n{header}\n")

                    # Attempt to read as text to find if there's malformed data
                    try:
                        header_str = header.decode("utf-8", errors="ignore")
                        print(f"Header as text: {header_str}")
                    except Exception as decode_err:
                        print(f"❌ Unable to decode header as text: {decode_err}")
                    continue

        print(f"✅ NPZ Object '{key}' contains {len(npz_content)} sub-objects.")

#
# Code to Verify HDF5
#
def verify_hdf5_object(s3_client, bucket_name, key):
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    data = response['Body'].read()

    if data[:8] != b"\x89HDF\r\n\x1a\n":
        print(f"❌ HDF5 Object '{key}' is not a valid HDF5 file. Signature mismatch.")
        return

    try:
        with h5py.File(io.BytesIO(data), 'r') as f:
            print(f"✅ HDF5 Object '{key}' contains {len(f.keys())} sub-objects.")
    except Exception as e:
        print(f"❌ Error opening HDF5 Object '{key}': {e}")



#
# Code to Verify TensorRecord
#

def mask_crc(crc):
    """
    Apply TensorFlow's CRC masking.
    """
    return ((crc >> 15) | (crc << 17)) + 0xa282ead8 & 0xFFFFFFFF


def unmask_crc(masked_crc):
    """
    Reverse the TensorFlow CRC masking.
    """
    rot = (masked_crc - 0xa282ead8) & 0xFFFFFFFF
    return ((rot >> 17) | (rot << 15)) & 0xFFFFFFFF


def verify_tfrecord_object(s3_client, bucket_name, key, index_key):
    print(f"{INFO_ICON} Processing TFRecord file '{key}' from bucket '{bucket_name}'")
    record_count = 0

    # Retrieve the TFRecord object from S3
    obj = s3_client.get_object(Bucket=bucket_name, Key=key)
    data = obj['Body'].read()

    # Retrieve the Index file from S3
    index_obj = s3_client.get_object(Bucket=bucket_name, Key=index_key)
    index_data = index_obj['Body'].read()

    # Parse index file
    index_entries = []
    for i in range(0, len(index_data), 16):
        if i + 16 > len(index_data):
            break
        offset, length = struct.unpack('<QQ', index_data[i:i+16])
        index_entries.append((offset, length))

    print(f"{SUCCESS_ICON} Found associated index file '{index_key}' with {len(index_entries)} entries.")

    cursor = 0
    total_length = len(data)

    for idx, (expected_offset, expected_length) in enumerate(index_entries):
        try:
            if cursor != expected_offset:
                print(f"{ERROR_ICON} Mismatch in offsets: Expected {expected_offset}, but cursor is at {cursor}")
                break

            # Read length (8 bytes)
            length_bytes = data[cursor:cursor + 8]
            if len(length_bytes) < 8:
                break
            length = struct.unpack('<Q', length_bytes)[0]
            cursor += 8

            # Verify if the length matches the expected length (index may include header & CRCs)
            if expected_length != length + 8 + 4 + 4:  # Data length + length header + length CRC + data CRC
                print(f"{ERROR_ICON} Mismatch in lengths: Expected {expected_length}, but found {length + 8 + 4 + 4}")
                continue

            # Read length masked CRC (4 bytes)
            masked_len_crc_bytes = data[cursor:cursor + 4]
            if len(masked_len_crc_bytes) < 4:
                break
            masked_len_crc = struct.unpack('<I', masked_len_crc_bytes)[0]
            cursor += 4

            # Verify length CRC
            computed_len_crc = zlib.crc32(length_bytes) & 0xFFFFFFFF
            if unmask_crc(masked_len_crc) != computed_len_crc:
                print(f"{ERROR_ICON} Invalid length CRC: {masked_len_crc} != {computed_len_crc}")
                continue

            # Read data
            data_block = data[cursor:cursor + length]
            if len(data_block) < length:
                print(f"{ERROR_ICON} Unexpected end of file")
                break
            cursor += length

            # Read data masked CRC (4 bytes)
            masked_data_crc_bytes = data[cursor:cursor + 4]
            if len(masked_data_crc_bytes) < 4:
                break
            masked_data_crc = struct.unpack('<I', masked_data_crc_bytes)[0]
            cursor += 4

            # Verify data CRC
            computed_data_crc = zlib.crc32(data_block) & 0xFFFFFFFF
            if unmask_crc(masked_crc=masked_data_crc) != computed_data_crc:
                print(f"{ERROR_ICON} Invalid data CRC: {masked_data_crc} != {computed_data_crc}")
                continue

            record_count += 1
            print(f"{SUCCESS_ICON} Valid record #{record_count} - Length: {len(data_block)}")

        except Exception as e:
            print(f"{ERROR_ICON} Error processing record: {e}")
            break

    if record_count > 0:
        print(f"{SUCCESS_ICON} Successfully processed {record_count} records.")
    else:
        print(f"{ERROR_ICON} No valid records found.")



def main(config_path):
    config = load_config(config_path)
    s3_client = boto3.client(
        's3',
        aws_access_key_id=config['s3_config']['aws_access_key_id'],
        aws_secret_access_key=config['s3_config']['aws_secret_access_key'],
        endpoint_url=config['s3_config']['endpoint_url'],
        region_name=config['s3_config']['region']
    )

    bucket_name = config['s3_config']['bucket_name']
    prefix = config['s3_config']['object_prefix']
    expected_format = config.get("format", "").upper()

    if expected_format not in {"HDF5", "NPZ", "TFRECORD"}:
        print(f"❌ Unsupported format specified in config: {expected_format}")
        return

    keys = list_objects(bucket_name, prefix, s3_client)
    tfrecord_files = {}

    # Store all keys by their base name
    for key in keys:
        if expected_format == "HDF5" and key.endswith('.h5'):
            verify_hdf5_object(s3_client, bucket_name, key)
        elif expected_format == "NPZ" and key.endswith('.npz'):
            verify_npz_object(s3_client, bucket_name, key)
        elif expected_format == "TFRECORD":
            if key.endswith('.tfrecord'):
                base_name = key[:-len('.tfrecord')]
                if base_name not in tfrecord_files:
                    tfrecord_files[base_name] = {}
                tfrecord_files[base_name]["tfrecord"] = key
            elif key.endswith('.tfrecord.idx'):
                base_name = key[:-len('.tfrecord.idx')]
                if base_name not in tfrecord_files:
                    tfrecord_files[base_name] = {}
                tfrecord_files[base_name]["index"] = key
        else:
            print(f"Skipping '{key}' as it does not match the expected format: {expected_format}")

    # Process each TFRecord file and its index (if exists)
    for base_name, files in tfrecord_files.items():
        tfrecord_key = files.get("tfrecord")
        index_key = files.get("index")

        if tfrecord_key:
            print(f"\n📂 Processing TFRecord file '{tfrecord_key}'")
            if index_key:
                print(f"📂 Found associated index file '{index_key}' for '{tfrecord_key}'")
            verify_tfrecord_object(s3_client, bucket_name, tfrecord_key, index_key=index_key)
        elif index_key:
            print(f"❌ Warning: Found index file '{index_key}' with no corresponding '.tfrecord' file.")



if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python verify_s3_objects.py <config_path>")
        sys.exit(1)
    main(sys.argv[1])

