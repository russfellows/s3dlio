import s3dlio
import tempfile
import os

print('=== Testing s3dlio Generic API ===')
print(f'Version: {s3dlio.__version__}')

# Create test directory and files
test_dir = tempfile.mkdtemp(prefix='s3dlio_test_')
test_file = os.path.join(test_dir, 'test_data.bin')
with open(test_file, 'wb') as f:
    f.write(b'Hello from s3dlio generic API test!')

test_uri = f'file://{test_file}'
test_dir_uri = f'file://{test_dir}/'

print(f'\nTest directory: {test_dir}')
print(f'Test file URI: {test_uri}')

# Test 1: New generic functions
print('\n--- Test 1: Generic Functions ---')

# list_keys
print('Testing list_keys()...')
keys = s3dlio.list_keys(test_dir_uri)
print(f'  list_keys result: {keys}')
assert 'test_data.bin' in keys, f'ERROR: Expected test_data.bin in keys, got {keys}'
print('  ✓ list_keys works correctly')

# list_full_uris  
print('Testing list_full_uris()...')
uris = s3dlio.list_full_uris(test_dir_uri)
print(f'  list_full_uris result: {uris}')
assert all(u.startswith('file://') for u in uris), 'ERROR: Scheme not preserved!'
assert test_uri in uris, f'ERROR: Expected {test_uri} in uris'
print('  ✓ list_full_uris works correctly')

# get_object
print('Testing get_object()...')
data = s3dlio.get_object(test_uri)
print(f'  get_object result type: {type(data)}')
print(f'  get_object result: {bytes(data)}')
assert bytes(data) == b'Hello from s3dlio generic API test!', 'ERROR: Data mismatch!'
print('  ✓ get_object works correctly')

# stat_object
print('Testing stat_object()...')
meta = s3dlio.stat_object(test_uri)
print(f'  stat_object result: {meta}')
assert meta is not None, 'ERROR: stat_object returned None!'
print('  ✓ stat_object works correctly')

print('\n--- Test 2: Deprecated Functions (should show warnings) ---')
import warnings

# Test deprecated functions work but warn
print('Testing list_keys_from_s3() [deprecated]...')
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    keys2 = s3dlio.list_keys_from_s3(test_dir_uri)
    if w and any(issubclass(warning.category, DeprecationWarning) for warning in w):
        print(f'  ✓ Deprecation warning raised')
    else:
        print('  ⚠ No deprecation warning raised')
    print(f'  Result: {keys2}')

print('\n--- Test 3: New Class Names Available ---')
print(f'  ObjectStoreIterableDataset: {s3dlio.ObjectStoreIterableDataset}')
print(f'  ObjectStoreMapDataset: {s3dlio.ObjectStoreMapDataset}')
print(f'  JaxIterable: {s3dlio.JaxIterable}')

print('\n--- Test 4: Deprecated Class Aliases ---')
print(f'  S3IterableDataset (deprecated): {s3dlio.S3IterableDataset}')
print(f'  S3MapDataset (deprecated): {s3dlio.S3MapDataset}')
print(f'  S3JaxIterable (deprecated): {s3dlio.S3JaxIterable}')

# Cleanup
os.remove(test_file)
os.rmdir(test_dir)
print(f'\n✓ Cleaned up test directory')

print('\n=== All Tests Passed! ===')

