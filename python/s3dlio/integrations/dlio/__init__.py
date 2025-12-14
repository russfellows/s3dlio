"""
s3dlio DLIO Benchmark Integration

This module provides storage backends for DLIO benchmark using s3dlio.

Two integration options are available:

OPTION 1: New storage type (RECOMMENDED)
    - Adds 'storage_type: s3dlio' option to DLIO
    - No overwriting of existing files
    - Requires small patch to DLIO (see generate_patch())
    
OPTION 2: Drop-in replacement  
    - Replaces s3_torch_storage.py
    - Works immediately but overwrites original
    - Use when you can't modify DLIO source

Usage (Option 1 - recommended):
    from s3dlio.integrations.dlio import install_s3dlio_storage, generate_patch
    
    # Generate patch file
    generate_patch('/path/to/dlio_benchmark')
    
    # Or install directly
    install_s3dlio_storage('/path/to/dlio_benchmark')

Usage (Option 2 - drop-in):
    from s3dlio.integrations.dlio import install_dropin_replacement
    install_dropin_replacement('/path/to/dlio_benchmark')
"""
import os


# =============================================================================
# Option 1: New storage type 's3dlio' (recommended)
# =============================================================================

def get_s3dlio_storage_path():
    """Return the path to s3dlio_storage.py (new storage type)."""
    return os.path.join(os.path.dirname(__file__), 's3dlio_storage.py')

def get_s3dlio_storage_content():
    """Return the content of s3dlio_storage.py."""
    with open(get_s3dlio_storage_path(), 'r') as f:
        return f.read()

def generate_patch(dlio_path=None, output_file=None):
    """
    Generate a patch file to add 's3dlio' storage type to DLIO.
    
    Args:
        dlio_path: Path to dlio_benchmark (for reference in patch)
        output_file: Where to write the patch (default: ./dlio_s3dlio.patch)
    
    Returns:
        Path to the generated patch file
    """
    patch = '''--- a/dlio_benchmark/common/enumerations.py
+++ b/dlio_benchmark/common/enumerations.py
@@ class StorageType(Enum):
     LOCAL_FS = 'local_fs'
     PARALLEL_FS = 'parallel_fs'
     S3 = 's3'
+    S3DLIO = 's3dlio'
 
     def __str__(self):
         return self.value

--- a/dlio_benchmark/storage/storage_factory.py
+++ b/dlio_benchmark/storage/storage_factory.py
@@ from dlio_benchmark.storage.file_storage import FileStorage
 from dlio_benchmark.storage.s3_storage import S3Storage
 from dlio_benchmark.common.enumerations import StorageType
 from dlio_benchmark.common.error_code import ErrorCodes
 
 class StorageFactory(object):
     def __init__(self):
         pass
 
     @staticmethod
     def get_storage(storage_type, namespace, framework=None):
         if storage_type == StorageType.LOCAL_FS:
             return FileStorage(namespace, framework)
+        elif storage_type == StorageType.S3DLIO:
+            from dlio_benchmark.storage.s3dlio_storage import S3dlioStorage
+            return S3dlioStorage(namespace, framework)
         elif storage_type == StorageType.S3:
             from dlio_benchmark.common.enumerations import FrameworkType
             if framework == FrameworkType.PYTORCH:
'''
    
    output = output_file or 'dlio_s3dlio.patch'
    with open(output, 'w') as f:
        f.write(patch)
    
    print(f"Generated patch file: {output}")
    print(f"\nTo apply:")
    print(f"  cd /path/to/dlio_benchmark")
    print(f"  patch -p1 < {os.path.abspath(output)}")
    print(f"\nOr manually add 'S3DLIO = s3dlio' to StorageType enum")
    print(f"and the elif branch to storage_factory.py")
    
    return output

def install_s3dlio_storage(dlio_path):
    """
    Install s3dlio as a new storage type in DLIO.
    
    This copies s3dlio_storage.py to DLIO and prints instructions
    for the small code changes needed.
    
    Args:
        dlio_path: Path to dlio_benchmark directory
    
    Returns:
        Path to installed file
    """
    import shutil
    
    target_dir = os.path.join(dlio_path, 'storage')
    if not os.path.isdir(target_dir):
        raise ValueError(f"Invalid DLIO path: {target_dir} does not exist")
    
    target_file = os.path.join(target_dir, 's3dlio_storage.py')
    
    # Copy the s3dlio storage file
    shutil.copy2(get_s3dlio_storage_path(), target_file)
    print(f"Installed: {target_file}")
    
    # Print instructions for code changes
    print(f"\n{'='*60}")
    print("MANUAL CHANGES REQUIRED:")
    print('='*60)
    print(f"\n1. Edit {dlio_path}/common/enumerations.py")
    print("   Add to StorageType enum:")
    print("       S3DLIO = 's3dlio'")
    print(f"\n2. Edit {dlio_path}/storage/storage_factory.py")
    print("   Add before 'elif storage_type == StorageType.S3:':")
    print("       elif storage_type == StorageType.S3DLIO:")
    print("           from dlio_benchmark.storage.s3dlio_storage import S3dlioStorage")
    print("           return S3dlioStorage(namespace, framework)")
    print(f"\n3. In your DLIO config, use:")
    print("       storage:")
    print("         storage_type: s3dlio")
    print("         storage_root: s3://bucket/prefix")
    print('='*60)
    
    return target_file


# =============================================================================
# Option 2: Drop-in replacement (overwrites s3_torch_storage.py)
# =============================================================================

def get_storage_file_path():
    """Return the path to s3_torch_storage.py drop-in replacement."""
    return os.path.join(os.path.dirname(__file__), 's3_torch_storage.py')

def get_storage_file_content():
    """Return the content of s3_torch_storage.py."""
    with open(get_storage_file_path(), 'r') as f:
        return f.read()

def install_dropin_replacement(dlio_path):
    """
    Install s3dlio as drop-in replacement for s3_torch_storage.py.
    
    WARNING: This overwrites the original s3_torch_storage.py!
    A backup is created with .bak extension.
    
    Args:
        dlio_path: Path to dlio_benchmark directory
    
    Returns:
        Path to the installed file
    """
    import shutil
    
    target_dir = os.path.join(dlio_path, 'storage')
    if not os.path.isdir(target_dir):
        raise ValueError(f"Invalid DLIO path: {target_dir} does not exist")
    
    target_file = os.path.join(target_dir, 's3_torch_storage.py')
    
    # Backup original if it exists
    if os.path.exists(target_file):
        backup_file = target_file + '.original.bak'
        if not os.path.exists(backup_file):
            shutil.copy2(target_file, backup_file)
            print(f"Backed up original to: {backup_file}")
    
    # Copy our replacement
    shutil.copy2(get_storage_file_path(), target_file)
    print(f"Installed s3dlio (drop-in mode) to: {target_file}")
    print("\nNote: This replaces s3torchconnector with s3dlio.")
    print("The class is still named S3PyTorchConnectorStorage for compatibility.")
    
    return target_file


# Legacy alias for backwards compatibility
install_to_dlio = install_dropin_replacement

