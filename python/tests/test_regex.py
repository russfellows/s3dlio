import s3dlio
import os
import shutil
import time

# --- Configuration ---
BUCKET_NAME = "russ-test5"
TEST_PREFIX = "test-recursive-py/"
S3_URI_PREFIX = f"s3://{BUCKET_NAME}/{TEST_PREFIX}"

def setup_test_files():
    """Create the bucket used for testing"""
    print(f"--- Creating S3 bucket {BUCKET_NAME}... ---")
    s3dlio.create_bucket(BUCKET_NAME)
    
    """Create a test file structure in S3 with unique filenames."""
    print("--- Setting up test files in S3... ---")
    
    # Create three unique local files to upload
    with open("dummy1.txt", "w") as f: f.write("test1")
    with open("dummy2.txt", "w") as f: f.write("test2")
    with open("dummy3.txt", "w") as f: f.write("test3")
    
    # Upload each unique file to its destination
    s3dlio.upload(["dummy1.txt"], f"{S3_URI_PREFIX}")
    s3dlio.upload(["dummy2.txt"], f"{S3_URI_PREFIX}data/")
    s3dlio.upload(["dummy3.txt"], f"{S3_URI_PREFIX}data/logs/")
    
    # Clean up local dummy files
    os.remove("dummy1.txt")
    os.remove("dummy2.txt")
    os.remove("dummy3.txt")

    print("Setup complete.")
    time.sleep(2) # Allow time for S3 to be consistent

def test_list():
    """Test the list function with and without recursion."""
    print("\n--- Testing s3dlio.list() ---")
    
    # Test non-recursive list
    non_recursive_list = s3dlio.list(S3_URI_PREFIX, recursive=False)
    print(f"Non-recursive list found {len(non_recursive_list)} items: {non_recursive_list}")
    assert len(non_recursive_list) == 2, "Non-recursive list should find 2 items"
    assert f"{TEST_PREFIX}dummy1.txt" in non_recursive_list # Check for dummy1.txt
    assert f"{TEST_PREFIX}data/" in non_recursive_list

    # Test recursive list
    recursive_list = s3dlio.list(S3_URI_PREFIX, recursive=True)
    print(f"Recursive list found {len(recursive_list)} items: {recursive_list}")
    assert len(recursive_list) == 3, "Recursive list should find 3 files"
    assert f"{TEST_PREFIX}dummy1.txt" in recursive_list # Check for dummy1.txt
    assert f"{TEST_PREFIX}data/dummy2.txt" in recursive_list # Check for dummy2.txt
    assert f"{TEST_PREFIX}data/logs/dummy3.txt" in recursive_list # Check for dummy3.txt
    print("✅ List tests passed!")

def test_download():
    """Test the download function."""
    print("\n--- Testing s3dlio.download() ---")
    
    # Test non-recursive download
    os.makedirs("py_download_nonrecursive", exist_ok=True)
    s3dlio.download(S3_URI_PREFIX, "py_download_nonrecursive", recursive=False)
    downloaded_files = os.listdir("py_download_nonrecursive")
    print(f"Non-recursive download got: {downloaded_files}")
    assert len(downloaded_files) == 1
    assert "dummy1.txt" in downloaded_files # Check for dummy1.txt

    # Test recursive download
    os.makedirs("py_download_recursive", exist_ok=True)
    s3dlio.download(S3_URI_PREFIX, "py_download_recursive", recursive=True)
    downloaded_files_rec = os.listdir("py_download_recursive")
    print(f"Recursive download got: {downloaded_files_rec}")
    assert len(downloaded_files_rec) == 3 # Should now pass
    assert "dummy1.txt" in downloaded_files_rec
    assert "dummy2.txt" in downloaded_files_rec
    assert "dummy3.txt" in downloaded_files_rec
    print("✅ Download tests passed!")


def cleanup():
    """Delete all test files from S3 and local directories."""
    print("\n--- Cleaning up test files... ---")
    s3dlio.delete(S3_URI_PREFIX, recursive=True)
    
    # Verify S3 cleanup
    remaining_files = s3dlio.list(S3_URI_PREFIX, recursive=True)
    assert len(remaining_files) == 0, "Cleanup failed, files still remain in S3!"
    
    # Clean up local directories
    if os.path.exists("py_download_nonrecursive"):
        shutil.rmtree("py_download_nonrecursive")
    if os.path.exists("py_download_recursive"):
        shutil.rmtree("py_download_recursive")

    """Create the bucket used for testing"""
    print(f"--- Deleting Empty S3 bucket {BUCKET_NAME}... ---")
    s3dlio.delete_bucket(BUCKET_NAME)

    print("✅ Cleanup successful!")


if __name__ == "__main__":
    try:
        setup_test_files()
        test_list()
        test_download()
    finally:
        cleanup()


