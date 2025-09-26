# s3dlio v0.8.3 - Universal Copy Operations & Multi-Backend Progress Tracking

## 🚀 Major Release Highlights

**s3dlio v0.8.3** represents a **transformative release** that evolves upload and download operations into **universal copy commands** that work seamlessly across **all storage backends** with **real-time progress tracking** and **advanced pattern matching**. This release delivers on the promise of treating storage operations like enhanced Unix `cp` commands, regardless of whether you're working with S3, Azure Blob Storage, local files, or high-performance DirectIO.

---

## 🎯 **Universal Copy Operations - The Game Changer**

### **One Interface, All Backends**
Upload and download commands now function as **universal copy operations** that automatically detect and work with any storage backend:

```bash
# Same command syntax works across ALL backends
s3-cli upload /local/data/*.log s3://bucket/logs/          # → Amazon S3
s3-cli upload /local/data/*.log az://container/logs/       # → Azure Blob  
s3-cli upload /local/data/*.log file:///backup/logs/       # → Local filesystem
s3-cli upload /local/data/*.log direct:///nvme/logs/       # → DirectIO (O_DIRECT)

# Download works the same way
s3-cli download s3://bucket/data/ ./local-data/            # From S3
s3-cli download az://container/files/ ./files/             # From Azure
s3-cli download file:///remote-mount/data/ ./data/         # From filesystem
```

### **Cross-Backend Workflows Made Simple**
Enable powerful data pipeline workflows with consistent interfaces:
```bash
# Multi-cloud data migration pipeline
s3-cli download s3://source-bucket/dataset/ ./temp/
s3-cli upload ./temp/* az://backup-container/dataset/
s3-cli upload ./temp/important/* direct:///nvme-cache/hot-data/
```

---

## 📊 **Real-Time Progress Tracking Across All Backends**

### **Consistent Progress Experience**
Every operation now displays **beautiful progress bars** with detailed metrics, regardless of storage backend:

```
UPLOAD:   [████████████████████████████████████████] 2.4 GB/2.4 GB (127.3 MiB/s, ETA: 0s)
DOWNLOAD: [██████████████████████▋                 ] 1.8 GB/2.4 GB (95.7 MiB/s, ETA: 6s)
```

**Progress Features:**
- ✅ **Real-time transfer rates** (MB/s, GiB/s)  
- ✅ **Accurate ETA calculations**
- ✅ **Multi-file operation tracking**
- ✅ **Dynamic total updates** (no more "0 B/0 B" displays)
- ✅ **Works across S3, Azure, file://, and direct:// backends**

---

## 🔍 **Advanced Pattern Matching & File Selection**

### **Powerful Pattern Support**
Choose files with precision using multiple pattern types:

```bash
# Glob patterns (shell-style wildcards)
s3-cli upload "/data/*.log" s3://bucket/logs/                    # All .log files
s3-cli upload "/data/backup_*.tar.gz" az://container/backups/    # Backup archives
s3-cli upload "/logs/app-??.log" s3://bucket/app-logs/          # Numbered log files

# Regular expressions (advanced matching)  
s3-cli upload "/data/.*\.json$" s3://bucket/json-files/         # All JSON files
s3-cli upload "/logs/app-[0-9]{4}\.log" az://container/logs/    # 4-digit app logs
s3-cli upload "/data/file_\d+\.csv$" s3://bucket/csv-data/      # Numbered CSV files

# Directory operations
s3-cli upload "/entire-directory/" s3://bucket/data/            # All files in directory
s3-cli download --recursive s3://bucket/dataset/ ./local/       # Recursive download
```

---

## 🏗️ **Robust Multi-Backend Architecture**

### **Automatic Backend Detection**
The system automatically selects the optimal backend based on URI scheme:
- **`s3://`** → Amazon S3 (with multiple engine options)
- **`az://` or `azure://`** → Azure Blob Storage  
- **`file://`** → Local file system operations
- **`direct://`** → High-performance DirectIO (O_DIRECT)

### **Intelligent Bucket/Container Management**
- **Smart creation**: Only creates buckets/containers when explicitly requested (`-c` flag)
- **Error handling**: Graceful handling of permission issues and missing containers
- **Cross-platform**: Consistent behavior across AWS, Azure, and local storage

---

## 🐍 **Enhanced Python API with Multi-Backend Support**

### **Seamless Python Integration**
The Python API now supports all backends with the same clean interface:

```python
import s3dlio

# Multi-backend operations with same function calls
s3dlio.upload(['/data/*.log'], 's3://bucket/logs/')              # S3 backend
s3dlio.upload(['/data/*.csv'], 'az://container/data/')           # Azure backend  
s3dlio.upload(['/data/*'], 'file:///backup/data/')              # File backend
s3dlio.upload(['/data/*'], 'direct:///nvme/cache/')             # DirectIO backend

# Download operations work identically
s3dlio.download('s3://bucket/dataset/', './local/', recursive=True)
s3dlio.download('az://container/files/', './files/', recursive=True)
```

### **Runtime Stability Improvements**
- ✅ **Fixed Tokio runtime nesting issues** that caused CLI panics
- ✅ **Robust async operation handling** for stable Python integration  
- ✅ **Backward compatibility** - existing Python code works unchanged
- ✅ **Enhanced error messages** for troubleshooting

---

## ⚡ **Performance & Compatibility**

### **Zero Performance Impact**
- **Preserved performance**: All existing benchmark results maintained
- **Concurrent operations**: Multi-file uploads/downloads retain parallel execution
- **Memory efficiency**: Minimal overhead for multi-backend abstraction
- **Streaming architecture**: Maintains zero-copy streaming where possible

### **Seamless Migration**
- **✅ 100% backward compatible** - all existing CLI commands work unchanged
- **✅ Python API compatibility** - existing scripts run without modification  
- **✅ Configuration preservation** - all settings and options preserved
- **✅ Immediate benefits** - enhanced capabilities available immediately upon upgrade

---

## 🔧 **Key Technical Improvements Since v0.7.9**

### **Architecture Enhancements**
- **Generic backend functions**: `generic_upload_files()` and `generic_download_objects()` 
- **ObjectStore trait abstraction**: Clean interface enabling easy addition of new storage backends
- **Async/await architecture**: Proper async implementation preventing runtime conflicts
- **Pattern matching engine**: Enhanced regex detection and glob handling

### **Critical Bug Fixes**  
- **Tokio runtime resolution**: Eliminated nested runtime creation causing CLI panics
- **Progress bar accuracy**: Fixed "0 B/0 B" display with dynamic total updates  
- **Pattern matching robustness**: Improved file discovery for glob and regex patterns
- **Memory management**: Enhanced streaming performance while adding multi-backend support

### **Quality of Life Improvements**
- **Better error messages**: Clear feedback for unsupported operations or invalid URIs
- **Graceful degradation**: Operations continue even if individual files fail (with warnings)
- **Enhanced logging**: Better debugging support with structured logging across backends
- **Improved documentation**: Comprehensive examples for all backends and use cases

---

## 📦 **Installation & Usage**

### **Rust CLI Installation**
```bash
git clone https://github.com/russfellows/s3dlio.git
cd s3dlio
cargo build --release
./target/release/s3-cli --help
```

### **Python Installation** 
```bash
pip install s3dlio
# or build from source for latest features:
git clone https://github.com/russfellows/s3dlio.git
cd s3dlio
./build_pyo3.sh && ./install_pyo3_wheel.sh
```

### **Quick Test Drive**
```bash
# Test multi-backend operations
echo "Hello s3dlio!" > test.txt
s3-cli upload test.txt file:///tmp/s3dlio-test/     # Local file backend
s3-cli download file:///tmp/s3dlio-test/ ./downloaded/
```

---

## 🎉 **What's Next?**

This release establishes **s3dlio** as the premier **universal storage library** for data-intensive applications. The unified copy interface, comprehensive backend support, and real-time progress tracking make it ideal for:

- **🤖 AI/ML Pipelines**: Seamless data movement between training environments
- **📊 Data Engineering**: Multi-cloud ETL workflows with consistent tooling  
- **☁️ Cloud Migration**: Simplified data movement between cloud providers
- **🚀 DevOps Automation**: Reliable backup and deployment scripts across storage types

**Coming Soon**: Enhanced Python progress callbacks, additional cloud backends (GCS), and performance optimizations for large-scale operations.

---

*For complete technical details, see the [full changelog](docs/Changelog.md) and visit our [GitHub repository](https://github.com/russfellows/s3dlio) for documentation and examples.*