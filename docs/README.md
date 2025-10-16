# s3dlio Documentation

**Last Updated:** October 15, 2025  
**Current Version:** v0.9.7  
**Documentation Status:** Organized and streamlined (15 essential documents)

---

## 📖 Quick Navigation

### 🚀 Getting Started
- **[../README.md](../README.md)** - Project overview, installation, quick start
- **[api/README.md](api/README.md)** - API documentation index

### 📋 Essential Guides

#### Process & Workflows
- **[RELEASE-CHECKLIST.md](RELEASE-CHECKLIST.md)** - Release process and checklist
- **[VERSION-MANAGEMENT.md](VERSION-MANAGEMENT.md)** - Versioning strategy
- **[TESTING-GUIDE.md](TESTING-GUIDE.md)** - Testing procedures and best practices
- **[Changelog.md](Changelog.md)** - Release history and notes

#### Technical References
- **[BACKEND-TESTING.md](BACKEND-TESTING.md)** - Comprehensive backend testing guide (S3, Azure, GCS, File, DirectIO)
- **[v0.9.4_BackEnd_Range-Summary.md](v0.9.4_BackEnd_Range-Summary.md)** - RangeEngine implementation analysis across all backends
- **[OPLOG-GUIDE.md](OPLOG-GUIDE.md)** - Operation logging and replay functionality
- **[ZERO-COPY-API-REFERENCE.md](ZERO-COPY-API-REFERENCE.md)** - Zero-copy patterns for high performance
- **[CONFIGURATION-HIERARCHY.md](CONFIGURATION-HIERARCHY.md)** - Configuration system explained
- **[ADAPTIVE-TUNING.md](ADAPTIVE-TUNING.md)** - Adaptive performance tuning features
- **[TFRECORD-INDEX-QUICKREF.md](TFRECORD-INDEX-QUICKREF.md)** - TFRecord indexing reference

#### Current Status
- **[DEPRECATION-NOTICE-v0.9.4.md](DEPRECATION-NOTICE-v0.9.4.md)** - Current API deprecations
- **[v0.9.4+_TODO.md](v0.9.4+_TODO.md)** - Outstanding work items and roadmap
- **[COMBINED_PERFORMANCE_RECOMMENDATIONS.md](COMBINED_PERFORMANCE_RECOMMENDATIONS.md)** - Performance optimization guide

---

## 📂 Directory Structure

```
docs/
├── README.md                              ← You are here
├── Changelog.md                           ← Release notes
│
├── Process & Workflows/
│   ├── RELEASE-CHECKLIST.md
│   ├── VERSION-MANAGEMENT.md
│   └── TESTING-GUIDE.md
│
├── Technical References/
│   ├── BACKEND-TESTING.md                 ← Backend testing strategies
│   ├── v0.9.4_BackEnd_Range-Summary.md    ← RangeEngine analysis
│   ├── OPLOG-GUIDE.md                     ← Operation logging
│   ├── ZERO-COPY-API-REFERENCE.md
│   ├── CONFIGURATION-HIERARCHY.md
│   ├── ADAPTIVE-TUNING.md
│   ├── TFRECORD-INDEX-QUICKREF.md
│   └── COMBINED_PERFORMANCE_RECOMMENDATIONS.md
│
├── Current Status/
│   ├── DEPRECATION-NOTICE-v0.9.4.md
│   └── v0.9.4+_TODO.md                    ← Roadmap
│
├── api/                                   ← API documentation
│   ├── README.md
│   ├── Environment_Variables.md
│   ├── python-api-v0.9.2.md               ← v0.9.2 Python API
│   ├── python-api-v0.9.3-addendum.md      ← v0.9.3 Python API
│   ├── rust-api-v0.9.2.md                 ← v0.9.2 Rust API
│   └── rust-api-v0.9.3-addendum.md        ← v0.9.3 Rust API
│
├── releases/                              ← v0.9.x release documentation
│   ├── v0.9.0_IMPLEMENTATION_PLAN.md
│   ├── v0.9.0-PRE-PUSH-REVIEW.md
│   ├── v0.9.0-TEST-SUMMARY.md
│   ├── v0.9.1-IMPLEMENTATION-PLAN.md
│   ├── v0.9.1-RELEASE-SUMMARY.md
│   ├── v0.9.1-ZERO-COPY-TEST-SUMMARY.md
│   ├── v0.9.2-IMPLEMENTATION-PLAN.md
│   ├── v0.9.2-RELEASE-SUMMARY.md
│   ├── v0.9.2_Test_Summary.md
│   ├── v0.9.3_TODO.md
│   └── POST-v0.9.1-TODO.md
│
├── implementation-plans/                  ← Feature implementation docs
│   ├── AZURE-RANGE-ENGINE-IMPLEMENTATION.md
│   ├── CANCELLATION-TOKEN-IMPLEMENTATION.md
│   ├── DIRECTIO-RANGE-ENGINE-ANALYSIS.md
│   ├── GCS_BACKEND_IMPLEMENTATION_PLAN.md
│   ├── GCS_Phase2_0-8-18.md
│   ├── GCS_TODO.md
│   ├── S3DLIO_OPLOG_IMPLEMENTATION_SUMMARY.md
│   ├── S3DLIO_OPLOG_INTEGRATION.md
│   └── OPLOG_STREAMING_ANALYSIS.md
│
├── testing/                               ← Testing documentation
│   ├── GCS-PAGINATION-TEST-GUIDE.md
│   ├── GCS-TESTING-SUMMARY.md
│   └── PAGINATION-ANALYSIS-ALL-BACKENDS.md
│
├── enhancement/                           ← Enhancement plans
│   ├── AI_ML_Realism_Enhancement_Plan.md
│   └── dl-driver-realism-epic-template.md
│
├── performance/                           ← Performance guides
│   ├── Performance_Profiling_Guide.md
│   ├── Performance_Optimization_Summary.md
│   ├── HowToZeroCopy.md
│   ├── MultiPart_README.md
│   └── O_DIRECT_Implementation.md
│
└── archive/                               ← Historical documents (v0.8.x and older)
    ├── releases/                          ← Old release notes (v0.8.x)
    ├── implementation-plans/              ← Old completed plans
    ├── api-versions/                      ← Old API docs (v0.7-v0.8)
    ├── testing/                           ← Old testing documents
    ├── performance/                       ← Old performance documents
    └── old-development/                   ← Historical development docs
```

---

## 🔍 Finding What You Need

### I want to...

#### Learn about s3dlio
→ Start with **[../README.md](../README.md)** (project root)

#### Use the Python API
→ **[api/python-api-v0.9.3-addendum.md](api/python-api-v0.9.3-addendum.md)**

#### Use the Rust API
→ **[api/rust-api-v0.9.3-addendum.md](api/rust-api-v0.9.3-addendum.md)**

#### Test a specific backend (S3, Azure, GCS, etc.)
→ **[BACKEND-TESTING.md](BACKEND-TESTING.md)**

#### Understand RangeEngine performance
→ **[v0.9.4_BackEnd_Range-Summary.md](v0.9.4_BackEnd_Range-Summary.md)**

#### Work with operation logs
→ **[OPLOG-GUIDE.md](OPLOG-GUIDE.md)**

#### Optimize performance
→ **[COMBINED_PERFORMANCE_RECOMMENDATIONS.md](COMBINED_PERFORMANCE_RECOMMENDATIONS.md)**  
→ **[performance/Performance_Profiling_Guide.md](performance/Performance_Profiling_Guide.md)**

#### Use zero-copy APIs
→ **[ZERO-COPY-API-REFERENCE.md](ZERO-COPY-API-REFERENCE.md)**  
→ **[performance/HowToZeroCopy.md](performance/HowToZeroCopy.md)**

#### Configure s3dlio
→ **[CONFIGURATION-HIERARCHY.md](CONFIGURATION-HIERARCHY.md)**  
→ **[api/Environment_Variables.md](api/Environment_Variables.md)**

#### Contribute to s3dlio
→ **[TESTING-GUIDE.md](TESTING-GUIDE.md)**  
→ **[RELEASE-CHECKLIST.md](RELEASE-CHECKLIST.md)**

#### Understand deprecated APIs
→ **[DEPRECATION-NOTICE-v0.9.4.md](DEPRECATION-NOTICE-v0.9.4.md)**

#### See what's coming next
→ **[v0.9.4+_TODO.md](v0.9.4+_TODO.md)**

#### Review release history
→ **[Changelog.md](Changelog.md)**  
→ **[releases/](releases/)** - v0.9.x release documentation

#### Review implementation details for features
→ **[implementation-plans/](implementation-plans/)** - Azure, GCS, DirectIO, OpLog implementations  
→ **[testing/](testing/)** - Backend-specific testing guides

#### Find historical documentation
→ **[archive/](archive/)** directory (v0.8.x and older, organized by category)

---

## 📚 API Documentation

### Current APIs (v0.9.3+)
- **[api/python-api-v0.9.3-addendum.md](api/python-api-v0.9.3-addendum.md)** - Python API reference (v0.9.3)
- **[api/rust-api-v0.9.3-addendum.md](api/rust-api-v0.9.3-addendum.md)** - Rust API reference (v0.9.3)
- **[api/Environment_Variables.md](api/Environment_Variables.md)** - Environment variable reference

### Recent API Versions (v0.9.x)
- **[api/python-api-v0.9.2.md](api/python-api-v0.9.2.md)** - Python API reference (v0.9.2)
- **[api/rust-api-v0.9.2.md](api/rust-api-v0.9.2.md)** - Rust API reference (v0.9.2)

### Old API Versions (Archived)
Historical API documentation for v0.8.x and earlier is available in **[archive/api-versions/](archive/api-versions/)**:
- v0.8.x, v0.7.x

---

## 🏗️ Architecture & Features

### Supported Backends (All Production-Ready)
1. **S3** - AWS S3, MinIO, Vast storage systems
2. **Azure Blob Storage** - Microsoft Azure
3. **Google Cloud Storage (GCS)** - Google Cloud
4. **Local Filesystem** (`file://`) - Standard filesystem I/O
5. **DirectIO** (`direct://`) - O_DIRECT bypass for NVMe/SSD

**See:** [BACKEND-TESTING.md](BACKEND-TESTING.md) for comprehensive backend information

### Key Features
- ✅ **Universal API** - Same interface for all backends
- ✅ **RangeEngine** - Concurrent range downloads (30-50% faster)
- ✅ **Zero-Copy** - High-performance data access patterns
- ✅ **Operation Logging** - Record and replay workloads
- ✅ **PyTorch Integration** - `S3IterableDataset` for ML training
- ✅ **Adaptive Tuning** - Automatic performance optimization
- ✅ **Configuration Hierarchy** - Flexible multi-level config system

---

## 🧪 Testing

### Test Execution
```bash
# Rust tests
cargo test --release --lib

# Python tests
./build_pyo3.sh && ./install_pyo3_wheel.sh
python tests/test_functionality.py

# Full test suite
./scripts/test_all.sh
```

**See:** [TESTING-GUIDE.md](TESTING-GUIDE.md) for comprehensive testing procedures

---

## 🚀 Performance

### Current Targets
- **Read (GET)**: ≥5 GB/s (50 Gb/s) sustained
- **Write (PUT)**: ≥2.5 GB/s (25 Gb/s) sustained
- **RangeEngine**: 30-50% improvement for files >4MB
- **Infrastructure**: Tested against bonded 100 Gb ports

**See:** 
- [COMBINED_PERFORMANCE_RECOMMENDATIONS.md](COMBINED_PERFORMANCE_RECOMMENDATIONS.md)
- [v0.9.4_BackEnd_Range-Summary.md](v0.9.4_BackEnd_Range-Summary.md)
- [performance/Performance_Profiling_Guide.md](performance/Performance_Profiling_Guide.md)

---

## 📝 Contributing

### Before Submitting Changes
1. ✅ Run tests: `cargo test --release --lib`
2. ✅ Check warnings: `cargo build --release` (must be zero warnings)
3. ✅ Run clippy: `cargo clippy --all-targets --all-features`
4. ✅ Test Python if applicable: `./build_pyo3.sh && ./install_pyo3_wheel.sh`
5. ✅ Update documentation if API changes
6. ✅ Update `Changelog.md`

**See:** [RELEASE-CHECKLIST.md](RELEASE-CHECKLIST.md) for full release process

---

## 🗂️ Archive

Historical documentation (v0.8.x and older) has been organized in **[archive/](archive/)**:

- **releases/** - Old release notes (v0.8.19, v0.8.21, pre-0.8.0)
- **implementation-plans/** - Completed older feature implementations
- **api-versions/** - Old API documentation (v0.7.x, v0.8.x)
- **testing/** - Historical testing documents (v0.8.x era)
- **performance/** - Old performance analyses (v0.8.1-v0.8.2)
- **old-development/** - Development summaries and phase completions

**Note**: All v0.9.x documentation has been restored to active docs (not archived) as it remains relevant.

---

## 🔗 External Resources

- **Main README**: [../README.md](../README.md)
- **GitHub Repository**: https://github.com/russfellows/s3dlio
- **Issue Tracker**: Use GitHub Issues for bug reports and feature requests
- **Crates.io**: https://crates.io/crates/s3dlio
- **PyPI**: https://pypi.org/project/s3dlio/

---

## 📊 Documentation Statistics

| Category | Document Count |
|----------|----------------|
| Essential Guides | 15 |
| API Documentation | 6 |
| v0.9.x Releases | 16 |
| Implementation Plans | 9 |
| Testing Documentation | 4 |
| Performance Guides | 5 |
| Enhancement Plans | 2 |
| **Total Active** | **57** |
| Archived Documents (v0.8.x and older) | 80+ |

**Organization Philosophy**: 
- **Main docs/**: Core guides and current documentation (15 essential docs)
- **Subdirectories**: Organized by category (api/, releases/, implementation-plans/, etc.)
- **Archive**: Historical v0.8.x and older documents preserved for reference

**Before Cleanup**: 117+ documents  
**After Cleanup**: 57 active + 80+ archived  
**Improvement**: Clear organization by version and category

---

## ❓ Questions or Issues?

- **Documentation unclear?** Open an issue on GitHub
- **Feature request?** See [v0.9.4+_TODO.md](v0.9.4+_TODO.md) or open an issue
- **Bug report?** Follow [TESTING-GUIDE.md](TESTING-GUIDE.md) to gather diagnostics

---

**Happy coding with s3dlio!** 🚀
