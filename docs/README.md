# s3dlio Documentation

> **Version**: v0.9.24 (November 2025)  
> **Status**: Production-Ready (All 5 Backends Active)

Welcome to s3dlio documentation! This README provides an overview of the documentation structure.

---

## 📚 Primary Documentation

These are the main user-facing guides:

| Document | Description |
|----------|-------------|
| **[CLI_GUIDE.md](CLI_GUIDE.md)** | Command-line interface reference (`s3-cli`) |
| **[PYTHON_API_GUIDE.md](PYTHON_API_GUIDE.md)** | Python API reference and examples |
| **[Changelog.md](Changelog.md)** | Complete release history (v0.7.0 → v0.9.24) |

---

## 📂 Documentation Structure

```
docs/
├── README.md                    ← You are here
├── CLI_GUIDE.md                 ← s3-cli command reference
├── PYTHON_API_GUIDE.md          ← Python API guide
├── Changelog.md                 ← Release history
│
├── api/                         ← API reference
│   └── Environment_Variables.md
│
├── supplemental/                ← Additional guides
│   ├── ADAPTIVE-TUNING.md
│   ├── CONFIGURATION-HIERARCHY.md
│   ├── GCS-BACKEND-SELECTION.md
│   ├── GCS-QUICK-START.md
│   ├── MULTI_ENDPOINT_GUIDE.md
│   ├── OPERATION_LOGGING.md
│   ├── OPLOG-GUIDE.md
│   ├── RELEASE-CHECKLIST.md
│   ├── STREAMING-ARCHITECTURE.md
│   ├── TFRECORD-INDEX-QUICKREF.md
│   ├── VERSION-MANAGEMENT.md
│   └── ZERO-COPY-API-REFERENCE.md
│
├── testing/                     ← Testing and validation
│   ├── TESTING-GUIDE.md
│   ├── BACKEND-TESTING.md
│   ├── GCS-PAGINATION-TEST-GUIDE.md
│   ├── GCS-TESTING-SUMMARY.md
│   └── PAGINATION-ANALYSIS-ALL-BACKENDS.md
│
├── performance/                 ← Performance tuning
│   ├── Performance_Profiling_Guide.md
│   ├── Performance_Optimization_Summary.md
│   ├── HowToZeroCopy.md
│   ├── MultiPart_README.md
│   └── O_DIRECT_Implementation.md
│
├── implementation-plans/        ← Feature implementations
│
├── enhancement/                 ← Future enhancement plans
│
├── bugs/                        ← Bug reports and investigations
│
└── archive/                     ← Historical documents (v0.8.x and older)
```

---

## 🔍 Quick Reference

### I want to...

| Task | Document |
|------|----------|
| Use the CLI (`s3-cli`) | [CLI_GUIDE.md](CLI_GUIDE.md) |
| Use the Python API | [PYTHON_API_GUIDE.md](PYTHON_API_GUIDE.md) |
| See what's new | [Changelog.md](Changelog.md) |
| Configure s3dlio | [supplemental/CONFIGURATION-HIERARCHY.md](supplemental/CONFIGURATION-HIERARCHY.md) |
| Use environment variables | [api/Environment_Variables.md](api/Environment_Variables.md) |
| Work with operation logs | [supplemental/OPLOG-GUIDE.md](supplemental/OPLOG-GUIDE.md) |
| Use multi-endpoint load balancing | [supplemental/MULTI_ENDPOINT_GUIDE.md](supplemental/MULTI_ENDPOINT_GUIDE.md) |
| Use zero-copy APIs | [supplemental/ZERO-COPY-API-REFERENCE.md](supplemental/ZERO-COPY-API-REFERENCE.md) |
| Test backends | [testing/BACKEND-TESTING.md](testing/BACKEND-TESTING.md) |
| Optimize performance | [performance/Performance_Profiling_Guide.md](performance/Performance_Profiling_Guide.md) |
| Prepare a release | [supplemental/RELEASE-CHECKLIST.md](supplemental/RELEASE-CHECKLIST.md) |

---

## 🏗️ Supported Backends

All backends are production-ready with the same universal API:

| Backend | URI Prefix | Description |
|---------|------------|-------------|
| **AWS S3** | `s3://` | Amazon S3, MinIO, Ceph, S3-compatible |
| **Azure Blob** | `az://` | Microsoft Azure Blob Storage |
| **Google Cloud Storage** | `gs://` | Google Cloud Storage |
| **Local Filesystem** | `file://` | Standard POSIX filesystem I/O |
| **DirectIO** | `direct://` | O_DIRECT bypass for NVMe/SSD |

**See:** [testing/BACKEND-TESTING.md](testing/BACKEND-TESTING.md) for backend-specific testing

---

## 🚀 Performance

### Targets
- **Read (GET)**: ≥5 GB/s (50 Gb/s) sustained
- **Write (PUT)**: ≥2.5 GB/s (25 Gb/s) sustained
- **RangeEngine**: 30-50% improvement for files >4MB

**See:** [performance/](performance/) for tuning guides

---

## 🔗 External Resources

- **Main README**: [../README.md](../README.md)
- **GitHub Repository**: https://github.com/russfellows/s3dlio
- **Crates.io**: https://crates.io/crates/s3dlio
- **PyPI**: https://pypi.org/project/s3dlio/

---

**Happy coding with s3dlio!** 🚀
