# s3dlio Enhancement Proposals

This directory contains design documents for proposed enhancements to the s3dlio library.

## Active Proposals

### 1. [Shortest Expected Waiting Time (SEWT) Load Balancing](SEWT_LOAD_BALANCING.md)

**Status**: Design Phase  
**Target Version**: v0.10.0  
**Complexity**: Medium (2-3 weeks)

Adds intelligent load balancing for multi-endpoint configurations using queueing theory. Routes requests based on `L_i × E[S_i]` (queue length × expected service time) to minimize wait time.

**Key Benefits**:
- **50-75% latency reduction** for heterogeneous endpoints (mixed fast/slow servers)
- **Minimal overhead**: ~120ns per request (~0.0002% of I/O latency)
- **Backward compatible**: New enum variant, no breaking changes

**Use Cases**:
- Multi-region deployments (US-East 10ms vs EU-West 80ms)
- Tiered storage (NVMe cache 2ms vs HDD archive 200ms)
- Mixed infrastructure (on-prem 5ms vs cloud 30ms)

**Implementation Highlights**:
- Exponential moving average (EMA) for service time tracking
- Lock-free atomic updates using CAS loop
- Per-endpoint statistics with ~16 bytes memory overhead

---

### 2. [Block Storage Interface](BLOCK_STORAGE_INTERFACE.md)

**Status**: Design Phase  
**Target Version**: v0.10.0 or v0.11.0  
**Complexity**: High (3-4 weeks)

Extends s3dlio to support raw block devices (`/dev/nvme0n1`, `/dev/sdb`) for maximum performance scenarios requiring direct hardware access.

**Key Benefits**:
- **99% of raw device throughput** (7 GB/s reads, 5 GB/s writes on NVMe Gen4)
- **No filesystem overhead**: Direct control over data placement
- **<1% CPU/memory overhead** vs raw device access

**Use Cases**:
- High-performance computing (HPC) with direct NVMe access
- AI/ML training pipelines with local NVMe cache
- Database/KV store backends (RocksDB, LMDB)
- Testing and benchmarking raw storage hardware
- Embedded systems with minimal overhead requirements

**Implementation Highlights**:
- `block://` URI scheme for device paths
- Metadata table for object-to-offset mapping (stored on device)
- Sector alignment (512/4096 bytes) with O_DIRECT support
- Crash recovery via copy-on-write metadata with checksums

---

## Comparison Matrix

| Feature | SEWT Load Balancing | Block Storage |
|---------|---------------------|---------------|
| **Performance Impact** | 0.0002% overhead | <1% overhead vs raw |
| **Memory Overhead** | 16 bytes/endpoint | 150MB for 1M objects |
| **Complexity** | Medium | High |
| **Platform Support** | All platforms | Linux first, expand later |
| **Breaking Changes** | None | None (new URI scheme) |
| **Testing Effort** | 1 week | 2 weeks (requires loopback) |
| **Documentation** | Update examples | New user guide section |

## Implementation Priority

**Recommended order**:

1. **SEWT Load Balancing** (v0.10.0)
   - Lower complexity, faster implementation
   - Immediate benefit for multi-endpoint users
   - No platform-specific code
   - Can be released as experimental feature quickly

2. **Block Storage Interface** (v0.10.0 or v0.11.0)
   - Higher complexity, more testing required
   - Platform-specific (Linux first)
   - Requires more comprehensive documentation
   - Consider user demand before prioritizing

**Alternative**: Implement both in parallel if resources permit (separate feature branches).

## Development Workflow

### Phase 1: SEWT (Weeks 1-3)

**Week 1**: Implementation
- Add `ShortestExpectedWaitingTime` enum variant
- Implement EMA tracking with AtomicU64
- Add latency instrumentation to ObjectStore methods
- Update `select_endpoint()` logic

**Week 2**: Testing
- Unit tests (EMA convergence, SEWT calculation)
- Integration tests (homogeneous + heterogeneous scenarios)
- Benchmark overhead measurements

**Week 3**: Documentation & Release
- Update module documentation with examples
- Add user guide section on choosing strategies
- Update CHANGELOG.md
- Create PR, review, merge
- Tag v0.10.0

### Phase 2: Block Storage (Weeks 4-7)

**Week 4**: Core Implementation
- Implement BlockStore struct
- Metadata table format and serialization
- Basic CRUD operations (get, put, delete, list)

**Week 5**: Robustness
- Crash recovery (copy-on-write metadata)
- Checksum validation
- Fragmentation management (coalescing)

**Week 6**: Testing
- Loopback device setup scripts
- Integration tests (CRUD, fragmentation, recovery)
- Performance benchmarks vs direct://

**Week 7**: Documentation & Release
- README section on block storage
- Comprehensive user guide
- Safety warnings and setup instructions
- Create PR, review, merge
- Tag v0.11.0 (or v0.10.1 if combined)

## Feature Flags

Both features should be opt-in to minimize dependencies and build times:

```toml
[features]
default = ["native-backends"]
native-backends = []
sewt-load-balancing = []  # Opt-in for SEWT (no extra deps)
block-storage = []         # Opt-in for block device support (Linux ioctl)
```

Users enable explicitly:
```bash
cargo build --features sewt-load-balancing
cargo build --features block-storage
cargo build --features sewt-load-balancing,block-storage  # Both
```

**Rationale**:
- Keeps default build lightweight
- Avoids platform-specific code in default builds
- Users explicitly opt into advanced features
- Easier to stabilize features incrementally

## Success Metrics

### SEWT Load Balancing

**Quantitative**:
- [ ] Selection overhead < 500ns (measured with criterion benchmark)
- [ ] 50%+ latency reduction in heterogeneous scenario (4 endpoints: 5ms, 10ms, 50ms, 100ms)
- [ ] Zero regressions in homogeneous scenario
- [ ] All unit/integration tests passing

**Qualitative**:
- [ ] Clear documentation with decision guide (when to use each strategy)
- [ ] User feedback positive (monitor GitHub issues)
- [ ] No stability issues in production use (monitor telemetry if available)

### Block Storage Interface

**Quantitative**:
- [ ] 99%+ of raw device throughput (measured with fio baseline)
- [ ] < 1% CPU overhead (measured with perf)
- [ ] < 200MB memory for 1M objects (measured with valgrind/heaptrack)
- [ ] Successful recovery from simulated crashes (100% success rate)

**Qualitative**:
- [ ] Comprehensive user guide with setup instructions
- [ ] Loopback testing works on Ubuntu 22.04 LTS
- [ ] Safety mechanisms prevent accidental data loss
- [ ] User feedback positive on API design

## Future Enhancements (Post-v0.11.0)

### SEWT Extensions

1. **Adaptive alpha**: Automatically tune EMA decay factor based on variance
2. **Weighted SEWT**: Manual weights for capacity-aware routing
3. **Percentile-based routing**: Use p95/p99 instead of mean (requires HDR histogram)
4. **Request size awareness**: Track EMA per size bucket

### Block Storage Extensions

1. **Multi-device RAID**: Stripe across multiple devices (4× throughput)
2. **ZNS SSD support**: Zone-aware allocation for append-only devices
3. **Tiered storage**: Auto-migrate cold data to slower devices
4. **Compression**: Zstd compression for space savings
5. **Replication**: Multi-copy for durability
6. **Windows/macOS support**: Expand beyond Linux

## Contributing

See individual design documents for detailed implementation specifications. Key resources:

- **s3dlio Architecture**: [../Architecture.md](../Architecture.md)
- **Development Guide**: [../Development.md](../Development.md)
- **Testing Strategy**: [../Testing.md](../Testing.md)

For questions or feedback on these proposals:
1. Open a GitHub issue with `[Enhancement]` prefix
2. Reference the specific proposal document
3. Provide use case context and requirements

---

**Document Status**: Active as of November 21, 2025  
**Maintainer**: s3dlio core team  
**Review Cycle**: Quarterly (or as needed for high-priority enhancements)
