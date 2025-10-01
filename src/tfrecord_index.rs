// TFRecord index generation compatible with NVIDIA DALI tfrecord2idx format
//
// Generates text-format index files: "{offset} {size}\n" (space-separated, ASCII)
// This format is compatible with:
// - NVIDIA DALI fn.readers.tfrecord(index_path=...)
// - TensorFlow tooling
// - Python ML pipelines
//
// Zero dependencies - uses only Rust standard library

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// TFRecord format constants
const TFRECORD_LENGTH_SIZE: usize = 8;  // u64 little-endian
const TFRECORD_CRC_SIZE: usize = 4;     // u32 CRC checksum

/// Index entry representing one TFRecord
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TfRecordIndexEntry {
    /// Byte offset from start of file
    pub offset: u64,
    /// Total size in bytes (length + crc + data + crc)
    pub size: u64,
}

/// Parse TFRecord file from bytes and extract index entries
///
/// Returns a vector of (offset, size) pairs for each TFRecord entry.
/// Compatible with NVIDIA DALI tfrecord2idx format.
///
/// # Arguments
/// * `data` - Raw TFRecord file bytes
///
/// # Returns
/// * `Ok(Vec<TfRecordIndexEntry>)` - Index entries for all records
/// * `Err(String)` - Error message if parsing fails
pub fn index_entries_from_bytes(data: &[u8]) -> Result<Vec<TfRecordIndexEntry>, String> {
    let mut entries = Vec::new();
    let mut pos = 0usize;

    while pos < data.len() {
        let start_offset = pos;

        // Check if we have enough bytes for header
        if pos + TFRECORD_LENGTH_SIZE + TFRECORD_CRC_SIZE > data.len() {
            break; // End of file
        }

        // Read length (u64 little-endian)
        let length_bytes = &data[pos..pos + TFRECORD_LENGTH_SIZE];
        let length = u64::from_le_bytes(
            length_bytes
                .try_into()
                .map_err(|e| format!("Failed to read length: {}", e))?,
        );

        pos += TFRECORD_LENGTH_SIZE;
        pos += TFRECORD_CRC_SIZE; // Skip length CRC

        // Check if we have enough bytes for data
        if pos + (length as usize) + TFRECORD_CRC_SIZE > data.len() {
            return Err(format!(
                "Truncated TFRecord at offset {}: expected {} bytes, only {} remaining",
                start_offset,
                length,
                data.len() - pos
            ));
        }

        // Skip data payload
        pos += length as usize;
        pos += TFRECORD_CRC_SIZE; // Skip data CRC

        let total_size = (pos - start_offset) as u64;

        entries.push(TfRecordIndexEntry {
            offset: start_offset as u64,
            size: total_size,
        });
    }

    Ok(entries)
}

/// Generate DALI-compatible index text from TFRecord bytes
///
/// Format: "{offset} {size}\n" for each record (space-separated ASCII)
///
/// # Arguments
/// * `data` - Raw TFRecord file bytes
///
/// # Returns
/// * `Ok(String)` - Index text in DALI format
/// * `Err(String)` - Error message if parsing fails
pub fn index_text_from_bytes(data: &[u8]) -> Result<String, String> {
    let entries = index_entries_from_bytes(data)?;
    let mut out = String::with_capacity(entries.len() * 32); // Estimate 32 bytes per line

    for e in entries {
        out.push_str(&format!("{} {}\n", e.offset, e.size));
    }

    Ok(out)
}

/// Read and parse a DALI-compatible index file
///
/// Reads an index file in NVIDIA DALI format and parses it into
/// structured index entries.
///
/// # Arguments
/// * `index_path` - Path to index file (.idx)
///
/// # Returns
/// * `Ok(Vec<TfRecordIndexEntry>)` - Parsed index entries
/// * `Err(String)` - Error message if operation fails
///
/// # Example
/// ```no_run
/// use s3dlio::tfrecord_index::read_index_file;
///
/// let entries = read_index_file("train.tfrecord.idx")
///     .expect("Failed to read index");
///
/// println!("Found {} records", entries.len());
/// for (i, entry) in entries.iter().enumerate() {
///     println!("Record {}: offset={}, size={}", i, entry.offset, entry.size);
/// }
/// ```
pub fn read_index_file<P: AsRef<Path>>(index_path: P) -> Result<Vec<TfRecordIndexEntry>, String> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(index_path.as_ref())
        .map_err(|e| format!("Failed to open index file: {}", e))?;

    let reader = BufReader::new(file);
    let mut entries = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("Failed to read line {}: {}", line_num + 1, e))?;
        let line = line.trim();

        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        // Parse "offset size" format
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 2 {
            return Err(format!(
                "Invalid format at line {}: expected 'offset size', got '{}'",
                line_num + 1,
                line
            ));
        }

        let offset = parts[0]
            .parse::<u64>()
            .map_err(|e| format!("Invalid offset at line {}: {}", line_num + 1, e))?;

        let size = parts[1]
            .parse::<u64>()
            .map_err(|e| format!("Invalid size at line {}: {}", line_num + 1, e))?;

        entries.push(TfRecordIndexEntry { offset, size });
    }

    Ok(entries)
}

/// Write DALI-compatible index file for a TFRecord file
///
/// Reads the TFRecord file, parses it, and writes an index file
/// in the format expected by NVIDIA DALI's tfrecord reader.
///
/// # Arguments
/// * `tfrecord_path` - Path to input TFRecord file
/// * `index_path` - Path to output index file (.idx)
///
/// # Returns
/// * `Ok(usize)` - Number of records indexed
/// * `Err(String)` - Error message if operation fails
///
/// # Example
/// ```no_run
/// use s3dlio::tfrecord_index::write_index_for_tfrecord_file;
///
/// let num_records = write_index_for_tfrecord_file(
///     "train.tfrecord",
///     "train.tfrecord.idx"
/// ).expect("Failed to create index");
///
/// println!("Indexed {} records", num_records);
/// ```
pub fn write_index_for_tfrecord_file<P: AsRef<Path>>(
    tfrecord_path: P,
    index_path: P,
) -> Result<usize, String> {
    // Read TFRecord file
    let mut file = File::open(tfrecord_path.as_ref())
        .map_err(|e| format!("Failed to open TFRecord file: {}", e))?;

    let mut data = Vec::new();
    file.read_to_end(&mut data)
        .map_err(|e| format!("Failed to read TFRecord file: {}", e))?;

    // Generate index text
    let index_text = index_text_from_bytes(&data)?;
    let num_records = index_text.lines().count();

    // Write index file
    let mut index_file = File::create(index_path.as_ref())
        .map_err(|e| format!("Failed to create index file: {}", e))?;

    index_file
        .write_all(index_text.as_bytes())
        .map_err(|e| format!("Failed to write index file: {}", e))?;

    Ok(num_records)
}

/// Streaming TFRecord indexer for large files
///
/// Processes TFRecord files incrementally without loading entire file into memory.
/// Useful for very large TFRecord files (>1GB).
///
/// # Example
/// ```no_run
/// use s3dlio::tfrecord_index::TfRecordIndexer;
///
/// let entries = TfRecordIndexer::new("large.tfrecord")
///     .expect("Failed to open file")
///     .index()
///     .expect("Failed to index");
///
/// println!("Found {} records", entries.len());
/// ```
pub struct TfRecordIndexer<R: Read + Seek> {
    reader: R,
}

impl<R: Read + Seek> TfRecordIndexer<R> {
    /// Create a new indexer from a reader
    pub fn from_reader(reader: R) -> Self {
        Self { reader }
    }

    /// Index the TFRecord file and return all entries
    pub fn index(&mut self) -> Result<Vec<TfRecordIndexEntry>, String> {
        let mut entries = Vec::new();
        self.reader
            .seek(SeekFrom::Start(0))
            .map_err(|e| format!("Failed to seek: {}", e))?;

        loop {
            let start_offset = self
                .reader
                .stream_position()
                .map_err(|e| format!("Failed to get position: {}", e))?;

            // Read length
            let mut length_buf = [0u8; TFRECORD_LENGTH_SIZE];
            match self.reader.read_exact(&mut length_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break, // End of file
                Err(e) => return Err(format!("Failed to read length: {}", e)),
            }

            let length = u64::from_le_bytes(length_buf);

            // Skip length CRC
            self.reader
                .seek(SeekFrom::Current(TFRECORD_CRC_SIZE as i64))
                .map_err(|e| format!("Failed to skip length CRC: {}", e))?;

            // Skip data payload
            self.reader
                .seek(SeekFrom::Current(length as i64))
                .map_err(|e| format!("Failed to skip data: {}", e))?;

            // Skip data CRC
            self.reader
                .seek(SeekFrom::Current(TFRECORD_CRC_SIZE as i64))
                .map_err(|e| format!("Failed to skip data CRC: {}", e))?;

            let end_offset = self
                .reader
                .stream_position()
                .map_err(|e| format!("Failed to get position: {}", e))?;

            let total_size = end_offset - start_offset;

            entries.push(TfRecordIndexEntry {
                offset: start_offset,
                size: total_size,
            });
        }

        Ok(entries)
    }

    /// Index the TFRecord file and write to an index file
    pub fn write_index<P: AsRef<Path>>(&mut self, index_path: P) -> Result<usize, String> {
        let entries = self.index()?;
        let num_records = entries.len();

        let index_file = File::create(index_path.as_ref())
            .map_err(|e| format!("Failed to create index file: {}", e))?;

        let mut writer = BufWriter::new(index_file);

        for e in &entries {
            writeln!(writer, "{} {}", e.offset, e.size)
                .map_err(|e| format!("Failed to write index entry: {}", e))?;
        }

        writer
            .flush()
            .map_err(|e| format!("Failed to flush index file: {}", e))?;

        Ok(num_records)
    }
}

impl TfRecordIndexer<File> {
    /// Create a new indexer from a file path
    pub fn new<P: AsRef<Path>>(tfrecord_path: P) -> Result<Self, String> {
        let file = File::open(tfrecord_path.as_ref())
            .map_err(|e| format!("Failed to open TFRecord file: {}", e))?;
        Ok(Self::from_reader(file))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a minimal valid TFRecord entry
    fn create_test_tfrecord(data: &[u8]) -> Vec<u8> {
        let mut record = Vec::new();

        // Length (u64 LE)
        let length = data.len() as u64;
        record.extend_from_slice(&length.to_le_bytes());

        // Length CRC (dummy for test)
        record.extend_from_slice(&[0u8; 4]);

        // Data
        record.extend_from_slice(data);

        // Data CRC (dummy for test)
        record.extend_from_slice(&[0u8; 4]);

        record
    }

    #[test]
    fn test_index_entries_from_bytes_single_record() {
        let data = b"test data";
        let tfrecord = create_test_tfrecord(data);

        let entries = index_entries_from_bytes(&tfrecord).expect("Failed to parse");

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].offset, 0);
        assert_eq!(entries[0].size, tfrecord.len() as u64);
    }

    #[test]
    fn test_index_entries_from_bytes_multiple_records() {
        let mut tfrecord = Vec::new();
        tfrecord.extend_from_slice(&create_test_tfrecord(b"first"));
        let second_offset = tfrecord.len();
        tfrecord.extend_from_slice(&create_test_tfrecord(b"second"));
        let third_offset = tfrecord.len();
        tfrecord.extend_from_slice(&create_test_tfrecord(b"third"));

        let entries = index_entries_from_bytes(&tfrecord).expect("Failed to parse");

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].offset, 0);
        assert_eq!(entries[1].offset, second_offset as u64);
        assert_eq!(entries[2].offset, third_offset as u64);
    }

    #[test]
    fn test_index_text_from_bytes() {
        let mut tfrecord = Vec::new();
        tfrecord.extend_from_slice(&create_test_tfrecord(b"first"));
        tfrecord.extend_from_slice(&create_test_tfrecord(b"second"));

        let index_text = index_text_from_bytes(&tfrecord).expect("Failed to generate index");

        // Verify format: "{offset} {size}\n"
        let lines: Vec<&str> = index_text.lines().collect();
        assert_eq!(lines.len(), 2);

        // First line should be "0 <size>"
        assert!(lines[0].starts_with("0 "));
        assert!(lines[0].contains(' '));

        // Second line should be "<offset> <size>"
        assert!(lines[1].contains(' '));

        // Verify space-separated format (DALI compatible)
        for line in lines {
            let parts: Vec<&str> = line.split(' ').collect();
            assert_eq!(parts.len(), 2, "Each line should have exactly 2 space-separated values");
            
            // Verify both parts are valid u64 numbers
            parts[0].parse::<u64>().expect("Offset should be valid u64");
            parts[1].parse::<u64>().expect("Size should be valid u64");
        }
    }

    #[test]
    fn test_empty_tfrecord() {
        let tfrecord = Vec::new();
        let entries = index_entries_from_bytes(&tfrecord).expect("Failed to parse");
        assert_eq!(entries.len(), 0);
    }

    #[test]
    fn test_truncated_tfrecord() {
        let mut tfrecord = create_test_tfrecord(b"test");
        tfrecord.truncate(tfrecord.len() - 5); // Remove last 5 bytes

        let result = index_entries_from_bytes(&tfrecord);
        assert!(result.is_err(), "Should fail on truncated record");
    }

    #[test]
    fn test_read_index_file() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a temporary index file
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let index_content = "0 100\n150 75\n300 200\n";
        temp_file
            .write_all(index_content.as_bytes())
            .expect("Failed to write");

        // Read it back
        let entries = read_index_file(temp_file.path()).expect("Failed to read index");

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].offset, 0);
        assert_eq!(entries[0].size, 100);
        assert_eq!(entries[1].offset, 150);
        assert_eq!(entries[1].size, 75);
        assert_eq!(entries[2].offset, 300);
        assert_eq!(entries[2].size, 200);
    }

    #[test]
    fn test_read_index_file_with_empty_lines() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create index with empty lines (should be skipped)
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let index_content = "0 100\n\n150 75\n\n\n300 200\n";
        temp_file
            .write_all(index_content.as_bytes())
            .expect("Failed to write");

        let entries = read_index_file(temp_file.path()).expect("Failed to read index");

        assert_eq!(entries.len(), 3, "Should skip empty lines");
    }

    #[test]
    fn test_read_index_file_invalid_format() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create index with invalid format
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let index_content = "0 100\ninvalid line\n300 200\n";
        temp_file
            .write_all(index_content.as_bytes())
            .expect("Failed to write");

        let result = read_index_file(temp_file.path());
        assert!(result.is_err(), "Should fail on invalid format");
    }

    #[test]
    fn test_roundtrip_index() {
        use tempfile::tempdir;

        // Create test TFRecord
        let mut tfrecord = Vec::new();
        tfrecord.extend_from_slice(&create_test_tfrecord(b"first"));
        tfrecord.extend_from_slice(&create_test_tfrecord(b"second"));
        tfrecord.extend_from_slice(&create_test_tfrecord(b"third"));

        // Write to temp directory
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let tfrecord_path = temp_dir.path().join("test.tfrecord");
        let index_path = temp_dir.path().join("test.idx");

        std::fs::write(&tfrecord_path, &tfrecord).expect("Failed to write TFRecord");

        // Generate index
        let num_written = write_index_for_tfrecord_file(&tfrecord_path, &index_path)
            .expect("Failed to create index");
        assert_eq!(num_written, 3);

        // Read index back
        let entries = read_index_file(&index_path).expect("Failed to read index");
        assert_eq!(entries.len(), 3);

        // Verify entries match original parse
        let original_entries = index_entries_from_bytes(&tfrecord).expect("Failed to parse");
        assert_eq!(entries.len(), original_entries.len());
        for (read, original) in entries.iter().zip(original_entries.iter()) {
            assert_eq!(read.offset, original.offset);
            assert_eq!(read.size, original.size);
        }
    }
}
