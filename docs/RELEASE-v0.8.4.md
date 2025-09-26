# s3dlio v0.8.4 - Critical Regex Pattern Matching Fix

## 🚨 **Critical Bug Fix Release**

**s3dlio v0.8.4** is a **critical bug fix release** that resolves a major regex pattern matching issue affecting all core commands (`list`, `get`, `delete`, `download`). This release ensures that pattern-based operations work correctly while maintaining proper directory hierarchy behavior.

---

## 🐛 **The Problem That Was Fixed**

### **Regex Patterns Returning "/" Instead of Objects**
Prior to v0.8.4, commands using regex patterns like `s3://mybucket/` would incorrectly return:
```bash
# ❌ BEFORE v0.8.4 - Wrong behavior
$ s3-cli list s3://mybucket/
/

Total objects: 1
```

Instead of the expected actual objects:
```bash  
# ✅ AFTER v0.8.4 - Correct behavior
$ s3-cli list s3://mybucket/
/object_0_of_10.dat
/object_1_of_10.dat
...
/object_9_of_10.dat

Total objects: 10
```

### **Root Cause Analysis**
The issue occurred because:
1. **S3 API Behavior**: When using `delimiter="/"`, S3 treats objects with leading slashes (e.g., `/object_0.dat`) as being in subdirectories
2. **CommonPrefixes vs Contents**: S3 returned these as `CommonPrefixes` (directory markers) instead of `Contents` (actual objects)
3. **Pattern Matching Logic**: The old code incorrectly processed CommonPrefixes as if they were object names, returning "/" instead of the actual objects

---

## ✅ **The Solution**

### **Smart CommonPrefixes Handling**
v0.8.4 introduces intelligent CommonPrefixes processing that:

#### **🎯 Distinguishes Prefix Types**
- **Single-character prefixes** (like `"/"`) → Recognized as containing objects with leading slashes
- **Multi-character prefixes** (like `"dir1/"`, `"subdir/"`) → Properly treated as true subdirectories

#### **🔄 Selective Recursive Processing**  
- For single-character prefixes, makes targeted queries to find actual objects underneath
- Applies regex patterns to the **actual object names**, not the prefix markers
- Maintains proper directory boundaries for multi-character prefixes

#### **📁 Preserves Directory Hierarchy**
```bash
# Non-recursive: Shows only root-level objects + objects with leading slashes
$ s3-cli list s3://mybucket/
root.txt                    # ✅ True root-level object
test.dat                   # ✅ True root-level object  
//object_0.dat            # ✅ Object with leading slash (logically root-level)
# (Does NOT show dir1/file.txt - properly excluded)

# Recursive: Shows everything including subdirectories  
$ s3-cli list s3://mybucket/ --recursive
root.txt                   # ✅ Root-level object
//object_0.dat            # ✅ Object with leading slash
dir1/file.txt             # ✅ Subdirectory object (only in recursive mode)
dir1/subdir/deep.txt      # ✅ Nested subdirectory object
```

---

## 🎯 **What's Fixed**

### **All Commands Now Work with Regex Patterns**

#### **LIST Command**
```bash
# Match specific patterns
$ s3-cli list "s3://mybucket/.*[0-5].*"
/object_0_of_10.dat
/object_1_of_10.dat  
...
/object_5_of_10.dat

# Match all objects
$ s3-cli list "s3://mybucket/"
# Now correctly shows all actual objects, not "/"
```

#### **GET Command**  
```bash
# Download objects matching pattern to memory (for benchmarking)
$ s3-cli get "s3://mybucket/.*test.*" --jobs 4
GET: [████████████████████] 120.5 MB/120.5 MB (45.2 MB/s)
```

#### **DELETE Command**
```bash
# Safely delete only objects matching the pattern
$ s3-cli delete "s3://mybucket/.*temp.*"
Deleting 3 objects…
Done.
```

#### **DOWNLOAD Command**
```bash  
# Download objects matching pattern to local directory
$ s3-cli download "s3://mybucket/.*backup.*" ./downloads/
DOWNLOAD: [████████████████████] 500.2 MB/500.2 MB (78.3 MB/s)
```

---

## 🔧 **Technical Details**

### **Code Changes**
- **Enhanced `list_objects()` function**: Improved CommonPrefixes vs Contents handling
- **Smart prefix detection**: Distinguishes between separator prefixes and directory prefixes  
- **Targeted recursive queries**: Only recurses for single-character prefixes containing objects
- **Pattern application fix**: Applies regex to actual object names, not prefix markers

### **Backwards Compatibility**
- **✅ 100% backwards compatible**: All existing functionality preserved
- **✅ No breaking changes**: Existing scripts and workflows continue to work unchanged
- **✅ Performance maintained**: No impact on performance, actually improved for pattern matching scenarios

### **Directory Behavior Preserved**
- **Non-recursive mode**: Continues to respect directory boundaries exactly as before
- **Recursive mode**: Continues to show all objects including subdirectories as before  
- **Directory navigation**: All existing directory listing and navigation semantics unchanged

---

## 🚀 **Upgrade Instructions**

### **Installation**
```bash
# Update via cargo
cargo install s3dlio

# Or download the latest release
# https://github.com/russfellows/s3dlio/releases/tag/v0.8.4
```

### **Verification**
Test the fix with your pattern-based commands:
```bash
# This should now show actual objects, not "/"
s3-cli list s3://yourbucket/

# Pattern matching should work correctly
s3-cli list "s3://yourbucket/.*pattern.*"
```

---

## 📋 **Migration Notes**

### **No Action Required**
- **Existing users**: No changes needed, all commands continue to work as before
- **Pattern-based workflows**: Will now work correctly instead of showing confusing "/" results
- **Scripts and automation**: No updates required, improved reliability

### **Benefits You'll See**
- **✅ Regex patterns work correctly**: Commands like `s3://bucket/.*pattern.*` now find actual objects
- **✅ Better user experience**: No more confusing "/" results when listing buckets
- **✅ Consistent behavior**: All four commands (`list`, `get`, `delete`, `download`) now handle patterns identically
- **✅ Maintained performance**: Same or better performance for all operations

---

## 🏆 **Summary**

s3dlio v0.8.4 resolves a critical issue that was causing regex pattern matching to fail across all core commands. The fix ensures that:

- **Pattern matching works correctly** for all commands
- **Directory boundaries are respected** in non-recursive mode  
- **Objects with leading slashes are handled properly** as root-level objects
- **All existing functionality continues unchanged** with improved reliability

This release maintains s3dlio's position as the premier multi-backend storage library while ensuring that pattern-based operations work as users expect.

**Upgrade immediately** to benefit from this critical fix! 🎯