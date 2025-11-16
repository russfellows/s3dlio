// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

// Tests for multi-array NPZ functionality

use anyhow::Result;
use ndarray::ArrayD;
use s3dlio::data_formats::npz::{build_multi_npz, list_npz_arrays, read_npz_array};

#[test]
fn test_multi_npz_two_arrays() -> Result<()> {
    // Create two different arrays
    let images = ArrayD::from_shape_vec(vec![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let labels = ArrayD::from_shape_vec(vec![2], vec![0.0f32, 1.0])?;
    
    // Build multi-array NPZ
    let arrays = vec![
        ("images", &images),
        ("labels", &labels),
    ];
    let npz_bytes = build_multi_npz(arrays)?;
    
    // Verify we can list arrays
    let names = list_npz_arrays(&npz_bytes)?;
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"images".to_string()));
    assert!(names.contains(&"labels".to_string()));
    
    // Read back and verify
    let images_read = read_npz_array(&npz_bytes, "images")?;
    assert_eq!(images_read.shape(), images.shape());
    assert_eq!(images_read.as_slice().unwrap(), images.as_slice().unwrap());
    
    let labels_read = read_npz_array(&npz_bytes, "labels")?;
    assert_eq!(labels_read.shape(), labels.shape());
    assert_eq!(labels_read.as_slice().unwrap(), labels.as_slice().unwrap());
    
    Ok(())
}

#[test]
fn test_multi_npz_many_arrays() -> Result<()> {
    // Create 5 different arrays with various shapes
    let arr1 = ArrayD::from_shape_vec(vec![10], (0..10).map(|i| i as f32).collect())?;
    let arr2 = ArrayD::from_shape_vec(vec![2, 5], (0..10).map(|i| (i * 2) as f32).collect())?;
    let arr3 = ArrayD::from_shape_vec(vec![1, 1, 10], (0..10).map(|i| (i * 3) as f32).collect())?;
    let arr4 = ArrayD::from_shape_vec(vec![5, 2], vec![1.0f32; 10])?;
    let arr5 = ArrayD::from_shape_vec(vec![1], vec![42.0f32])?;
    
    // Build NPZ with all arrays
    let arrays = vec![
        ("vector", &arr1),
        ("matrix", &arr2),
        ("tensor3d", &arr3),
        ("ones", &arr4),
        ("scalar", &arr5),
    ];
    let npz_bytes = build_multi_npz(arrays)?;
    
    // Verify all arrays listed
    let names = list_npz_arrays(&npz_bytes)?;
    assert_eq!(names.len(), 5);
    assert!(names.contains(&"vector".to_string()));
    assert!(names.contains(&"matrix".to_string()));
    assert!(names.contains(&"tensor3d".to_string()));
    assert!(names.contains(&"ones".to_string()));
    assert!(names.contains(&"scalar".to_string()));
    
    // Verify each array
    let vector_read = read_npz_array(&npz_bytes, "vector")?;
    assert_eq!(vector_read.shape(), &[10]);
    
    let matrix_read = read_npz_array(&npz_bytes, "matrix")?;
    assert_eq!(matrix_read.shape(), &[2, 5]);
    
    let tensor_read = read_npz_array(&npz_bytes, "tensor3d")?;
    assert_eq!(tensor_read.shape(), &[1, 1, 10]);
    
    let ones_read = read_npz_array(&npz_bytes, "ones")?;
    assert_eq!(ones_read.shape(), &[5, 2]);
    assert!(ones_read.iter().all(|&x| x == 1.0));
    
    let scalar_read = read_npz_array(&npz_bytes, "scalar")?;
    assert_eq!(scalar_read.shape(), &[1]);
    assert_eq!(scalar_read[[0]], 42.0);
    
    Ok(())
}

#[test]
fn test_multi_npz_with_npy_extension() -> Result<()> {
    // Test that .npy extension is handled correctly (not duplicated)
    let arr = ArrayD::from_shape_vec(vec![3], vec![1.0f32, 2.0, 3.0])?;
    
    let arrays = vec![
        ("data.npy", &arr),  // Already has .npy extension
    ];
    let npz_bytes = build_multi_npz(arrays)?;
    
    let names = list_npz_arrays(&npz_bytes)?;
    assert_eq!(names.len(), 1);
    assert_eq!(names[0], "data");  // Extension stripped
    
    // Should be readable by both names
    let arr_read = read_npz_array(&npz_bytes, "data")?;
    assert_eq!(arr_read.shape(), &[3]);
    
    Ok(())
}

#[test]
fn test_multi_npz_empty_arrays_error() {
    // Should fail with empty array list
    let arrays: Vec<(&str, &ArrayD<f32>)> = vec![];
    let result = build_multi_npz(arrays);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("zero arrays"));
}

#[test]
fn test_multi_npz_pytorch_style() -> Result<()> {
    // Simulate PyTorch dataset: images + labels + metadata
    let batch_size = 4;
    let image_size = 28 * 28;  // MNIST-style
    
    // Images: [batch_size, 28, 28]
    let images = ArrayD::from_shape_vec(
        vec![batch_size, 28, 28],
        (0..batch_size * image_size).map(|i| (i % 256) as f32 / 255.0).collect()
    )?;
    
    // Labels: [batch_size]
    let labels = ArrayD::from_shape_vec(
        vec![batch_size],
        vec![3.0, 7.0, 2.0, 9.0]
    )?;
    
    // Metadata: single value (epoch number or similar)
    let epoch = ArrayD::from_shape_vec(vec![1], vec![5.0])?;
    
    let arrays = vec![
        ("images", &images),
        ("labels", &labels),
        ("epoch", &epoch),
    ];
    
    let npz_bytes = build_multi_npz(arrays)?;
    
    // Verify structure
    let names = list_npz_arrays(&npz_bytes)?;
    assert_eq!(names.len(), 3);
    
    // Read back in typical PyTorch order
    let images_loaded = read_npz_array(&npz_bytes, "images")?;
    let labels_loaded = read_npz_array(&npz_bytes, "labels")?;
    let epoch_loaded = read_npz_array(&npz_bytes, "epoch")?;
    
    assert_eq!(images_loaded.shape(), &[batch_size, 28, 28]);
    assert_eq!(labels_loaded.shape(), &[batch_size]);
    assert_eq!(epoch_loaded.shape(), &[1]);
    assert_eq!(epoch_loaded[[0]], 5.0);
    
    Ok(())
}
