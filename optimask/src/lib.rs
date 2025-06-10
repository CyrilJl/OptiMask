use numpy::{PyReadonlyArray2}; // Keep existing imports, removed ToPyArray
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::cmp::max; // Keep existing imports

// ... (previously implemented functions remain unchanged) ...
#[pyfunction]
fn is_decreasing(h: Vec<u32>) -> bool {
    if h.is_empty() {
        return true;
    }
    for i in 0..(h.len() - 1) {
        if h[i] < h[i + 1] {
            return false;
        }
    }
    true
}

#[pyfunction]
fn has_nan_in_subset(x_array: PyReadonlyArray2<f64>, rows: Vec<usize>, cols: Vec<usize>) -> bool {
    let x = x_array.as_array();
    for &r in &rows {
        for &c in &cols {
            if x[[r, c]].is_nan() {
                return true;
            }
        }
    }
    false
}

#[pyfunction]
fn groupby_max(a: Vec<u32>, b: Vec<u32>, n: usize) -> Vec<u32> {
    let mut ret = vec![0u32; n];
    for k in 0..a.len() {
        let ak = a[k] as usize;
        if ak < n {
            ret[ak] = max(ret[ak], b[k] + 1);
        }
    }
    ret
}

#[pyfunction]
fn numba_apply_permutation(p: Vec<u32>, x: Vec<u32>) -> Vec<u32> {
    let n = p.len();
    let m = x.len();
    let mut rank = vec![0u32; n];
    let mut result = vec![0u32; m];

    for i in 0..n {
        rank[p[i] as usize] = i as u32;
    }

    for i in 0..m {
        result[i] = rank[x[i] as usize];
    }
    result
}

#[pyfunction]
fn numba_apply_permutation_inplace(p: Vec<u32>, mut x: Vec<u32>) -> Vec<u32> {
    let n = p.len();
    let mut rank = vec![0u32; n];

    for i in 0..n {
        rank[p[i] as usize] = i as u32;
    }

    for i in 0..x.len() {
        x[i] = rank[x[i] as usize];
    }
    x
}

#[pyfunction]
fn apply_p_step(p_step: Vec<u32>, a: Vec<u32>, b: Vec<u32>) -> (Vec<u32>, Vec<u32>) {
    let len_p = p_step.len();
    let mut ret_a = Vec::with_capacity(len_p);
    let mut ret_b = Vec::with_capacity(len_p);

    for k in 0..len_p {
        let pk = p_step[k] as usize;
        if pk < a.len() && pk < b.len() {
            ret_a.push(a[pk]);
            ret_b.push(b[pk]);
        }
    }
    (ret_a, ret_b)
}

#[pyfunction]
fn compute_to_keep(size: usize, index_with_nan: Vec<u32>, permutation: Vec<u32>, split: usize) -> Vec<u32> {
    let mut mask = vec![0u8; size];

    for i in 0..split {
        if let Some(&p_val) = permutation.get(i) {
            if let Some(&nan_idx) = index_with_nan.get(p_val as usize) {
                if (nan_idx as usize) < size {
                    mask[nan_idx as usize] = 1;
                }
            }
        }
    }

    let mut result_count = 0;
    for i in 0..size {
        if mask[i] == 0 {
            result_count += 1;
        }
    }

    let mut result = Vec::with_capacity(result_count);
    for i in 0..size {
        if mask[i] == 0 {
            result.push(i as u32);
        }
    }
    result
}

#[pyfunction]
fn _preprocess(x_array: PyReadonlyArray2<f64>) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let x = x_array.as_array();
    let m = x.shape()[0];
    let n = x.shape()[1];

    let mut iy_vec: Vec<u32> = Vec::with_capacity(m * n); // Max possible size
    let mut ix_vec: Vec<u32> = Vec::with_capacity(m * n); // Max possible size

    let mut cols_index_mapper: Vec<i32> = vec![-1; n];
    let mut rows_with_nan_vec: Vec<u32> = Vec::with_capacity(m);

    let mut n_rows_with_nan: u32 = 0;
    let mut n_cols_with_nan: u32 = 0;
    // cnt is implicitly handled by iy_vec.len()

    for i in 0..m { // Iterate rows
        let mut row_has_nan = false;
        for j in 0..n { // Iterate columns
            if x[[i, j]].is_nan() {
                row_has_nan = true;
                iy_vec.push(n_rows_with_nan);

                if cols_index_mapper[j] >= 0 {
                    ix_vec.push(cols_index_mapper[j] as u32);
                } else {
                    ix_vec.push(n_cols_with_nan);
                    cols_index_mapper[j] = n_cols_with_nan as i32;
                    n_cols_with_nan += 1;
                }
            }
        }
        if row_has_nan {
            rows_with_nan_vec.push(i as u32);
            n_rows_with_nan += 1;
        }
    }

    // Construct cols_with_nan from cols_index_mapper
    let mut temp_cols: Vec<(u32, u32)> = Vec::new();
    for j in 0..n {
        if cols_index_mapper[j] >= 0 {
            // Store (mapped_index, original_column_index)
            temp_cols.push((cols_index_mapper[j] as u32, j as u32));
        }
    }
    // Sort by the mapped_index to replicate Numba's argsort behavior on the filtered+mapped array
    temp_cols.sort_unstable_by_key(|k| k.0);

    let cols_with_nan_vec: Vec<u32> = temp_cols.into_iter().map(|(_, original_col_idx)| original_col_idx).collect();

    (iy_vec, ix_vec, rows_with_nan_vec, cols_with_nan_vec)
}

#[pymodule]
fn optimask_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_decreasing, m)?)?;
    m.add_function(wrap_pyfunction!(has_nan_in_subset, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_max, m)?)?;
    m.add_function(wrap_pyfunction!(numba_apply_permutation, m)?)?;
    m.add_function(wrap_pyfunction!(numba_apply_permutation_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(apply_p_step, m)?)?;
    m.add_function(wrap_pyfunction!(compute_to_keep, m)?)?;
    m.add_function(wrap_pyfunction!(_preprocess, m)?)?;
    Ok(())
}
