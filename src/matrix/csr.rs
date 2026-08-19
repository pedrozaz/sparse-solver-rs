/// Compressed Sparse Row (CSR) matrix format for efficient sparse linear algebra.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    pub(crate) values: Vec<f64>,
    pub(crate) col_indices: Vec<usize>,
    pub(crate) row_ptr: Vec<usize>,
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
}

impl CsrMatrix {
    /// Returns the number of rows in the matrix.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Returns the number of columns in the matrix.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Returns the number of non-zero entries stored in the matrix.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Returns a slice of non-zero values stored in the matrix.
    #[inline]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Returns a slice of columns indices corresponding to non-zero values.
    #[inline]
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Returns the row pointer array used in CSR representation.
    #[inline]
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Constructs a `CsrMatrix` from a list of triplets `(row, col, value)`.
    ///
    /// Duplicate entries at the same `(row, col)` position are summed together.
    ///
    /// # Panics
    /// Panics if any triplet index is out of bounds for the given `nrows` and `ncols`
    ///
    /// # Example
    /// ```
    /// use sparse_solver_rs::matrix::CsrMatrix;
    ///
    /// let triplets = vec![(0, 0, 1.0), (0, 1, 2.0), (1, 1, 3.0)];
    /// let matrix = CsrMatrix::from_triplets(2, 2, &triplets);
    /// assert_eq!(matrix.nnz(), 3);
    /// ```
    pub fn from_triplets(nrows: usize, ncols: usize, triplets: &[(usize, usize, f64)]) -> Self {
        for &(r, c, _) in triplets {
            assert!(
                r < nrows,
                "Row index {} out of bounds for nrows {}",
                r,
                nrows
            );
            assert!(
                c < ncols,
                "Columns index {} out of bounds for ncols {}",
                c,
                ncols
            );
        }

        let mut sorted = triplets.to_vec();
        sorted.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptr = vec![0; nrows + 1];

        let mut current_row = 0;

        for &(r, c, val) in &sorted {
            if val == 0.0 {
                continue;
            }

            // Update row pointers if we advanced to a new row
            while current_row < r {
                current_row += 1;
                row_ptr[current_row] = values.len();
            }

            // If entry at same (row, col) exists in current_row, accumulate value
            if values.len() > row_ptr[r] && col_indices.last() == Some(&c) {
                if let Some(last_val) = values.last_mut() {
                    *last_val += val;
                }
                continue;
            }

            values.push(val);
            col_indices.push(c);
        }

        // Fill remaining row pointers for empty trailing rows
        while current_row < nrows {
            current_row += 1;
            row_ptr[current_row] = values.len();
        }

        Self {
            values,
            col_indices,
            row_ptr,
            nrows,
            ncols,
        }
    }

    /// Performs sparse matrix-vector multiplication $y = A \cdot x$.
    ///
    /// # Panics
    /// Panics if `x.len() != self.ncols()`.
    ///
    /// # Example
    /// ```
    /// use sparse_solver_rs::matrix::CsrMatrix;
    ///
    /// let triplets = vec![(0, 0, 2.0), (1, 1, 3.0)];
    /// let matrix = CsrMatrix::from_triplets(2, 2, &triplets);
    /// let x = vec![1.0, 2.0];
    /// let y = matrix.spmv(&x);
    /// assert_eq!(y, vec![2.0, 6.0]);
    /// ```
    pub fn spmv(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(
            x.len(),
            self.ncols,
            "Input vector length {} does not match matrix columns {}",
            x.len(),
            self.ncols
        );

        let mut y = vec![0.0; self.nrows];
        for (y_i, window) in y.iter_mut().zip(self.row_ptr.windows(2)) {
            let start = window[0];
            let end = window[1];
            let mut sum = 0.0;
            for k in start..end {
                sum += self.values[k] * x[self.col_indices[k]];
            }
            *y_i = sum;
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_csr_from_triplets_and_spmv() {
        // Matrix:
        // [1.0, 2.0, 0.0]
        // [0.0, 3.0, 4.0]
        // [5.0, 0.0, 6.0]
        let triplets = vec![
            (0, 0, 1.0),
            (0, 1, 2.0),
            (1, 1, 3.0),
            (1, 2, 4.0),
            (2, 0, 5.0),
            (2, 2, 6.0),
        ];

        let csr = CsrMatrix::from_triplets(3, 3, &triplets);

        assert_eq!(csr.nrows(), 3);
        assert_eq!(csr.ncols(), 3);
        assert_eq!(csr.nnz(), 6);
        assert_eq!(csr.values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(csr.col_indices(), &[0, 1, 1, 2, 0, 2]);
        assert_eq!(csr.row_ptr(), &[0, 2, 4, 6]);

        let x = vec![1.0, 2.0, 3.0];
        let y = csr.spmv(&x);
        // y[0] = 1*1 + 2*2 = 5
        // y[1] = 3*2 + 4*3 = 18
        // y[2] = 5*1 + 6*3 = 23
        assert_eq!(y, vec![5.0, 18.0, 23.0]);
    }

    #[test]
    fn test_csr_duplicates_and_zeros() {
        let triplets = vec![(0, 0, 1.0), (0, 0, 2.0), (0, 1, 0.0), (1, 1, 4.0)];

        let csr = CsrMatrix::from_triplets(2, 2, &triplets);
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.values(), &[3.0, 4.0]);
        assert_eq!(csr.col_indices(), &[0, 1]);
        assert_eq!(csr.row_ptr(), &[0, 1, 2]);
    }

    proptest! {
        #[test]
        fn prop_spmv_matches_dense(
            nrows in 1..=20usize,
            ncols in 1..=20usize,
            dense in prop::collection::vec(prop::collection::vec(-100.0..100.0f64, 1..=20), 1..=20),
            x in prop::collection::vec(-100.0..100.0f64, 1..=20)
        ) {
            let actual_rows = nrows.min(dense.len());
            let actual_cols = ncols.min(x.len());

            let mut triplets = Vec::new();
            for (r, row) in dense.iter().enumerate().take(actual_rows) {
                for (c, &val) in row.iter().enumerate().take(actual_cols) {
                    if val.abs() > 1e-6 {
                        triplets.push((r, c, val));
                    }
                }
            }

            let csr = CsrMatrix::from_triplets(actual_rows, actual_cols, &triplets);
            let x_slice = &x[..actual_cols];
            let y_csr = csr.spmv(x_slice);

            // Compute dense matrix-vector product for comparison
            let mut y_dense = vec![0.0; actual_rows];
            for (y_r, row) in y_dense.iter_mut().zip(dense.iter().take(actual_rows)) {
                for (&val, &x_c) in row.iter().zip(x_slice.iter()).take(actual_cols) {
                    if val.abs() > 1e-6 {
                        *y_r += val * x_c;
                    }
                }
            }

            prop_assert_eq!(y_csr.len(), y_dense.len());
            for (a, b) in y_csr.iter().zip(y_dense.iter()) {
                prop_assert!((a - b).abs() < 1e-9, "Mismatch: CSR {} vs Dense {}", a, b);
            }
        }
    }
}
