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

            // If entry at same (row, col) exists, accumulate value
            if current_row == r && col_indices.last() == Some(&c) {
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
