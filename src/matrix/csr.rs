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
}
