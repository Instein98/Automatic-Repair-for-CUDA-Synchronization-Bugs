// cudamatrix/cu-kernels.cu

// Copyright 2009-2012  Karel Vesely
//                2013  Ehsan Variani
//                2013  Johns Hopkins University (author: Daniel Povey)
//                2013  Hainan Xu
//                2013  Xiaohui Zhang
//           2013-2015  Guoguo Chen
//           2016-2018  Shiyin Kang
//                2017  Hossein Hadian, Daniel Galvez

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

// In this file is the CUDA code of the CUDA kernels, plus the ANSI-C wrappers

//#include <cfloat>
//#include <limits>
//#include <math_constants.h>
//#include "cudamatrix/cu-kernels-ansi.h"



/***********************************************************************
 * Generic __device__ functions
 */
template<typename Real>
__device__
static Real _sum_reduce(Real buffer[]) {
  // Total number of active threads
  int32_cuda nTotalThreads = blockDim.x;
  __syncthreads();
  // perform tree-based reduction (sum)
  while (nTotalThreads > 1) {
    int32_cuda halfPoint = ((1 + nTotalThreads) >> 1); // divide by two
    // only the first half of the threads will be active.
    if (threadIdx.x >= halfPoint) { // was <
      // Get the shared value stored by another thread
      Real temp = 0.0;
      if (threadIdx.x < nTotalThreads) { // was +halfPoint
        temp = buffer[threadIdx.x]; // was +halfPoint
      }
      buffer[threadIdx.x - halfPoint] += temp;
    }
    __syncthreads();  // BD
    nTotalThreads = ((1 + nTotalThreads) >> 1); // divide by two.
  }
  // the result
  return buffer[0];
}

/***********************************************************************
 * CUDA kernels
 * the functions are templated to have the float/double operations
 */

/*
 * CuMatrix
 */

template<typename Real>
__global__
static void _copy_low_upp(Real* A, MatrixDim dimA) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i <= j || i >= dimA.rows)
    return;  // expression ??
  int index_1 = i * dimA.stride + j;
  int index_2 = j * dimA.stride + i;
  A[index_2] = A[index_1];
}

template<typename Real>
__global__
static void _copy_upp_low(Real* A, MatrixDim dimA) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (j <= i || j >= dimA.rows)
    return;
  int index_1 = i * dimA.stride + j;
  int index_2 = j * dimA.stride + i;
  A[index_2] = A[index_1];
}

// mat += diag(vec) * mat2.
template<typename Real>
__global__
static void _add_diag_vec_mat(Real alpha, Real *mat, MatrixDim mat_dim,
                              const Real *vec, const Real *mat2,
                              int mat2_row_stride, int mat2_col_stride,
                              Real beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // column index
  int j = blockIdx.y * blockDim.y + threadIdx.y; // row index

  int index = j * mat_dim.stride + i, index2 = j * mat2_row_stride
      + i * mat2_col_stride;

  if (i < mat_dim.cols && j < mat_dim.rows) {
    mat[index] = alpha * vec[j] * mat2[index2] + beta * mat[index];
  }
}

template<typename Real, typename OtherReal>
__global__
static void _copy_from_tp(Real* A, const OtherReal* B, MatrixDim dmat) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;  // col index
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
  if (i < dmat.cols && j < dmat.rows) {
    int32_cuda index_B = (j * (j + 1) / 2) + i;
    int32_cuda index_A = j * dmat.stride + i;
    if (i <= j) {
      A[index_A] = B[index_B];
    } else {
      A[index_A] = 0.0;
    }
  }
}

template<typename Real, typename OtherReal>
__global__
static void _copy_from_tp_trans(Real* A, const OtherReal* B, MatrixDim dmat) {
  // we interpret these indexes oppositely from normal, but it doesn't
  // matter as it's invoked in a symmetric way.
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  // transpose the indices used to index the source TpMatrix.
  if (i < dmat.rows && j < dmat.cols) {
    int32_cuda index_B = (j * (j + 1) / 2) + i;
    int32_cuda index_A = i * dmat.stride + j;
    if (i <= j) {
      A[index_A] = B[index_B];
    } else {
      A[index_A] = 0.0;
    }
  }
}

template<typename Real, typename OtherReal>
__global__
static void _copy_from_mat(Real* mat_out, const OtherReal* mat_in,
                           MatrixDim d_out, MatrixDim d_in) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;  // col-index
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;  // row-index.
  int32_cuda index_out = i + j * d_out.stride;
  int32_cuda index_in = i + j * d_in.stride;
  if (i < d_out.cols && j < d_out.rows)
    mat_out[index_out] = static_cast<Real>(mat_in[index_in]);
}

template<int TileDim, typename Real, typename OtherReal>
__global__
static void _copy_from_mat_trans(Real* mat_out, const OtherReal* mat_in,
                                 MatrixDim d_out, MatrixDim d_in) {
  // Use shared meme to achieve both coalesced memory reading and writing
  // '+1' to avoid bank conflict when reading sbuf
  __shared__ Real sbuf[TileDim][TileDim + 1];
  const int x = 50;
  int a = x++;
  a = ++x;
  a = a + x * 10;
  a = a * x + 10;
  a = a * (x + 10);
  const int32_cuda i_in = blockIdx.y * TileDim + threadIdx.y; // row-index
  const int32_cuda j_in = blockIdx.x * TileDim + threadIdx.x; // col-index
  const int32_cuda tile_stride_in = CU1DBLOCK / TileDim * d_in.stride;
  int32_cuda index_in = i_in * d_in.stride + j_in;

# pragma unroll
  for (int i = 0; i < TileDim; i += CU1DBLOCK / TileDim) {
    if (i_in + i < d_in.rows && j_in < d_in.cols) {
      sbuf[threadIdx.y + i][threadIdx.x] = static_cast<Real>(mat_in[index_in]);
    }
    index_in += tile_stride_in;
  }
  __syncthreads();

  // Grid is transposed, but block is not yet.
  // Warp (blockDim.x) is always along the row-dim.
  const int32_cuda i_out = blockIdx.x * TileDim + threadIdx.y;
  const int32_cuda j_out = blockIdx.y * TileDim + threadIdx.x;
  const int32_cuda tile_stride_out = CU1DBLOCK / TileDim * d_out.stride;
  int32_cuda index_out = i_out * d_out.stride + j_out;

# pragma unroll
  for (int i = 0; i < TileDim; i += CU1DBLOCK / TileDim) {
    if (i_out + i < d_out.rows && j_out < d_out.cols) {
      // block is tranposed when reading sbuf
      mat_out[index_out] = sbuf[threadIdx.x][threadIdx.y + i];
    }
    index_out += tile_stride_out;
  }
}

// Copy from CSR sparse matrix to dense matrix
//
// We use warpSize threads per row to access only the nnz elements.
// Every CU1DBLOCK/warpSize rows share one thread block.
// 1D grid to cover all rows.
template<typename Real, typename OtherReal>
__global__
static void _copy_from_smat(Real* mat, MatrixDim mat_dim,
                            const int* smat_row_ptr, const int* smat_col_idx,
                            const OtherReal* smat_val) {
  const int i = blockIdx.x * blockDim.y + threadIdx.y; // row idx
  if (i < mat_dim.rows) {
    const int nz_start = smat_row_ptr[i];
    const int nz_end = smat_row_ptr[i + 1];
    for (int nz_id = nz_start + threadIdx.x; nz_id < nz_end; nz_id +=
        warpSize) {
      const int j = smat_col_idx[nz_id]; // col idx
      mat[i * mat_dim.stride + j] = static_cast<Real>(smat_val[nz_id]);
    }
  }
}


/// Select a subset of the rows of a CSR SparseMatrix.
/// Sets 'out' to only the rows of 'in' that are listed
/// in 'row_indexes'.  'row_indexes' must be sorted and unique,
/// and satisfy 0 <= row_indexes[i] < in.size().
///
/// Note: 'out_row_ptr' is an input parameter that is calculated before
/// calling this kernel function
///
/// We use warpSize threads per row to access only the nnz elements.
/// Every CU1DBLOCK/warpSize rows share one thread block.
/// 1D grid to cover all selected rows.
template<typename Real>
__global__
static void _select_rows(const int* out_row_ptr, int* out_col_idx,
                         Real* out_val, const int* row_indexes,
                         const int num_selected_rows, const int* in_row_ptr,
                         const int* in_col_idx, const Real* in_val) {
  const int out_i = blockIdx.x * blockDim.y + threadIdx.y; // out row idx
  if (out_i < num_selected_rows) {
    const int in_i = row_indexes[out_i];
    const int in_row_start = in_row_ptr[in_i];
    const int out_row_start = out_row_ptr[out_i];
    const int row_length = in_row_ptr[in_i + 1] - in_row_start;
    for (int k = threadIdx.x; k < row_length; k += warpSize) {
      const int in_n = in_row_start + k;
      const int out_n = out_row_start + k;
      out_col_idx[out_n] = in_col_idx[in_n];
      out_val[out_n] = in_val[in_n];
    }
  }
}

// mat += alpha * smat
//
// We use warpSize threads per row to access only the nonzero elements.
// Every CU1DBLOCK/warpSize rows share one thread block.
// 1D grid to cover all rows of smat.
template<typename Real>
__global__
static void _add_smat(Real* mat, MatrixDim mat_dim, Real alpha,
                      const int* smat_row_ptr, const int* smat_col_idx,
                      const Real* smat_val) {
  const int i = blockIdx.x * blockDim.y + threadIdx.y; // row idx
  if (i < mat_dim.rows) {
    const int row_start = smat_row_ptr[i];
    const int row_end = smat_row_ptr[i + 1];
    for (int n = row_start + threadIdx.x; n < row_end; n += warpSize) {
      const int j = smat_col_idx[n]; // col idx of smat
      mat[i * mat_dim.stride + j] += alpha * smat_val[n];
    }
  }
}

// mat += alpha * smat^T
//
// We use warpSize threads per row to access only the nonzero elements.
// Every CU1DBLOCK/warpSize rows share one thread block.
// 1D grid to cover all rows of smat.
template<typename Real>
__global__
static void _add_smat_trans(Real* mat, MatrixDim mat_dim, Real alpha,
                            const int* smat_row_ptr, const int* smat_col_idx,
                            const Real* smat_val) {
  const int i = blockIdx.x * blockDim.y + threadIdx.y; // row idx
  if (i < mat_dim.cols) {
    const int row_start = smat_row_ptr[i];
    const int row_end = smat_row_ptr[i + 1];
    for (int n = row_start + threadIdx.x; n < row_end; n += warpSize) {
      const int j = smat_col_idx[n]; // col idx of smat
      mat[j * mat_dim.stride + i] += alpha * smat_val[n];
    }
  }
}

/// For each element x of the matrix, set it to
/// (x < 0 ? exp(x) : x + 1).
/// Use block/grid sizes for simple matrix ops
template<typename T>
__global__
static void _apply_exp_special(T* out, MatrixDim out_dim, const T* in,
                               int in_stride) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < out_dim.rows && j < out_dim.cols) {
    T x = in[i * in_stride + j];
    if (x < T(0)) {
      out[i * out_dim.stride + j] = exp(x);
    } else {
      out[i * out_dim.stride + j] = x + T(1);
    }
  }
}

/// Fill the array 'data' with the sequence [base ... base + length)
/// Use 1D block and 1D grid
template<typename T>
__global__
static void _sequence(T* data, int length, T base) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length) {
    data[i] = base + T(i);
  }
}

// Copy from CSR sparse matrix to transposed dense matrix
//
// We use warpSize threads per row to access only the nnz elements.
// Every CU1DBLOCK/warpSize rows share one thread block.
// 1D grid to cover all rows.
template<typename Real, typename OtherReal>
__global__
static void _copy_from_smat_trans(Real* mat, MatrixDim mat_dim,
                                  const int* smat_row_ptr,
                                  const int* smat_col_idx,
                                  const OtherReal* smat_val) {
  const int i = blockIdx.x * blockDim.y + threadIdx.y; // row idx of smat
  if (i < mat_dim.cols) {
    const int nz_start = smat_row_ptr[i];
    const int nz_end = smat_row_ptr[i + 1];
    for (int nz_id = nz_start + threadIdx.x; nz_id < nz_end; nz_id +=
        warpSize) {
      const int j = smat_col_idx[nz_id]; // col idx of smat
      mat[j * mat_dim.stride + i] = static_cast<Real>(smat_val[nz_id]);
    }
  }
}

// First stage of trace(mat * smat^T)
// We use warpSize threads per row to access only the nnz elements.
// Every CU1DBLOCK/warpSize rows share one thread block.
// 1D grid to cover all rows of smat.
template<typename Real>
__global__
static void _trace_mat_smat_trans(const Real* mat, MatrixDim mat_dim,
                                  const int* smat_row_ptr,
                                  const int* smat_col_idx, const Real* smat_val,
                                  Real* trace_vec) {
  const int i = blockIdx.x * blockDim.y + threadIdx.y; // row idx of smat
  if (i < mat_dim.rows) {
    const int nz_start = smat_row_ptr[i];
    const int nz_end = smat_row_ptr[i + 1];
    for (int nz_id = nz_start + threadIdx.x; nz_id < nz_end; nz_id +=
        warpSize) {
      const int j = smat_col_idx[nz_id]; // col idx of smat
      trace_vec[nz_id] = mat[i * mat_dim.stride + j] * smat_val[nz_id];
    }
  }
}

// First stage of trace(mat * smat)
// We use warpSize threads per row to access only the nnz elements.
// Every CU1DBLOCK/warpSize rows share one thread block.
// 1D grid to cover all rows of smat.
template<typename Real>
__global__
static void _trace_mat_smat(const Real* mat, MatrixDim mat_dim,
                            const int* smat_row_ptr, const int* smat_col_idx,
                            const Real* smat_val, Real* trace_vec) {
  const int i = blockIdx.x * blockDim.y + threadIdx.y; // row idx of smat
  if (i < mat_dim.cols) {
    const int nz_start = smat_row_ptr[i];
    const int nz_end = smat_row_ptr[i + 1];
    for (int nz_id = nz_start + threadIdx.x; nz_id < nz_end; nz_id +=
        warpSize) {
      const int j = smat_col_idx[nz_id]; // col idx of smat
      trace_vec[nz_id] = mat[j * mat_dim.stride + i] * smat_val[nz_id];
    }
  }
}

template<typename Real>
__global__
static void _apply_exp(Real* mat, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride;
  if (i < d.cols && j < d.rows) {
    mat[index] = exp(mat[index]);
  }
}

template<typename Real>
__global__
static void _apply_exp_limited(Real* mat, MatrixDim d,
                               Real lower_limit, Real upper_limit) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride;
  if (i < d.cols && j < d.rows) {
    Real x = mat[index];
    // I'm writing !(x >= lower_limit) instead of (x < lower_limit) so that
    // nan's will be set to the lower-limit.
    if (!(x >= lower_limit))
      x = lower_limit;
    else if (x > upper_limit)
      x = upper_limit;
    mat[index] = exp(x);
  }
}


template<typename Real>
__global__
static void _scale_diag_packed(Real* mat, Real value, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = ((i + 1) * (i + 2) / 2) - 1;
  if (i < dim) {
    mat[index] = value * mat[index];
  }
}

template<typename Real>
__global__
static void _set_diag(Real* mat, Real value, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = i + i * d.stride;
  if (i < d.rows && i < d.cols) {
    mat[index] = value;
  }
}

template<typename Real>
__global__
static void _set_diag_packed(Real* mat, Real value, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = ((i + 1) * (i + 2) / 2) - 1;
  if (i < dim) {
    mat[index] = value;
  }
}

template<typename Real>
__global__
static void _add_diag_packed(Real* mat, Real value, int dim) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda index = ((i + 1) * (i + 2) / 2) - 1;
  if (i < dim) {
    mat[index] = mat[index] + value;
  }
}

template<typename Real>
__global__
static void _set_const(Real* mat, Real value, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;  // column
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;  // row
  int32_cuda index = i + j * d.stride;
  if (i < d.cols && j < d.rows)
    mat[index] = value;
}

template<typename Real>
__global__
static void _set_zero_above_diag(Real* mat, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride;
  if (i < d.cols && j < i)
    mat[index] = 0.0;
}

template<typename Real>
__global__
static void _add(Real* mat, Real value, MatrixDim d) {
  int32_cuda i = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda j = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index = i + j * d.stride;
  if (i < d.cols && j < d.rows)
    mat[index] = mat[index] + value;
}