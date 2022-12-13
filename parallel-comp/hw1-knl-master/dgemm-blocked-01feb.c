#include <immintrin.h>
#include <stdio.h>
const char* dgemm_desc = "Simple blocked dgemm.";
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#define PFETCH_DIST_A 0
#define PFETCH_DIST_B 0

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block_microkernel(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // Declare Register Variables
    __m512d Ar_0, Ar_1, Ar_2, Ar_3, Br_0, Br_1, Br_2, Br_3; 
    __m512d Cr_0, Cr_1, Cr_2, Cr_3, Cr_4, Cr_5, Cr_6, Cr_7, Cr_8, Cr_9, Cr_10, Cr_11, Cr_12, Cr_13, Cr_14, Cr_15;
    // For each row i of A 
    for (int m = 0; m < M; m+=32) {
        // For each column j of B
        for (int n = 0; n < N; n+=4) {
          Cr_0 = _mm512_load_pd(C + m + n*lda);
          Cr_1 = _mm512_load_pd(C + m + (n+1)*lda);
          Cr_2 = _mm512_load_pd(C + m + (n+2)*lda);
          Cr_3 = _mm512_load_pd(C + m + (n+3)*lda);

          Cr_4 = _mm512_load_pd(C + m+8 + n*lda);
          Cr_5 = _mm512_load_pd(C + m+8 + (n+1)*lda);
          Cr_6 = _mm512_load_pd(C + m+8 + (n+2)*lda);
          Cr_7 = _mm512_load_pd(C + m+8 + (n+3)*lda);

          Cr_8 = _mm512_load_pd(C + m+16 + n*lda);
          Cr_9 = _mm512_load_pd(C + m+16 + (n+1)*lda);
          Cr_10 = _mm512_load_pd(C + m+16 + (n+2)*lda);
          Cr_11 = _mm512_load_pd(C + m+16 + (n+3)*lda);

          Cr_12 = _mm512_load_pd(C + m+24 + n*lda);
          Cr_13 = _mm512_load_pd(C + m+24 + (n+1)*lda);
          Cr_14 = _mm512_load_pd(C + m+24 + (n+2)*lda);
          Cr_15 = _mm512_load_pd(C + m+24 + (n+3)*lda);

          for (int k = 0; k < K; ++k) {
              Ar_0 = _mm512_load_pd(A + m + k*lda);
              Ar_1 = _mm512_load_pd(A + (m+8) + k*lda);
              Ar_2 = _mm512_load_pd(A + (m+16) + k*lda);
              Ar_3 = _mm512_load_pd(A + (m+24) + k*lda);
            //   _mm_prefetch(A + m+32 + k*lda + PFETCH_DIST_A, _MM_HINT_T0); 

              Br_0 = _mm512_set1_pd(*(B + n*lda + k));
              Br_1 = _mm512_set1_pd(*(B + (n+1)*lda + k));
              Br_2 = _mm512_set1_pd(*(B + (n+2)*lda + k));
              Br_3 = _mm512_set1_pd(*(B + (n+3)*lda + k));
            //   _mm_prefetch(B + (n+8)*lda + k + PFETCH_DIST_B, _MM_HINT_T0);

              Cr_0 = _mm512_fmadd_pd (Ar_0, Br_0, Cr_0); 
              Cr_1 = _mm512_fmadd_pd (Ar_0, Br_1, Cr_1); 
              Cr_2 = _mm512_fmadd_pd (Ar_0, Br_2, Cr_2); 
              Cr_3 = _mm512_fmadd_pd (Ar_0, Br_3, Cr_3); 

              Cr_4 = _mm512_fmadd_pd (Ar_1, Br_0, Cr_4); 
              Cr_5 = _mm512_fmadd_pd (Ar_1, Br_1, Cr_5); 
              Cr_6 = _mm512_fmadd_pd (Ar_1, Br_2, Cr_6); 
              Cr_7 = _mm512_fmadd_pd (Ar_1, Br_3, Cr_7); 

              Cr_8 = _mm512_fmadd_pd (Ar_2, Br_0, Cr_8); 
              Cr_9 = _mm512_fmadd_pd (Ar_2, Br_1, Cr_9); 
              Cr_10 = _mm512_fmadd_pd (Ar_2, Br_2, Cr_10); 
              Cr_11 = _mm512_fmadd_pd (Ar_2, Br_3, Cr_11); 

              Cr_12 = _mm512_fmadd_pd (Ar_3, Br_0, Cr_12); 
              Cr_13 = _mm512_fmadd_pd (Ar_3, Br_1, Cr_13); 
              Cr_14 = _mm512_fmadd_pd (Ar_3, Br_2, Cr_14); 
              Cr_15 = _mm512_fmadd_pd (Ar_3, Br_3, Cr_15); 
          } 
          //store into 4x4 square
          _mm512_store_pd(C + m + n*lda, Cr_0);
          _mm512_store_pd(C + m + (n+1)*lda, Cr_1);
          _mm512_store_pd(C + m + (n+2)*lda, Cr_2);
          _mm512_store_pd(C + m + (n+3)*lda, Cr_3);

          _mm512_store_pd(C + m+8 + n*lda, Cr_4);
          _mm512_store_pd(C + m+8 + (n+1)*lda, Cr_5);
          _mm512_store_pd(C + m+8 + (n+2)*lda, Cr_6);
          _mm512_store_pd(C + m+8 + (n+3)*lda, Cr_7);

          _mm512_store_pd(C + m+16 + n*lda, Cr_8);
          _mm512_store_pd(C + m+16 + (n+1)*lda, Cr_9);
          _mm512_store_pd(C + m+16 + (n+2)*lda, Cr_10);
          _mm512_store_pd(C + m+16 + (n+3)*lda, Cr_11);

          _mm512_store_pd(C + m+24 + n*lda, Cr_12);
          _mm512_store_pd(C + m+24 + (n+1)*lda, Cr_13);
          _mm512_store_pd(C + m+24 + (n+2)*lda, Cr_14);
          _mm512_store_pd(C + m+24 + (n+3)*lda, Cr_15);
        }
    }
}

// Copies matrix into allocated and padded matrix
static double* pad(int lda, int lda_padded, double* M) {
    double* M_padded = (double*) _mm_malloc(lda_padded * lda_padded * sizeof(double), 64);
    for (int i = 0; i < lda; i++) {
        for (int j = 0; j < lda; j++) {
            M_padded[i * lda_padded + j] = M[i * lda + j];
        }
    }
    return M_padded;
}

// Copies padded matrix into original, unpadded matrix
static void unpad(int lda, int lda_padded, double* M, double* M_padded) {
    for (int i = 0; i < lda; i++) {
        for (int j = 0; j < lda; j++) {
            M[i * lda + j] = M_padded[i * lda_padded + j];
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
    int lda_padded = lda;
    if (lda % 32 != 0) {
        lda_padded += 32 - (lda % 32);
    }
    // Repack A to row-major order and pad A to be lda-by-lda_padded
    double* A_padded = pad(lda, lda_padded, A);
    // Pad B to be lda_padded-by-lda
    double* B_padded = pad(lda, lda_padded, B);
    // Pad C to be lda_padded-by-lda
    double* C_padded = pad(lda, lda_padded, C);

    for (int k = 0; k < lda_padded; k += BLOCK_SIZE) {
        for (int n = 0; n < lda_padded; n += BLOCK_SIZE) {
            for (int m = 0; m < lda_padded; m += BLOCK_SIZE) {
                int K = min(BLOCK_SIZE, lda_padded - k);
                int M = min(BLOCK_SIZE, lda_padded - m);                
                int N = min(BLOCK_SIZE, lda_padded - n);            
                do_block_microkernel(lda_padded, M, N, K, A_padded + m + k*lda_padded, B_padded + k + n*lda_padded, C_padded + m + n*lda_padded);       
            }
        }
    }
    unpad(lda, lda_padded, C, C_padded);
    _mm_free(A_padded);
    _mm_free(B_padded);    
}
