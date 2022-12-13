#include <immintrin.h>
#include <stdio.h>
const char* dgemm_desc = "Simple blocked dgemm.";
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

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
            // load 32*4 C block to registers     
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

            // prefetch next 32*4 C block to cache, 32 rows away
            _mm_prefetch(C + m+32 + n*lda , _MM_HINT_T0);
            _mm_prefetch(C + m+32 + n*lda + 8 , _MM_HINT_T0);
            _mm_prefetch(C + m+32 + n*lda + 16 , _MM_HINT_T0);
            _mm_prefetch(C + m+32 + n*lda + 24 , _MM_HINT_T0);

            _mm_prefetch(C + m+32 + (n+1)*lda , _MM_HINT_T0);
            _mm_prefetch(C + m+32 + (n+1)*lda + 8 , _MM_HINT_T0);
            _mm_prefetch(C + m+32 + (n+1)*lda + 16 , _MM_HINT_T0);
            _mm_prefetch(C + m+32 + (n+1)*lda + 24 , _MM_HINT_T0);

            _mm_prefetch(C + m+32 + (n+2)*lda , _MM_HINT_T0);
            _mm_prefetch(C + m+32 + (n+2)*lda + 8 , _MM_HINT_T0);
            _mm_prefetch(C + m+32 + (n+2)*lda + 16 , _MM_HINT_T0);
            _mm_prefetch(C + m+32 + (n+2)*lda + 24 , _MM_HINT_T0);  

            _mm_prefetch(C + m+32 + (n+3)*lda , _MM_HINT_T0);
            _mm_prefetch(C + m+32 + (n+3)*lda + 8 , _MM_HINT_T0);
            _mm_prefetch(C + m+32 + (n+3)*lda + 16 , _MM_HINT_T0);
            _mm_prefetch(C + m+32 + (n+3)*lda + 24 , _MM_HINT_T0);

            // prefetch 8*4 B block to cache, use for first 8 kernel opreations
            _mm_prefetch(B + n*lda, _MM_HINT_T0);
            _mm_prefetch(B + (n+1)*lda, _MM_HINT_T0);
            _mm_prefetch(B + (n+2)*lda, _MM_HINT_T0);
            _mm_prefetch(B + (n+3)*lda, _MM_HINT_T0);        

            for (int k = 0; k < K; ++k) {
                Ar_0 = _mm512_load_pd(A + m + k*lda);
                Ar_1 = _mm512_load_pd(A + (m+8) + k*lda);
                Ar_2 = _mm512_load_pd(A + (m+16) + k*lda);
                Ar_3 = _mm512_load_pd(A + (m+24) + k*lda);

                Br_0 = _mm512_set1_pd(*(B + n*lda + k));
                Br_1 = _mm512_set1_pd(*(B + (n+1)*lda + k));
                Br_2 = _mm512_set1_pd(*(B + (n+2)*lda + k));
                Br_3 = _mm512_set1_pd(*(B + (n+3)*lda + k));

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
          //store into 32*4 C block
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

    // Pad A to be lda_padded-by-lda
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

                // initial prefetch for 32*4 C block
                _mm_prefetch(C_padded + m + n*lda_padded, _MM_HINT_T0);
                _mm_prefetch(C_padded + m + n*lda_padded + 8, _MM_HINT_T0);
                _mm_prefetch(C_padded + m + n*lda_padded + 16, _MM_HINT_T0);
                _mm_prefetch(C_padded + m + n*lda_padded + 24, _MM_HINT_T0);
                _mm_prefetch(C_padded + m + (n+1)*lda_padded, _MM_HINT_T0);
                _mm_prefetch(C_padded + m + (n+1)*lda_padded + 8, _MM_HINT_T0);
                _mm_prefetch(C_padded + m + (n+1)*lda_padded + 16, _MM_HINT_T0);
                _mm_prefetch(C_padded + m + (n+1)*lda_padded + 24, _MM_HINT_T0); 
                _mm_prefetch(C_padded + m + (n+2)*lda_padded, _MM_HINT_T0);
                _mm_prefetch(C_padded + m + (n+2)*lda_padded + 8, _MM_HINT_T0);
                _mm_prefetch(C_padded + m + (n+2)*lda_padded + 16, _MM_HINT_T0);
                _mm_prefetch(C_padded + m + (n+2)*lda_padded + 24, _MM_HINT_T0);                    
                _mm_prefetch(C_padded + m + (n+3)*lda_padded, _MM_HINT_T0);
                _mm_prefetch(C_padded + m + (n+3)*lda_padded + 8, _MM_HINT_T0);
                _mm_prefetch(C_padded + m + (n+3)*lda_padded + 16, _MM_HINT_T0);
                _mm_prefetch(C_padded + m + (n+3)*lda_padded + 24, _MM_HINT_T0);              
                                                                 
                do_block_microkernel(lda_padded, M, N, K, A_padded + m + k*lda_padded, B_padded + k + n*lda_padded, C_padded + m + n*lda_padded);       
            }
        }
    }

    unpad(lda, lda_padded, C, C_padded);
    _mm_free(A_padded);
    _mm_free(B_padded);    
}
