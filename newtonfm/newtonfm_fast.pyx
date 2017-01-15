import numpy as np
import cython
from cython cimport floating

# Lotsa stuff below had nogil on it originally (from lightfm)

cdef extern from "math.h":
    double sqrt(double)
    double exp(double)
    double log(double)
    double floor(double)


# cdef extern from "stdlib.h":
#     ctypedef void const_void "const void"
#     void qsort(void *base, int nmemb, int size,
#                int(*compar)(const_void *, const_void *))
#     void* bsearch(const void *key, void *base, int nmemb, int size,
#                   int(*compar)(const_void *, const_void *))
#     void random_shuffle(void *first, void *last)


cdef slice_csr_rows(X, int [:] rows):

	cdef double [:] data
	cdef int [:] cols
	for row in rows:

	sliced = sp.coo_matrix()



def pcg(X, floating [:, :] Q, floating [:, :] G, D, double lambda_, bool do_pcond, double sub_rate):

    dtype = np.float64 if floating is double else np.float32

    cdef double zeta = 0.3
    cdef int cg_max_iter = 100

    if sub_rate < 1:
    	cdef int l = X.shape[0]
    	cdef int [:] whole = np.random.permutation(l)
    	cdef int max_idx = np.max((1, int(floor(sub_rate * l))))
    	cdef int [:] selected = np.sort(whole[:max_idx])

