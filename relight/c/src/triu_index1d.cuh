#pragma once

#include <cuda.h>
#include <cuda_runtime.h>


__device__ __forceinline__ void triu_index1d(int i, int j, int n, int &index_1d) {
	int ii, jj;
	if (i < j) {
		ii = i;
		jj = j;
	} else {
		ii = j;
		jj = i;
	}

	// ii = 0, offset = 0
	// ii = 1, offset = n
	// ii = 2, offset = n + (n - 1)
	// ii = 3, offset = n + (n - 1) + (n - 2)
	// ii = m, offset = n + (n - 1) + ... + (n - m + 1) = (n  + (n - m + 1)) * m / 2
	int offset = (n + (n - ii + 1)) * ii / 2;
	index_1d = offset + jj - ii;
}


__device__ __forceinline__ void index1d_triu(int index_1d, int n, int &i, int &j) {
	int m = n - 1;
	while (index_1d > m) {
		index_1d -= m;
		m -= 1;
	}
	i = m;
	j = index_1d + m;
}
