include ../../common.mk
NAME=kmeans_sparse
FUTHARK_BENCH_OPTIONS=
JAC_SPEEDUP=False
OBJ_SPEEDUP=True

.PHONY: results
results: kmeans_sparse_k10.json kmeans_sparse_pytorch.json kmeans_sparse_manual_k10.json kmeans_sparse_jax.json
	make gen_results
