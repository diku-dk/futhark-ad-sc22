-- Manual version of sparse-kmeans for k=10
-- ==
-- entry: calculate_objective
-- compiled input @ data/movielens.in.gz output @ data/movielens.out
-- compiled input @ data/nytimes.in.gz
-- compiled input @ data/scrna.in.gz

import "kmeans_sparse_manual"

entry calculate_objective [nnz][np1] 
         (values: [nnz]f32)
         (indices_data: [nnz]i64) 
         (pointers: [np1]i64) =

  let fix_iter = false
  let threshold = 0.005f32
  let num_iterations = 10 --250i64
  let k = 10i64

  let (_delta, _i, cluster_centers) =
    kmeans_seq_rows fix_iter threshold num_iterations k
                    values indices_data pointers

  in  cluster_centers
