-- AD version of sparse-kmeans for k=10

-- ==
-- entry: calculate_objective
-- compiled input @ data/movielens.in.gz output @ data/movielens.out
-- compiled input @ data/nytimes.in.gz
-- compiled input @ data/scrna.in.gz

import "kmeans_sparse"

entry calculate_objective [nnz][np1] 
         (values: [nnz]f32)
         (indices_data: [nnz]i64) 
         (pointers: [np1]i64) =
  let fix_iter = false
  let threshold = 0.005f32
  let num_iterations = 10i32 --250i32
  let k = 10i64

  let (cluster_centers, _num_its) =
      kmeansSpAD k threshold num_iterations fix_iter
                 values
                 indices_data
                 pointers
  in cluster_centers

