-- ==
-- entry: calculate_objective
-- input @ data/n10000.in.gz
-- input @ data/n30000.in.gz

-- ==
-- entry: calculate_jacobian
-- input @ data/n10000.in.gz
-- input @ data/n30000.in.gz

import "lib/github.com/diku-dk/linalg/linalg"
module la = mk_linalg f32

def helmholtz [n] (R: f32) (T: f32) (b: [n]f32) (A: [n][n]f32) (xs: [n]f32) : f32 =
  let bxs = la.dotprod b xs
  let term1 = map (\x -> f32.log (x / (1 - bxs))) xs |> f32.sum
  let term2 = la.dotprod xs (la.matvecmul_row A xs) / (f32.sqrt(8) * bxs)
  let term3 = (1 + (1 + f32.sqrt(2)) * bxs) / (1 + (1 - f32.sqrt(2)) * bxs) |> f32.log
  in R * T * term1 - term2 * term3

entry foo [n] (R: f32) (T: f32) (b: [n]f32) (A: [n][n]f32) (xs: [n]f32)  =
  (R, T, b, A, xs)
				  
entry calculate_objective = helmholtz
											      
entry calculate_jacobian [n] (R: f32) (T: f32) (b: [n]f32) (A: [n][n]f32) (xs: [n]f32) =
  vjp (helmholtz R T b A) xs 1.0
