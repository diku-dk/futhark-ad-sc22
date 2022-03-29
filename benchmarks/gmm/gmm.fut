let fst (x,_) = x

let snd (_,y) = y

let sumBy 'a (f : a -> f32)  (xs : []a) : f32 = map f xs |> f32.sum

let l2normSq (v : []f32) = sumBy (\x -> x * x) v

let logsumexp = sumBy f32.exp >-> f32.log

let frobeniusNormSq (mat : [][]f32) = sumBy (\x -> x * x) (flatten mat)

let unpackQ [d] (logdiag: [d]f32) (lt: []f32) : [d][d]f32  =
  tabulate_2d d d (\j i ->
                    if i < j then 0
                    else if i == j then f32.exp logdiag[i]
                    else lt[d * j + i - j - 1 - j * (j + 1) / 2])

let logGammaDistrib (a : f32) (p : i64) =
  0.25 * f32.i64 p * f32.i64 (p - 1) * f32.log f32.pi +
  ((1...p) |> sumBy (\j -> f32.lgamma (a + 0.5 * f32.i64 (1 - j))))

let logsumexp_DArray (arr : []f32) =
    let mx = f32.maximum arr
    let sumShiftedExp = arr |> sumBy (\x -> f32.exp (x - mx))
    in f32.log sumShiftedExp + mx

let logWishartPrior [k] (qs: [k][][]f32) (sums: [k]f32) wishartGamma wishartM p =
    let n = p + wishartM + 1
    let c = f32.i64 (n * p) * (f32.log wishartGamma - 0.5 * f32.log 2) -
            (logGammaDistrib (0.5 * f32.i64 n) p)
    let frobenius = sumBy frobeniusNormSq qs
    let sumQs = f32.sum sums
    in 0.5 * wishartGamma * wishartGamma * frobenius -
       f32.i64 wishartM * sumQs - f32.i64 k * c

let matmultr [d][n] (a: [d][n]f32) (b: [d][d]f32) : [n][d]f32 =
    map (\a_col ->
            map (\b_col ->
                    map2 (*) a_col b_col  |> f32.sum
                ) (transpose b)
        ) (transpose a)

let gmmObjective [d][k][n]
                 (alphas: [k]f32)
                 (means:  [k][d]f32)
                 (icf:    [k][] f32)
                 (xtr:    [d][n]f32)
                 (wishartGamma: f32)
                 (wishartM: i64) =
    let constant = -(f32.i64 n * f32.i64 d * 0.5 * f32.log (2 * f32.pi))
    let logdiags = icf[:,:d]
    let lts = icf[:,d:]
    let qs = map2 unpackQ logdiags lts
    let diffs = tabulate_3d k d n
            (\ b q a -> xtr[q,a] - means[b, q]) |> opaque
    let qximeans_mats = map2 matmultr diffs qs |> opaque  -- : [k][n][d]f32
    let tmp1 = map (map l2normSq) qximeans_mats |> opaque -- tmp1 : [k][n]f32

    let sumQs= map f32.sum logdiags -- sumQs : [k]f32
    let tmp2 = map3(\ (row: [n]f32) alpha sumQ ->
                        map (\ el ->
                                -0.5 * el + alpha + sumQ
                            ) row
                   ) tmp1 alphas sumQs
    let tmp3  = map logsumexp_DArray (transpose tmp2)
    let slse  = f32.sum tmp3
    in constant + slse  - f32.i64 n * logsumexp alphas +
       logWishartPrior qs sumQs wishartGamma wishartM d

let grad f x = vjp f x 1f32

entry calculate_objective [d][k][n]
                          (alphas: [k]f32)
                          (means: [k][d]f32)
                          (icf: [k][]f32)
                          (x:   [n][d]f32)
                          (w_gamma: f32) (w_m: i64) =
  gmmObjective alphas means icf (transpose x) w_gamma w_m

entry calculate_jacobian [d][k][n]
                         (alphas: [k]f32)
                         (means: [k][d]f32)
                         (icf: [k][]f32)
                         (x:   [n][d]f32)
                         (w_gamma: f32) (w_m: i64) =
  let (alphas_, means_, icf_) =
    grad (\(a, m, i) -> gmmObjective a m i (transpose x) w_gamma w_m) (alphas, means, icf)
  in (alphas_, means_, icf_)

-- ==
-- entry: calculate_objective
-- compiled input @ data/f32/1k/gmm_d64_K200.in.gz
-- compiled input @ data/f32/1k/gmm_d128_K200.in.gz
-- compiled input @ data/f32/10k/gmm_d32_K200.in.gz
-- compiled input @ data/f32/10k/gmm_d64_K25.in.gz
-- compiled input @ data/f32/10k/gmm_d128_K25.in.gz
-- compiled input @ data/f32/10k/gmm_d128_K200.in.gz

-- ==
-- entry: calculate_jacobian
-- compiled input @ data/f32/1k/gmm_d64_K200.in.gz
-- compiled input @ data/f32/1k/gmm_d128_K200.in.gz
-- compiled input @ data/f32/10k/gmm_d32_K200.in.gz
-- compiled input @ data/f32/10k/gmm_d64_K25.in.gz
-- compiled input @ data/f32/10k/gmm_d128_K25.in.gz
-- compiled input @ data/f32/10k/gmm_d128_K200.in.gz
