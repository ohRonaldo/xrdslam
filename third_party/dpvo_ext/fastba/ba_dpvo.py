import cuda_ba

neighbors = cuda_ba.neighbors
reproject = cuda_ba.reproject


def bundle_adjust_dpvo(poses,
                       patches,
                       intrinsics,
                       target,
                       weight,
                       lmbda,
                       ii,
                       jj,
                       kk,
                       t0,
                       t1,
                       iterations=2):
    return cuda_ba.forward(poses.data, patches, intrinsics, target, weight,
                           lmbda, ii, jj, kk, t0, t1, iterations)
