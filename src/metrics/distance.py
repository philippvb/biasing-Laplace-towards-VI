from torch import Tensor

def wasserstein_diag_gauss(mu_x:Tensor, cov_x:Tensor, mu_y:Tensor, cov_y:Tensor) -> Tensor:
    square = (mu_x - mu_y).square().mean()
    cov_sum = cov_x + cov_y
    cov_root = (cov_x * cov_y).sqrt()
    return square + (cov_sum - 2 * cov_root).mean()