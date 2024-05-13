import torch

from neurograph.models.decomp import moving_avg, series_decomp, series_decomp_multi


def test_moving_avg():
    b, r, t = 3, 32, 16
    x = torch.randn(b, r, t)

    for k in (1, 3, 5):
        mv = moving_avg(k)
        y_1 = mv(x)
        assert x.shape == y_1.shape

        sd = series_decomp(k)
        season_1, trend_1 = sd(x)
        assert x.shape == season_1.shape
        assert x.shape == trend_1.shape

        sdm = series_decomp_multi([k, k + 1, 2 * k])
        season_2, trend_2 = sdm(x)
        assert x.shape == season_2.shape
        assert x.shape == trend_2.shape
