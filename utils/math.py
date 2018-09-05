from numpy.core.umath import float_power, exp, sqrt, pi, log

TAU = 2 * pi


def decay_function_exponential_with_decay_factor(decay_factor) -> callable:
    # Decay formula for activation a, original activation a_0, decay factor d, time t:
    #   a = a_0 d^t
    #
    # In traditional formulation of exponential decay, this is equivalent to:
    #   a = a_0 e^(-λt)
    # where λ is the decay constant.
    #
    # I.e.
    #   d = e^(-λ)
    #   λ = - ln d
    assert 0 < decay_factor <= 1

    def decay_function(age, original_activation):
        return original_activation * (decay_factor ** age)

    return decay_function


def decay_function_exponential_with_half_life(half_life) -> callable:
    assert half_life > 0
    # Using notation from above, with half-life hl
    #   λ = ln 2 / ln hl
    #   d = 2 ^ (- 1 / hl)
    decay_factor = float_power(2, - 1 / half_life)
    return decay_function_exponential_with_decay_factor(decay_factor)


def decay_function_gaussian_with_sd(sd, height_coef=1, centre=0) -> callable:
    """Gaussian decay with sd specifying the number of ticks."""
    assert height_coef > 0
    assert sd > 0

    def decay_function(age, original_activation):
        height = original_activation * height_coef
        return height * exp((-1) * (((age - centre) ** 2) / (2 * sd * sd)))

    return decay_function


# TODO: ensure that realigned zero should be the default
def decay_function_lognormal_with_sd(sd: float, realign_zero: bool = True) -> callable:
    """
    Lognormal decay with sd specifying the number of ticks.
    :param sd:
    :param realign_zero:
        If True, decay proceeds from zero point.
        If False, decay proceeds according to an actual lognormal PDF curve.
    :return:
    """

    height = _lognormal_modal_height(0, sd)
    if realign_zero:
        mode = _lognormal_mode(0, sd)

        def decay_function(age, original_activation):
            return original_activation * _lognormal_pdf(age + mode, 0, sd) / height
    else:

        def decay_function(age, original_activation):
            return original_activation * _lognormal_pdf(age, 0, sd) / height

    return decay_function


def _lognormal_pdf(x, mu, sigma):
    """Lognormal PDF."""
    coef = 1 / (x * sigma * sqrt(TAU))
    expo = exp((-1) * ((log(x) - mu) ** 2) / (2 * sigma * sigma))
    return coef * expo


def _lognormal_mode(mu, sigma):
    """The mode of the lognormal function."""
    return exp(mu - (sigma * sigma))


def _lognormal_modal_height(mu, sigma):
    """The height of the lognormal function at its mode. This is its maximum height."""
    return _lognormal_pdf(_lognormal_mode(mu, sigma), mu, sigma)

