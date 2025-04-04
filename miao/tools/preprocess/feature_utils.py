import polars as pl

las = ['la1', 'la2', 'la5', 'la10', 'la15', 'la30', 'la60', 'la120', 'la180', 'la240', 'la300', 'la360']

def ema(col, length=100)->pl.Expr:
    target_col = pl.col(col)
    ema = target_col.ewm_mean(span=length)
    ema_reduced = target_col / ema
    return ema_reduced

def p_change(col, length=1)->pl.Expr:
    '''变化率，percentage change'''
    clip_col = pl.col(col).clip(lower_bound=1e-5)  # todo 加一个动态下界
    target_col = clip_col.log()
    return target_col - target_col.shift(length)

def zscore(col, length=100)->pl.Expr:
    '''zscore'''
    target_col = pl.col(col)
    mean = target_col.rolling_mean(length)
    std = target_col.rolling_std(length)
    zscore = (target_col - mean) / std
    zscore = zscore.clip(-4, 4)
    return zscore

def featclip(feat: pl.Expr) -> pl.Expr:
    """
    1 filter quantile 0.01~0.99
    """
    featclip = feat.clip(
        feat.quantile(0.01),
        feat.quantile(0.99),
    )
    return sanitise(featclip)

def sanitise(pl_col) -> pl.Expr:
    """
    replace inf, -inf, NaN, and null to 0
    """
    return (
        pl.when(pl_col.is_infinite() | pl_col.is_nan() | pl_col.is_null())
        .then(0)
        .otherwise(pl_col)
    )