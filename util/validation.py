import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


def sample_representative_validation_set_across_attributes(
        gdf: gpd.GeoDataFrame,
        target_attrs: list[str],
        representative_attrs: list[str],
        val_size: float,
    ) -> np.typing.NDArray[np.bool_]:
    """
    Generate a representative validation set for inferring missing building attributes
    by selecting buildings that are close to the unlabeled buildings in feature space
    for multiple attributes (equally represented).
    """
    # Generate attribute-specific validation masks and combine them
    val_masks = [sample_representative_validation_set(gdf, attr, representative_attrs, val_size) for attr in target_attrs]
    val_mask = np.logical_or(*val_masks)

    # Limit the size of the validation set to the specified fraction of labeled data
    n_labeled = (gdf[target_attrs].notna().any(axis=1)).sum()
    n = min(int(n_labeled * val_size), val_mask.sum())
    val_idx = gdf[val_mask].sample(n).index
    val_mask = gdf.index.isin(val_idx)

    return val_mask


def sample_representative_validation_set(
        gdf: gpd.GeoDataFrame,
        attr: str,
        representative_attrs: list[str],
        val_size: float,
    ) -> np.typing.NDArray[np.bool_]:
    """
    Generate a representative validation set for inferring missing building attributes
    by selecting buildings that are close to the unlabeled buildings in feature space.
    """
    gdf = gdf[representative_attrs + [attr]].replace([np.inf, -np.inf], np.nan)

    for col in representative_attrs:
        gdf[col] = gdf[col].fillna(gdf[col].mean())

    na_mask = gdf[attr].isna()
    df_labeled = gdf[~na_mask]
    df_unlabeled = gdf[na_mask]

    # return a random sample mask when there are no missing attributes
    if df_unlabeled.empty:
        return np.random.rand(len(gdf)) < val_size

    # return empty sample mask when there are no attributes
    if df_labeled.empty:
        return np.zeros(len(gdf), dtype=bool)

    X_labeled = df_labeled[representative_attrs]
    X_unlabeled = df_unlabeled[representative_attrs]

    # increase number of neighbors when unlabeled_ratio is below desired validation set size (val_size)
    unlabeled_ratio = len(df_unlabeled) / len(df_labeled)
    n = int(np.ceil((1 / unlabeled_ratio) * val_size))

    # determine representatives among labeled data that are close to unlabeled data
    nn = NearestNeighbors(n_neighbors=n).fit(X_labeled)
    dist, idx = nn.kneighbors(X_unlabeled)
    representative_indices = X_labeled.index[np.unique(idx.flatten())]

    # decrease number of representatives when unlabeled_ratio is above desired validation set size (val_size)
    target_size = int(len(df_labeled) * val_size)
    if len(representative_indices) > target_size:
        representative_indices = np.random.choice(representative_indices, target_size, replace=False)

    val_mask = gdf.index.isin(representative_indices)

    return val_mask
