"""Meridional-grid and blade-angle-law helpers shared by the quasi-3D
throughflow examples (hub-to-shroud + blade-to-blade coupled models)."""

import numpy as np

from adet.geometry import BezierCurve


def resample_by_arc_length(
    curve: BezierCurve, n_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a (finely discretized) curve at ``n_points`` equally
    spaced arc-length stations."""
    ds = np.hypot(np.diff(curve.z_coords), np.diff(curve.r_coords))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    s_stations = np.linspace(0.0, s[-1], n_points)
    z = np.interp(s_stations, s, curve.z_coords)
    r = np.interp(s_stations, s, curve.r_coords)
    return z, r


def build_meridional_grid(
    hub_z: np.ndarray,
    hub_r: np.ndarray,
    shroud_z: np.ndarray,
    shroud_r: np.ndarray,
    n_span: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid of intermediate quasi streamsurfaces, by linear interpolation
    along the hub-to-shroud quasi-orthogonals.

    Returns
    -------
    Z, R : ndarray, shape (n_stream, n_span)
        Axis 0 is streamwise (hub/shroud curve direction), axis 1 is
        spanwise (hub -> shroud).
    """
    eta = np.linspace(0.0, 1.0, n_span)[None, :]
    z = hub_z[:, None] * (1.0 - eta) + shroud_z[:, None] * eta
    r = hub_r[:, None] * (1.0 - eta) + shroud_r[:, None] * eta
    return z, r


def spanwise_normal_curvature(z: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Normal curvature :math:`1/\\mathfrak{R}_n` of each quasi
    streamsurface, projected onto the local hub-to-shroud direction."""
    n_stream, n_span = z.shape

    n_z = z[:, -1] - z[:, 0]
    n_r = r[:, -1] - r[:, 0]
    n_mag = np.hypot(n_z, n_r)
    n_z, n_r = n_z / n_mag, n_r / n_mag

    inv_Rn = np.empty((n_stream, n_span))
    for j in range(n_span):
        ds = np.hypot(np.diff(z[:, j]), np.diff(r[:, j]))
        s = np.concatenate([[0.0], np.cumsum(ds)])
        dz_ds = np.gradient(z[:, j], s)
        dr_ds = np.gradient(r[:, j], s)
        d2z_ds2 = np.gradient(dz_ds, s)
        d2r_ds2 = np.gradient(dr_ds, s)
        inv_Rn[:, j] = d2z_ds2 * n_z + d2r_ds2 * n_r

    return inv_Rn


def midline_tangent_angle(z_mid: np.ndarray, r_mid: np.ndarray) -> np.ndarray:
    """Local meridional streamline tangent angle :math:`\\gamma(s)`
    (0 = axial, 90 deg = radial), evaluated on the midspan track."""
    ds = np.hypot(np.diff(z_mid), np.diff(r_mid))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    dz_ds = np.gradient(z_mid, s)
    dr_ds = np.gradient(r_mid, s)
    return np.arctan2(dr_ds, dz_ds)


def front_loaded_beta_law(
    s_hat: np.ndarray,
    beta_in: float,
    beta_out: float,
    split_frac: float,
    frac_at_split: float,
    half_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Front-loaded, two-slope blade-angle law beta(s_hat): quick turn from
    ``beta_in`` covering ``frac_at_split`` of the total change by
    ``s_hat=split_frac``, then an almost linear, low-slope approach to
    ``beta_out`` at s_hat=1, blended smoothly (cubic smoothstep) across
    the split so beta and d(beta)/d(s_hat) are both continuous.

    Returns
    -------
    beta, dbeta_dshat : ndarray
        Both evaluated at ``s_hat``; ``dbeta_dshat`` is d(beta)/d(s_hat),
        not yet converted to the physical d(beta)/ds.
    """
    beta_split = beta_in + frac_at_split * (beta_out - beta_in)
    slope1 = (beta_split - beta_in) / split_frac
    slope2 = (beta_out - beta_split) / (1.0 - split_frac)

    line1 = beta_in + slope1 * s_hat
    line2 = beta_split + slope2 * (s_hat - split_frac)

    t_w = np.clip((s_hat - (split_frac - half_width)) / (2 * half_width), 0.0, 1.0)
    w = t_w * t_w * (3.0 - 2.0 * t_w)
    in_window = (t_w > 0.0) & (t_w < 1.0)
    dw_dshat = np.where(in_window, 6.0 * t_w * (1.0 - t_w) / (2 * half_width), 0.0)

    beta = (1.0 - w) * line1 + w * line2
    dbeta_dshat = (1.0 - w) * slope1 + w * slope2 + dw_dshat * (line2 - line1)
    return beta, dbeta_dshat
