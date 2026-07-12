"""Shared definitions for equatorial-wave diagnostics.

The easterly-wave box follows Rios-Berrios et al. (2023, J. Climate):
westward zonal wavenumbers 6--28 and periods 2.5--7 days. It is a
rectangular TD/easterly-wave filter, not a shallow-water normal-mode filter.
"""

MISSING_EQUIVALENT_DEPTH = -999.0

WAVE_VARIABILITY_FIELDS = (
    "Precip",
    "OLR",
    "PW",
    "cldfrac",
    "u850",
    "u250",
    "u100",
    "v850",
    "v250",
    "w500",
    "w250",
    "T250",
    "T100",
)

CORE_WAVE_NAMES = ("Kelvin", "Easterly", "ER")

WAVE_BOXES = {
    "Kelvin": dict(
        k=(1, 14), period=(2.5, 20.0), depth=(8, 90),
        label="Kelvin", filter_name="Kelvin", components=("symmetric",),
    ),
    "EIG": dict(
        k=(1, 14), period=(1.8, 4.5), depth=(8, 90),
        label="EIG", filter_name="IG0", components=("antisymmetric",),
    ),
    "IG_sym": dict(
        k=(-15, -1), period=(1.4, 2.6), depth=(8, 90),
        label="IG sym", filter_name="IG1", components=("symmetric",),
    ),
    "IG_antisym": dict(
        k=(-15, -1), period=(1.25, 2.0), depth=(8, 90),
        label="IG antisym", filter_name="IG2", components=("antisymmetric",),
    ),
    "Easterly": dict(
        k=(-28, -6), period=(2.5, 7.0),
        depth=(MISSING_EQUIVALENT_DEPTH, MISSING_EQUIVALENT_DEPTH),
        label="Easterly", filter_name="None",
        components=("symmetric", "antisymmetric"),
    ),
    "ER": dict(
        k=(-10, -1), period=(10.0, 40.0), depth=(8, 90),
        label="ER", filter_name="ER", components=("symmetric",),
    ),
}

WAVE_LATITUDE_PROJECTION = {
    "Kelvin": "symmetric",
    "EIG": "antisymmetric",
    "IG_antisym": "antisymmetric",
    # Published easterly-wave filtering uses rainfall averaged across the
    # complete tropical latitude band, without imposing Matsuno parity.
    "Easterly": "direct",
    "ER": "symmetric",
}
