"""Shared definitions for equatorial-wave diagnostics.

The easterly-wave box follows Rios-Berrios et al. (2023, J. Climate):
westward zonal wavenumbers 6--28 and periods 2.5--7 days. It is a
rectangular TD/easterly-wave filter, not a shallow-water normal-mode filter.
"""

MISSING_EQUIVALENT_DEPTH = -999.0
WAVE_CONFIG_VERSION = "equatorial-wave-bands-v3-strict-schema"
FILTERED_WAVE_PRODUCT_VERSION = (
    "equatorial-wave-filtered-v5-tapered-record"
)
EASTWARD_INTRASEASONAL_FILTER = "Eastward_intraseasonal"

WAVE_VARIABILITY_FIELDS = (
    "Precip",
    "OLR",
    "PW",
    "high_cloud",
    "active_high_cloud",
    "stratiform_high_cloud",
    "nonprecipitating_high_cloud",
    "ttl_reaching_high_cloud",
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

WAVE_BOXES = {
    EASTWARD_INTRASEASONAL_FILTER: dict(
        k=(1, 5),
        period=(20.0, 100.0),
        depth=(MISSING_EQUIVALENT_DEPTH, MISSING_EQUIVALENT_DEPTH),
        label="Eastward intraseasonal",
        filter_name="None",
        components=("symmetric",),
    ),
    "Kelvin": dict(
        k=(1, 14), period=(2.5, 20.0), depth=(8, 90),
        label="Kelvin", filter_name="Kelvin", components=("symmetric",),
    ),
    "EIG": dict(
        k=(1, 14), period=(1.8, 4.5), depth=(8, 90),
        label="EIG", filter_name="IG0", components=("antisymmetric",),
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
    EASTWARD_INTRASEASONAL_FILTER: "symmetric",
    "Kelvin": "symmetric",
    "EIG": "antisymmetric",
    "IG_antisym": "antisymmetric",
    # Published easterly-wave filtering uses rainfall averaged across the
    # complete tropical latitude band, without imposing Matsuno parity.
    "Easterly": "direct",
    "ER": "symmetric",
}

# Diagnostic-specific collections prevent the 20--100-day band from being
# inserted into calculations whose lag windows or dispersion assumptions are
# designed for shorter-period CCEWs.
ALL_CACHED_FILTERS = tuple(WAVE_BOXES)
WAVE_RESPONSE_FILTERS = (
    EASTWARD_INTRASEASONAL_FILTER,
    "Kelvin",
    "Easterly",
    "ER",
)
VERTICAL_STRUCTURE_FILTERS = ALL_CACHED_FILTERS
TOP_HEAVINESS_FILTERS = (
    EASTWARD_INTRASEASONAL_FILTER,
    "Kelvin",
    "ER",
    "Easterly",
)
SHORT_LAG_FILTERS = ("Kelvin", "Easterly", "ER")
SHORT_PERIOD_CRITICAL_LEVEL_FILTERS = (
    "Kelvin",
    "EIG",
    "IG_antisym",
    "Easterly",
    "ER",
)
EDDY_FLUX_FILTERS = SHORT_PERIOD_CRITICAL_LEVEL_FILTERS
LPO_LIFECYCLE_FILTERS = WAVE_RESPONSE_FILTERS
