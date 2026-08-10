"""Shared reference definition for the tropical tropopause layer.

The fixed synthesis bounds follow Fueglistaler et al. (2009): a TTL base at
150 hPa, 355 K, or about 14 km and a top at 70 hPa, 425 K, or about 18.5 km.
Height and pressure bounds are coordinate representations of one reference
layer.  A cloud reaches the TTL when its top pressure is at most 150 hPa.
"""

TTL_BASE_PRESSURE_HPA = 150.0
TTL_TOP_PRESSURE_HPA = 70.0
TTL_PRESSURE_BOUNDS_HPA = (
    TTL_TOP_PRESSURE_HPA,
    TTL_BASE_PRESSURE_HPA,
)

TTL_BASE_HEIGHT_KM = 14.0
TTL_TOP_HEIGHT_KM = 18.5
TTL_HEIGHT_BOUNDS_KM = (
    TTL_BASE_HEIGHT_KM,
    TTL_TOP_HEIGHT_KM,
)
TTL_HEIGHT_BOUNDS_M = tuple(
    1000.0 * height_km for height_km in TTL_HEIGHT_BOUNDS_KM
)

TTL_REFERENCE = (
    "Fueglistaler et al. (2009), doi:10.1029/2008RG000267; "
    "150-70 hPa, approximately 14-18.5 km"
)
