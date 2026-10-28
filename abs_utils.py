"""Shared helpers for the ABS challenge model."""
import numpy as np

BALL_RADIUS_IN = 1.45   # the zone test counts any part of the ball touching the zone


def edge_dist(x_ft, z_ft, top_ft, bot_ft):
    """Distance in inches from the edge of the strike zone, plus the ball radius.

    x_ft, z_ft : pitch location at the front of the plate (feet)
    top_ft, bot_ft : zone top and bottom (feet)
    Positive = inside the zone (the pitch is a strike), negative = outside.
    A challenge wins exactly when this is > 0.
    """
    x = np.asarray(x_ft, dtype="float64")
    z = np.asarray(z_ft, dtype="float64")
    top = np.asarray(top_ft, dtype="float64")
    bot = np.asarray(bot_ft, dtype="float64")
    dx = 8.5 - np.abs(x * 12)                       # inches inside the plate edge
    dz = np.minimum((top - z) * 12, (z - bot) * 12)  # inches inside top / bottom
    inside = (dx >= 0) & (dz >= 0)
    outside = -np.hypot(np.maximum(-dx, 0), np.maximum(-dz, 0))
    return np.where(inside, np.minimum(dx, dz), outside) + BALL_RADIUS_IN


def zone_from_height(height_in):
    """ABS zone for a batter: bottom 27% and top 53.5% of height (in feet).

    NOTE: these percentages were typed from memory and never confirmed
    against Savant's ABS documentation. Check them before relying on results.
    """
    h = np.asarray(height_in, dtype="float64")
    return 0.535 * h / 12, 0.27 * h / 12   # top_ft, bot_ft
