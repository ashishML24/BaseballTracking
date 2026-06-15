from .config import FOCAL_LENGTH_MM, PIXEL_SIZE_MM, CX, CY
import numpy as np

def backproject(u, v, Z):
    """
    Back-projects a 2D image point with known depth into 3D camera coordinates
    using a pinhole camera model.

    This function converts pixel-space coordinates of the detected ball center
    into a 3D point in the camera coordinate system, assuming the depth along
    the optical axis is known. It is a core geometric operation used for 3D
    reconstruction from monocular images.

    ------------------------------------------------------------------------
    INPUTS
    ------------------------------------------------------------------------
    u : float
        Horizontal pixel coordinate of the detected ball center in the image
        (measured in pixels).

    v : float
        Vertical pixel coordinate of the detected ball center in the image
        (measured in pixels).

    Z : float
        Depth of the ball center along the camera optical axis, expressed in
        meters. This depth is typically estimated using the known physical
        radius of the ball and its apparent size in the image.

    ------------------------------------------------------------------------
    CAMERA PARAMETERS (GLOBAL CONSTANTS)
    ------------------------------------------------------------------------
    FOCAL_LENGTH_MM : float
        Camera focal length in millimeters.

    PIXEL_SIZE_MM : float
        Physical size of a pixel in millimeters.

    CX, CY : float
        Principal point coordinates (image center) in pixel units.

    ------------------------------------------------------------------------
    OUTPUT
    ------------------------------------------------------------------------
    X_cam : np.ndarray of shape (3,)
        3D coordinates of the point in the camera coordinate system, expressed
        in meters as:
            [X, Y, Z]

    ------------------------------------------------------------------------
    MATHEMATICAL FORMULATION
    ------------------------------------------------------------------------
    The effective focal length in pixel units is computed as:

        fx = FOCAL_LENGTH_MM / PIXEL_SIZE_MM

    Using the pinhole camera model, the back-projection equations are:

        X = (u - CX) * Z / fx
        Y = (v - CY) * Z / fx
        Z = Z

    These equations map the image point onto a ray passing through the camera
    center and scale it to the specified depth.

    ------------------------------------------------------------------------
    ASSUMPTIONS
    ------------------------------------------------------------------------
    - The camera follows an ideal pinhole model.
    - Camera intrinsics (focal length, pixel size, principal point) are known
      and accurate.
    - Lens distortion effects are negligible or have been corrected upstream.
    - The depth Z is reliable and expressed in consistent units.

    ------------------------------------------------------------------------
    NOTES
    ------------------------------------------------------------------------
    - All computations are performed in metric units (meters).
    - The returned point lies in the camera coordinate frame; further
      transformations may be required to express it in a world coordinate
      system.
    """

    fx = FOCAL_LENGTH_MM / PIXEL_SIZE_MM
    return np.array([
        (u - CX) * Z / fx,
        (v - CY) * Z / fx,
        Z
    ])