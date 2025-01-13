import numpy as np

def compute_pitch_and_yaw(lookatxyz, _xyz):
    """
    Computes pitch and yaw given lookatxyz and _xyz in NumPy.
    
    Args:
        lookatxyz (numpy.ndarray): Target coordinates (N, 3) in the range [0, 1].
        _xyz (numpy.ndarray): Current position coordinates (N, 3).
        
    Returns:
        tuple: (yaw, pitch) where:
            - yaw (numpy.ndarray): Yaw angles in radians, range [-pi, pi].
            - pitch (numpy.ndarray): Pitch angles in radians, range [-pi/3, pi/2].
    """

    # Compute dxyz
    dxyz = lookatxyz - _xyz + 1e-6

    # Calculate yaw (-pi to pi)
    yaw = np.arctan2(dxyz[:, 1], dxyz[:, 0])

    # Calculate pitch (-pi/2 to pi/2)
    pitch = np.arctan2(dxyz[:, 2], np.sqrt(dxyz[:, 0]**2 + dxyz[:, 1]**2))

    return yaw, pitch

# Example Inputs
lookatxyz = np.array([[-0.2, -0.2, 10.2]])  # (N, 3) in range [0, 1]
_xyz = np.array([[-3.1, -1.6, 2.6]])       # (N, 3) real-world positions

# Compute pitch and yaw
yaw, pitch = compute_pitch_and_yaw(lookatxyz, _xyz)

# Print Results
print("Yaw:", yaw)
print("Pitch:", pitch)
