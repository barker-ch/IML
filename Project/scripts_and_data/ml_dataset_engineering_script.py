import argparse
import os
import numpy as np
import pandas as pd

# Original column names for convenience
COL_START_X = "Start Position (X)"
COL_START_Y = "Start Position (Y)"
COL_START_Z = "Start Position (Z)"
COL_DEST_X  = "Destination Position (X)"
COL_DEST_Y  = "Destination Position (Y)"
COL_DEST_Z  = "Destination Position (Z)"

COL_STATIC  = "Static Object Count"
COL_DYNAMIC = "Dynamic Object Count"

COL_OBS_POS_X = "Obstacle Position (X)"
COL_OBS_POS_Y = "Obstacle Position (Y)"
COL_OBS_POS_Z = "Obstacle Position (Z)"

COL_OBS_VEL_X = "Obstacle Velocity (X)"
COL_OBS_VEL_Y = "Obstacle Velocity (Y)"
COL_OBS_VEL_Z = "Obstacle Velocity (Z)"

COL_PATH_LEN  = "Path Length (m)"
COL_OBS_H     = "Obstacle Height (m)"
COL_OBS_W     = "Obstacle Width (m)"
COL_OBS_D     = "Obstacle Depth (m)"

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the robot navigation dataset."""

    # 3D distance between start and destination
    dx = df[COL_DEST_X] - df[COL_START_X]
    dy = df[COL_DEST_Y] - df[COL_START_Y]
    dz = df[COL_DEST_Z] - df[COL_START_Z]

    df["StartDest_Distance"]   = np.sqrt(dx**2 + dy**2 + dz**2)
    df["StartDest_DistanceXY"] = np.sqrt(dx**2 + dy**2)  # ground distance
    df["StartDest_DeltaZ"]     = dz.abs()

    # Obstacle velocity magnitude
    df["Obstacle_Speed"] = np.sqrt(
        df[COL_OBS_VEL_X]**2 +
        df[COL_OBS_VEL_Y]**2 +
        df[COL_OBS_VEL_Z]**2
    )

    # Obstacle volume (assuming rectangular block)
    df["Obstacle_Volume"] = df[COL_OBS_H] * df[COL_OBS_W] * df[COL_OBS_D]

    # Total and ratio of dynamic obstacles
    total_obs = df[COL_STATIC] + df[COL_DYNAMIC]
    df["Total_Obstacle_Count"]   = total_obs
    # Avoid division by zero
    df["Dynamic_Obstacle_Ratio"] = np.where(
        total_obs > 0,
        df[COL_DYNAMIC] / total_obs,
        0.0
    )

    # Obstacle density per meter of path
    # (again guard against division by zero)
    df["Obstacle_Density_PerMeter"] = np.where(
        df[COL_PATH_LEN] > 0,
        total_obs / df[COL_PATH_LEN],
        0.0
    )

    return df

def main(input_csv: str, output_csv: str, overwrite: bool = False):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if os.path.exists(output_csv) and not overwrite:
        raise FileExistsError(
            f"Output file '{output_csv}' already exists. "
            f"Use --overwrite to replace it."
        )

    # Load dataset
    df = pd.read_csv(input_csv)
    original_shape = df.shape
    original_cols = set(df.columns)

    # Add new features
    df_enhanced = add_engineered_features(df)
    new_shape = df_enhanced.shape
    enhanced_cols = set(df_enhanced.columns)

    # Identify newly added columns
    added_cols = list(enhanced_cols - original_cols)

    # Save enhanced dataset
    df_enhanced.to_csv(output_csv, index=False)

    print("=== Feature Engineering Summary ===")
    print(f"Loaded: {input_csv}")
    print(f"Original shape: {original_shape[0]} rows × {original_shape[1]} columns")
    print(f"Enhanced shape: {new_shape[0]} rows × {new_shape[1]} columns")
    print()
    print("New columns added:")
    for c in added_cols:
        print(f"  - {c}")
    print()
    print(f"Saved enhanced dataset to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add engineered features to robot navigation dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="pathfinding_robot_navigation_dataset.csv",
        help="Path to the original CSV file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pathfinding_robot_navigation_dataset_engineered.csv",
        help="Path for the enhanced CSV file."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists."
    )
    args = parser.parse_args()

    main(args.input, args.output, overwrite=args.overwrite)
