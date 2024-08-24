# janky-charuco-calibrator

Detects a charuco board in a live stream from a Basler camera and overlays the board detections.

## Usage

1. Install with:

    ```
    conda env create -f environment.yml
    ```

2. Activate the environment:

    ```
    conda activate janky-charuco-calibrator
    ```

3. Call `main.py`, optionally with the camera serial number:

    ```
    python main.py
    ```

    ```
    python main.py 24750370
    ```

4. While running, press **S** to save the raw image or **Q** to quit.
