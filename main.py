import cv2
from pypylon import pylon
import sys


def select_camera_by_serial(serial_number=None):
    # Get the list of all connected devices
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()

    if serial_number:
        # Search for the device with the specified serial number
        for device in devices:
            if device.GetSerialNumber() == serial_number:
                return pylon.InstantCamera(tl_factory.CreateDevice(device))
        raise ValueError(f"Camera with serial number {serial_number} not found.")
    else:
        # Use the first available camera if no serial number is provided
        if len(devices) == 0:
            raise ValueError("No cameras found.")
        return pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))


def detect_charuco_board(img):
    # Define the ArUco dictionary and the ChArUco board parameters

    # Old OpenCV API (pre 4.5.0?):
    # aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    # charuco_board = cv2.aruco.CharucoBoard_create(6, 12, 1, 0.8, aruco_dict)
    
    # marker_bits = 4
    # dict_size = 1000
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    squares_x, squares_y = 6, 12
    square_length = 24.0
    marker_length = 18.75
    charuco_board = cv2.aruco.CharucoBoard(
        [squares_x, squares_y],
        square_length,
        marker_length,
        aruco_dict
    )
    
    parameters = cv2.aruco.DetectorParameters()

    # Aniposelib:
    # parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    # parameters.adaptiveThreshWinSizeMin = 50
    # parameters.adaptiveThreshWinSizeMax = 700
    # parameters.adaptiveThreshWinSizeStep = 50
    # parameters.adaptiveThreshConstant = 0

    # Adjusting key parameters to improve robustness
    parameters.adaptiveThreshWinSizeMin = 5  # Increase window size for adaptive thresholding
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.adaptiveThreshConstant = 7  # Adjust the constant for adaptive thresholding

    parameters.minMarkerPerimeterRate = 0.03  # Adjust the perimeter rate for marker detection
    parameters.maxMarkerPerimeterRate = 4.0

    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # Enable subpixel corner refinement
    parameters.minOtsuStdDev = 5.0  # Adjust the minimum Otsu's threshold standard deviation


    # Detect ArUco markers in the image
    corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

    # If markers are detected, refine them and detect the ChArUco board
    if ids is not None and len(ids) > 0:
        cv2.aruco.refineDetectedMarkers(img, charuco_board, corners, ids, rejected)
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, img, charuco_board)

        if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 0:
            # Draw the detected ChArUco board
            cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)

            # Get the top-left and bottom-right corners of the board
            top_left_corner = charuco_corners[0][0]  # First detected corner
            bottom_right_corner = charuco_corners[-1][0]  # Last detected corner

            return img, top_left_corner, bottom_right_corner, True

    return img, None, None, False


def main(serial_number=None):
    # Select the camera based on the serial number
    camera = select_camera_by_serial(serial_number)

    if serial_number is None:
        serial_number = camera.GetDeviceInfo().GetSerialNumber()

    # Start grabbing continuously (camera.GetMaxNumBuffer() buffers)
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    # Convert the grabbed images to OpenCV format and display them.
    converter = pylon.ImageFormatConverter()

    # Convert to OpenCV BGR format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        charuco_img = None
        if grabResult.GrabSucceeded():
            # Access the image data
            image = converter.Convert(grabResult)
            img = image.GetArray()

            # Copy the raw image before any overlays
            raw_img = img.copy()

            # Detect the ChArUco board in the image
            img, top_left, bottom_right, board_detected = detect_charuco_board(img)

            # Overlay general illumination stats
            prc5 = np.percentile(img.flatten(), 5)
            prc50 = np.percentile(img.flatten(), 50)
            prc95 = np.percentile(img.flatten(), 95)
            illumination_text = f"5% / 50% / 95%: {prc5:.1f} / {prc50:.1f} / {prc95:.1f}"
            cv2.putText(img, illumination_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            if board_detected:
                # Draw text overlay with pixel coordinates of the top-left and bottom-right corners
                top_left_text = f"Top Left: ({int(top_left[0])}, {int(top_left[1])})"
                bottom_right_text = f"Bottom Right: ({int(bottom_right[0])}, {int(bottom_right[1])})"

                cv2.putText(img, top_left_text, (int(top_left[0]), int(top_left[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img, bottom_right_text, (int(bottom_right[0]), int(bottom_right[1]) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                charuco_img = img

            # Display the image in a window
            # window_title = f'Basler Camera {serial_number} - Board Detected' if board_detected else f'Basler Camera {serial_number}'
            window_title = f'Basler Camera {serial_number}'
            cv2.imshow(window_title, img)

            # Exit the loop if the user presses the 'q' key
            key = cv2.waitKey(1) & 0xFF

            if cv2.waitKey(1) & key == ord('q'):
                break
            elif key == ord('s'):
                # Save the raw image before overlays
                filename = f"{serial_number}.raw.png"
                cv2.imwrite(filename, raw_img)
                print(f"Saved raw image to {filename}")

                if charuco_img is not None:
                    filename = f"{serial_number}.charuco.png"
                    cv2.imwrite(filename, charuco_img)
                    print(f"Saved charuco image to {filename}")

        grabResult.Release()

    # Release the camera and close any OpenCV windows
    camera.StopGrabbing()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    serial_number = sys.argv[1] if len(sys.argv) > 1 else None
    main(serial_number)
