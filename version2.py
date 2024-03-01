import cv2
import numpy as np

def get_connected_components(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    # Define the codec and create VideoWriter object
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Thresholding to segment objects
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        components = []
        for label in range(1, num_labels):
            component = np.zeros_like(gray, dtype=np.uint8)
            component[labels == label] = 255
            components.append(component)
        
        #fft of component
        connected_components=components    
        fft_results = []
        for component in connected_components:
        # Zero-padding to the nearest power of 2
           rows, cols = component.shape
           padded_rows = int(2 ** np.ceil(np.log2(rows)))
           padded_cols = int(2 ** np.ceil(np.log2(cols)))
           padded_component = np.zeros((padded_rows, padded_cols), dtype=np.uint8)
           padded_component[:rows, :cols] = component

        # Perform FFT
           fft_result = np.fft.fft2(padded_component)
           fft_shifted = np.fft.fftshift(fft_result)
           magnitude_spectrum = 20 * np.log(np.abs(fft_shifted) + 1)

           fft_results.append(magnitude_spectrum)

           return fft_results
        # Draw bounding boxes around each connected component
    #     for i in range(1, num_labels):
    #         x, y, w, h, area = stats[i]
    #         if area > 50:  # Filter out small components
    #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #     # Display the resulting frame
    #     cv2.imshow('Frame', frame)
    #     out.write(frame)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # # Release resources
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()

# Path to the thermal video
video_path = '1705951007967.mp4'

# Call the function to extract connected components
fft_results=get_connected_components(video_path)
for i, fft_result in enumerate(fft_results):
    cv2.imshow(f'FFT Component {i}', fft_result.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

