import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import importlib.util
import sys
import easyocr
import time
import re
import argparse
from PIL import Image, ImageDraw, ImageChops, ImageEnhance
from collections import defaultdict
from scipy import stats
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

# Add MantraNet directory to path if needed
if "MantraNet" not in sys.path:
    sys.path.append("MantraNet")

# Load the mantranet module
spec = importlib.util.spec_from_file_location("mantranet", "mantranet.py")
mantranet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mantranet)

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load pre-trained MantraNet model
model = mantranet.pre_trained_model(weight_path='MantraNetv4.pt', device=device)
model.eval()
print("✅ Model Loaded Successfully")

# Initialize EasyOCR reader
print("Initializing EasyOCR reader...")
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
print("✅ OCR initialized successfully")


def detect_individual_text_elements(image):
    """
    Precise detection of individual text elements using multiple techniques
    optimized for isolating each text fragment as its own region.

    Args:
        image: Input image (OpenCV format)

    Returns:
        List of text regions as (x, y, width, height)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    height, width = gray.shape
    text_regions = []

    # --- Method 1: Fine-grained contour detection ---
    # Try multiple thresholding techniques to catch different text characteristics
    binary_versions = []

    # Standard adaptive thresholding
    binary1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    binary_versions.append(binary1)

    # Mean adaptive thresholding
    binary2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 5)
    binary_versions.append(binary2)

    # Global Otsu thresholding
    _, binary3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    binary_versions.append(binary3)

    # Special processing for sequences of digits
    for binary in binary_versions:
        # Use small, precise kernels to avoid merging text
        kernel_sizes = [(1, 1), (2, 1), (3, 1), (5, 1), (2, 2)]
        for kw, kh in kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
            dilated = cv2.dilate(binary, kernel, iterations=1)

            # Find contours - use RETR_EXTERNAL for outer contours only
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Process each contour with very permissive criteria
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Accept even very small regions
                if w > 2 and h > 2:
                    # Add minimal padding to ensure text is contained
                    x_pad = max(0, x - 1)
                    y_pad = max(0, y - 1)
                    w_pad = min(width - x_pad, w + 2)
                    h_pad = min(height - y_pad, h + 2)

                    text_regions.append((x_pad, y_pad, w_pad, h_pad))

    # --- Method 2: Component analysis for digits and small text ---
    # Connected component analysis to find isolated characters
    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_versions[0], connectivity=8)

    # Skip the first component (background)
    for i in range(1, nb_components):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Very permissive filtering to catch individual characters
        if w > 2 and h > 2 and area > 5:
            # Ensure each component has a region
            x_pad = max(0, x - 2)
            y_pad = max(0, y - 2)
            w_pad = min(width - x_pad, w + 4)
            h_pad = min(height - y_pad, h + 4)

            text_regions.append((x_pad, y_pad, w_pad, h_pad))

    # --- Method 3: MSER for character detection ---
    try:
        mser = cv2.MSER_create(
            delta=5,  # Small delta for stability
            min_area=5,  # Very small min area to catch small characters
            max_area=2000,  # Reasonable max area
            max_variation=0.25  # Lower variation for more stable regions
        )

        regions, _ = mser.detectRegions(gray)

        for region in regions:
            x, y, w, h = cv2.boundingRect(region)

            # Accept very small regions
            if w > 2 and h > 2:
                # Add minimal padding
                x_pad = max(0, x - 1)
                y_pad = max(0, y - 1)
                w_pad = min(width - x_pad, w + 2)
                h_pad = min(height - y_pad, h + 2)

                text_regions.append((x_pad, y_pad, w_pad, h_pad))
    except Exception as e:
        print(f"MSER detection error: {e}")

    # --- Method 4: Horizontal digit sequence detection ---
    # Special processing to catch sequences like "555555555"
    # Create horizontal projections
    horiz_proj = np.sum(binary_versions[0], axis=0)

    # Find runs of digits using horizontal projection
    in_run = False
    run_start = 0

    for i in range(width):
        if horiz_proj[i] > 0 and not in_run:
            in_run = True
            run_start = i
        elif horiz_proj[i] == 0 and in_run:
            in_run = False
            run_end = i

            # Get the vertical bounds
            vertical_indices = np.where(binary_versions[0][:, run_start:run_end].sum(axis=1) > 0)[0]
            if len(vertical_indices) > 0:
                y_start = vertical_indices.min()
                y_end = vertical_indices.max() + 1

                # Create region with a bit of padding
                x_pad = max(0, run_start - 2)
                y_pad = max(0, y_start - 2)
                w_pad = min(width - x_pad, (run_end - run_start) + 4)
                h_pad = min(height - y_pad, (y_end - y_start) + 4)

                text_regions.append((x_pad, y_pad, w_pad, h_pad))

    # --- Method 5: Direct detection of repetitive digit patterns (like 555555555) ---
    for binary in binary_versions:
        # Create a dilated version to connect sequences of the same digit
        digit_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        digit_dilated = cv2.dilate(binary, digit_kernel, iterations=1)

        # Find contours of potential digit sequences
        digit_contours, _ = cv2.findContours(digit_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in digit_contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Look for horizontally elongated regions (typical of digit sequences)
            if w > h * 2 and h > 3:
                # Extract region and check for uniformity (characteristic of repeated digits)
                region = binary[y:y + h, x:x + w]

                # Check for regularity in the pattern (optional)
                row_sum = np.sum(region, axis=0)
                reg_score = np.std(row_sum[row_sum > 0]) / (np.mean(row_sum[row_sum > 0]) + 1e-6)

                # Lower reg_score indicates more regular pattern
                if reg_score < 0.5 or w > 5 * h:  # More likely to be a sequence
                    x_pad = max(0, x - 3)
                    y_pad = max(0, y - 3)
                    w_pad = min(width - x_pad, w + 6)
                    h_pad = min(height - y_pad, h + 6)

                    text_regions.append((x_pad, y_pad, w_pad, h_pad))

    # --- Method 6: Precise detection for digit patterns ---
    # A more targeted approach specifically for detecting digit sequences
    for y in range(0, height, 10):  # Scan the image in horizontal strips
        y_end = min(y + 30, height)
        strip = gray[y:y_end, :]

        # Apply threshold to the strip
        _, strip_binary = cv2.threshold(strip, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Find horizontal runs of white pixels (potential digit sequences)
        for row in range(strip_binary.shape[0]):
            row_data = strip_binary[row, :]
            run_start = None

            for col in range(row_data.shape[0]):
                if row_data[col] > 0 and run_start is None:
                    run_start = col
                elif row_data[col] == 0 and run_start is not None:
                    run_end = col
                    run_length = run_end - run_start

                    # Check if this might be a digit or part of a sequence
                    if run_length > 2:
                        # Expand vertically to find full height
                        y_check = y + row
                        y_start = y_check
                        y_end = y_check

                        # Expand upward
                        for check_y in range(y_check, max(0, y_check - 20), -1):
                            check_row = gray.shape[0] - 1 if check_y >= gray.shape[0] else check_y
                            if np.any(binary_versions[0][check_row, run_start:run_end] > 0):
                                y_start = check_y
                            else:
                                break

                        # Expand downward
                        for check_y in range(y_check, min(height, y_check + 20)):
                            check_row = 0 if check_y < 0 else check_y
                            if check_row < gray.shape[0] and np.any(
                                    binary_versions[0][check_row, run_start:run_end] > 0):
                                y_end = check_y
                            else:
                                break

                        # Create region with padding
                        x_pad = max(0, run_start - 2)
                        y_pad = max(0, y_start - 2)
                        w_pad = min(width - x_pad, run_length + 4)
                        h_pad = min(height - y_pad, (y_end - y_start) + 4)

                        if h_pad > 2:  # Ensure it's not just noise
                            text_regions.append((x_pad, y_pad, w_pad, h_pad))

                    run_start = None

            # Handle case where sequence runs to the end of the row
            if run_start is not None:
                run_end = row_data.shape[0]
                run_length = run_end - run_start

                if run_length > 2:
                    # Create region with padding
                    x_pad = max(0, run_start - 2)
                    y_pad = max(0, y + row - 2)
                    w_pad = min(width - x_pad, run_length + 4)
                    h_pad = min(height - y_pad, 10)  # Reasonable height estimate

                    text_regions.append((x_pad, y_pad, w_pad, h_pad))

    # --- Post-processing: Smart merging ---
    # Instead of simply merging overlapping regions, we need a smarter approach
    # that keeps individual text elements separate while merging within the same element
    merged_regions = []

    # Sort by area (smaller first) to prioritize precise regions
    text_regions.sort(key=lambda r: r[2] * r[3])

    # Group regions that likely belong to the same text element
    for r in text_regions:
        x, y, w, h = r

        # Check if this region is mostly contained in any existing region
        contained = False
        for i, existing in enumerate(merged_regions):
            ex, ey, ew, eh = existing

            # Calculate overlap
            overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
            overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
            overlap_area = overlap_x * overlap_y

            # If significant overlap with existing region
            if overlap_area > 0.7 * (w * h):
                contained = True
                break

            # Check if they are horizontally adjacent and similar height (same text line)
            # but not too far apart (different words)
            is_same_line = (abs((y + h / 2) - (ey + eh / 2)) < max(h, eh) * 0.5)
            is_close = ((x >= ex and x <= ex + ew + 10) or (ex >= x and ex <= x + w + 10))

            # If same line and close, they might be part of the same word
            if is_same_line and is_close and min(h, eh) / max(h, eh) > 0.5:
                # Merge horizontally
                new_x = min(x, ex)
                new_y = min(y, ey)
                new_w = max(x + w, ex + ew) - new_x
                new_h = max(y + h, ey + eh) - new_y

                merged_regions[i] = (new_x, new_y, new_w, new_h)
                contained = True
                break

        if not contained:
            merged_regions.append(r)

    # Add OCR-based detection
    try:
        # Run OCR on the full image for precise text localization
        ocr_results = reader.readtext(gray, paragraph=False, min_size=3, low_text=0.3)

        for detection in ocr_results:
            bbox, text, confidence = detection

            # Convert points to rectangle
            x_min = min(point[0] for point in bbox)
            y_min = min(point[1] for point in bbox)
            x_max = max(point[0] for point in bbox)
            y_max = max(point[1] for point in bbox)

            # Create region with minimal padding
            x = max(0, int(x_min) - 2)
            y = max(0, int(y_min) - 2)
            w = min(width - x, int(x_max - x_min) + 4)
            h = min(height - y, int(y_max - y_min) + 4)

            # Add to regions if not contained in existing ones
            contained = False
            for ex, ey, ew, eh in merged_regions:
                if (x >= ex and y >= ey and x + w <= ex + ew and y + h <= ey + eh):
                    contained = True
                    break

            if not contained and w > 2 and h > 2:
                merged_regions.append((x, y, w, h))
    except Exception as e:
        print(f"OCR-based detection error: {e}")

    # Return the merged regions
    return merged_regions


def fourier_compression_analysis(image):
    """
    Apply Fourier transform to detect compression artifacts

    Args:
        image: Input image

    Returns:
        compression_mask: Mask highlighting potential compression artifacts
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply Fourier transform
    f = np.float32(gray)
    dft = cv2.dft(f, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Compute magnitude spectrum (logarithmic scale)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)

    # Normalize to 0-1
    magnitude_spectrum_norm = cv2.normalize(magnitude_spectrum, None, 0, 1, cv2.NORM_MINMAX)

    # Find regular patterns in the frequency domain (potential compression artifacts)
    # These typically appear as regular grid patterns or strong horizontal/vertical lines

    # Detect grid patterns
    sobelx = cv2.Sobel(magnitude_spectrum_norm, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(magnitude_spectrum_norm, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude
    gradient_magnitude = cv2.magnitude(sobelx, sobely)

    # Threshold to detect strong edges in the frequency domain
    _, freq_edges = cv2.threshold(gradient_magnitude, 0.2, 1, cv2.THRESH_BINARY)

    # Apply inverse Fourier transform to see where these patterns manifest in the spatial domain
    # First, create a mask from the frequency domain edges
    mask = np.zeros_like(dft_shift)
    mask[:, :, 0] = freq_edges
    mask[:, :, 1] = freq_edges

    # Apply the mask to the shifted DFT
    fshift_masked = dft_shift * mask

    # Shift back
    f_ishift = np.fft.ifftshift(fshift_masked)

    # Inverse DFT
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize to create the compression artifact mask
    compression_mask = cv2.normalize(img_back, None, 0, 1, cv2.NORM_MINMAX)

    return compression_mask, magnitude_spectrum_norm


def detect_discontinuous_distributions(mask, min_size=10):
    """
    Detect regions with discontinuous probability distributions,
    which are likely genuine forgeries rather than compression artifacts

    Args:
        mask: Probability mask from MantraNet
        min_size: Minimum size of regions to consider

    Returns:
        discontinuity_mask: Binary mask highlighting discontinuous regions
    """
    # Calculate gradient to find discontinuities
    gx = cv2.Sobel(mask, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mask, cv2.CV_32F, 0, 1, ksize=3)

    # Compute gradient magnitude
    gradient_mag = cv2.magnitude(gx, gy)

    # Threshold to find strong edges (discontinuities)
    _, discontinuity_mask = cv2.threshold(gradient_mag, 0.05, 1, cv2.THRESH_BINARY)

    # Remove small regions (likely noise)
    # Convert to uint8 for morphological operations
    disc_mask_uint8 = (discontinuity_mask * 255).astype(np.uint8)

    # Apply morphological closing to connect nearby edges
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(disc_mask_uint8, cv2.MORPH_CLOSE, kernel)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)

    # Create mask keeping only sufficiently large components
    filtered_mask = np.zeros_like(disc_mask_uint8)
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered_mask[labels == i] = 255

    # Dilate to include nearby high-probability areas
    dilated = cv2.dilate(filtered_mask, kernel, iterations=2)

    # Convert back to float32 [0-1]
    result = dilated.astype(np.float32) / 255.0

    return result


def differential_analysis(mantranet_mask, compression_mask, threshold=0.3):
    """
    Compare MantraNet mask with compression artifact mask to identify
    true forgeries (high in MantraNet, low in compression mask)

    Args:
        mantranet_mask: Forgery probability mask from MantraNet
        compression_mask: Compression artifact mask from Fourier analysis
        threshold: Threshold for differential detection

    Returns:
        differential_mask: Mask highlighting likely genuine forgeries
    """
    # Normalize both masks to 0-1 range
    mantranet_norm = cv2.normalize(mantranet_mask, None, 0, 1, cv2.NORM_MINMAX)
    compression_norm = cv2.normalize(compression_mask, None, 0, 1, cv2.NORM_MINMAX)

    # Calculate the difference: high in MantraNet but low in compression
    # This highlights areas that MantraNet thinks are forged but aren't due to compression
    differential = mantranet_norm - compression_norm

    # Threshold to keep only significantly positive differences
    differential[differential < threshold] = 0

    # Scale to 0-1
    if np.max(differential) > 0:
        differential = differential / np.max(differential)

    return differential


def identify_critical_text_regions(image, ocr_results):
    """
    Identify critical text regions like ID numbers and dates

    Args:
        image: Input image
        ocr_results: OCR detection results

    Returns:
        critical_regions: List of critical regions with region info
    """
    height, width = image.shape[:2]
    critical_regions = []

    # Define patterns for critical information
    id_patterns = [
        re.compile(r'[A-Z]{1,2}\d{5,8}'),  # ID numbers like BJ547892
        re.compile(r'\d{5,}')  # Numeric IDs
    ]

    date_patterns = [
        re.compile(r'\d{1,2}[\.\/\-]\d{1,2}[\.\/\-]\d{2,4}'),  # DD.MM.YYYY
        re.compile(r'\d{2,4}[\.\/\-]\d{1,2}[\.\/\-]\d{1,2}')  # YYYY.MM.DD
    ]

    # First pass: find by content
    for i, (x, y, w, h, text, conf) in enumerate(ocr_results):
        if not text:
            continue

        # Check for ID numbers
        is_id = any(pattern.search(text) for pattern in id_patterns)

        # Check for dates
        is_date = any(pattern.search(text) for pattern in date_patterns)

        if is_id:
            critical_regions.append({
                'region_id': i,
                'position': (x, y, w, h),
                'text': text,
                'type': 'id_number',
                'confidence': conf
            })
        elif is_date:
            critical_regions.append({
                'region_id': i,
                'position': (x, y, w, h),
                'text': text,
                'type': 'date',
                'confidence': conf
            })

    # Second pass: find by position if no critical regions found
    if not critical_regions:
        # Analyze possible regions by position
        bottom_regions = []

        for i, (x, y, w, h, text, conf) in enumerate(ocr_results):
            # Look for regions in the bottom half
            if y > height * 0.5:
                bottom_regions.append((i, x, y, w, h, text, conf))

        # Sort by y position (bottom to top)
        bottom_regions.sort(key=lambda r: r[2] + r[3], reverse=True)

        # Bottom left is often ID number
        left_regions = [r for r in bottom_regions if r[1] < width * 0.5]
        if left_regions:
            i, x, y, w, h, text, conf = left_regions[0]
            critical_regions.append({
                'region_id': i,
                'position': (x, y, w, h),
                'text': text,
                'type': 'id_number_by_position',
                'confidence': conf
            })

        # Middle regions often contain dates
        middle_regions = [r for r in bottom_regions
                          if r[1] > width * 0.3 and r[1] < width * 0.7]
        if middle_regions:
            for i, x, y, w, h, text, conf in middle_regions[:2]:  # Take top 2
                critical_regions.append({
                    'region_id': i,
                    'position': (x, y, w, h),
                    'text': text,
                    'type': 'date_by_position',
                    'confidence': conf
                })

    return critical_regions


# Integration of ELA Analysis from the second code
class ErrorLevelAnalyzer:
    """
    Class for performing Error Level Analysis (ELA) on images
    to detect potential forgeries based on JPEG compression artifacts.
    """

    def __init__(self, quality=90, error_scale=20, opacity=0.95):
        """
        Initialize the ELA detector.

        Args:
            quality: JPEG compression quality (0-100)
            error_scale: Scale factor for error visualization
            opacity: Opacity for overlay visualizations
        """
        self.quality = quality
        self.error_scale = error_scale
        self.opacity = opacity

    def perform_ela(self, image_path, output_dir="output"):
        """
        Perform Error Level Analysis on the input image.

        Args:
            image_path: Path to the input image
            output_dir: Directory to save output images

        Returns:
            Tuple of (ELA image as numpy array, path to the ELA result image)
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            # Load the original image
            original = Image.open(image_path)

            # Get the filename without extension
            filename = os.path.splitext(os.path.basename(image_path))[0]

            # Convert to RGB if necessary (for PNG files with transparency)
            if original.mode == 'RGBA':
                print("Converting RGBA image to RGB mode")
                # Create a white background
                background = Image.new('RGB', original.size, (255, 255, 255))
                # Paste the image on the background (respecting transparency)
                background.paste(original, mask=original.split()[3])
                original = background
            elif original.mode != 'RGB':
                print(f"Converting {original.mode} image to RGB mode")
                original = original.convert('RGB')

            # Save the image with specified quality
            temp_path = os.path.join(output_dir, f"{filename}_temp.jpg")
            original.save(temp_path, quality=self.quality)

            # Load the saved image
            resaved = Image.open(temp_path)

            # Calculate the difference between the original and resaved images
            diff = ImageChops.difference(original, resaved)

            # Apply error scale to make differences more visible
            diff = ImageEnhance.Brightness(diff).enhance(self.error_scale)

            # Save the ELA result
            ela_path = os.path.join(output_dir, f"{filename}_ela.png")
            diff.save(ela_path)

            # Convert to numpy array for further processing
            ela_image = np.array(diff)

            # Remove the temporary file
            os.remove(temp_path)

            return ela_image, ela_path

        except Exception as e:
            raise Exception(f"Error in ELA processing: {str(e)}")

    def detect_ela_regions(self, ela_image, threshold=20):
        """
        Detect potentially edited regions based on ELA results.

        Args:
            ela_image: ELA image as numpy array
            threshold: Brightness threshold for detecting edited regions

        Returns:
            Binary mask of detected regions
        """
        # Convert to grayscale if needed
        if len(ela_image.shape) == 3:
            gray = cv2.cvtColor(ela_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = ela_image.copy()

        # Threshold to identify high error areas
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask


def generate_intersection_image(image_path, mantranet_mask, ela_mask, output_dir="output",
                                mantra_threshold=0.3, ela_threshold=0.3):
    """
    Generate an image showing only the intersection of artifacts detected by both
    MantraNet and ELA methods.

    Args:
        image_path: Path to the original document image
        mantranet_mask: The forgery probability mask from MantraNet (normalized 0-1)
        ela_mask: The ELA detection mask (normalized 0-1)
        output_dir: Directory to save the output image
        mantra_threshold: Threshold for MantraNet detection
        ela_threshold: Threshold for ELA detection

    Returns:
        Path to the saved intersection image
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the original image
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert to RGB for visualization
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    height, width = original.shape[:2]

    # Ensure masks are the same size as the original image
    if mantranet_mask.shape[:2] != (height, width):
        mantranet_mask_resized = cv2.resize(mantranet_mask, (width, height))
    else:
        mantranet_mask_resized = mantranet_mask

    if ela_mask.shape[:2] != (height, width):
        ela_mask_resized = cv2.resize(ela_mask, (width, height))
    else:
        ela_mask_resized = ela_mask

    # Threshold the masks
    mantra_binary = (mantranet_mask_resized > mantra_threshold).astype(np.float32)
    ela_binary = (ela_mask_resized > ela_threshold).astype(np.float32)

    # Calculate intersection
    intersection = mantra_binary * ela_binary

    # Create visualization
    plt.figure(figsize=(20, 10))

    # Original image
    plt.subplot(131)
    plt.title("Original Image")
    plt.imshow(original_rgb)
    plt.axis('off')

    # MantraNet + ELA overlay
    plt.subplot(132)
    plt.title("Model (Red) & ELA (Blue) Detections")
    plt.imshow(original_rgb)

    # Show MantraNet detections in red
    mantra_viz = np.zeros((height, width, 4))
    mantra_viz[:, :, 0] = 1.0  # Red channel
    mantra_viz[:, :, 3] = mantra_binary * 0.5  # Alpha channel
    plt.imshow(mantra_viz)

    # Show ELA detections in blue
    ela_viz = np.zeros((height, width, 4))
    ela_viz[:, :, 2] = 1.0  # Blue channel
    ela_viz[:, :, 3] = ela_binary * 0.5  # Alpha channel
    plt.imshow(ela_viz)

    plt.axis('off')

    # Intersection only
    plt.subplot(133)
    plt.title("Intersection: Detected by Both Methods")
    plt.imshow(original_rgb)

    # Show intersection in purple
    intersection_viz = np.zeros((height, width, 4))
    intersection_viz[:, :, 0] = 1.0  # Red channel
    intersection_viz[:, :, 2] = 1.0  # Blue channel
    intersection_viz[:, :, 3] = intersection * 0.7  # Alpha channel
    plt.imshow(intersection_viz)

    # Find contours of intersection regions
    intersection_uint8 = (intersection * 255).astype(np.uint8)
    contours, _ = cv2.findContours(intersection_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around intersection regions
    intersection_with_boxes = original_rgb.copy()
    for contour in contours:
        # Skip tiny regions (likely noise)
        if cv2.contourArea(contour) < 50:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(intersection_with_boxes, (x, y), (x + w, y + h), (255, 0, 255), 2)

    plt.imshow(intersection_with_boxes)
    plt.axis('off')

    plt.tight_layout()
    plt.suptitle("Forgery Artifacts Detected by Both Model and ELA", fontsize=16)
    plt.subplots_adjust(top=0.9)

    # Save the image
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{filename}_intersection_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Intersection analysis saved to: {output_path}")
    return output_path


def integrated_forgery_detection(image_path, output_dir="output", threshold=0.3,
                                 numerical_threshold=0.2, ocr_confidence=0.1,
                                 ela_quality=90, ela_error_scale=20, ela_threshold=20):
    """
    Integrated document forgery detection using MantraNet + ELA + Fourier analysis
    with focused attention on critical text regions and discontinuous distributions

    Args:
        image_path: Path to document image
        output_dir: Directory to save output images
        threshold: General forgery detection threshold
        numerical_threshold: Lower threshold for numerical fields
        ocr_confidence: Minimum confidence for OCR detection
        ela_quality: JPEG quality for ELA (0-100)
        ela_error_scale: Scale factor for ELA visualization
        ela_threshold: Threshold for ELA region detection
    """
    start_time = time.time()
    print(f"Starting integrated forgery detection: {image_path}")

    # Check if file exists
    if not os.path.isfile(image_path):
        raise ValueError(f"File does not exist: {image_path}")

    # Load image
    full_image = cv2.imread(image_path)
    if full_image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert to RGB for visualization
    full_image_rgb = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
    height, width = full_image.shape[:2]

    # Create temp directory
    temp_dir = os.path.join(output_dir, "temp_regions")
    os.makedirs(temp_dir, exist_ok=True)

    # Step 1: Basic preprocessing to reduce noise
    print("Preprocessing image...")
    gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(full_image, (3, 3), 0)

    # Step 2: Apply Fourier analysis to detect compression artifacts
    print("Analyzing compression artifacts using Fourier analysis...")
    compression_mask, freq_spectrum = fourier_compression_analysis(full_image)

    # Step 3: Apply ELA analysis
    print("Performing Error Level Analysis (ELA)...")
    ela_analyzer = ErrorLevelAnalyzer(
        quality=ela_quality,
        error_scale=ela_error_scale,
        opacity=0.95
    )
    ela_image, ela_path = ela_analyzer.perform_ela(image_path, output_dir)

    # Detect regions from ELA
    ela_regions_mask = ela_analyzer.detect_ela_regions(ela_image, threshold=ela_threshold)

    # Convert to float32 and normalize to 0-1 range for consistency with other masks
    ela_mask_float = ela_regions_mask.astype(np.float32) / 255.0

    # Step 4: Detect text regions
    print("Detecting text regions...")
    text_regions = detect_individual_text_elements(full_image)
    print(f"Found {len(text_regions)} text regions")

    # Step 5: Apply OCR to detected regions
    print("Applying OCR to text regions...")
    ocr_results = []

    for i, (x, y, w, h) in enumerate(text_regions):
        try:
            # Extract region
            region_img = full_image[y:y + h, x:x + w]

            # Skip if region is too small
            if w < 3 or h < 3 or region_img.size == 0:
                ocr_results.append((x, y, w, h, "", 0.0))
                continue

            # Save to temp file
            region_path = os.path.join(temp_dir, f"region_{i}.jpg")
            cv2.imwrite(region_path, region_img)

            # Apply OCR
            result = reader.readtext(region_path, paragraph=False,
                                     min_size=3, low_text=0.3)

            # Process result
            if result:
                texts = []
                confidences = []

                for detection in result:
                    bbox, text, confidence = detection
                    if confidence >= ocr_confidence:
                        texts.append(text)
                        confidences.append(confidence)

                if texts:
                    combined_text = " ".join(texts)
                    avg_confidence = sum(confidences) / len(confidences)
                    ocr_results.append((x, y, w, h, combined_text, avg_confidence))
                else:
                    ocr_results.append((x, y, w, h, "", 0.0))
            else:
                ocr_results.append((x, y, w, h, "", 0.0))

        except Exception as e:
            print(f"Error in OCR for region {i}: {e}")
            ocr_results.append((x, y, w, h, "", 0.0))

    print(f"Completed OCR for {len(ocr_results)} regions")

    # Step 6: Identify critical text regions (ID numbers, dates, etc.)
    print("Identifying critical text regions...")
    critical_regions = identify_critical_text_regions(full_image, ocr_results)
    print(f"Found {len(critical_regions)} critical regions")

    # Step 7: Apply MantraNet to all regions
    print("Applying Model to all regions...")

    # Regular regions analysis
    region_results = []
    combined_mask = np.zeros((height, width), dtype=np.float32)

    for i, (x, y, w, h, text, conf) in enumerate(ocr_results):
        try:
            # Skip if too small
            if w < 3 or h < 3:
                continue

            # Extract region
            region_img = full_image[y:y + h, x:x + w]
            region_path = os.path.join(temp_dir, f"mantra_region_{i}.jpg")
            cv2.imwrite(region_path, region_img)

            # Apply MantraNet
            region_mask = mantranet.check_forgery(model, img_path=region_path, device=device)

            # Convert tensor to numpy
            if isinstance(region_mask, torch.Tensor):
                region_mask = region_mask.squeeze().detach().cpu().numpy()

            # Check if critical region
            is_critical = any(r['region_id'] == i for r in critical_regions)

            # Determine appropriate threshold
            if is_critical:
                region_threshold = numerical_threshold * 0.7  # Much lower for critical
            elif bool(re.search(r'\d', text)):
                region_threshold = numerical_threshold
            else:
                region_threshold = threshold

            # Calculate metrics
            max_prob = np.max(region_mask)
            avg_prob = np.mean(region_mask)

            # Store result
            result = {
                'region_id': i,
                'position': (x, y, w, h),
                'text': text,
                'is_critical': is_critical,
                'max_prob': max_prob,
                'avg_prob': avg_prob,
                'threshold': region_threshold,
                'mask': region_mask
            }
            region_results.append(result)

            # Update combined mask
            try:
                combined_mask[y:y + h, x:x + w] = np.maximum(combined_mask[y:y + h, x:x + w], region_mask)
            except:
                pass

        except Exception as e:
            print(f"Error processing region {i}: {e}")

    # Step 8: Enhanced analysis for critical regions integrating ELA
    print("Performing enhanced analysis for critical regions with ELA integration...")
    critical_results = []

    for region in critical_regions:
        region_id = region['region_id']
        x, y, w, h = region['position']
        region_type = region['type']

        # Find the regular analysis result
        base_result = next((r for r in region_results if r['region_id'] == region_id), None)
        if not base_result:
            continue

        # Extract region for additional analysis
        region_img = full_image[y:y + h, x:x + w]

        # 1. Fourier analysis for this region
        compression_local, _ = fourier_compression_analysis(region_img)

        # 2. Get MantraNet mask for this region
        mantranet_local = base_result['mask']

        # 3. Extract ELA results for this region
        try:
            # Get the corresponding region from ELA results
            ela_local = ela_mask_float[y:y + h, x:x + w]
            # Ensure shape compatibility
            if ela_local.shape != mantranet_local.shape:
                ela_local = cv2.resize(ela_local, (mantranet_local.shape[1], mantranet_local.shape[0]))
        except:
            # If there's an issue, create an empty mask
            ela_local = np.zeros_like(mantranet_local)

        # 4. Apply differential analysis between MantraNet and compression
        differential_mantra_comp = differential_analysis(mantranet_local, compression_local, threshold=0.2)

        # 5. Apply differential analysis between MantraNet and ELA
        # (high in both indicates potential forgery)
        differential_mantra_ela = np.maximum(mantranet_local, ela_local) - np.minimum(mantranet_local, ela_local)
        differential_mantra_ela = differential_mantra_ela * np.maximum(mantranet_local, ela_local)

        # Threshold to keep only significant values
        differential_mantra_ela[differential_mantra_ela < 0.2] = 0
        if np.max(differential_mantra_ela) > 0:
            differential_mantra_ela = differential_mantra_ela / np.max(differential_mantra_ela)

        # 6. Detect discontinuous distributions
        discontinuity_mask = detect_discontinuous_distributions(mantranet_local)

        # 7. Weight and combine results - integrate all detection methods
        # Give higher weight to areas detected by multiple methods
        combined_analysis = (differential_mantra_comp * 0.3 +
                             differential_mantra_ela * 0.4 +
                             discontinuity_mask * 0.3)

        # Additional boost where ELA detects forgery
        combined_analysis = np.maximum(combined_analysis, ela_local * 0.8)

        # Calculate metrics
        max_combined = np.max(combined_analysis)
        avg_combined = np.mean(combined_analysis)

        # Very low threshold for critical regions
        threshold_used = 0.15

        # Determine if suspicious
        is_suspicious = max_combined > threshold_used

        # Store enhanced result
        enhanced_result = {
            'region_id': region_id,
            'position': (x, y, w, h),
            'type': region_type,
            'text': base_result['text'],
            'max_prob': base_result['max_prob'],
            'avg_prob': base_result['avg_prob'],
            'differential_max': max_combined,
            'differential_avg': avg_combined,
            'mantranet_mask': mantranet_local,
            'compression_mask': compression_local,
            'ela_mask': ela_local,
            'differential_mask_mantra_comp': differential_mantra_comp,
            'differential_mask_mantra_ela': differential_mantra_ela,
            'discontinuity_mask': discontinuity_mask,
            'combined_mask': combined_analysis,
            'threshold': threshold_used,
            'is_suspicious': is_suspicious,
            'ela_detection': np.max(ela_local) > 0.3  # Flag if ELA detected something
        }
        critical_results.append(enhanced_result)

    # Step 9: Create visualizations
    plt.figure(figsize=(20, 15))

    # Original image with text regions
    plt.subplot(231)
    plt.title("Detected Text Regions")
    region_viz = full_image_rgb.copy()

    for i, (x, y, w, h, text, conf) in enumerate(ocr_results):
        # Color based on region type
        is_critical = any(r['region_id'] == i for r in critical_regions)

        if is_critical:
            color = (255, 0, 255)  # Magenta for critical
        elif text and re.search(r'\d', text):
            color = (255, 0, 0)  # Red for numbers
        else:
            color = (0, 255, 0)  # Green for regular text

        cv2.rectangle(region_viz, (x, y), (x + w, y + h), color, 1)

    plt.imshow(region_viz)
    plt.axis('off')

    # MantraNet analysis
    plt.subplot(232)
    plt.title("Model Forgery Analysis")
    plt.imshow(full_image_rgb)
    plt.imshow(combined_mask, alpha=0.5, cmap='hot')
    plt.axis('off')

    # ELA analysis
    plt.subplot(233)
    plt.title("Artifact Analysis of Edited Area Analyses")
    ela_rgb = cv2.cvtColor(ela_image, cv2.COLOR_BGR2RGB) if len(ela_image.shape) == 3 else ela_image
    plt.imshow(ela_rgb)
    plt.axis('off')

    # Compression artifacts
    plt.subplot(234)
    plt.title("Compression Artifacts ")
    plt.imshow(full_image_rgb)
    plt.imshow(compression_mask, alpha=0.5, cmap='cool')
    plt.axis('off')

    # Combined MantraNet + ELA mask
    plt.subplot(235)
    plt.title("Model + Artifact Analysis of Edited Area Analyses")

    # Create combined visualization
    combined_mantra_ela = np.zeros((height, width), dtype=np.float32)
    for result in critical_results:
        x, y, w, h = result['position']
        try:
            # Use the combined analysis that integrates MantraNet and ELA
            combined_mantra_ela[y:y + h, x:x + w] = result['combined_mask']
        except:
            pass

    plt.imshow(full_image_rgb)
    plt.imshow(combined_mantra_ela, alpha=0.6, cmap='hot')
    plt.axis('off')

    # Final analysis - regions detected by both methods
    plt.subplot(236)
    plt.title("Final Analysis (Detected Forgeries)")

    # Create a mask for ELA-detected regions across the whole image
    ela_detection_mask = np.zeros((height, width), dtype=np.uint8)
    ela_detection_mask = ela_regions_mask

    # Create visualization with rectangles for the final result
    final_viz = full_image_rgb.copy()
    final_combined_mask = np.zeros((height, width), dtype=np.float32)

    # Integrate suspicious regions from critical analysis with ELA detection
    for result in critical_results:
        if result['is_suspicious']:
            x, y, w, h = result['position']

            # Strong indicator: region flagged by both methods
            is_detected_by_both = result['ela_detection']

            # Color coding: red for both detections, orange for MantraNet only
            color = (255, 0, 0) if is_detected_by_both else (255, 165, 0)
            line_thickness = 2 if is_detected_by_both else 1

            # Draw rectangle around suspicious region
            cv2.rectangle(final_viz, (x, y), (x + w, y + h), color, line_thickness)

            # Add label with detection type
            detection_type = f"{result['type']}"
            if is_detected_by_both:
                detection_type += " (Model +Artifact Analysis of Edited Area )"
            else:
                detection_type += " (Model only)"

            cv2.putText(final_viz, detection_type, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Update the combined mask
            try:
                final_combined_mask[y:y + h, x:x + w] = result['combined_mask']
            except:
                pass

    # Also highlight any ELA-detected regions not already covered
    contours, _ = cv2.findContours(ela_regions_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Check if this contour overlaps with any critical suspicious region
        x, y, w, h = cv2.boundingRect(contour)

        # Skip small regions (likely noise)
        if w < 10 or h < 10:
            continue

        # Skip if this region overlaps with any already detected critical region
        overlap = False
        for result in critical_results:
            if result['is_suspicious']:
                rx, ry, rw, rh = result['position']
                # Check for overlap
                if (x < rx + rw and x + w > rx and
                        y < ry + rh and y + h > ry):
                    overlap = True
                    break

        if not overlap:
            # This is an ELA-only detection
            cv2.rectangle(final_viz, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(final_viz, "ELA only", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Add to final mask
            tmp_mask = np.zeros((h, w), dtype=np.float32)
            tmp_mask.fill(0.7)  # Lower confidence for ELA-only
            try:
                final_combined_mask[y:y + h, x:x + w] = np.maximum(
                    final_combined_mask[y:y + h, x:x + w], tmp_mask)
            except:
                pass

    plt.imshow(final_viz)
    plt.imshow(final_combined_mask, alpha=0.4, cmap='hot')
    plt.axis('off')

    plt.tight_layout()
    plt.suptitle("Integrated Forgery Analysis ", fontsize=16)
    plt.subplots_adjust(top=0.92)

    # Save visualization
    viz_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_integrated_analysis.png")
    plt.savefig(viz_path)

    # Create detail view for suspicious critical regions
    suspicious_critical = [r for r in critical_results if r['is_suspicious']]

    if suspicious_critical:
        plt.figure(figsize=(20, 15))
        plt.suptitle("Detailed Analysis of Detected Forgeries", fontsize=16)

        for i, result in enumerate(suspicious_critical[:6]):
            x, y, w, h = result['position']
            region_type = result['type']
            text = result['text']

            # Skip if we've reached the maximum number of plots
            if i >= 6:
                break

            # Row 1: Original with combined mask
            plt.subplot(3, 6, i + 1)
            plt.title(f"{region_type}: '{text}'")
            plt.imshow(full_image_rgb[y:y + h, x:x + w])
            plt.imshow(result['combined_mask'], alpha=0.5, cmap='hot')
            plt.axis('off')

            # Row 2: ELA vs MantraNet
            plt.subplot(3, 6, i + 7)
            plt.title("ELA vs Model")

            # Ensure shape compatibility
            ela_mask_display = result['ela_mask']
            mantra_mask_display = result['mantranet_mask']
            if ela_mask_display.shape != mantra_mask_display.shape:
                ela_mask_display = cv2.resize(ela_mask_display,
                                              (mantra_mask_display.shape[1], mantra_mask_display.shape[0]))

            # Side by side comparison
            comparison = np.hstack([
                ela_mask_display,
                mantra_mask_display
            ])

            plt.imshow(comparison, cmap='hot')
            plt.axis('off')

            # Row 3: Combined differential analysis
            plt.subplot(3, 6, i + 13)
            plt.title(f"Combined Score: {result['differential_max']:.3f}")

            # Show the final combined analysis
            plt.imshow(result['combined_mask'], cmap='hot')
            plt.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # Save the detailed visualization
        detail_path = os.path.join(output_dir,
                                   f"{os.path.splitext(os.path.basename(image_path))[0]}_detailed_analysis.png")
        plt.savefig(detail_path)

    # Create the intersection image showing only regions detected by both methods
    print("\nGenerating intersection analysis image...")

    # Create binary mask from MantraNet results
    mantranet_full_mask = np.zeros((height, width), dtype=np.float32)
    for result in region_results:
        x, y, w, h = result['position']
        try:
            mantranet_full_mask[y:y + h, x:x + w] = np.maximum(
                mantranet_full_mask[y:y + h, x:x + w], result['mask'])
        except:
            pass

    # Generate and save the intersection image
    intersection_path = generate_intersection_image(
        image_path=image_path,
        mantranet_mask=mantranet_full_mask,
        ela_mask=ela_mask_float,
        output_dir=output_dir
    )

    # Step 10: Print analysis results
    print("\n===== INTEGRATED FORGERY ANALYSIS RESULTS =====")
    print(f"Total regions analyzed: {len(region_results)}")
    print(f"Critical regions identified: {len(critical_regions)}")
    print(f"Suspicious critical regions (with Model): {len(suspicious_critical)}")

    # Count of ELA detections
    ela_regions_count = len(contours)
    print(f"Regions flagged by ELA: {ela_regions_count}")

    # Regions detected by both methods
    both_detected = [r for r in suspicious_critical if r['ela_detection']]
    print(f"Regions detected by both Model and ELA: {len(both_detected)}")

    # Print detailed results for suspicious regions
    if suspicious_critical:
        print("\nDetected forgeries in critical regions:")
        for i, result in enumerate(suspicious_critical):
            x, y, w, h = result['position']
            region_type = result['type']
            text = result['text']
            diff_max = result['differential_max']

            detection_methods = []
            if np.max(result['mantranet_mask']) > 0.3:
                detection_methods.append("MantraNet")
            if result['ela_detection']:
                detection_methods.append("ELA")

            detection_str = " & ".join(detection_methods)

            print(f"{i + 1}. {region_type.upper()}: '{text}' at ({x},{y})")
            print(f"   Detected by: {detection_str}")
            print(f"   Integrated score: {diff_max:.3f} (threshold: {result['threshold']:.3f})")
            print(f"   MantraNet probability: {result['max_prob']:.3f}")

    # Analysis conclusion
    print("\n===== FORGERY ANALYSIS CONCLUSIONS =====")

    if suspicious_critical:
        print(f"⚠️ CRITICAL FORGERY DETECTED: Found {len(suspicious_critical)} forged critical regions")

        # Add confidence level based on detection method agreement
        high_confidence = [r for r in suspicious_critical if r['ela_detection']]
        if high_confidence:
            print(f"⚠️ HIGH CONFIDENCE FORGERY: {len(high_confidence)} regions detected by both Model and ELA")

        # Group by type
        id_forgeries = [r for r in suspicious_critical if 'id' in r['type'].lower()]
        date_forgeries = [r for r in suspicious_critical if 'date' in r['type'].lower()]

        if id_forgeries:
            print(f"⚠️ ID NUMBER FORGERY: Detected {len(id_forgeries)} forged ID fields")
            for result in id_forgeries:
                detection_methods = []
                if np.max(result['mantranet_mask']) > 0.3:
                    detection_methods.append("MantraNet")
                if result['ela_detection']:
                    detection_methods.append("ELA")

                detection_str = " & ".join(detection_methods)
                print(
                    f"  - ID field: '{result['text']}' (Detected by: {detection_str}, Score: {result['differential_max']:.3f})")

        if date_forgeries:
            print(f"⚠️ DATE FORGERY: Detected {len(date_forgeries)} forged date fields")
            for result in date_forgeries:
                detection_methods = []
                if np.max(result['mantranet_mask']) > 0.3:
                    detection_methods.append("MantraNet")
                if result['ela_detection']:
                    detection_methods.append("ELA")

                detection_str = " & ".join(detection_methods)
                print(
                    f"  - Date field: '{result['text']}' (Detected by: {detection_str}, Score: {result['differential_max']:.3f})")
    else:
        print("✓ NO CRITICAL FORGERIES DETECTED: No evidence of manipulation in critical text fields")

    # Clean up
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    end_time = time.time()
    print(f"\nIntegrated analysis completed in {end_time - start_time:.2f} seconds")

    return {
        'critical_regions': critical_regions,
        'suspicious_regions': suspicious_critical,
        'ela_regions_count': ela_regions_count,
        'both_detected_count': len(both_detected) if 'both_detected' in locals() else 0,
        'ela_path': ela_path,
        'visualization_path': viz_path,
        'detailed_path': detail_path if suspicious_critical else None,
        'intersection_path': intersection_path,
        'execution_time': end_time - start_time
    }


def main():
    """
    Main function for running the integrated document forgery detection.
    This allows the script to be run as a standalone program.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Integrated Document Forgery Detection")
    parser.add_argument("--image", type=str, help="Path to the document image")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.6, help="General forgery detection threshold")
    parser.add_argument("--numerical_threshold", type=float, default=0.4, help="Threshold for numerical fields")
    parser.add_argument("--ocr_confidence", type=float, default=0.2, help="Minimum OCR confidence")
    parser.add_argument("--ela_quality", type=int, default=90, help="JPEG quality for ELA (0-100)")
    parser.add_argument("--ela_error_scale", type=int, default=20, help="Error scaling for ELA visualization")
    parser.add_argument("--ela_threshold", type=int, default=20, help="Threshold for ELA region detection")

    args = parser.parse_args()

    # Use the first specified image path, or fall back to default
    image_path = args.image if args.image else r"C:\Users\reda.benkirane\Downloads\sofac\sofac\image (7).jpeg"

    print(f"Processing document: {image_path}")
    print(f"Output directory: {args.output_dir}")

    # Run integrated forgery detection
    results = integrated_forgery_detection(
        image_path=image_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        numerical_threshold=args.numerical_threshold,
        ocr_confidence=args.ocr_confidence,
        ela_quality=args.ela_quality,
        ela_error_scale=args.ela_error_scale,
        ela_threshold=args.ela_threshold
    )

    print("\n===== OUTPUT IMAGES =====")
    print(f"1. ELA Analysis: {results['ela_path']}")
    print(f"2. MantraNet Analysis: {results['visualization_path']}")
    print(f"3. Intersection Analysis: {results['intersection_path']}")

    if results['detailed_path']:
        print(f"4. Detailed Analysis: {results['detailed_path']}")

    print(f"\nAnalysis completed in {results['execution_time']:.2f} seconds")
    print(f"Found {results['both_detected_count']} regions detected by both Model and ELA")

    print("Analysis complete! Results are saved in the output directory.")


# Example usage
if __name__ == "__main__":
    main()