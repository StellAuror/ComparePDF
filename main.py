import cv2
import numpy as np

def find_L_shapes(image):
    """
    Detect L-shaped contours in the image using edge detection and contour analysis.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate edges to make them more continuous
    dilated = cv2.dilate(edges, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    L_shapes = []
    for cnt in contours:
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check for L-shape by looking for 3 or 4 vertices
        if 3 <= len(approx) <= 4:
            # Compute angles between vertices to find right angles
            angles = []
            for i in range(len(approx)):
                pt1 = approx[i][0]
                pt2 = approx[(i + 1) % len(approx)][0]
                pt3 = approx[(i + 2) % len(approx)][0]

                v1 = pt2 - pt1
                v2 = pt3 - pt2
                angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                angles.append(np.degrees(angle))

            # Check if we have two near 90-degree angles
            right_angles = [angle for angle in angles if 85 <= angle <= 95]
            if len(right_angles) >= 2:
                L_shapes.append(approx)

    return L_shapes

def compare_pdfs(template_path, artwork_path, output_image_path, dpi=300, transparency=0.5):
    # Extract first page of each PDF as image
    template_img = extract_page_as_image(template_path, 0, dpi)
    artwork_img = extract_page_as_image(artwork_path, 0, dpi)

    # Find L-shapes in both images
    L_shapes_template = find_L_shapes(template_img)
    L_shapes_artwork = find_L_shapes(artwork_img)

    if not L_shapes_template or not L_shapes_artwork:
        print("No L-shapes found in one or both images.")
        return

    # Align and overlay artwork on template
    result_img = align_and_overlay(template_img, artwork_img, L_shapes_template[0], L_shapes_artwork[0], transparency)

    # Save the result
    cv2.imwrite(output_image_path, result_img)
    print(f"Overlay image saved to {output_image_path}")

# Example usage
template_pdf = 'template.pdf'
artwork_pdf = 'artwork.pdf'
output_image = 'overlay_result.png'

compare_pdfs(template_pdf, artwork_pdf, output_image, dpi=300, transparency=0.5)



with open(".gitignore", "w") as f:
    f.write("__pycache__/\n*.pyc")
