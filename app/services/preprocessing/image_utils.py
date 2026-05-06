import cv2
import numpy as np
from PIL import Image, ImageOps


class ImageUtils:
    @staticmethod
    def pil_to_cv(image: Image.Image) -> np.ndarray:
        image = ImageOps.exif_transpose(image)
        rgb = image.convert("RGB")
        array = np.array(rgb)

        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    @staticmethod
    def cv_to_pil(image: np.ndarray) -> Image.Image:
        if len(image.shape) == 2:
            return Image.fromarray(image)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @staticmethod
    def read_image(path: str) -> np.ndarray:
        image = Image.open(path)
        return ImageUtils.pil_to_cv(image)

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def resize_if_too_large(
        image: np.ndarray,
        max_side: int = 2800,
    ) -> np.ndarray:
        height, width = image.shape[:2]
        longest = max(height, width)

        if longest <= max_side:
            return image

        scale = max_side / longest
        new_width = int(width * scale)
        new_height = int(height * scale)

        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return cv2.fastNlMeansDenoising(
                image,
                None,
                h=10,
                templateWindowSize=7,
                searchWindowSize=21,
            )

        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=8,
            hColor=8,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        gray = ImageUtils.to_grayscale(image)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        return clahe.apply(gray)

    @staticmethod
    def adaptive_threshold(image: np.ndarray) -> np.ndarray:
        gray = ImageUtils.to_grayscale(image)

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        return cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

    @staticmethod
    def otsu_threshold(image: np.ndarray) -> np.ndarray:
        gray = ImageUtils.to_grayscale(image)

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        _, thresholded = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        return thresholded

    @staticmethod
    def remove_small_noise(binary_image: np.ndarray) -> np.ndarray:
        kernel = np.ones((2, 2), np.uint8)

        opened = cv2.morphologyEx(
            binary_image,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        return opened

    @staticmethod
    def deskew(image: np.ndarray) -> tuple[np.ndarray, float]:
        gray = ImageUtils.to_grayscale(image)

        inverted = cv2.bitwise_not(gray)

        coords = np.column_stack(np.where(inverted > 0))

        if len(coords) < 100:
            return image, 0.0

        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.3 or abs(angle) > 15:
            return image, 0.0

        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rotated, float(angle)

    @staticmethod
    def find_document_contour(image: np.ndarray) -> np.ndarray | None:
        resized = ImageUtils.resize_if_too_large(image, max_side=1600)

        ratio_y = image.shape[0] / resized.shape[0]
        ratio_x = image.shape[1] / resized.shape[1]

        gray = ImageUtils.to_grayscale(resized)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 180)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:8]:
            area = cv2.contourArea(contour)
            image_area = resized.shape[0] * resized.shape[1]

            if area < image_area * 0.15:
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(approx) == 4:
                approx = approx.reshape(4, 2).astype("float32")
                approx[:, 0] *= ratio_x
                approx[:, 1] *= ratio_y

                return approx

        return None

    @staticmethod
    def order_points(points: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")

        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]

        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]

        return rect

    @staticmethod
    def perspective_correct(image: np.ndarray, points: np.ndarray) -> np.ndarray:
        rect = ImageUtils.order_points(points)

        top_left, top_right, bottom_right, bottom_left = rect

        width_a = np.linalg.norm(bottom_right - bottom_left)
        width_b = np.linalg.norm(top_right - top_left)
        max_width = int(max(width_a, width_b))

        height_a = np.linalg.norm(top_right - bottom_right)
        height_b = np.linalg.norm(top_left - bottom_left)
        max_height = int(max(height_a, height_b))

        if max_width < 100 or max_height < 100:
            return image

        destination = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )

        matrix = cv2.getPerspectiveTransform(rect, destination)

        warped = cv2.warpPerspective(
            image,
            matrix,
            (max_width, max_height),
        )

        return warped

    @staticmethod
    def crop_document_if_found(image: np.ndarray) -> tuple[np.ndarray, bool]:
        contour = ImageUtils.find_document_contour(image)

        if contour is None:
            return image, False

        corrected = ImageUtils.perspective_correct(image, contour)

        return corrected, True

    @staticmethod
    def encode_png(image: np.ndarray) -> bytes:
        success, buffer = cv2.imencode(".png", image)

        if not success:
            raise RuntimeError("Could not encode image as PNG")

        return buffer.tobytes()