# OCR Languages
PADDLE_LANG = 'en'
EASYOCR_LANG = ['en']
TESSERACT_LANG = 'eng'

# GPU Settings
USE_GPU = True
PADDLE_USE_MP = True

# Output Settings
OUTPUT_DIR = "results"
SAVE_ANNOTATED_IMAGES = True
IMAGE_FORMAT = "png"
NDJSON_PREFIX = "ocr_results"
IMAGE_SUFFIX = "_with_boxes"

# Image Extensions
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]

# Bounding Box Colors (BGR format)
COLORS = {
    "paddle": (0, 0, 255),      # Red
    "easyocr": (0, 255, 0),     # Green  
    "tesseract": (255, 0, 0)    # Blue
}

# Box Drawing Settings
BOX_LINE_THICKNESS = 2
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# Processing Settings
# MAX_CONCURRENT_ENGINES = 3
# CONTINUE_ON_ERROR = True

# Confidence Thresholds
MIN_TEXT_CONFIDENCE = 0.5
MIN_BOX_CONFIDENCE = 0.3