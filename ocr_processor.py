import argparse
import cv2
import numpy as np
import glob
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing

import tqdm
import tqdm.contrib.concurrent


class OCRProcessor:
    """Character recognition using template matching"""
    
    def __init__(self):
        """Initialize OCR processor with hardcoded configuration values"""
        # ROI extraction
        self.roi_height = 50
        
        # Preprocessing
        self.threshold_value = 30
        self.morphology_kernel = np.ones((3, 3), np.uint8)
        
        # Template matching
        self.glyph_height_threshold = 30
        self.contour_gap_threshold = 10
        self.glyph_width_tolerance = 10
        self.height_diff_tolerance = 3
        self.width_diff_tolerance = 3
        
        # Template characters
        self.template_chars_large = "NGOKSEC0123456789"
        self.template_chars_small = "WHSRmm0123456789."
        
        # Template rendering
        self.template_scale_large = 1.5
        self.template_scale_small = 1.0
        self.template_thickness = 2
        self.font = cv2.FONT_HERSHEY_DUPLEX
        
        # Build templates once during initialization
        self.templates = self._build_templates()
    
    @staticmethod
    def render_glyph(ch: str, scale: float, thickness: int) -> np.ndarray:
        """Render a single glyph character as binary image"""
        (w, h), baseline = cv2.getTextSize(
            ch, cv2.FONT_HERSHEY_DUPLEX, scale, thickness
        )
        
        pad = 3
        img = np.zeros((h + baseline + pad*2, w + pad*2), dtype=np.uint8)
        
        cv2.putText(
            img, ch, (pad, h + pad),
            cv2.FONT_HERSHEY_DUPLEX, scale, 255, thickness, cv2.LINE_AA
        )
        
        _, img = cv2.threshold(
            img, 20, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
        # Crop to content
        ys, xs = np.where(img > 0)
        img = img[ys.min():ys.max()+1, xs.min():xs.max()+1]
        
        return img
    
    def _build_templates(self) -> dict:
        """Build template dictionary for all character sets"""
        return {
            'large': {ch: self.render_glyph(ch, self.template_scale_large, self.template_thickness) 
                     for ch in self.template_chars_large},
            'small': {ch: self.render_glyph(ch, self.template_scale_small, self.template_thickness) 
                     for ch in self.template_chars_small},
        }
    
    def process_file(self, path: str) -> str:
        """
        Process an image file and return recognized text
        
        Args:
            path: Path to the image file
            
        Returns:
            Recognized text string
        """
        # Load image
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not load image from {path}")
        
        # Extract ROI
        roi = img[:self.roi_height, :]
        
        # Convert to binary
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.morphology_kernel)
        
        # Find contours
        contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        # Match templates
        matches = []
        contours_sorted = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        last_x = 0
        
        for i, contour in enumerate(contours_sorted):
            x, y, w, h = cv2.boundingRect(contour)
            glyph_img = binary[y:y+h, x:x+w]
                        
            best_ch = '?'
            best_score = -1
            
            # Choose template size based on glyph height
            size_key = 'large' if h > self.glyph_height_threshold else 'small'
            
            for ch, tmpl in self.templates[size_key].items():
                # Check if dimensions are close
                if (abs(tmpl.shape[0] - h) > self.height_diff_tolerance or 
                    abs(tmpl.shape[1] - w) > self.width_diff_tolerance):
                    continue
                
                # Resize template to match glyph height
                scale = h / tmpl.shape[0]
                tmpl_resized = cv2.resize(
                    tmpl, (int(tmpl.shape[1] * scale), h),
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Skip if widths differ too much
                if abs(tmpl_resized.shape[1] - w) > self.glyph_width_tolerance:
                    continue
                
                # Match template
                res = cv2.matchTemplate(
                    glyph_img, tmpl_resized,
                    cv2.TM_CCOEFF_NORMED
                )
                _, score, _, _ = cv2.minMaxLoc(res)
                
                if score > best_score:
                    best_score = score
                    best_ch = ch
            
            if best_ch == '1':
                x -= 2
                w += 2

            # Add space if gap is large enough
            if x - last_x > self.contour_gap_threshold:
                matches.append((' ', 1.0))
            last_x = x + w

            # print(f"Glyph {i}: recognized as '{size_key} {best_ch}' with score {best_score:.3f}")
            matches.append((best_ch, best_score))
        
        return ''.join(ch for ch, score in matches)
    
    def batch_process(self, paths: list[str], output_csv: str) -> int:
        """
        Process multiple image files matching a glob pattern and write results to CSV
        
        Args:
            pattern: Glob pattern for image files (e.g., "images/**/*.jpg")
            output_csv: Path to output CSV file
            
        Returns:
            Number of files processed
        """
        # results = tqdm.contrib.concurrent.process_map(
        #     self.process_file,
        #     paths,
        #     chunksize=10,
        # )

        results = []
        for path in tqdm.tqdm(paths):
            results.append(self.process_file(path))

        # Write results to CSV
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['File Path', 'Recognized Text'])
            
            for path, text in zip(paths, results):
                writer.writerow([path, text])
        
        print(f"\nBatch processing complete. Results written to {output_csv}")
        return len(paths)

    def batch_process_parallel(
        self,
        paths: list[str],
        output_csv: str,
        max_workers: int | None = None,
        use_processes: bool = False,
    ) -> int:
        """
        Parallel batch processing with threads or processes and CSV output.

        Args:
            paths: List of image file paths to process
            output_csv: Path to output CSV file
            max_workers: Number of workers (defaults to sensible value)
            use_processes: True to use ProcessPoolExecutor, else ThreadPoolExecutor

        Returns:
            Number of files processed
        """
        results_map: dict[str, str] = {}

        if use_processes:
            # Use a standalone worker to avoid pickling bound methods
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(_process_path_worker, p): p for p in paths}
                for future in tqdm.tqdm(as_completed(future_map), total=len(future_map)):
                    p = future_map[future]
                    try:
                        text = future.result()
                    except Exception as e:
                        text = f"ERROR: {e}"
                    results_map[p] = text
        else:
            # Threads work well with OpenCV (C extensions release GIL)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(self.process_file, p): p for p in paths}
                for future in tqdm.tqdm(as_completed(future_map), total=len(future_map)):
                    p = future_map[future]
                    try:
                        text = future.result()
                    except Exception as e:
                        text = f"ERROR: {e}"
                    results_map[p] = text

        # Write results to CSV in original order
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['File Path', 'Recognized Text'])
            for p in paths:
                writer.writerow([p, results_map.get(p, '')])

        print(f"\nParallel batch processing complete. Results written to {output_csv}")
        return len(paths)


def main():
    ap = argparse.ArgumentParser(description="OCR Processor")

    ap.add_argument(
        "-i",
        "--input_pattern",
        type=str,
        required=True,
        default="../251017~251223/*/*/*/Result/*.jpg",
        help="Glob pattern for input image files"
    )

    ap.add_argument(
        "-o",
        "--output_csv",
        type=str,
        required=False,
        default="ocr_results.csv",
        help="Path to output CSV file"
    )

    ap.add_argument( # count files and exit
        "-c",
        "--count_only",
        action="store_true",
        help="Only count files matching the input pattern and exit"
    )

    ap.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        help="Run batch processing in parallel"
    )

    ap.add_argument(
        "-P",
        "--processes",
        action="store_true",
        help="Use process-based parallelism (default is threads)"
    )

    ap.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers"
    )

    args = ap.parse_args()

    ocr_processor = OCRProcessor()
    input_pattern = args.input_pattern
    output_csv_path = args.output_csv
    # Find all files matching the pattern
    file_paths = glob.glob(input_pattern, recursive=True)
    
    if not file_paths:
        print(f"No files found matching pattern: {input_pattern}")
        return
    
    if args.count_only:
        print(f"Number of files matching pattern '{input_pattern}': {len(file_paths)}")
        return
    
    # Sort for consistent ordering
    file_paths.sort()

    if args.parallel:
        ocr_processor.batch_process_parallel(
            file_paths,
            output_csv_path,
            max_workers=args.workers,
            use_processes=args.processes,
        )
    else:
        ocr_processor.batch_process(file_paths, output_csv_path)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

def _process_path_worker(path: str) -> str:
    """Standalone worker to process a single path in a separate process."""
    # Create a fresh processor per process to avoid shared state issues
    processor = OCRProcessor()
    return processor.process_file(path)
