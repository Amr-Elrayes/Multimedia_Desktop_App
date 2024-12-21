import tkinter as tk
from tkinter import Button, Label, Toplevel, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy.fftpack
from tkinter import simpledialog
from collections import Counter



class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Application")
        self.root.geometry("1200x700")
        self.root.state('zoomed')  # Start in full-screen mode

        # Initialize variables
        self.image = None
        self.original_image = None
        self.image_path = None
        self.active_button = None  # Track the active button

        # Create UI
        self.create_widgets()

    def create_widgets(self):
        # Right frame for buttons
        button_frame = tk.Frame(self.root, bg="#2E3440")
        button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # Canvas with scrollbar for buttons
        canvas = tk.Canvas(button_frame, bg="#2E3440", highlightthickness=0, width=240)
        scrollbar = tk.Scrollbar(button_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#2E3440")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        # Enable mouse wheel scrolling
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", lambda ev: self._on_mouse_wheel(ev, canvas)))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Buttons
        self.button_style = {
            "font": ("Arial", 12, "bold"),
            "bg": "#5E81AC",
            "fg": "white",
            "activebackground": "#81A1C1",
            "activeforeground": "black",
            "width": 25,
            "pady": 5,
            "relief": tk.FLAT,
        }

        buttons = [
            ("Read Image", self.load_image),
            ("Save Image", self.save_image),
            ("Reset to Original", self.reset_to_original),  # Modified here
            ("Change RGB to Grayscale", self.convert_to_grayscale),
            ("Resize Image", self.resize_image),
            ("Rotate Image", self.rotate_image),
            ("Translate Image", self.translate_image),
            ("Thresholding", self.thresholding),
            ("Image Sampling", self.sample_image),
            ("Image Quantization", self.quantize_image),
            ("Image Histogram", self.display_histogram),
            ("CMY Model", self.convert_to_cmy),
            ("HSI Model", self.convert_to_hsi),
            ("LAB Model", self.convert_to_lab),
            ("Ordered dithering", self.apply_ordered_dithering),
            ("Median Cut", self.apply_median_cut_quantization),
            ("Apply DCT & IDCT", self.apply_dct_and_idct),
            ("DWT", self.apply_Dwt_transform),
            ("Invers DWT", self.perform_inverse_Dwt_transform),
            ("Wavelet", self.apply_wavelet_transform_multi_level),
            ("Invers Wavelet", self.display_inverse_wavelet_reconstruction),
            ("Haar", self.apply_haar_wavelet_transform),
            ("Huffman", self.huffman_encoding),
            ("Shannon", self.shannon_fano_encoding),
            ("Arithmetic Encoding", self.arithmetic_encoding),
        ]

        self.buttons = []  # Store button widgets for styling

        for text, command in buttons:
            btn = tk.Button(
                scrollable_frame,
                text=text,
                command=lambda cmd=command, b=text: self.set_active_button(cmd, b),
                **self.button_style,
            )
            btn.pack(padx=5, pady=5, fill=tk.X)
            self.buttons.append(btn)

        # Canvas for displaying images
        self.original_canvas = tk.Canvas(self.root, bg="#ECEFF4", width=600, height=700)
        self.original_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.modified_canvas = tk.Canvas(self.root, bg="#ECEFF4", width=600, height=700)
        self.modified_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _on_mouse_wheel(self, event, canvas):
        canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def set_active_button(self, command, button_text):
        # Reset all button styles
        for btn in self.buttons:
            btn.config(bg="#5E81AC")  # Default color

        # Highlight the active button except for "Reset to Original"
        if button_text != "Reset to Original" and button_text != "Save Image" and button_text != "Read Image":
            for btn in self.buttons:
                if btn.cget("text") == button_text:
                    btn.config(bg="#A3BE8C")  # Active color

        # Execute the command
        command()

    # Image operations
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not file_path:
            return
        self.image_path = file_path
        self.image = cv2.imread(file_path)
        self.original_image = self.image.copy()
        self.display_image(self.original_image, self.original_canvas)
        self.clear_canvas(self.modified_canvas)

    def save_image(self):
        if self.image is None:
            messagebox.showwarning("Warning", "No image to save.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if not file_path:
            return
        cv2.imwrite(file_path, self.image)
        messagebox.showinfo("Success", "Image saved successfully!")

    def reset_to_original(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "No image loaded.")
            return
        # Reset the image to the original one
        self.image = self.original_image.copy()
        self.display_image(self.image, self.original_canvas)
        # Clear the modified canvas
        self.clear_canvas(self.modified_canvas)

    def convert_to_grayscale(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        self.display_image(self.image, self.modified_canvas)

    # Placeholder for other functions

    def resize_image(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        resize_window = tk.Toplevel(self.root)
        resize_window.title("Resize Image")

        tk.Label(resize_window, text="Width:").grid(row=0, column=0, padx=10, pady=5)
        width_entry = tk.Entry(resize_window)
        width_entry.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(resize_window, text="Height:").grid(row=1, column=0, padx=10, pady=5)
        height_entry = tk.Entry(resize_window)
        height_entry.grid(row=1, column=1, padx=10, pady=5)

        def apply_resize():
            try:
                width = int(width_entry.get())
                height = int(height_entry.get())
                resized_image = cv2.resize(self.original_image, (width, height))
                self.image = resized_image
                self.display_image(self.image, self.modified_canvas)
                resize_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid integer dimensions.")

        apply_button = tk.Button(resize_window, text="Apply", command=apply_resize)
        apply_button.grid(row=2, column=0, columnspan=2, pady=10)



    def rotate_image(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # Create a new window for rotating the image
        rotate_window = tk.Toplevel(self.root)
        rotate_window.title("Rotate Image")

        # Create a label and entry for the angle input
        tk.Label(rotate_window, text="Angle (degrees):").grid(row=0, column=0, padx=10, pady=5)
        angle_entry = tk.Entry(rotate_window)
        angle_entry.grid(row=0, column=1, padx=10, pady=5)

        def apply_rotation():
            try:
                angle = float(angle_entry.get())

                # Use the processed image if available, otherwise use the original image
                image_to_rotate =  self.original_image.copy()

                # Calculate the rotation matrix
                height, width = image_to_rotate.shape[:2]
                image_center = (width / 2, height / 2)
                rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1)

                # Calculate the sine and cosine of the angle
                abs_cos = abs(rotation_matrix[0, 0])
                abs_sin = abs(rotation_matrix[0, 1])

                # Compute the new bounding dimensions of the image
                new_width = int(width * abs_cos + height * abs_sin)
                new_height = int(width * abs_sin + height * abs_cos)

                # Adjust the rotation matrix to take into account translation
                rotation_matrix[0, 2] += (new_width / 2) - image_center[0]
                rotation_matrix[1, 2] += (new_height / 2) - image_center[1]

                # Perform the actual rotation
                rotated_image = cv2.warpAffine(image_to_rotate, rotation_matrix, (new_width, new_height))

                # Display the rotated image
                self.display_image(rotated_image, self.modified_canvas)
                self.image = rotated_image

                rotate_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid angle.")

        # Apply button to apply the rotation
        apply_button = tk.Button(rotate_window, text="Apply", command=apply_rotation)
        apply_button.grid(row=1, column=0, columnspan=2, pady=10)




    def translate_image(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # First window to get the matrix dimensions
        dimensions_window = tk.Toplevel(self.root)
        dimensions_window.title("Translation Matrix Dimensions")

        tk.Label(dimensions_window, text="Number of Rows:").grid(row=0, column=0, padx=10, pady=5)
        rows_entry = tk.Entry(dimensions_window)
        rows_entry.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(dimensions_window, text="Number of Columns:").grid(row=1, column=0, padx=10, pady=5)
        cols_entry = tk.Entry(dimensions_window)
        cols_entry.grid(row=1, column=1, padx=10, pady=5)

        def proceed_to_matrix_input():
            try:
                rows = int(rows_entry.get())
                cols = int(cols_entry.get())

                if rows != 2 or cols != 3:
                    messagebox.showerror("Error", "Matrix must be 2x3 for affine transformation.")
                    return

                dimensions_window.destroy()

                # Second window to input matrix values
                matrix_window = tk.Toplevel(self.root)
                matrix_window.title("Enter Translation Matrix Values")

                matrix_entries = []

                for i in range(rows):
                    row_entries = []
                    for j in range(cols):
                        entry = tk.Entry(matrix_window, width=10)
                        entry.grid(row=i, column=j, padx=5, pady=5)
                        row_entries.append(entry)
                    matrix_entries.append(row_entries)

                def apply_translation():
                    try:
                        # Read matrix values from entries
                        matrix_values = [
                            [float(matrix_entries[i][j].get()) for j in range(cols)]
                            for i in range(rows)
                        ]
                        
                        # Construct the translation matrix for cv2.warpAffine
                        translation_matrix = np.float32(matrix_values)

                        # Apply the transformation using cv2.warpAffine
                        translated_image = cv2.warpAffine(self.original_image, translation_matrix, 
                        (self.original_image.shape[1], self.original_image.shape[0]))

                        # Show the translated image
                        self.image = translated_image
                        self.display_image(self.image, self.modified_canvas)
                        matrix_window.destroy()
                    except ValueError:
                        messagebox.showerror("Error", "Please ensure all inputs are valid numbers.")

                tk.Button(matrix_window, text="Apply", command=apply_translation).grid(row=rows, column=0, columnspan=cols, pady=10)

            except ValueError:
                messagebox.showerror("Error", "Please enter valid dimensions.")

        tk.Button(dimensions_window, text="Next", command=proceed_to_matrix_input).grid(row=2, column=0, columnspan=2, pady=10)

    def sample_image(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        try:
            # Prompt user for cropping dimensions
            crop_window = tk.Toplevel()
            crop_window.title("Enter Crop Dimensions")

            tk.Label(crop_window, text="Start Y:").grid(row=0, column=0)
            start_y = tk.Entry(crop_window)
            start_y.grid(row=0, column=1)

            tk.Label(crop_window, text="End Y:").grid(row=1, column=0)
            end_y = tk.Entry(crop_window)
            end_y.grid(row=1, column=1)

            tk.Label(crop_window, text="Start X:").grid(row=2, column=0)
            start_x = tk.Entry(crop_window)
            start_x.grid(row=2, column=1)

            tk.Label(crop_window, text="End X:").grid(row=3, column=0)
            end_x = tk.Entry(crop_window)
            end_x.grid(row=3, column=1)

            def apply_crop():
                try:
                    sy = int(start_y.get())
                    ey = int(end_y.get())
                    sx = int(start_x.get())
                    ex = int(end_x.get())

                # Perform cropping on the original image
                    crop_img = self.original_image[sy:ey, sx:ex]
                    self.image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    self.display_image(self.image, self.modified_canvas)
                    crop_window.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Invalid dimensions or cropping failed: {e}")

            tk.Button(crop_window, text="Crop", command=apply_crop).grid(row=4, columnspan=2)
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def quantize_image(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        try:
        # Prompt user for quantization level
            quant_window = tk.Toplevel()
            quant_window.title("Select Quantization Level")

            tk.Label(quant_window, text="Choose quantization level:").pack()

            level_var = tk.StringVar(value="8")

            tk.Radiobutton(quant_window, text="4-bit", variable=level_var, value="4").pack(anchor="w")
            tk.Radiobutton(quant_window, text="8-bit", variable=level_var, value="8").pack(anchor="w")
            tk.Radiobutton(quant_window, text="16-bit", variable=level_var, value="16").pack(anchor="w")
            tk.Radiobutton(quant_window, text="32-bit", variable=level_var, value="32").pack(anchor="w")

            def apply_quantization():
                try:
                    level = int(level_var.get())
                    num_bins = 2 ** (level // 4)  # Convert bit level to number of bins
                    bins = np.linspace(0, self.original_image.max(), num_bins)
                    quantized_img = np.digitize(self.original_image, bins)
                    quantized_img = (np.vectorize(bins.tolist().__getitem__)(quantized_img - 1).astype(int))

                # Ensure values are in valid range for display
                    quantized_img = np.clip(quantized_img, 0, 255)

                    self.image = cv2.cvtColor(quantized_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                    self.display_image(self.image, self.modified_canvas)
                    quant_window.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Quantization failed: {e}")

            tk.Button(quant_window, text="Apply", command=apply_quantization).pack()
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
            
            
            
    def display_histogram(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        try:
        # Convert to grayscale if the image is in color
            if len(self.original_image.shape) == 3:
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.original_image

        # Calculate the histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        # Plot the histogram
            plt.figure()
            plt.hist(gray.ravel(), 256, [0, 256])
            plt.title('Histogram')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to display histogram: {e}")
            
    def convert_to_cmy(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        try:
        # Normalize the original image
            normalized_img = self.original_image / 255.0

        # Extract individual color channels
            B, G, R = normalized_img[:, :, 0], normalized_img[:, :, 1], normalized_img[:, :, 2]

        # Compute CMY channels
            C = 1 - R
            M = 1 - G
            Y = 1 - B

        # Merge CMY channels to form the CMY image
            cmy_image = cv2.merge([C, M, Y])

        # Convert back to 8-bit image
            cmy_image = (cmy_image * 255).astype(np.uint8)

        # Display the CMY image
            self.image = cmy_image
            self.display_image(self.image, self.modified_canvas)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to convert to CMY: {e}")
            
    def convert_to_hsi(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        try:
        # Convert the original image to HSI (HSV_FULL in OpenCV)
            hsi_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV_FULL)

        # Display the HSI image
            self.image = hsi_image
            self.display_image(self.image, self.modified_canvas)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to convert to HSI: {e}")
            
    def convert_to_lab(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        try:
        # Convert the original image to LAB color space
            lab_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2LAB)

        # Display the LAB image
            self.image = lab_image
            self.display_image(self.image, self.modified_canvas)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to convert to LAB: {e}")
            
    def apply_ordered_dithering(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        try:
            def get_dither_matrix(rows, cols):
                try:
                    matrix_window = tk.Toplevel()
                    matrix_window.title("Enter Dither Matrix Values")

                    matrix_entries = []
                    for i in range(rows):
                        row_entries = []
                        for j in range(cols):
                            entry = tk.Entry(matrix_window, width=5)
                            entry.grid(row=i, column=j, padx=5, pady=5)
                            row_entries.append(entry)
                        matrix_entries.append(row_entries)

                    def apply_matrix():
                        try:
                            dither_matrix = []
                            for i in range(rows):
                                row = []
                                for j in range(cols):
                                    value = int(matrix_entries[i][j].get())
                                    row.append(value)
                                dither_matrix.append(row)

                            perform_dithering(dither_matrix)
                            matrix_window.destroy()
                        except Exception as e:
                            messagebox.showerror("Error", f"Invalid matrix values: {e}")

                    tk.Button(matrix_window, text="Apply", command=apply_matrix).grid(row=rows, columnspan=cols, pady=10)
                except Exception as e:
                    messagebox.showerror("Error", f"Invalid dimensions: {e}")

            def perform_dithering(dither_matrix):
                try:
                # Convert the original image to grayscale
                    gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

                # Get dimensions of the dither matrix
                    n = len(dither_matrix)

                # Normalize the Image to Match Dither Matrix Range
                    new_range_divider = 256 / (n * n + 1)
                    normalized_img = gray_image // new_range_divider

                # Apply Ordered Dithering
                    rows, columns = gray_image.shape
                    dithered_img = np.zeros_like(gray_image)

                    for x in range(rows):
                        for y in range(columns):
                            i = x % n
                            j = y % n

                            if normalized_img[x, y] > dither_matrix[i][j]:
                                dithered_img[x, y] = 255  # White
                            else:
                                dithered_img[x, y] = 0  # Black

                # Display the Dithered Image
                    self.image = dithered_img
                    self.display_image(self.image, self.modified_canvas)

                except Exception as e:
                    messagebox.showerror("Error", f"Failed to apply dithering: {e}")

        # Prompt user for dither matrix dimensions directly
            dim_prompt_window = tk.Toplevel()
            dim_prompt_window.title("Enter Dither Matrix Dimensions")

            tk.Label(dim_prompt_window, text="Rows:").grid(row=0, column=0, padx=5, pady=5)
            row_var = tk.StringVar()
            tk.Entry(dim_prompt_window, textvariable=row_var, width=10).grid(row=0, column=1, padx=5, pady=5)

            tk.Label(dim_prompt_window, text="Columns:").grid(row=1, column=0, padx=5, pady=5)
            col_var = tk.StringVar()
            tk.Entry(dim_prompt_window, textvariable=col_var, width=10).grid(row=1, column=1, padx=5, pady=5)

            def proceed_to_matrix():
                try:
                    rows = int(row_var.get())
                    cols = int(col_var.get())
                    if rows > 0 and cols > 0:
                        dim_prompt_window.destroy()
                        get_dither_matrix(rows, cols)
                    else:
                        raise ValueError("Dimensions must be positive integers.")
                except ValueError as e:
                    messagebox.showerror("Error", f"Invalid dimensions: {e}")

            tk.Button(dim_prompt_window, text="Next", command=proceed_to_matrix).grid(row=2, columnspan=2, pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def apply_median_cut_quantization(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        try:
            def perform_quantization(num_colors):
                try:
                # Convert the original image to RGB (if not already)
                    rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

                    # Flatten the image into a 2D array of pixels (R, G, B)
                    pixels = rgb_image.reshape((-1, 3))

                # Find the most dominant colors using histogram binning
                    hist_bins = np.linspace(0, 256, num_colors + 1, endpoint=True)
                    quantized_pixels = np.zeros_like(pixels)

                    for channel in range(3):  # Process R, G, B channels separately
                        channel_values = pixels[:, channel]
                        indices = np.digitize(channel_values, bins=hist_bins) - 1
                        quantized_channel = (hist_bins[indices] + hist_bins[indices + 1]) // 2
                        quantized_pixels[:, channel] = quantized_channel

                # Reshape the quantized pixels back to the original image shape
                    quantized_image = quantized_pixels.reshape(rgb_image.shape).astype(np.uint8)

                # Convert back to BGR for OpenCV compatibility
                    self.image = cv2.cvtColor(quantized_image, cv2.COLOR_RGB2BGR)
                    self.display_image(self.image, self.modified_canvas)

                except Exception as e:
                    messagebox.showerror("Error", f"Failed to apply quantization: {e}")

        # Prompt user for the number of colors
            prompt_window = tk.Toplevel()
            prompt_window.title("Enter Number of Colors")

            tk.Label(prompt_window, text="Number of Colors:").grid(row=0, column=0, padx=5, pady=5)
            colors_var = tk.StringVar()
            tk.Entry(prompt_window, textvariable=colors_var, width=10).grid(row=0, column=1, padx=5, pady=5)

            def apply_quantization():
                try:
                    num_colors = int(colors_var.get())
                    if num_colors > 0:
                        prompt_window.destroy()
                        perform_quantization(num_colors)
                    else:
                        raise ValueError("Number of colors must be a positive integer.")
                except ValueError as e:
                    messagebox.showerror("Error", f"Invalid input: {e}")

            tk.Button(prompt_window, text="Apply", command=apply_quantization).grid(row=1, columnspan=2, pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
            
    def apply_dct_and_idct(self):

        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        try:
            import numpy as np
            from scipy.fftpack import dct, idct
            import cv2

            def dct_2d(block):
                """Apply 2D Discrete Cosine Transform."""
                return dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

            def idct_2d(block):
                """Apply 2D Inverse Discrete Cosine Transform."""
                return idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

            def perform_dct_and_idct(block_size, scaling_factor):
                """Function to process the image with DCT and IDCT, then show them vertically."""
            # Convert the image to grayscale
                original_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

            # Image dimensions
                img_height, img_width = original_gray.shape
                dct_result = np.zeros((img_height, img_width), dtype=np.float32)

            # Apply DCT to blocks
                for i in range(0, img_height, block_size):
                    for j in range(0, img_width, block_size):
                        block = original_gray[i:i+block_size, j:j+block_size]
                        if block.shape == (block_size, block_size):
                            dct_block = dct_2d(block)
                            dct_result[i:i+block_size, j:j+block_size] = dct_block

            # Apply Log scaling to DCT result
                dct_visual = np.log(np.abs(dct_result) * scaling_factor + 1)

            # Normalize DCT result to range [0, 255]
                dct_visual_normalized = cv2.normalize(dct_visual, None, 0, 255, cv2.NORM_MINMAX)
                dct_visual_uint8 = np.uint8(dct_visual_normalized)

            # Now apply IDCT to the DCT result
                idct_result = np.zeros_like(dct_result, dtype=np.float32)

                for i in range(0, img_height, block_size):
                    for j in range(0, img_width, block_size):
                        block = dct_result[i:i+block_size, j:j+block_size]
                        if block.shape == (block_size, block_size):
                            idct_result[i:i+block_size, j:j+block_size] = idct_2d(block)

            # Normalize IDCT result to range [0, 255]
                idct_result_normalized = np.clip(idct_result, 0, 255).astype(np.uint8)

            # Add titles to DCT and IDCT images
                dct_title = "DCT Effect"
                idct_title = "IDCT Reconstruction"

            # Add text to the top of DCT image
                dct_image_with_title = cv2.putText(dct_visual_uint8, dct_title, (10, 30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Add text to the top of IDCT image
                idct_image_with_title = cv2.putText(idct_result_normalized, idct_title, (10, 30),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Combine DCT and IDCT results vertically
                combined_image = np.vstack((dct_image_with_title, idct_image_with_title))

            # Convert combined image to BGR and display it
                self.image = cv2.cvtColor(combined_image, cv2.COLOR_GRAY2BGR)
                self.display_image(self.image, self.modified_canvas)

        # Input window for block size and scaling factor
            def get_user_input():
                input_window = tk.Toplevel()
                input_window.title("DCT & IDCT Parameters")

                block_size_var = tk.StringVar(value="8")
                scaling_factor_var = tk.StringVar(value="1.0")

                tk.Label(input_window, text="Block Size (e.g., 8):").grid(row=0, column=0, padx=5, pady=5)
                tk.Entry(input_window, textvariable=block_size_var, width=10).grid(row=0, column=1, padx=5, pady=5)

                tk.Label(input_window, text="Scaling Factor (e.g., 1.0):").grid(row=1, column=0, padx=5, pady=5)
                tk.Entry(input_window, textvariable=scaling_factor_var, width=10).grid(row=1, column=1, padx=5, pady=5)

                def apply_parameters():
                    try:
                        block_size = int(block_size_var.get())
                        scaling_factor = float(scaling_factor_var.get())

                        if block_size <= 0 or block_size > 32:
                            raise ValueError("Block size must be between 1 and 32.")
                        if scaling_factor <= 0:
                            raise ValueError("Scaling factor must be greater than 0.")

                        input_window.destroy()
                        perform_dct_and_idct(block_size, scaling_factor)

                    except ValueError as e:
                        messagebox.showerror("Invalid Input", f"Error: {e}")

                tk.Button(input_window, text="Apply", command=apply_parameters).grid(row=2, columnspan=2, pady=10)

            get_user_input()

        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")


    def apply_Dwt_transform(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        try:
            import numpy as np
            import pywt # type: ignore
            import cv2
            import matplotlib.pyplot as plt

            def compute_approximation_coefficients(img):
                """Compute and return approximation coefficients (cA)."""
                c = pywt.dwt2(img, 'db5')
                return c[0]

            def compute_horizontal_details(img):
                """Compute and return horizontal detail coefficients (cH)."""
                c = pywt.dwt2(img, 'db5')
                return c[1][0]

            def compute_vertical_details(img):
                """Compute and return vertical detail coefficients (cV)."""
                c = pywt.dwt2(img, 'db5')
                return c[1][1]

            def compute_diagonal_details(img):
                """Compute and return diagonal detail coefficients (cD)."""
                c = pywt.dwt2(img, 'db5')
                return c[1][2]

            def display_coefficients(coefficients, title):
                """Display the given coefficients in the GUI."""
                plt.figure(figsize=[5, 5])
                plt.imshow(coefficients, cmap="gray")
                plt.title(title)
                plt.xticks([])
                plt.yticks([])
                plt.show()

            def perform_wavelet_transform(selection):
                """Perform wavelet transform based on user selection."""
                # Convert the image to grayscale
                original_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

                if selection == "cA":
                    cA = compute_approximation_coefficients(original_gray)
                    display_coefficients(cA, "Approximation coefficients (cA)")
                elif selection == "cH":
                    cH = compute_horizontal_details(original_gray)
                    display_coefficients(cH, "Horizontal detail (cH)")
                elif selection == "cV":
                    cV = compute_vertical_details(original_gray)
                    display_coefficients(cV, "Vertical detail (cV)")
                elif selection == "cD":
                    cD = compute_diagonal_details(original_gray)
                    display_coefficients(cD, "Diagonal detail (cD)")
                elif selection == "All":
                    cA = compute_approximation_coefficients(original_gray)
                    cH = compute_horizontal_details(original_gray)
                    cV = compute_vertical_details(original_gray)
                    cD = compute_diagonal_details(original_gray)

                # Plot all coefficients
                    plt.figure(figsize=[10, 10])

                    plt.subplot(2, 2, 1)
                    plt.imshow(cA, cmap="gray")
                    plt.title("Approximation coefficients (cA)")
                    plt.xticks([])
                    plt.yticks([])

                    plt.subplot(2, 2, 2)
                    plt.imshow(cH, cmap="gray")
                    plt.title("Horizontal detail (cH)")
                    plt.xticks([])
                    plt.yticks([])

                    plt.subplot(2, 2, 3)
                    plt.imshow(cV, cmap="gray")
                    plt.title("Vertical detail (cV)")
                    plt.xticks([])
                    plt.yticks([])

                    plt.subplot(2, 2, 4)
                    plt.imshow(cD, cmap="gray")
                    plt.title("Diagonal detail (cD)")
                    plt.xticks([])
                    plt.yticks([])

                    plt.show()

        # Input window for wavelet transform
            def get_user_input():
                input_window = tk.Toplevel()
                input_window.title("Wavelet Transform Selection")

                tk.Label(input_window, text="Select the type of coefficients to display:").pack(padx=10, pady=5)

                selection_var = tk.StringVar(value="All")

                options = ["All", "cA", "cH", "cV", "cD"]

                for option in options:
                    tk.Radiobutton(input_window, text=option, variable=selection_var, value=option).pack(anchor="w", padx=10)

                def apply_selection():
                    selection = selection_var.get()
                    input_window.destroy()
                    perform_wavelet_transform(selection)

                tk.Button(input_window, text="Apply", command=apply_selection).pack(pady=10)

            get_user_input()

        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def perform_inverse_Dwt_transform(self):
        """Perform the Inverse DWT to reconstruct the original image."""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        try:
            import numpy as np
            import pywt # type: ignore
            import cv2

        # Convert the image to grayscale
            original_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # Perform DWT and IDWT
            c = pywt.dwt2(original_gray, 'db5')
            reconstructed_image = pywt.idwt2(c, 'db5')

        # Normalize the reconstructed image to fit the range [0, 255]
            reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

        # Convert to BGR for displaying in the main canvas
            self.image = cv2.cvtColor(reconstructed_image, cv2.COLOR_GRAY2BGR)
            self.display_image(self.image, self.modified_canvas)

        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")


    def apply_wavelet_transform_multi_level(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        try:
            import numpy as np
            import pywt # type: ignore
            import cv2
            import matplotlib.pyplot as plt

            def compute_approximation_coefficients(img):
                """Compute and return approximation coefficients (cA) using multi-level wavelet."""
                coeffs = pywt.wavedec2(img, 'db5', level=3)
                return coeffs[0]

            def compute_horizontal_details(img):
                """Compute and return horizontal detail coefficients (cH) using multi-level wavelet."""
                coeffs = pywt.wavedec2(img, 'db5', level=3)
                return coeffs[1][0]

            def compute_vertical_details(img):
                """Compute and return vertical detail coefficients (cV) using multi-level wavelet."""
                coeffs = pywt.wavedec2(img, 'db5', level=3)
                return coeffs[1][1]

            def compute_diagonal_details(img):
                """Compute and return diagonal detail coefficients (cD) using multi-level wavelet."""
                coeffs = pywt.wavedec2(img, 'db5', level=3)
                return coeffs[1][2]

            def display_coefficients(coefficients, title):
                """Display the given coefficients in a separate window."""
                plt.figure(figsize=[5, 5])
                plt.imshow(coefficients, cmap="gray")
                plt.title(title)
                plt.xticks([])  # Hide x-axis ticks
                plt.yticks([])  # Hide y-axis ticks
                plt.show()

            def perform_wavelet_transform(selection):
                """Perform wavelet transform based on user selection."""
                original_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

                if selection == "cA":
                    cA = compute_approximation_coefficients(original_gray)
                    display_coefficients(cA, "Approximation coefficients (cA)")

                elif selection == "cH":
                    cH = compute_horizontal_details(original_gray)
                    display_coefficients(cH, "Horizontal detail (cH)")

                elif selection == "cV":
                    cV = compute_vertical_details(original_gray)
                    display_coefficients(cV, "Vertical detail (cV)")

                elif selection == "cD":
                    cD = compute_diagonal_details(original_gray)
                    display_coefficients(cD, "Diagonal detail (cD)")

                elif selection == "All":
                    cA = compute_approximation_coefficients(original_gray)
                    cH = compute_horizontal_details(original_gray)
                    cV = compute_vertical_details(original_gray)
                    cD = compute_diagonal_details(original_gray)

                    # Create a single window to display all coefficients in a grid
                    plt.figure(figsize=[10, 10])

                    # Plot the approximation coefficients (cA)
                    plt.subplot(2, 2, 1)
                    plt.imshow(cA, cmap="gray")
                    plt.title("Approximation coefficients (cA)")
                    plt.xticks([])  # Hide x-axis ticks
                    plt.yticks([])  # Hide y-axis ticks

                    # Plot the horizontal detail coefficients (cH)
                    plt.subplot(2, 2, 2)
                    plt.imshow(cH, cmap="gray")
                    plt.title("Horizontal detail (cH)")
                    plt.xticks([])  # Hide x-axis ticks
                    plt.yticks([])  # Hide y-axis ticks

                    # Plot the vertical detail coefficients (cV)
                    plt.subplot(2, 2, 3)
                    plt.imshow(cV, cmap="gray")
                    plt.title("Vertical detail (cV)")
                    plt.xticks([])  # Hide x-axis ticks
                    plt.yticks([])  # Hide y-axis ticks

                    # Plot the diagonal detail coefficients (cD)
                    plt.subplot(2, 2, 4)
                    plt.imshow(cD, cmap="gray")
                    plt.title("Diagonal detail (cD)")
                    plt.xticks([])  # Hide x-axis ticks
                    plt.yticks([])  # Hide y-axis ticks

                    plt.show()


        # Input window for wavelet transform
            def get_user_input():
                input_window = tk.Toplevel()
                input_window.title("Wavelet Transform Selection")

                tk.Label(input_window, text="Select the type of coefficients to display:").pack(padx=10, pady=5)

                selection_var = tk.StringVar(value="All")

                options = ["All", "cA", "cH", "cV", "cD"]

                for option in options:
                    tk.Radiobutton(input_window, text=option, variable=selection_var, value=option).pack(anchor="w", padx=10)

                def apply_selection():
                    selection = selection_var.get()
                    input_window.destroy()
                    perform_wavelet_transform(selection)

                tk.Button(input_window, text="Apply", command=apply_selection).pack(pady=10)

            get_user_input()

        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")


    def display_inverse_wavelet_reconstruction(self):
        """Display the inverse DWT reconstruction of the image."""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        try:
            import numpy as np
            import pywt  # type: ignore
            import cv2
        # Convert the image to grayscale
            original_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # Perform DWT
            c = pywt.dwt2(original_gray, 'db5')

        # Perform inverse DWT reconstruction
            imgrec = pywt.waverec2(c, 'db5')

        # Normalize the reconstructed image to fit the range [0, 255]
            imgrec = np.clip(imgrec, 0, 255).astype(np.uint8)
            reconstructed_image_bgr = cv2.cvtColor(imgrec, cv2.COLOR_GRAY2BGR)
            self.image = reconstructed_image_bgr
            self.display_image(self.image, self.modified_canvas)

        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")




    def apply_haar_wavelet_transform(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        try:
            import numpy as np
            import pywt # type: ignore
            import cv2
            import matplotlib.pyplot as plt

            def perform_wavelet_transform(selection):
                """Perform Haar wavelet transform and display based on user selection."""
                original_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

                # Perform 2D Haar wavelet transform
                coeffs2 = pywt.dwt2(original_gray, 'haar')
                LL, (LH, HL, HH) = coeffs2

                if selection == "LL":
                    plt.figure(figsize=[6, 6])
                    plt.imshow(LL, cmap="gray")
                    plt.title("Approximation (LL)")
                    plt.axis('off')
                    plt.show()

                elif selection == "LH":
                    plt.figure(figsize=[6, 6])
                    plt.imshow(LH, cmap="gray")
                    plt.title("Horizontal Detail (LH)")
                    plt.axis('off')
                    plt.show()

                elif selection == "HL":
                    plt.figure(figsize=[6, 6])
                    plt.imshow(HL, cmap="gray")
                    plt.title("Vertical Detail (HL)")
                    plt.axis('off')
                    plt.show()

                elif selection == "HH":
                    plt.figure(figsize=[6, 6])
                    plt.imshow(HH, cmap="gray")
                    plt.title("Diagonal Detail (HH)")
                    plt.axis('off')
                    plt.show()

                elif selection == "All":
                # Plot all components in a single window
                    plt.figure(figsize=[12, 8])

                # Original Image
                    plt.subplot(2, 3, 1)
                    plt.imshow(original_gray, cmap="gray")
                    plt.title("Original Image")
                    plt.axis('off')

                # Approximation (LL)
                    plt.subplot(2, 3, 2)
                    plt.imshow(LL, cmap="gray")
                    plt.title("Approximation (LL)")
                    plt.axis('off')

                # Horizontal Detail (LH)
                    plt.subplot(2, 3, 3)
                    plt.imshow(LH, cmap="gray")
                    plt.title("Horizontal Detail (LH)")
                    plt.axis('off')

                # Vertical Detail (HL)
                    plt.subplot(2, 3, 4)
                    plt.imshow(HL, cmap="gray")
                    plt.title("Vertical Detail (HL)")
                    plt.axis('off')

                # Diagonal Detail (HH)
                    plt.subplot(2, 3, 5)
                    plt.imshow(HH, cmap="gray")
                    plt.title("Diagonal Detail (HH)")
                    plt.axis('off')

                    plt.show()

        # Input window for wavelet transform
            def get_user_input():
                input_window = tk.Toplevel()
                input_window.title("Wavelet Transform Selection")

                tk.Label(input_window, text="Select the type of coefficients to display:").pack(padx=10, pady=5)

                selection_var = tk.StringVar(value="All")

                options = ["All", "LL", "LH", "HL", "HH"]

                for option in options:
                    tk.Radiobutton(input_window, text=option, variable=selection_var, value=option).pack(anchor="w", padx=10)

                def apply_selection():
                    selection = selection_var.get()
                    input_window.destroy()
                    perform_wavelet_transform(selection)

                tk.Button(input_window, text="Apply", command=apply_selection).pack(pady=10)

            get_user_input()

        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")



    def huffman_encoding(self):
        """Unified Huffman Encoding function for text or image."""
        try:
        # Helper function: Build the Huffman Tree
            def make_tree(nodes):
                """Build the Huffman Tree."""
                class NodeTree:
                    def __init__(self, left=None, right=None):
                        self.left = left
                        self.right = right

                    def children(self):
                        return (self.left, self.right)

            # ترتيب العقد حسب التردد بشكل تصاعدي
                nodes = sorted(nodes, key=lambda x: x[1])

                while len(nodes) > 1:
                    (key1, c1) = nodes[0]
                    (key2, c2) = nodes[1]
                    nodes = nodes[2:]
                    node = NodeTree(key1, key2)
                    nodes.append((node, c1 + c2))
                    nodes = sorted(nodes, key=lambda x: x[1])  # إعادة الترتيب بعد كل عملية دمج

                return nodes[0][0]

        # Helper function: Generate Huffman Codes from the tree
            def huffman_code_tree(node, left=True, code=''):
                """Generate Huffman Codes from the tree."""
                if isinstance(node, (str, int, np.uint8)):
                    return {node: code}
                (l, r) = node.children()
                d = dict()
                d.update(huffman_code_tree(l, True, code + '0'))
                d.update(huffman_code_tree(r, False, code + '1'))
                return d

        # Helper function: Display Huffman results
            def display_huffman_results(encoding):
                """Display Huffman Encoding results in a new window."""
                results_window = tk.Toplevel(self.root)
                results_window.title("Huffman Encoding Results")

                tk.Label(results_window, text="Huffman Encoding Results", font=("Arial", 14, "bold")).pack(pady=10)

                text_area = tk.Text(results_window, wrap="word", font=("Courier", 10), height=15, width=50)
                text_area.pack(padx=10, pady=10)

                for key, code in encoding.items():
                    text_area.insert("end", f"'{key}': {code}\n")

                text_area.config(state="disabled")

        # Function for Huffman Encoding on text
            def huffman_on_text():
                try:
                    user_text = simpledialog.askstring("Input Text", "Enter your text for Huffman Encoding:")

                    if not user_text:
                        messagebox.showwarning("Warning", "No text entered. Please enter some text.")
                        return

                # Calculate character frequencies
                    txt_count = Counter(user_text)
                    freq = [(k, v) for k, v in txt_count.items()]

                # Build the Huffman Tree
                    root = make_tree(freq)

                # Generate Huffman codes
                    encoding = huffman_code_tree(root)

                # Display the results
                    display_huffman_results(encoding)

                except Exception as e:
                    messagebox.showerror("Error", f"An unexpected error occurred: {e}")

        # Function for Huffman Encoding on image
            def huffman_on_image():
                try:
                    if self.original_image is None:
                        messagebox.showwarning("Warning", "No image loaded. Please load an image first.")
                        return

                # Convert image to grayscale if it's colored
                    gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY) if len(self.original_image.shape) == 3 else self.original_image

                # Flatten the image into a 1D array
                    pixel_values = gray_image.flatten()

                # Calculate pixel frequencies
                    pixel_count = Counter(pixel_values)
                    freq = [(k, v) for k, v in pixel_count.items()]

                # Build the Huffman Tree
                    root = make_tree(freq)

                # Generate Huffman codes
                    encoding = huffman_code_tree(root)

                # Display the results
                    display_huffman_results(encoding)

                except Exception as e:
                    messagebox.showerror("Error", f"An unexpected error occurred: {e}")

        # Main Window: Ask the user to choose Text or Image
            menu_window = tk.Toplevel(self.root)
            menu_window.title("Huffman Encoding")
            tk.Label(menu_window, text="Choose Data Type for Huffman Encoding", font=("Arial", 14, "bold")).pack(pady=10)

            tk.Button(menu_window, text="Text", command=lambda: [menu_window.destroy(), huffman_on_text()]).pack(pady=5)
            tk.Button(menu_window, text="Image", command=lambda: [menu_window.destroy(), huffman_on_image()]).pack(pady=5)

        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")



    def shannon_fano_encoding(self):
        """Unified Shannon-Fano Encoding function for text or image."""
        try:
        # Helper function: Build Shannon-Fano Codes
            def shannon_fano(nodes, prefix='', codebook={}):
                """Recursive function to generate Shannon-Fano codes."""
                if len(nodes) == 1:
                    codebook[nodes[0][0]] = prefix
                    return

            # Split nodes into two groups with nearly equal frequency sums
                total = sum([freq for _, freq in nodes])
                cumulative = 0
                split_index = 0
                for i, (_, freq) in enumerate(nodes):
                    cumulative += freq
                    if cumulative >= total / 2:
                        split_index = i + 1
                        break

                left = nodes[:split_index]
                right = nodes[split_index:]

            # Assign '0' to the left group and '1' to the right group
                shannon_fano(left, prefix + '0', codebook)
                shannon_fano(right, prefix + '1', codebook)

                return codebook

        # Helper function: Display Shannon-Fano results
            def display_shannon_fano_results(encoding):
                """Display Shannon-Fano Encoding results in a new window."""
                results_window = tk.Toplevel(self.root)
                results_window.title("Shannon-Fano Encoding Results")

                tk.Label(results_window, text="Shannon-Fano Encoding Results", font=("Arial", 14, "bold")).pack(pady=10)

                text_area = tk.Text(results_window, wrap="word", font=("Courier", 10), height=15, width=50)
                text_area.pack(padx=10, pady=10)

                for key, code in encoding.items():
                    text_area.insert("end", f"'{key}': {code}\n")

                text_area.config(state="disabled")

        # Function for Shannon-Fano Encoding on text
            def shannon_fano_on_text():
                try:
                    user_text = simpledialog.askstring("Input Text", "Enter your text for Shannon-Fano Encoding:")

                    if not user_text:
                        messagebox.showwarning("Warning", "No text entered. Please enter some text.")
                        return

                # Calculate character frequencies
                    txt_count = Counter(user_text)
                    freq = [(k, v) for k, v in txt_count.items()]

                # Sort frequencies in descending order
                    freq = sorted(freq, key=lambda x: x[1], reverse=True)

                # Generate Shannon-Fano codes
                    encoding = shannon_fano(freq)

                # Display the results
                    display_shannon_fano_results(encoding)

                except Exception as e:
                    messagebox.showerror("Error", f"An unexpected error occurred: {e}")

        # Function for Shannon-Fano Encoding on image
            def shannon_fano_on_image():
                try:
                    if self.original_image is None:
                        messagebox.showwarning("Warning", "No image loaded. Please load an image first.")
                        return

                # Convert image to grayscale if it's colored
                    gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY) if len(self.original_image.shape) == 3 else self.original_image

                # Flatten the image into a 1D array
                    pixel_values = gray_image.flatten()

                # Calculate pixel frequencies
                    pixel_count = Counter(pixel_values)
                    freq = [(k, v) for k, v in pixel_count.items()]

                # Sort frequencies in descending order
                    freq = sorted(freq, key=lambda x: x[1], reverse=True)

                # Generate Shannon-Fano codes
                    encoding = shannon_fano(freq)

                # Display the results
                    display_shannon_fano_results(encoding)

                except Exception as e:
                    messagebox.showerror("Error", f"An unexpected error occurred: {e}")

        # Main Window: Ask the user to choose Text or Image
            menu_window = tk.Toplevel(self.root)
            menu_window.title("Shannon-Fano Encoding")
            tk.Label(menu_window, text="Choose Data Type for Shannon-Fano Encoding", font=("Arial", 14, "bold")).pack(pady=10)

            tk.Button(menu_window, text="Text", command=lambda: [menu_window.destroy(), shannon_fano_on_text()]).pack(pady=5)
            tk.Button(menu_window, text="Image", command=lambda: [menu_window.destroy(), shannon_fano_on_image()]).pack(pady=5)

        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")





    def arithmetic_encoding(self):
        """Perform Arithmetic Encoding on either text or image based on user choice."""
        try:
            import numpy as np
            from PIL import Image
            import tkinter as tk
            from tkinter import simpledialog, Toplevel, Label, Text, Button, messagebox

        # لحساب احتمالات الرموز
            def calculate_probabilities(data):
                unique, counts = np.unique(data, return_counts=True)
                total = np.sum(counts)
                probabilities = counts / total
                return dict(zip(unique, probabilities))

        # لحساب الفواصل التراكمية
            def compute_intervals(probabilities):
                intervals = {}
                low = 0.0
                for symbol, prob in sorted(probabilities.items()):
                    high = low + prob
                    intervals[symbol] = (low, high)
                    low = high
                return intervals

        # التشفير الحسابي
            def arithmetic_encode(data, intervals):
                low = 0.0
                high = 1.0
                for symbol in data:
                    symbol_low, symbol_high = intervals[symbol]
                    range_width = high - low
                    high = low + range_width * symbol_high
                    low = low + range_width * symbol_low
                return low

        # عرض النتائج في نافذة منفصلة
            def display_results(encoded_value, probabilities, intervals):
                result_window = Toplevel(self.root)
                result_window.title("Arithmetic Encoding Results")

                Label(result_window, text="Arithmetic Encoding Results", font=("Arial", 14, "bold")).pack(pady=10)

                text_area = Text(result_window, wrap="word", font=("Courier", 10), height=20, width=60)
                text_area.pack(padx=10, pady=10)

            # عرض القيم المشفرة
                text_area.insert("end", f"Encoded Value: {encoded_value}\n\n")
                text_area.insert("end", "Probabilities:\n")
                for symbol, prob in probabilities.items():
                    text_area.insert("end", f"'{symbol}': {prob:.6f}\n")

                text_area.insert("end", "\nIntervals:\n")
                for symbol, (low, high) in intervals.items():
                    text_area.insert("end", f"'{symbol}': ({low:.6f}, {high:.6f})\n")

                text_area.config(state="disabled")

        # معالجة النصوص
            def process_text():
                user_text = simpledialog.askstring("Input Text", "Enter text for Arithmetic Encoding:")
                if not user_text:
                    messagebox.showwarning("Warning", "No text entered. Please enter some text.")
                    return

                data = list(user_text)
                probabilities = calculate_probabilities(data)
                intervals = compute_intervals(probabilities)
                encoded_value = arithmetic_encode(data, intervals)

                display_results(encoded_value, probabilities, intervals)

        # معالجة الصور
            def process_image():
                if self.original_image is None:
                    messagebox.showwarning("Warning", "No image loaded. Please load an image first.")
                    return

            # تحويل الصورة إلى تدرج رمادي إذا كانت ملونة
                gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY) if len(self.original_image.shape) == 3 else self.original_image
                image_data = gray_image.flatten()

                probabilities = calculate_probabilities(image_data)
                intervals = compute_intervals(probabilities)
                encoded_value = arithmetic_encode(image_data, intervals)

                display_results(encoded_value, probabilities, intervals)

        # نافذة اختيار نوع البيانات
            def choose_data_type():
                menu_window = Toplevel(self.root)
                menu_window.title("Arithmetic Encoding")
                Label(menu_window, text="Choose Data Type for Arithmetic Encoding", font=("Arial", 14, "bold")).pack(pady=10)

                Button(menu_window, text="Text", command=lambda: [menu_window.destroy(), process_text()]).pack(pady=5)
                Button(menu_window, text="Image", command=lambda: [menu_window.destroy(), process_image()]).pack(pady=5)

        # استدعاء نافذة الاختيار
            choose_data_type()

        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
    def thresholding(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # Function to apply thresholding
        def apply_threshold(threshold_value):
            # Convert to grayscale if necessary
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
            
            # Convert back to BGR for consistent display
            self.image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)
            self.display_image(self.image, self.modified_canvas)

        # Function for automatic thresholding (Otsu's method)
        def automatic_thresholding():
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            
            # Otsu's thresholding
            _, otsu_thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to BGR for consistent display
            self.image = cv2.cvtColor(otsu_thresholded_image, cv2.COLOR_GRAY2BGR)
            self.display_image(self.image, self.modified_canvas)

        # Create an initial window for thresholding mode selection
        mode_window = tk.Toplevel(self.root)
        mode_window.title("Thresholding Mode")

        tk.Label(mode_window, text="Select Thresholding Mode:", font=("Arial", 12)).pack(pady=10)

        # Manual thresholding button
        def open_manual_thresholding():
            mode_window.destroy()

            # Create another window to enter threshold value
            manual_window = tk.Toplevel(self.root)
            manual_window.title("Manual Thresholding")

            tk.Label(manual_window, text="Enter Threshold Value (0-255):").grid(row=0, column=0, padx=10, pady=5)
            threshold_entry = tk.Entry(manual_window)
            threshold_entry.grid(row=0, column=1, padx=10, pady=5)

            def apply_manual_threshold():
                try:
                    threshold_value = int(threshold_entry.get())
                    if 0 <= threshold_value <= 255:
                        apply_threshold(threshold_value)
                        manual_window.destroy()
                    else:
                        messagebox.showerror("Error", "Threshold value must be between 0 and 255.")
                except ValueError:
                    messagebox.showerror("Error", "Please enter a valid integer threshold value.")

            # Apply button
            tk.Button(manual_window, text="Apply", command=apply_manual_threshold).grid(row=1, column=0, columnspan=2, pady=10)

        # Buttons for selecting mode
        tk.Button(mode_window, text="Manual Thresholding", width=25, command=open_manual_thresholding).pack(pady=5)
        tk.Button(mode_window, text="Automatic Thresholding (Otsu)", width=25, command=lambda: [automatic_thresholding(), mode_window.destroy()]).pack(pady=5)


    def display_image(self, img, canvas):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.thumbnail((canvas.winfo_width(), canvas.winfo_height()))
        img_tk = ImageTk.PhotoImage(img)
        canvas.image = img_tk
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    def clear_canvas(self, canvas):
        canvas.delete("all")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()