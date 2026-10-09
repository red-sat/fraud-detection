import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
import threading
import subprocess
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import webbrowser
import json
from datetime import datetime
import math


class ModernForgeryDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OCP SOLUTION")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)
        self.root.configure(bg='#f8fafc')

        # Modern color scheme
        self.colors = {
            'primary': '#6366f1',
            'primary_dark': '#4f46e5',
            'secondary': '#ec4899',
            'surface': '#ffffff',
            'surface_2': '#f8fafc',
            'surface_3': '#f1f5f9',
            'text': '#1e293b',
            'text_muted': '#64748b',
            'border': '#e2e8f0',
            'success': '#10b981',
            'warning': '#f59e0b',
            'error': '#ef4444'
        }

        # Set up custom styles
        self.setup_styles()

        # Variables
        self.image_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="output")
        self.setup_parameters()
        self.result_paths = {}
        self.analysis_running = False

        # Create the modern GUI
        self.create_modern_widgets()

        # Bind window events
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_styles(self):
        """Setup modern ttk styles"""
        style = ttk.Style()

        # Configure modern button style
        style.configure(
            "Modern.TButton",
            padding=(20, 12),
            font=('Segoe UI', 10, 'bold'),
            borderwidth=0,
            focuscolor='none'
        )

        # Primary button style
        style.configure(
            "Primary.TButton",
            background=self.colors['primary'],
            foreground='white',
            padding=(25, 15),
            font=('Segoe UI', 11, 'bold'),
            borderwidth=0,
            focuscolor='none'
        )

        # Card frame style
        style.configure(
            "Card.TFrame",
            background=self.colors['surface'],
            relief='flat',
            borderwidth=1
        )

        # Header label style
        style.configure(
            "Header.TLabel",
            background=self.colors['surface'],
            foreground=self.colors['text'],
            font=('Segoe UI', 14, 'bold'),
            padding=(0, 10)
        )

        # Subheader style
        style.configure(
            "Subheader.TLabel",
            background=self.colors['surface'],
            foreground=self.colors['text_muted'],
            font=('Segoe UI', 10),
            padding=(0, 5)
        )

    def setup_parameters(self):
        """Initialize parameter variables"""
        self.params = {
            'threshold': tk.DoubleVar(value=0.6),
            'numerical_threshold': tk.DoubleVar(value=0.4),
            'ocr_confidence': tk.DoubleVar(value=0.2),
            'ela_quality': tk.IntVar(value=90),
            'ela_error_scale': tk.IntVar(value=20),
            'ela_threshold': tk.IntVar(value=20)
        }

    def create_modern_widgets(self):
        """Create the modern GUI layout"""
        # Main container with padding
        main_container = tk.Frame(self.root, bg=self.colors['surface_2'], padx=30, pady=20)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Header section
        self.create_header(main_container)

        # Main content area
        content_frame = tk.Frame(main_container, bg=self.colors['surface_2'])
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))

        # Left panel (controls)
        left_panel = self.create_left_panel(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        # Right panel (results)
        right_panel = self.create_right_panel(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def create_header(self, parent):
        """Create the modern header section"""
        header_frame = tk.Frame(parent, bg=self.colors['surface'], height=120)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        header_frame.pack_propagate(False)

        # Add subtle border
        border_frame = tk.Frame(header_frame, bg=self.colors['border'], height=1)
        border_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Header content
        header_content = tk.Frame(header_frame, bg=self.colors['surface'])
        header_content.pack(expand=True, fill=tk.BOTH, padx=30, pady=20)

        # Icon and title
        title_frame = tk.Frame(header_content, bg=self.colors['surface'])
        title_frame.pack(side=tk.LEFT, fill=tk.Y)

        # App icon (using Unicode symbol)
        icon_label = tk.Label(
            title_frame,
            text="🛡️",
            font=('Segoe UI', 32),
            bg=self.colors['surface'],
            fg=self.colors['primary']
        )
        icon_label.pack(side=tk.LEFT, padx=(0, 15))

        # Title and subtitle
        text_frame = tk.Frame(title_frame, bg=self.colors['surface'])
        text_frame.pack(side=tk.LEFT, fill=tk.Y)

        title_label = tk.Label(
            text_frame,
            text="OCP SOLUTIONS",
            font=('Segoe UI', 24, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['text']
        )
        title_label.pack(anchor=tk.W)

        subtitle_label = tk.Label(
            text_frame,
            text="",
            font=('Segoe UI', 12),
            bg=self.colors['surface'],
            fg=self.colors['text_muted']
        )
        subtitle_label.pack(anchor=tk.W)

        # Status indicator
        status_frame = tk.Frame(header_content, bg=self.colors['surface'])
        status_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.status_indicator = tk.Canvas(status_frame, width=12, height=12, bg=self.colors['surface'],
                                          highlightthickness=0)
        self.status_indicator.pack(side=tk.RIGHT, padx=(10, 0), pady=20)
        self.status_indicator.create_oval(2, 2, 10, 10, fill=self.colors['success'], outline='')

        self.status_label = tk.Label(
            status_frame,
            text="Ready",
            font=('Segoe UI', 10),
            bg=self.colors['surface'],
            fg=self.colors['text_muted']
        )
        self.status_label.pack(side=tk.RIGHT, pady=20)

    def create_left_panel(self, parent):
        """Create the left control panel with guaranteed button visibility"""
        panel = tk.Frame(parent, bg=self.colors['surface'], width=400)
        panel.pack_propagate(False)

        # Panel content with padding
        content = tk.Frame(panel, bg=self.colors['surface'])
        content.pack(fill=tk.BOTH, expand=True, padx=25, pady=25)

        # Create file section
        self.create_file_section(content)

        # Create parameters section with remaining space
        self.create_parameters_section_compact(content)

        # ALWAYS VISIBLE Action buttons at the bottom - LAST so they stay visible
        self.create_action_buttons_fixed(content)

        return panel

    def create_file_section(self, parent):
        """Create the file selection section - compact version"""
        # Section header
        file_header = tk.Label(
            parent,
            text="📁 Document Selection",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['text']
        )
        file_header.pack(anchor=tk.W, pady=(0, 10))

        # Upload area - smaller
        self.upload_frame = tk.Frame(
            parent,
            bg=self.colors['surface_3'],
            relief=tk.FLAT,
            bd=2,
            cursor='hand2'
        )
        self.upload_frame.pack(fill=tk.X, pady=(0, 8))
        self.upload_frame.bind("<Button-1>", lambda e: self.browse_image())

        upload_content = tk.Frame(self.upload_frame, bg=self.colors['surface_3'])
        upload_content.pack(expand=True, fill=tk.BOTH, padx=15, pady=20)

        # Upload icon and text - smaller
        upload_icon = tk.Label(
            upload_content,
            text="📤",
            font=('Segoe UI', 24),
            bg=self.colors['surface_3']
        )
        upload_icon.pack()
        upload_icon.bind("<Button-1>", lambda e: self.browse_image())

        upload_text = tk.Label(
            upload_content,
            text="Click to select document\nSupports: PNG, JPG, JPEG, TIF, TIFF, BMP",
            font=('Segoe UI', 9),
            bg=self.colors['surface_3'],
            fg=self.colors['text_muted'],
            justify=tk.CENTER
        )
        upload_text.pack(pady=(8, 0))
        upload_text.bind("<Button-1>", lambda e: self.browse_image())

        # File info display
        self.file_info_frame = tk.Frame(parent, bg=self.colors['surface'])

        # Output directory - compact
        output_label = tk.Label(
            parent,
            text="📂 Output Directory:",
            font=('Segoe UI', 9, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['text']
        )
        output_label.pack(anchor=tk.W, pady=(15, 5))

        output_container = tk.Frame(parent, bg=self.colors['surface'])
        output_container.pack(fill=tk.X, pady=(0, 8))

        self.output_entry = tk.Entry(
            output_container,
            textvariable=self.output_dir,
            font=('Segoe UI', 9),
            bg=self.colors['surface_3'],
            fg=self.colors['text'],
            relief=tk.FLAT,
            bd=5
        )
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        output_btn = tk.Button(
            output_container,
            text="Browse",
            command=self.browse_output_dir,
            font=('Segoe UI', 8),
            bg=self.colors['primary'],
            fg='white',
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=6,
            cursor='hand2'
        )
        output_btn.pack(side=tk.RIGHT)

    def create_parameters_section_compact(self, parent):
        """Create compact parameters section with limited height"""
        # Section header
        params_header = tk.Label(
            parent,
            text="⚙️ Analysis Parameters",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['text']
        )
        params_header.pack(anchor=tk.W, pady=(15, 10))

        # Create limited-height frame for parameters to save space for button
        params_main_frame = tk.Frame(parent, bg=self.colors['surface'], height=180)
        params_main_frame.pack(fill=tk.X, expand=False, pady=(0, 15))
        params_main_frame.pack_propagate(False)  # Keep fixed height

        # Create scrollable area within fixed frame
        canvas = tk.Canvas(params_main_frame, bg=self.colors['surface'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(params_main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['surface'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Parameter definitions - compact
        param_configs = [
            ('threshold', 'General Threshold', 0.1, 1.0, 0.1, '🎯'),
            ('numerical_threshold', 'Numerical Detection', 0.1, 1.0, 0.1, '🔢'),
            ('ocr_confidence', 'OCR Confidence', 0.1, 1.0, 0.1, '📝'),
            ('ela_quality', 'ELA Quality', 70, 100, 5, '🔍'),
            ('ela_error_scale', 'ELA Error Scale', 5, 50, 5, '📊'),
            ('ela_threshold', 'ELA Threshold', 5, 50, 5, '⚡')
        ]

        for param_key, label, min_val, max_val, step, icon in param_configs:
            self.create_parameter_slider_compact(
                scrollable_frame, param_key, label, min_val, max_val, step, icon
            )

    def create_parameter_slider_compact(self, parent, param_key, label, min_val, max_val, step, icon):
        """Create a very compact parameter slider"""
        param_frame = tk.Frame(parent, bg=self.colors['surface'])
        param_frame.pack(fill=tk.X, pady=(0, 5))

        # Label row
        label_frame = tk.Frame(param_frame, bg=self.colors['surface'])
        label_frame.pack(fill=tk.X, pady=(0, 2))

        param_label = tk.Label(
            label_frame,
            text=f"{icon} {label}",
            font=('Segoe UI', 8),
            bg=self.colors['surface'],
            fg=self.colors['text']
        )
        param_label.pack(side=tk.LEFT)

        # Value display
        value_var = tk.StringVar()
        value_var.set(str(self.params[param_key].get()))

        value_label = tk.Label(
            label_frame,
            textvariable=value_var,
            font=('Segoe UI', 8, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['primary']
        )
        value_label.pack(side=tk.RIGHT)

        # Slider - very compact
        slider = tk.Scale(
            param_frame,
            from_=min_val,
            to=max_val,
            resolution=step,
            orient=tk.HORIZONTAL,
            variable=self.params[param_key],
            showvalue=0,
            bg=self.colors['surface'],
            fg=self.colors['text'],
            highlightthickness=0,
            troughcolor=self.colors['surface_3'],
            activebackground=self.colors['primary'],
            font=('Segoe UI', 7),
            length=280,
            width=15  # Thinner slider
        )
        slider.pack(fill=tk.X, pady=(0, 1))

        # Update value display when slider changes
        def update_value(*args):
            value_var.set(str(self.params[param_key].get()))

        self.params[param_key].trace('w', update_value)

    def create_action_buttons_fixed(self, parent):
        """Create action buttons that are GUARANTEED to be visible at the bottom"""
        # Add some spacing before the button
        spacer = tk.Frame(parent, bg=self.colors['surface'], height=20)
        spacer.pack(fill=tk.X)

        # Add visual separator
        separator = tk.Frame(parent, bg=self.colors['border'], height=1)
        separator.pack(fill=tk.X, pady=(0, 15))

        # MAIN ANALYZE BUTTON - Large and impossible to miss
        self.analyze_btn = tk.Button(
            parent,
            text=" START ANALYSIS",
            command=self.run_analysis,
            font=('Segoe UI', 16, 'bold'),
            bg=self.colors['primary'],
            fg='white',
            relief=tk.FLAT,
            bd=0,
            padx=30,
            pady=20,
            cursor='hand2',
            activebackground=self.colors['primary_dark']
        )
        self.analyze_btn.pack(fill=tk.X, pady=(0, 10))

        # Progress bar frame (initially hidden)
        self.progress_frame = tk.Frame(parent, bg=self.colors['surface'])

        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            mode='indeterminate'
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))

        self.progress_label = tk.Label(
            self.progress_frame,
            text="",
            font=('Segoe UI', 9),
            bg=self.colors['surface'],
            fg=self.colors['text_muted']
        )
        self.progress_label.pack()

    def create_right_panel(self, parent):
        """Create the right results panel"""
        panel = tk.Frame(parent, bg=self.colors['surface'])
        panel.pack_propagate(False)

        # Panel content with padding
        content = tk.Frame(panel, bg=self.colors['surface'])
        content.pack(fill=tk.BOTH, expand=True, padx=25, pady=25)

        # Results header
        results_header = tk.Label(
            content,
            text="📊 Analysis Results",
            font=('Segoe UI', 14, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['text']
        )
        results_header.pack(anchor=tk.W, pady=(0, 15))

        # Create notebook for tabbed interface
        self.create_results_notebook(content)

        return panel

    def create_results_notebook(self, parent):
        """Create a tabbed interface for results"""
        # Custom style for notebook
        style = ttk.Style()
        style.configure('Modern.TNotebook', background=self.colors['surface'])
        style.configure('Modern.TNotebook.Tab', padding=[20, 10])

        self.notebook = ttk.Notebook(parent, style='Modern.TNotebook')
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Output tab
        self.create_output_tab()

        # Preview tab
        self.create_preview_tab()

        # Results tab
        self.create_results_tab()

    def create_output_tab(self):
        """Create the output/console tab"""
        output_frame = tk.Frame(self.notebook, bg=self.colors['surface'])
        self.notebook.add(output_frame, text='📟 Console Output')

        # Output text area
        text_frame = tk.Frame(output_frame, bg='#1e293b', relief=tk.FLAT, bd=1)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        self.output_text = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            font=('Consolas', 10),
            bg='#1e293b',
            fg='#e2e8f0',
            insertbackground='#e2e8f0',
            relief=tk.FLAT,
            bd=10
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Initial message
        self.output_text.insert(tk.END, "ForgeShield AI Console\n")
        self.output_text.insert(tk.END, "=" * 50 + "\n")
        self.output_text.insert(tk.END, "Ready to analyze documents...\n\n")

    def create_preview_tab(self):
        """Create the image preview tab with zoom functionality"""
        preview_frame = tk.Frame(self.notebook, bg=self.colors['surface'])
        self.notebook.add(preview_frame, text='🖼️ Preview')

        # Zoom controls at top
        controls_frame = tk.Frame(preview_frame, bg=self.colors['surface'])
        controls_frame.pack(fill=tk.X, padx=15, pady=(15, 5))

        # Zoom buttons
        zoom_out_btn = tk.Button(
            controls_frame,
            text="🔍➖ Zoom Out",
            command=self.zoom_out,
            font=('Segoe UI', 9),
            bg=self.colors['primary'],
            fg='white',
            relief=tk.FLAT,
            bd=0,
            padx=15,
            pady=8,
            cursor='hand2'
        )
        zoom_out_btn.pack(side=tk.LEFT, padx=(0, 5))

        zoom_in_btn = tk.Button(
            controls_frame,
            text="🔍➕ Zoom In",
            command=self.zoom_in,
            font=('Segoe UI', 9),
            bg=self.colors['primary'],
            fg='white',
            relief=tk.FLAT,
            bd=0,
            padx=15,
            pady=8,
            cursor='hand2'
        )
        zoom_in_btn.pack(side=tk.LEFT, padx=(0, 5))

        zoom_reset_btn = tk.Button(
            controls_frame,
            text="🎯 Fit to Window",
            command=self.zoom_reset,
            font=('Segoe UI', 9),
            bg=self.colors['secondary'],
            fg='white',
            relief=tk.FLAT,
            bd=0,
            padx=15,
            pady=8,
            cursor='hand2'
        )
        zoom_reset_btn.pack(side=tk.LEFT, padx=(0, 5))

        # Zoom level display
        self.zoom_label = tk.Label(
            controls_frame,
            text="Zoom: 100%",
            font=('Segoe UI', 9, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['text']
        )
        self.zoom_label.pack(side=tk.RIGHT, padx=(5, 0))

        # Instructions
        instructions = tk.Label(
            controls_frame,
            text="💡 Use mouse wheel to zoom • Click and drag to pan",
            font=('Segoe UI', 8),
            bg=self.colors['surface'],
            fg=self.colors['text_muted']
        )
        instructions.pack(side=tk.RIGHT, padx=(5, 15))

        # Scrollable preview container
        preview_container = tk.Frame(preview_frame, bg=self.colors['surface_3'], relief=tk.SUNKEN, bd=1)
        preview_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        # Create canvas with scrollbars for panning large images
        self.preview_canvas = tk.Canvas(
            preview_container,
            bg=self.colors['surface_3'],
            highlightthickness=0
        )

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(preview_container, orient=tk.VERTICAL, command=self.preview_canvas.yview)
        h_scrollbar = ttk.Scrollbar(preview_container, orient=tk.HORIZONTAL, command=self.preview_canvas.xview)

        self.preview_canvas.configure(
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set
        )

        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.preview_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Image display area inside canvas
        self.image_display = tk.Label(
            self.preview_canvas,
            text="📷\n\nNo image selected\n\nChoose a document to preview",
            font=('Segoe UI', 12),
            bg=self.colors['surface_3'],
            fg=self.colors['text_muted'],
            justify=tk.CENTER
        )

        # Create window in canvas
        self.canvas_image_window = self.preview_canvas.create_window(
            0, 0, anchor=tk.NW, window=self.image_display
        )

        # Initialize zoom variables
        self.current_image = None
        self.original_image = None
        self.zoom_factor = 1.0
        self.canvas_width = 400
        self.canvas_height = 300

        # Bind mouse events for zoom and pan
        self.preview_canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.preview_canvas.bind("<Button-4>", self.on_mouse_wheel)  # Linux
        self.preview_canvas.bind("<Button-5>", self.on_mouse_wheel)  # Linux
        self.preview_canvas.bind("<ButtonPress-1>", self.start_pan)
        self.preview_canvas.bind("<B1-Motion>", self.do_pan)
        self.preview_canvas.bind("<Configure>", self.on_canvas_configure)

        # Pan variables
        self.pan_start_x = 0
        self.pan_start_y = 0

    def create_results_tab(self):
        """Create the results action tab"""
        results_frame = tk.Frame(self.notebook, bg=self.colors['surface'])
        self.notebook.add(results_frame, text='📈 Results')

        # Results container
        results_container = tk.Frame(results_frame, bg=self.colors['surface'])
        results_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Result buttons grid
        buttons_frame = tk.Frame(results_container, bg=self.colors['surface'])
        buttons_frame.pack(fill=tk.X, pady=(0, 20))

        # Create result buttons
        self.result_buttons = {}
        button_configs = [
            ('ela', '🔬 ELA Analysis', 'View Error Level Analysis results'),
            ('main', '🧠 AI Analysis', 'View MantraNet AI detection results'),
            ('intersection', '🎯 Intersection', 'View combined analysis overlay'),
            ('detailed', '📋 Detailed Report', 'View comprehensive analysis report'),
            ('folder', '📁 Open Folder', 'Open output folder in explorer')
        ]

        for i, (key, text, tooltip) in enumerate(button_configs):
            if key == 'folder':
                # Special handling for folder button - spans both columns
                btn = tk.Button(
                    buttons_frame,
                    text=text,
                    command=self.open_output_folder,
                    font=('Segoe UI', 10, 'bold'),
                    bg=self.colors['secondary'],
                    fg='white',
                    relief=tk.FLAT,
                    bd=0,
                    padx=20,
                    pady=15,
                    cursor='hand2'
                )
                btn.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky='ew')
            else:
                # Regular result buttons
                row = i // 2
                col = i % 2

                btn = tk.Button(
                    buttons_frame,
                    text=text,
                    state=tk.DISABLED,
                    font=('Segoe UI', 10),
                    bg=self.colors['surface_3'],
                    fg=self.colors['text_muted'],
                    relief=tk.FLAT,
                    bd=0,
                    padx=20,
                    pady=15,
                    cursor='hand2'
                )
                btn.grid(row=row, column=col, padx=5, pady=5, sticky='ew')

            self.result_buttons[key] = btn

            # Configure grid weights
            if i < 2:  # Only for first row
                buttons_frame.grid_columnconfigure(col % 2, weight=1)

        # Results summary area
        summary_label = tk.Label(
            results_container,
            text="Analysis Summary",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['text']
        )
        summary_label.pack(anchor=tk.W, pady=(0, 10))

        self.results_summary = tk.Text(
            results_container,
            height=10,
            font=('Segoe UI', 10),
            bg=self.colors['surface_3'],
            fg=self.colors['text'],
            relief=tk.FLAT,
            bd=10,
            wrap=tk.WORD
        )
        self.results_summary.pack(fill=tk.BOTH, expand=True)

        # Initial message
        self.results_summary.insert(tk.END, "No analysis results yet.\n\nRun an analysis to see detailed results here.")
        self.results_summary.config(state=tk.DISABLED)

    def browse_image(self):
        """Enhanced file browser with preview"""
        file_path = filedialog.askopenfilename(
            title="Select Document Image",
            filetypes=[
                ("All Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg;*.jpeg"),
                ("TIFF files", "*.tif;*.tiff"),
                ("BMP files", "*.bmp")
            ]
        )

        if file_path:
            self.image_path.set(file_path)
            self.display_file_info(file_path)
            self.display_image_preview(file_path)
            self.update_status("Document selected - Ready for analysis", self.colors['success'])

    def display_file_info(self, file_path):
        """Display selected file information"""
        # Clear previous file info
        for widget in self.file_info_frame.winfo_children():
            widget.destroy()

        # File info display
        self.file_info_frame.pack(fill=tk.X, pady=(10, 0))

        try:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            file_name = os.path.basename(file_path)

            info_text = f"📁 {file_name}\n📏 {file_size:.2f} MB"

            info_label = tk.Label(
                self.file_info_frame,
                text=info_text,
                font=('Segoe UI', 9),
                bg=self.colors['surface'],
                fg=self.colors['text'],
                justify=tk.LEFT
            )
            info_label.pack(anchor=tk.W, pady=5)

        except Exception as e:
            print(f"Error displaying file info: {e}")

    def display_image_preview(self, image_path):
        """Enhanced image preview with zoom support"""
        try:
            # Open and store original image
            self.original_image = Image.open(image_path)

            # Reset zoom
            self.zoom_factor = 1.0
            self.update_zoom_display()

            # Initial display
            self.update_image_display()

        except Exception as e:
            self.image_display.configure(
                text=f"❌\n\nPreview Error\n\n{str(e)}",
                image=""
            )
            self.current_image = None
            self.original_image = None

    def update_image_display(self):
        """Update the image display based on current zoom"""
        if self.original_image is None:
            return

        try:
            # Calculate new size based on zoom
            orig_width, orig_height = self.original_image.size
            new_width = int(orig_width * self.zoom_factor)
            new_height = int(orig_height * self.zoom_factor)

            # Resize image
            if self.zoom_factor != 1.0:
                resized_image = self.original_image.resize(
                    (new_width, new_height),
                    Image.Resampling.LANCZOS if self.zoom_factor > 1.0 else Image.Resampling.LANCZOS
                )
            else:
                resized_image = self.original_image

            # Convert to PhotoImage
            self.current_image = ImageTk.PhotoImage(resized_image)

            # Update display
            self.image_display.configure(image=self.current_image, text="")

            # Update canvas scroll region
            self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all"))

            # Center image if smaller than canvas
            self.center_image_if_needed()

        except Exception as e:
            print(f"Error updating image display: {e}")

    def center_image_if_needed(self):
        """Center the image in canvas if it's smaller than the canvas"""
        if self.current_image is None:
            return

        # Get canvas size
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()

        # Get image size
        img_width = self.current_image.width()
        img_height = self.current_image.height()

        # Calculate position to center
        x = max(0, (canvas_width - img_width) // 2)
        y = max(0, (canvas_height - img_height) // 2)

        # Update window position
        self.preview_canvas.coords(self.canvas_image_window, x, y)

    def zoom_in(self):
        """Zoom in on the image"""
        if self.original_image is None:
            return

        self.zoom_factor = min(self.zoom_factor * 1.2, 5.0)  # Max 500% zoom
        self.update_image_display()
        self.update_zoom_display()

    def zoom_out(self):
        """Zoom out on the image"""
        if self.original_image is None:
            return

        self.zoom_factor = max(self.zoom_factor / 1.2, 0.1)  # Min 10% zoom
        self.update_image_display()
        self.update_zoom_display()

    def zoom_reset(self):
        """Reset zoom to fit window"""
        if self.original_image is None:
            return

        # Get canvas size
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not ready yet, use default
            canvas_width = 600
            canvas_height = 400

        # Get image size
        img_width, img_height = self.original_image.size

        # Calculate scale to fit
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        self.zoom_factor = min(scale_x, scale_y, 1.0)  # Don't zoom larger than original

        self.update_image_display()
        self.update_zoom_display()

    def update_zoom_display(self):
        """Update the zoom percentage display"""
        zoom_percent = int(self.zoom_factor * 100)
        self.zoom_label.configure(text=f"Zoom: {zoom_percent}%")

    def on_mouse_wheel(self, event):
        """Handle mouse wheel zoom"""
        if self.original_image is None:
            return

        # Determine zoom direction
        if event.delta > 0 or event.num == 4:  # Zoom in
            self.zoom_factor = min(self.zoom_factor * 1.1, 5.0)
        else:  # Zoom out
            self.zoom_factor = max(self.zoom_factor / 1.1, 0.1)

        self.update_image_display()
        self.update_zoom_display()

    def start_pan(self, event):
        """Start panning operation"""
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.preview_canvas.configure(cursor="fleur")

    def do_pan(self, event):
        """Perform panning operation"""
        if self.current_image is None:
            return

        # Calculate movement
        delta_x = event.x - self.pan_start_x
        delta_y = event.y - self.pan_start_y

        # Move the canvas view
        self.preview_canvas.scan_dragto(event.x, event.y, gain=1)

        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def on_canvas_configure(self, event):
        """Handle canvas resize"""
        # Update canvas size tracking
        self.canvas_width = event.width
        self.canvas_height = event.height

        # Re-center image if needed
        if self.current_image and self.zoom_factor <= 1.0:
            self.center_image_if_needed()

    def browse_output_dir(self):
        """Browse for output directory"""
        dir_path = filedialog.askdirectory(title="Select Output Directory")
        if dir_path:
            self.output_dir.set(dir_path)

    def run_analysis(self):
        """Run the analysis with modern UI feedback"""
        if not self.image_path.get():
            messagebox.showerror("Error", "Please select a document image first")
            return

        if self.analysis_running:
            return

        # Create output directory if needed
        output_dir = self.output_dir.get()
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                messagebox.showerror("Error", f"Could not create output directory: {str(e)}")
                return

        self.start_analysis_ui()

        # Start analysis in separate thread
        threading.Thread(target=self._run_analysis_thread, daemon=True).start()

    def start_analysis_ui(self):
        """Start analysis UI state"""
        self.analysis_running = True
        self.analyze_btn.config(
            text="⏳ ANALYZING...",
            state=tk.DISABLED,
            bg=self.colors['text_muted']
        )

        # Show and start progress bar
        self.progress_frame.pack(fill=tk.X, pady=(10, 0))
        self.progress_bar.start(10)
        self.progress_label.config(text="Initializing analysis...")

        # Switch to output tab
        self.notebook.select(0)

        # Clear previous results
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, f"Starting analysis: {datetime.now().strftime('%H:%M:%S')}\n")
        self.output_text.insert(tk.END, "=" * 60 + "\n\n")

        # Update status
        self.update_status("Analysis in progress...", self.colors['warning'])

    def _run_analysis_thread(self):
        """Run analysis in background thread"""
        try:
            # Build command
            command = [
                sys.executable,
                "model.py",
                "--image", self.image_path.get(),
                "--output_dir", self.output_dir.get(),
                "--threshold", str(self.params['threshold'].get()),
                "--numerical_threshold", str(self.params['numerical_threshold'].get()),
                "--ocr_confidence", str(self.params['ocr_confidence'].get()),
                "--ela_quality", str(self.params['ela_quality'].get()),
                "--ela_error_scale", str(self.params['ela_error_scale'].get()),
                "--ela_threshold", str(self.params['ela_threshold'].get())
            ]

            self.update_output(f"Command: {' '.join(command)}\n\n")

            # Run process
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=False,
                bufsize=1
            )

            # Process output
            for line_bytes in iter(process.stdout.readline, b''):
                try:
                    line = line_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    line = line_bytes.decode('latin-1')

                self.update_output(line)
                self.parse_result_paths(line)

            return_code = process.wait()

            if return_code == 0:
                self.root.after(0, self.analysis_complete_success)
            else:
                self.root.after(0, lambda: self.analysis_complete_error(return_code))

        except Exception as e:
            self.root.after(0, lambda: self.analysis_complete_error(str(e)))

    def parse_result_paths(self, line):
        """Parse result file paths from output"""
        if line.startswith("1. ELA Analysis:"):
            self.result_paths['ela'] = line.split(":", 1)[1].strip()
        elif line.startswith("2. MantraNet Analysis:"):
            self.result_paths['main'] = line.split(":", 1)[1].strip()
        elif line.startswith("3. Intersection Analysis:"):
            self.result_paths['intersection'] = line.split(":", 1)[1].strip()
        elif line.startswith("4. Detailed Analysis:"):
            self.result_paths['detailed'] = line.split(":", 1)[1].strip()

    def update_output(self, text):
        """Thread-safe output update"""
        self.root.after(0, lambda: self._append_output(text))

    def _append_output(self, text):
        """Append text to output widget"""
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)

    def analysis_complete_success(self):
        """Handle successful analysis completion"""
        self.analysis_running = False

        # Update UI
        self.analyze_btn.config(
            text="✅ ANALYSIS COMPLETE",
            state=tk.NORMAL,
            bg=self.colors['success']
        )

        # Stop progress bar
        self.progress_bar.stop()
        self.progress_label.config(text="Analysis completed successfully!")

        # Update status
        self.update_status("Analysis completed successfully", self.colors['success'])

        # Enable result buttons and update summary
        self.enable_result_buttons()
        self.update_results_summary()

        # Auto-switch to results tab
        self.notebook.select(2)

        # Reset button after delay
        self.root.after(3000, self.reset_analyze_button)

    def analysis_complete_error(self, error):
        """Handle analysis error"""
        self.analysis_running = False

        # Update UI
        self.analyze_btn.config(
            text="❌ ANALYSIS FAILED",
            state=tk.NORMAL,
            bg=self.colors['error']
        )

        # Stop progress bar
        self.progress_bar.stop()
        self.progress_label.config(text=f"Analysis failed: {error}")

        # Update status
        self.update_status(f"Analysis failed: {error}", self.colors['error'])

        # Show error message
        self.update_output(f"\n❌ Analysis failed: {error}\n")

        # Reset button after delay
        self.root.after(3000, self.reset_analyze_button)

    def reset_analyze_button(self):
        """Reset analyze button to default state"""
        self.analyze_btn.config(
            text="🚀 START ANALYSIS",
            bg=self.colors['primary']
        )

    def enable_result_buttons(self):
        """Enable result buttons based on available files"""
        button_commands = {
            'ela': lambda: self.open_result_image('ela'),
            'main': lambda: self.open_result_image('main'),
            'intersection': lambda: self.open_result_image('intersection'),
            'detailed': lambda: self.open_result_image('detailed')
        }

        for key, btn in self.result_buttons.items():
            if key in self.result_paths and os.path.isfile(self.result_paths[key]):
                btn.config(
                    state=tk.NORMAL,
                    bg=self.colors['primary'],
                    fg='white',
                    command=button_commands[key]
                )
            else:
                btn.config(
                    state=tk.DISABLED,
                    bg=self.colors['surface_3'],
                    fg=self.colors['text_muted']
                )

    def update_results_summary(self):
        """Update the results summary text"""
        summary_text = f"Analysis completed at {datetime.now().strftime('%H:%M:%S')}\n\n"
        summary_text += "📊 Generated Results:\n"
        summary_text += "=" * 40 + "\n\n"

        result_descriptions = {
            'ela': "🔬 ELA Analysis - Error Level Analysis highlighting potential forgeries",
            'main': "🧠 AI Analysis - MantraNet deep learning detection results",
            'intersection': "🎯 Intersection Analysis - Combined overlay of all detection methods",
            'detailed': "📋 Detailed Report - Comprehensive analysis with statistics"
        }

        available_count = 0
        for key, description in result_descriptions.items():
            if key in self.result_paths and os.path.isfile(self.result_paths[key]):
                summary_text += f"✅ {description}\n    📁 {os.path.basename(self.result_paths[key])}\n\n"
                available_count += 1
            else:
                summary_text += f"❌ {description}\n    ⚠️ Not generated\n\n"

        summary_text += f"\n📈 Summary: {available_count}/4 analysis types completed\n"

        if available_count > 0:
            summary_text += "\n💡 Click the buttons above to view individual results."

        # Update summary widget
        self.results_summary.config(state=tk.NORMAL)
        self.results_summary.delete(1.0, tk.END)
        self.results_summary.insert(tk.END, summary_text)
        self.results_summary.config(state=tk.DISABLED)

    def open_result_image(self, result_key):
        """Open and display result image"""
        if result_key not in self.result_paths:
            messagebox.showerror("Error", "Result file not found")
            return

        file_path = self.result_paths[result_key]
        if not os.path.isfile(file_path):
            messagebox.showerror("Error", f"Result file does not exist: {file_path}")
            return

        try:
            # Display in preview tab
            self.display_image_preview(file_path)
            self.notebook.select(1)  # Switch to preview tab

            # Try to open in system default viewer with better error handling
            try:
                if sys.platform == 'win32':
                    # Try multiple methods for Windows
                    try:
                        os.startfile(file_path)
                    except OSError:
                        # If no association, try common image viewers
                        subprocess.run(['explorer', file_path], check=False)
                elif sys.platform == 'darwin':  # macOS
                    subprocess.run(['open', file_path])
                else:  # Linux and others
                    subprocess.run(['xdg-open', file_path])
            except Exception as open_error:
                # If external opening fails, just show in preview (which already worked)
                messagebox.showinfo("Image Displayed",
                                    f"Image displayed in Preview tab.\n\n"
                                    f"Could not open with system viewer: {str(open_error)}\n\n"
                                    f"File location: {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Could not display result image: {str(e)}")

    def open_output_folder(self):
        """Open the output folder in system file explorer"""
        output_dir = self.output_dir.get()

        if not os.path.exists(output_dir):
            messagebox.showerror("Error", f"Output directory does not exist: {output_dir}")
            return

        try:
            if sys.platform == 'win32':
                os.startfile(output_dir)
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', output_dir])
            else:  # Linux and others
                subprocess.run(['xdg-open', output_dir])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open output folder: {str(e)}")

    def update_status(self, message, color):
        """Update status indicator and message"""
        self.status_label.config(text=message)

        # Update status indicator color
        self.status_indicator.delete("all")
        self.status_indicator.create_oval(2, 2, 10, 10, fill=color, outline='')

    def save_session(self):
        """Save current session configuration"""
        try:
            session_data = {
                'image_path': self.image_path.get(),
                'output_dir': self.output_dir.get(),
                'parameters': {key: var.get() for key, var in self.params.items()},
                'timestamp': datetime.now().isoformat()
            }

            session_file = filedialog.asksaveasfilename(
                title="Save Session Configuration",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")]
            )

            if session_file:
                with open(session_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
                messagebox.showinfo("Success", "Session configuration saved successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Could not save session: {str(e)}")

    def load_session(self):
        """Load session configuration"""
        try:
            session_file = filedialog.askopenfilename(
                title="Load Session Configuration",
                filetypes=[("JSON files", "*.json")]
            )

            if session_file:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)

                # Restore configuration
                self.image_path.set(session_data.get('image_path', ''))
                self.output_dir.set(session_data.get('output_dir', 'output'))

                parameters = session_data.get('parameters', {})
                for key, value in parameters.items():
                    if key in self.params:
                        self.params[key].set(value)

                messagebox.showinfo("Success", "Session configuration loaded successfully!")

                # Update display if image exists
                if self.image_path.get() and os.path.isfile(self.image_path.get()):
                    self.display_image_preview(self.image_path.get())
                    self.display_file_info(self.image_path.get())

        except Exception as e:
            messagebox.showerror("Error", f"Could not load session: {str(e)}")

    def show_about(self):
        """Show about dialog"""
        about_text = """
ForgeShield AI - Document Authentication Tool

Version: 2.0
Created: 2025

Advanced document forgery detection using:
• Error Level Analysis (ELA)
• MantraNet Deep Learning
• OCR Text Analysis
• Multi-layered Detection

Features:
✓ Modern intuitive interface
✓ Real-time analysis progress
✓ Multiple detection algorithms
✓ Comprehensive result visualization
✓ Batch processing support

For support and updates, visit:
github.com/forgeshield-ai
        """

        messagebox.showinfo("About ForgeShield AI", about_text)

    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Document...", command=self.browse_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Set Output Directory...", command=self.browse_output_dir)
        file_menu.add_separator()
        file_menu.add_command(label="Save Session...", command=self.save_session, accelerator="Ctrl+S")
        file_menu.add_command(label="Load Session...", command=self.load_session, accelerator="Ctrl+L")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing, accelerator="Ctrl+Q")

        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Start Analysis", command=self.run_analysis, accelerator="F5")
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Reset Parameters", command=self.reset_parameters)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Console Output", command=lambda: self.notebook.select(0))
        view_menu.add_command(label="Image Preview", command=lambda: self.notebook.select(1))
        view_menu.add_command(label="Results", command=lambda: self.notebook.select(2))

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        help_menu.add_command(label="About", command=self.show_about)

        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.browse_image())
        self.root.bind('<Control-s>', lambda e: self.save_session())
        self.root.bind('<Control-l>', lambda e: self.load_session())
        self.root.bind('<Control-q>', lambda e: self.on_closing())
        self.root.bind('<F5>', lambda e: self.run_analysis())

    def reset_parameters(self):
        """Reset all parameters to default values"""
        defaults = {
            'threshold': 0.6,
            'numerical_threshold': 0.4,
            'ocr_confidence': 0.2,
            'ela_quality': 90,
            'ela_error_scale': 20,
            'ela_threshold': 20
        }

        for key, value in defaults.items():
            self.params[key].set(value)

    def show_user_guide(self):
        """Show user guide window"""
        guide_window = tk.Toplevel(self.root)
        guide_window.title("ForgeShield AI - User Guide")
        guide_window.geometry("600x500")
        guide_window.configure(bg=self.colors['surface'])

        # Guide content
        guide_text = scrolledtext.ScrolledText(
            guide_window,
            wrap=tk.WORD,
            font=('Segoe UI', 10),
            bg=self.colors['surface'],
            fg=self.colors['text'],
            padx=20,
            pady=20
        )
        guide_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        guide_content = """
FORGESHIELD AI - USER GUIDE

Getting Started:
1. Click "Click to select document" or drag & drop an image file
2. Adjust analysis parameters if needed (defaults work well)
3. Click "START ANALYSIS" to begin detection
4. View results in the Results tab

Supported Formats:
• PNG, JPG, JPEG - Most common image formats
• TIF, TIFF - High-quality document scans
• BMP - Uncompressed bitmap images

Analysis Parameters:

🎯 General Threshold (0.1-1.0)
Controls overall detection sensitivity. Higher values = stricter detection.

🔢 Numerical Threshold (0.1-1.0)
Specifically targets number/text alterations. Lower values catch subtle changes.

📝 OCR Confidence (0.1-1.0)
Minimum confidence for text recognition. Lower values process more text.

🔍 ELA Quality (70-100)
JPEG compression level for Error Level Analysis. 90 is recommended.

📊 ELA Error Scale (5-50)
Amplification of compression errors. Higher values show more details.

⚡ ELA Threshold (5-50)
Sensitivity for ELA anomaly detection. Adjust based on image quality.

Result Types:

🔬 ELA Analysis
Shows compression inconsistencies that indicate manipulation.

🧠 AI Analysis
Deep learning detection using MantraNet algorithm.

🎯 Intersection Analysis
Combined view of all detection methods for comprehensive analysis.

📋 Detailed Report
Statistical analysis and confidence scores for detected forgeries.

Tips for Best Results:
• Use high-resolution scans (300+ DPI)
• Ensure good lighting and contrast
• Avoid heavily compressed JPEG images
• Compare multiple analysis types for verification

Troubleshooting:
• If analysis fails, check file permissions and disk space
• Large files may take several minutes to process
• Ensure Python dependencies are installed correctly
        """

        guide_text.insert(tk.END, guide_content)
        guide_text.config(state=tk.DISABLED)

    def on_closing(self):
        """Handle application closing"""
        if self.analysis_running:
            if messagebox.askokcancel("Quit", "Analysis is running. Are you sure you want to quit?"):
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    """Main application entry point"""
    root = tk.Tk()

    # Set application icon if available
    try:
        # You can add an icon file here
        # root.iconbitmap('icon.ico')
        pass
    except:
        pass

    # Create and configure the application
    app = ModernForgeryDetectionGUI(root)
    app.create_menu_bar()

    # Center the window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")

    # Start the application
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Application Error", f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()