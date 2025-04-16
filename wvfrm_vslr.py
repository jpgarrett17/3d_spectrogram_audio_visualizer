import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl
import sys
from scipy import interpolate
import statsmodels.api as sm
import librosa
import os
import math
from math import pi


class Visualizer(object):
    """ A class to visualize a given song"""

    def __init__(self, path=None):
        """Initialize the visualizer, optionally with an audio path"""
        # Initialize Qt application
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        if path:
            self.initialize_gui()
            self.process_audio(path)

    def initialize_gui(self):
        """Initialize the GUI components"""
        self.w = gl.GLViewWidget()
        self.w.setGeometry(0, 0, 1920, 1080)
        self.w.setWindowTitle('Music Visualizer')
        self.w.setBackgroundColor('w')
        self.Vector = QtGui.QVector3D
        self.w.setCameraPosition(pos=None, distance=900, azimuth=270, elevation=50)
        self.w.show()

    def process_audio(self, path):
        """Process the audio file and prepare it for visualization"""
        print("Loading audio file...")
        # Primary feature extraction from the audio
        self.time_series, self.sample_rate = librosa.load(path)
        print("Audio file loaded. Processing audio features...")

        self.hop_length = 64  # Fidelity (# of "lines")
        print("Computing STFT...")
        self.stft = np.abs(librosa.stft(self.time_series, hop_length=self.hop_length, n_fft=2048))
        print("STFT computed. Calculating onset strength...")

        self.oenv = librosa.onset.onset_strength(y=self.time_series, sr=self.sample_rate,
                                                 hop_length=self.hop_length)
        print("Detecting onsets...")
        self.onset_frames = librosa.onset.onset_detect(onset_envelope=self.oenv, sr=self.sample_rate)
        self.beat_boolean = np.array([1 if i in self.onset_frames else 0 for i in
                                      range(0, len(librosa.times_like(self.stft)))])
        print("Computing spectral features...")
        self.spectral_centroid = librosa.feature.spectral_centroid(y=self.time_series, sr=self.sample_rate)
        self.chroma = librosa.feature.chroma_stft(S=self.stft, sr=self.sample_rate)
        self.n_chromas = self.chroma.shape[0]

        # Extracting and Manipulating features for visualization
        print("Processing visualization data...")
        self.tempo_mult = 20
        self.spectr_mult = 120  # Originally "60" (Controls height of amplitude)
        self.chroma_tracer_offset_height = self.tempo_mult + self.spectr_mult * 1.8

        print("Generating tempogram...")
        self.tempo_final = self.get_tempogram()
        print("Generating spectrogram...")
        self.spectrogram_final = self.get_spectrogram()
        self.spectro_beat_final = self.get_spectrogram()
        print("Calculating camera positions...")
        self.camera_x = self.get_camera_x_position()
        self.chroma_tracer_z = self.get_chroma_tracer_z()

        # Defining the presentation parameters
        self.tempo_chunks = self.tempo_final.shape[0]
        self.spectro_beat_chunks = self.spectro_beat_final.shape[0]
        self.window_length = 400
        self.matrix_offset = 1

        print("Preprocessing data for visualization...")
        # Extending the data to make the initial representation of the first timestamp seem continuous
        self.tempo_final = self.data_extender(self.tempo_final)
        self.chroma = self.data_extender(self.chroma)
        self.spectro_beat_final = self.data_extender(self.spectro_beat_final)
        self.beat_boolean = self.data_extender(self.beat_boolean)
        self.camera_x = self.data_extender(self.camera_x)
        self.chroma_tracer_z = self.data_extender(self.chroma_tracer_z)

        # Initializing the relevant data generators
        self.tempo_gen = self.data_sample_gen(self.tempo_final, self.matrix_offset)
        self.chroma_gen = self.data_sample_gen(self.chroma, self.matrix_offset, op_data_as_cood=False)
        self.specto_beat_gen = self.data_sample_gen(self.spectro_beat_final, self.matrix_offset, along_y=True)
        self.beat_gen = self.data_sample_gen(self.beat_boolean, self.matrix_offset, ip_data_1d=True)
        self.cam_x_gen = self.data_sample_gen(self.camera_x, self.matrix_offset, ip_data_1d=True)
        self.chroma_tracer_z_gen = self.data_sample_gen(self.chroma_tracer_z, self.matrix_offset, ip_data_1d=True)

        # Getting the first set of data sample
        self.specto_beat_sample = next(self.specto_beat_gen)
        self.beat_sample = next(self.beat_gen)
        self.cam_x_sample = next(self.cam_x_gen)
        self.chroma_tracer_z_sample = next(self.chroma_tracer_z_gen)

        # Sets the folder to store the frames
        self.total_frames = self.tempo_final.shape[1]
        self.img_path = os.path.dirname(path)
        os.makedirs(self.img_path, exist_ok=True)

        # Initializing Dictionaries to store the graph items
        self.traces = {}
        self.tempo_traces = {}
        self.circles = {}
        self.chroma_tracers = {}
        self.spectro_beat_tracers = {}

        # Initializing a tracker for observing the run
        self.tracker_gen = self.tracker(self.total_frames)
        next(self.tracker_gen)

        print("Initialization complete.")

    def initialize_visuals(self):
        """Initialize the visual elements for the animation"""
        print("Initializing visual elements...")
        # Spectrobeat visuals initialization
        start_pt = 0
        for i in range(self.window_length):
            pts = np.zeros((384, 3))
            start_pt += start_pt
            self.spectro_beat_tracers[i] = gl.GLLinePlotItem(pos=pts,
                                                             color=pg.mkColor(0, 0, 0, 255),  # Black color
                                                             antialias=True,
                                                             width=self.beat_sample[i])
            self.w.addItem(self.spectro_beat_tracers[i])
        print("Visual elements initialized.")

    def update(self):
        """
        This will update the graph items which were initialized by initialize_visuals()
        Every update will be saved as an image
        """
        # update tracker
        tracker_id = next(self.tracker_gen)

        # Updating the Spectro Beat array
        new_specto_beat_sample = next(self.specto_beat_gen)
        new_beat_sample = next(self.beat_gen)
        start_pt = 0
        for i in range(self.window_length):
            pts = new_specto_beat_sample[start_pt:start_pt + self.spectro_beat_chunks]
            start_pt += self.spectro_beat_chunks
            self.spectro_beat_tracers[i].setData(pos=pts,
                                                 color=pg.mkColor(0, 0, 0, 255),  # Black color
                                                 antialias=True,
                                                 width=1)
        # Updating the Camera
        new_cam_x_sample = next(self.cam_x_gen)

    def animation(self):
        """Start the animation timer"""
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(5)
        self.start()

    def capture_full_spectrogram(self, output_path="full_spectrogram.png"):
        """
        Generate a single image showing the entire audio spectrogram
        with interactive camera control before saving
        """
        print("Generating full spectrogram visualization...")

        # Clear any existing items
        for key in list(self.spectro_beat_tracers.keys()):
            self.w.removeItem(self.spectro_beat_tracers[key])
            del self.spectro_beat_tracers[key]

        # Get full spectrogram dimensions
        print(f"Spectrogram dimensions: {self.spectro_beat_final.shape}")

        # Create the 3D coordinates for the full spectrogram
        xx, yy = np.meshgrid(np.arange(self.spectro_beat_final.shape[1]),
                             np.arange(self.spectro_beat_final.shape[0]))

        # Create points array with proper shape
        pts = np.vstack((yy.ravel(), xx.ravel(), self.spectro_beat_final.ravel())).T

        print(f"Created {pts.shape[0]} points for wireframe visualization.")

        # Create a full wireframe
        self.current_wireframe = gl.GLLinePlotItem(pos=pts,
                                                   color=pg.mkColor(0, 0, 0, 255),  # Black color
                                                   antialias=True,
                                                   width=1)
        self.w.addItem(self.current_wireframe)

        # Initial camera position
        self.w.setCameraPosition(distance=1500, elevation=40, azimuth=250)

        # Create a control UI
        self.create_camera_controls(output_path)

        # Start the app
        self.start()

    def create_camera_controls(self, output_path):
        """
        Create a control panel for adjusting camera settings before saving the image
        """
        # Create a control window
        self.control_window = QtWidgets.QWidget()
        self.control_window.setWindowTitle("Visualization Controls")
        layout = QtWidgets.QVBoxLayout()
        self.control_window.setLayout(layout)

        # Add a tab widget for organization
        tab_widget = QtWidgets.QTabWidget()
        layout.addWidget(tab_widget)

        # Camera tab
        camera_tab = QtWidgets.QWidget()
        camera_layout = QtWidgets.QVBoxLayout()
        camera_tab.setLayout(camera_layout)

        # Display current camera values
        camera_position = self.w.cameraPosition()
        distance = camera_position.length()

        # Camera distance slider
        distance_layout = QtWidgets.QHBoxLayout()
        distance_label = QtWidgets.QLabel("Distance:")
        self.distance_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.distance_slider.setMinimum(500)
        self.distance_slider.setMaximum(3000)
        self.distance_slider.setValue(int(distance))
        self.distance_slider.valueChanged.connect(self.update_camera)
        self.distance_value = QtWidgets.QLabel(str(int(distance)))
        distance_layout.addWidget(distance_label)
        distance_layout.addWidget(self.distance_slider)
        distance_layout.addWidget(self.distance_value)
        camera_layout.addLayout(distance_layout)

        # Camera elevation slider
        elevation_layout = QtWidgets.QHBoxLayout()
        elevation_label = QtWidgets.QLabel("Elevation:")
        self.elevation_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.elevation_slider.setMinimum(0)
        self.elevation_slider.setMaximum(90)
        self.elevation_slider.setValue(40)  # Default value
        self.elevation_slider.valueChanged.connect(self.update_camera)
        self.elevation_value = QtWidgets.QLabel("40")
        elevation_layout.addWidget(elevation_label)
        elevation_layout.addWidget(self.elevation_slider)
        elevation_layout.addWidget(self.elevation_value)
        camera_layout.addLayout(elevation_layout)

        # Camera azimuth slider
        azimuth_layout = QtWidgets.QHBoxLayout()
        azimuth_label = QtWidgets.QLabel("Azimuth:")
        self.azimuth_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.azimuth_slider.setMinimum(0)
        self.azimuth_slider.setMaximum(360)
        self.azimuth_slider.setValue(250)  # Default value
        self.azimuth_slider.valueChanged.connect(self.update_camera)
        self.azimuth_value = QtWidgets.QLabel("250")
        azimuth_layout.addWidget(azimuth_label)
        azimuth_layout.addWidget(self.azimuth_slider)
        azimuth_layout.addWidget(self.azimuth_value)
        camera_layout.addLayout(azimuth_layout)

        # 2D Mode toggle
        two_d_layout = QtWidgets.QHBoxLayout()
        self.two_d_checkbox = QtWidgets.QCheckBox("2D Mode")
        self.two_d_checkbox.stateChanged.connect(self.toggle_2d_mode)
        two_d_layout.addWidget(self.two_d_checkbox)
        camera_layout.addLayout(two_d_layout)

        # Center X position slider
        center_x_layout = QtWidgets.QHBoxLayout()
        center_x_label = QtWidgets.QLabel("Center X:")
        self.center_x_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.center_x_slider.setMinimum(-500)
        self.center_x_slider.setMaximum(500)
        self.center_x_slider.setValue(0)
        self.center_x_slider.valueChanged.connect(self.update_camera)
        self.center_x_value = QtWidgets.QLabel("0")
        center_x_layout.addWidget(center_x_label)
        center_x_layout.addWidget(self.center_x_slider)
        center_x_layout.addWidget(self.center_x_value)
        camera_layout.addLayout(center_x_layout)

        # Center Y position slider
        center_y_layout = QtWidgets.QHBoxLayout()
        center_y_label = QtWidgets.QLabel("Center Y:")
        self.center_y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.center_y_slider.setMinimum(-500)
        self.center_y_slider.setMaximum(500)
        self.center_y_slider.setValue(0)
        self.center_y_slider.valueChanged.connect(self.update_camera)
        self.center_y_value = QtWidgets.QLabel("0")
        center_y_layout.addWidget(center_y_label)
        center_y_layout.addWidget(self.center_y_slider)
        center_y_layout.addWidget(self.center_y_value)
        camera_layout.addLayout(center_y_layout)

        # Center Z position slider
        center_z_layout = QtWidgets.QHBoxLayout()
        center_z_label = QtWidgets.QLabel("Center Z:")
        self.center_z_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.center_z_slider.setMinimum(-500)
        self.center_z_slider.setMaximum(500)
        self.center_z_slider.setValue(0)
        self.center_z_slider.valueChanged.connect(self.update_camera)
        self.center_z_value = QtWidgets.QLabel("0")
        center_z_layout.addWidget(center_z_label)
        center_z_layout.addWidget(self.center_z_slider)
        center_z_layout.addWidget(self.center_z_value)
        camera_layout.addLayout(center_z_layout)

        # Add camera tab
        tab_widget.addTab(camera_tab, "Camera")

        # Appearance tab
        appearance_tab = QtWidgets.QWidget()
        appearance_layout = QtWidgets.QVBoxLayout()
        appearance_tab.setLayout(appearance_layout)

        # Color section
        color_group = QtWidgets.QGroupBox("Colors")
        color_layout = QtWidgets.QVBoxLayout()
        color_group.setLayout(color_layout)

        # Line color picker
        line_color_layout = QtWidgets.QHBoxLayout()
        line_color_label = QtWidgets.QLabel("Line Color:")
        self.line_color_button = QtWidgets.QPushButton()
        self.line_color_button.setFixedSize(50, 20)
        self.current_line_color = pg.mkColor(0, 0, 0, 255)
        self.line_color_button.setStyleSheet(f"background-color: rgb(0, 0, 0);")
        self.line_color_button.clicked.connect(self.pick_line_color)
        line_color_layout.addWidget(line_color_label)
        line_color_layout.addWidget(self.line_color_button)
        color_layout.addLayout(line_color_layout)

        # Background color picker
        bg_color_layout = QtWidgets.QHBoxLayout()
        bg_color_label = QtWidgets.QLabel("Background Color:")
        self.bg_color_button = QtWidgets.QPushButton()
        self.bg_color_button.setFixedSize(50, 20)
        self.current_bg_color = pg.mkColor(255, 255, 255, 255)
        self.bg_color_button.setStyleSheet(f"background-color: rgb(255, 255, 255);")
        self.bg_color_button.clicked.connect(self.pick_bg_color)
        bg_color_layout.addWidget(bg_color_label)
        bg_color_layout.addWidget(self.bg_color_button)
        color_layout.addLayout(bg_color_layout)

        appearance_layout.addWidget(color_group)

        # Wireframe settings
        wireframe_group = QtWidgets.QGroupBox("Wireframe")
        wireframe_layout = QtWidgets.QVBoxLayout()
        wireframe_group.setLayout(wireframe_layout)

        # Line width slider
        line_width_layout = QtWidgets.QHBoxLayout()
        line_width_label = QtWidgets.QLabel("Line Width:")
        self.line_width_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.line_width_slider.setMinimum(1)
        self.line_width_slider.setMaximum(5)
        self.line_width_slider.setValue(1)
        self.line_width_slider.valueChanged.connect(self.update_wireframe)
        self.line_width_value = QtWidgets.QLabel("1")
        line_width_layout.addWidget(line_width_label)
        line_width_layout.addWidget(self.line_width_slider)
        line_width_layout.addWidget(self.line_width_value)
        wireframe_layout.addLayout(line_width_layout)

        # Horizontal stride slider (to make wireframe more sparse)
        h_stride_layout = QtWidgets.QHBoxLayout()
        h_stride_label = QtWidgets.QLabel("Horizontal Stride:")
        self.h_stride_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.h_stride_slider.setMinimum(1)
        self.h_stride_slider.setMaximum(10)
        self.h_stride_slider.setValue(1)
        self.h_stride_slider.valueChanged.connect(self.update_wireframe)
        self.h_stride_value = QtWidgets.QLabel("1")
        h_stride_layout.addWidget(h_stride_label)
        h_stride_layout.addWidget(self.h_stride_slider)
        h_stride_layout.addWidget(self.h_stride_value)
        wireframe_layout.addLayout(h_stride_layout)

        # Vertical stride slider (to make wireframe more sparse)
        v_stride_layout = QtWidgets.QHBoxLayout()
        v_stride_label = QtWidgets.QLabel("Vertical Stride:")
        self.v_stride_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.v_stride_slider.setMinimum(1)
        self.v_stride_slider.setMaximum(10)
        self.v_stride_slider.setValue(1)
        self.v_stride_slider.valueChanged.connect(self.update_wireframe)
        self.v_stride_value = QtWidgets.QLabel("1")
        v_stride_layout.addWidget(v_stride_label)
        v_stride_layout.addWidget(self.v_stride_slider)
        v_stride_layout.addWidget(self.v_stride_value)
        wireframe_layout.addLayout(v_stride_layout)

        # Compression/averaging controls
        compression_layout = QtWidgets.QVBoxLayout()

        # Checkbox to enable/disable time compression
        compress_checkbox_layout = QtWidgets.QHBoxLayout()
        self.downsample_checkbox = QtWidgets.QCheckBox("Enable Time Compression")
        self.downsample_checkbox.setChecked(True)  # Enable by default
        self.downsample_checkbox.stateChanged.connect(self.update_wireframe)
        compress_checkbox_layout.addWidget(self.downsample_checkbox)
        compression_layout.addLayout(compress_checkbox_layout)

        # Slider for compression amount
        downsample_slider_layout = QtWidgets.QHBoxLayout()
        downsample_label = QtWidgets.QLabel("Compression Amount:")
        self.downsample_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.downsample_slider.setMinimum(1)
        self.downsample_slider.setMaximum(10)
        self.downsample_slider.setValue(3)  # Default value
        self.downsample_slider.valueChanged.connect(self.update_wireframe)
        self.downsample_value = QtWidgets.QLabel("3")
        self.downsample_slider.valueChanged.connect(
            lambda v: self.downsample_value.setText(str(v))
        )
        downsample_slider_layout.addWidget(downsample_label)
        downsample_slider_layout.addWidget(self.downsample_slider)
        downsample_slider_layout.addWidget(self.downsample_value)
        compression_layout.addLayout(downsample_slider_layout)

        wireframe_layout.addLayout(compression_layout)

        appearance_layout.addWidget(wireframe_group)

        # Add appearance tab
        tab_widget.addTab(appearance_tab, "Appearance")

        # Save button
        save_layout = QtWidgets.QHBoxLayout()
        save_button = QtWidgets.QPushButton("Save Image")
        save_button.clicked.connect(lambda: self._save_and_quit(output_path))
        change_output_button = QtWidgets.QPushButton("Change Output File...")
        change_output_button.clicked.connect(lambda: self._change_output_file(output_path))
        save_layout.addWidget(save_button)
        save_layout.addWidget(change_output_button)
        layout.addLayout(save_layout)

        # Store reference to wireframe
        self.current_wireframe = None

        # Show the control window
        self.control_window.show()

    def pick_line_color(self):
        """Open a color picker dialog for line color"""
        color_dialog = QtWidgets.QColorDialog(self.control_window)
        color = color_dialog.getColor(initial=QtGui.QColor(0, 0, 0), title="Select Line Color")

        if color.isValid():
            self.current_line_color = pg.mkColor(color.red(), color.green(), color.blue(), 255)
            self.line_color_button.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});")
            self.update_wireframe()

    def pick_bg_color(self):
        """Open a color picker dialog for background color"""
        color_dialog = QtWidgets.QColorDialog(self.control_window)
        color = color_dialog.getColor(initial=QtGui.QColor(255, 255, 255), title="Select Background Color")

        if color.isValid():
            self.current_bg_color = pg.mkColor(color.red(), color.green(), color.blue(), 255)
            self.bg_color_button.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});")
            self.w.setBackgroundColor(self.current_bg_color)

    def toggle_2d_mode(self):
        """Toggle between 2D and 3D viewing modes"""
        if self.two_d_checkbox.isChecked():
            # 2D mode - set near top-down view
            self.elevation_slider.setValue(89)
            self.azimuth_slider.setValue(270)
        else:
            # 3D mode - set to angled view
            self.elevation_slider.setValue(40)
            self.azimuth_slider.setValue(250)

        # Update the camera
        self.update_camera()

    def update_wireframe(self):
        """Update the wireframe appearance based on settings"""
        # Update labels
        self.line_width_value.setText(str(self.line_width_slider.value()))
        self.h_stride_value.setText(str(self.h_stride_slider.value()))
        self.v_stride_value.setText(str(self.v_stride_slider.value()))

        # Get stride values
        h_stride = self.h_stride_slider.value()
        v_stride = self.v_stride_slider.value()
        line_width = self.line_width_slider.value()

        # Check if downsample/compression is enabled
        downsample_enabled = False
        downsample_factor = 1
        max_length = 100  # Maximum desired length

        if hasattr(self, 'downsample_checkbox') and self.downsample_checkbox.isChecked():
            downsample_enabled = True
            if hasattr(self, 'downsample_slider'):
                downsample_factor = self.downsample_slider.value()

        # Remove existing wireframe
        if self.current_wireframe is not None:
            self.w.removeItem(self.current_wireframe)

        # Clear all line plot items from the view to ensure no duplicates
        for item in list(self.w.items):
            if isinstance(item, gl.GLLinePlotItem):
                self.w.removeItem(item)

        # Apply downsampling if enabled
        if downsample_enabled:
            # Calculate current length and determine needed compression
            current_length = self.spectro_beat_final.shape[1]

            if current_length > max_length:
                # Calculate how many frames to merge
                # Using the slider value to determine compression factor
                frames_to_merge = max(1, int(current_length / (max_length / downsample_factor)))

                # Reshape and average to reduce time dimension
                time_steps = self.spectro_beat_final.shape[1]
                # Calculate how many complete chunks we can make
                complete_chunks = (time_steps // frames_to_merge) * frames_to_merge

                # Use only the complete chunks for reshaping
                usable_data = self.spectro_beat_final[:, :complete_chunks]

                # Reshape to group frames together
                reshaped = usable_data.reshape(
                    self.spectro_beat_final.shape[0],
                    -1,
                    frames_to_merge
                )

                # Average each group
                compressed_data = np.mean(reshaped, axis=2)

                working_data = compressed_data
            else:
                working_data = self.spectro_beat_final
        else:
            working_data = self.spectro_beat_final

        # Create 3D coordinates with the specified stride for a sparser wireframe
        xx, yy = np.meshgrid(
            np.arange(0, working_data.shape[1], h_stride),
            np.arange(0, working_data.shape[0], v_stride)
        )

        # Get z values using the sparser grid
        z_values = working_data[::v_stride, ::h_stride]

        # Create points array
        pts = np.vstack((yy.ravel(), xx.ravel(), z_values.ravel())).T

        # Create a new wireframe
        self.current_wireframe = gl.GLLinePlotItem(
            pos=pts,
            color=self.current_line_color,
            antialias=True,
            width=line_width
        )
        self.w.addItem(self.current_wireframe)

    def update_camera(self):
        """Update the camera based on slider values"""
        distance = self.distance_slider.value()
        elevation = self.elevation_slider.value()
        azimuth = self.azimuth_slider.value()
        center_x = self.center_x_slider.value()
        center_y = self.center_y_slider.value()
        center_z = self.center_z_slider.value()

        # Update labels
        self.distance_value.setText(str(distance))
        self.elevation_value.setText(str(elevation))
        self.azimuth_value.setText(str(azimuth))
        self.center_x_value.setText(str(center_x))
        self.center_y_value.setText(str(center_y))
        self.center_z_value.setText(str(center_z))

        # Set center position
        center = QtGui.QVector3D(center_x, center_y, center_z)

        # Update camera
        self.w.setCameraPosition(pos=center, distance=distance, elevation=elevation, azimuth=azimuth)

    def _change_output_file(self, current_output_path):
        """Open a dialog to change the output file path"""
        file_dialog = QtWidgets.QFileDialog()
        new_path, _ = file_dialog.getSaveFileName(
            None,
            "Save Visualization As",
            current_output_path,
            "Images (*.png *.jpg *.jpeg);;All Files (*)"
        )

        if new_path:
            # Return the new path and notify the user
            print(f"Output path changed to: {new_path}")
            return new_path
        return current_output_path

    def browse_audio_file(self):
        """Open a file dialog to browse for an audio file"""
        file_dialog = QtWidgets.QFileDialog()
        audio_file, _ = file_dialog.getOpenFileName(
            None,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)"
        )

        if audio_file:
            # Update the audio path in the UI
            self.audio_path_input.setText(audio_file)
            return audio_file
        return None

    def browse_save_location(self):
        """Open a file dialog to select where to save the visualization"""
        file_dialog = QtWidgets.QFileDialog()
        save_path, _ = file_dialog.getSaveFileName(
            None,
            "Save Visualization As",
            os.path.join(os.path.expanduser("~"), "visualization.png"),
            "Images (*.png *.jpg *.jpeg);;All Files (*)"
        )

        if save_path:
            # Update the save path in the UI
            self.save_path_input.setText(save_path)
            return save_path
        return None

    def show_startup_dialog(self):
        """Show a dialog to select input and output files before starting visualization"""
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Audio Visualization Settings")
        layout = QtWidgets.QVBoxLayout()

        # Audio file selection
        audio_group = QtWidgets.QGroupBox("Select Audio File")
        audio_layout = QtWidgets.QHBoxLayout()

        self.audio_path_input = QtWidgets.QLineEdit()
        browse_audio_btn = QtWidgets.QPushButton("Browse...")
        browse_audio_btn.clicked.connect(lambda: self.browse_audio_file())

        audio_layout.addWidget(self.audio_path_input)
        audio_layout.addWidget(browse_audio_btn)
        audio_group.setLayout(audio_layout)

        # Output file selection
        output_group = QtWidgets.QGroupBox("Select Output Location")
        output_layout = QtWidgets.QHBoxLayout()

        self.save_path_input = QtWidgets.QLineEdit()
        # Default to the same directory as the audio file but with .png extension
        browse_output_btn = QtWidgets.QPushButton("Browse...")
        browse_output_btn.clicked.connect(lambda: self.browse_save_location())

        output_layout.addWidget(self.save_path_input)
        output_layout.addWidget(browse_output_btn)
        output_group.setLayout(output_layout)

        # OK and Cancel buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        # Add all widgets to dialog
        layout.addWidget(audio_group)
        layout.addWidget(output_group)
        layout.addWidget(button_box)
        dialog.setLayout(layout)

        # Show dialog and get result
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            return self.audio_path_input.text(), self.save_path_input.text()
        else:
            return None, None

    def _save_and_quit(self, output_path):
        """Helper function to save the image and quit the application"""
        # Get current camera and appearance settings for reference
        distance = self.distance_slider.value() if hasattr(self, 'distance_slider') else 0
        elevation = self.elevation_slider.value() if hasattr(self, 'elevation_slider') else 0
        azimuth = self.azimuth_slider.value() if hasattr(self, 'azimuth_slider') else 0
        center_x = self.center_x_slider.value() if hasattr(self, 'center_x_slider') else 0
        center_y = self.center_y_slider.value() if hasattr(self, 'center_y_slider') else 0
        center_z = self.center_z_slider.value() if hasattr(self, 'center_z_slider') else 0

        # Appearance settings
        line_width = self.line_width_slider.value() if hasattr(self, 'line_width_slider') else 1
        h_stride = self.h_stride_slider.value() if hasattr(self, 'h_stride_slider') else 1
        v_stride = self.v_stride_slider.value() if hasattr(self, 'v_stride_slider') else 1

        # Compression settings
        compression_enabled = self.downsample_checkbox.isChecked() if hasattr(self, 'downsample_checkbox') else False
        compression_amount = self.downsample_slider.value() if hasattr(self, 'downsample_slider') else 1

        # Get color values
        line_color = self.current_line_color if hasattr(self, 'current_line_color') else pg.mkColor(0, 0, 0, 255)
        bg_color = self.current_bg_color if hasattr(self, 'current_bg_color') else pg.mkColor(255, 255, 255, 255)

        # Save the image
        self.w.grabFramebuffer().save(output_path)
        print(f"Full spectrogram image saved to: {output_path}")

        # Save settings to a text file for reference
        settings_path = os.path.splitext(output_path)[0] + "_settings.txt"
        with open(settings_path, 'w') as f:
            f.write(f"Visualization Settings:\n\n")
            f.write(f"Camera Settings:\n")
            f.write(f"Distance: {distance}\n")
            f.write(f"Elevation: {elevation}\n")
            f.write(f"Azimuth: {azimuth}\n")
            f.write(f"Center X: {center_x}\n")
            f.write(f"Center Y: {center_y}\n")
            f.write(f"Center Z: {center_z}\n\n")
            f.write(f"Appearance Settings:\n")
            f.write(f"Line Width: {line_width}\n")
            f.write(f"Horizontal Stride: {h_stride}\n")
            f.write(f"Vertical Stride: {v_stride}\n")
            f.write(f"Time Compression Enabled: {compression_enabled}\n")
            f.write(f"Compression Amount: {compression_amount}\n")
            f.write(f"Line Color: RGB({line_color.red()}, {line_color.green()}, {line_color.blue()})\n")
            f.write(f"Background Color: RGB({bg_color.red()}, {bg_color.green()}, {bg_color.blue()})\n")
        print(f"Visualization settings saved to: {settings_path}")

        # Quit the application
        QtWidgets.QApplication.instance().quit()

    def data_extender(self, data):
        """
        This helps in adding additional data at the very beginning to ensure that the first frame isn't blank
        :param data: 1D or 2D numpy array
        :return: extended data
        """
        if len(data.shape) == 2:
            arr = np.clip(data[:, 0], 0, 0)
            append_this = np.tile(arr, (self.window_length - 1, 1)).T
            return np.hstack((append_this, data))
        elif len(data.shape) == 1:
            return np.hstack((np.ones(self.window_length - 1) * data[0], data))
        else:
            raise ValueError("Can't handle data with more than 2 dimensions")


    def data_sample_gen(self, data, offset, op_data_as_cood=True, along_y=False, ip_data_1d=False):
        """
        This is a generator which gives us an updated window of the original data which needs to be plotted
        :param data: data from which the sample needs to be created
        :param offset: the steps by which the new window needs to be offset
        :param op_data_as_cood: if True this provides data with coordinates. if False it provides just the data-points
        :param along_y: if True this provides data with coordinates such that the GLLinePlotItem plots line along
        the y-axis
        :param ip_data_1d: True if the input data is 1D
        :return: yields the relevant data sample
        """
        starter_ind = 0
        while True:
            if ip_data_1d:
                sample = data[starter_ind: starter_ind + self.window_length]
                yield sample
                starter_ind = (starter_ind + offset) % len(data)
            else:
                sample = data.T[starter_ind: starter_ind + self.window_length].T
                if op_data_as_cood:
                    if along_y:
                        xx, yy = np.meshgrid(np.arange(sample.shape[0]), np.arange(sample.shape[1]))
                        sample_cood = np.vstack((xx.ravel(), yy.ravel(), sample.ravel(order="F"))).T
                        yield sample_cood
                    else:
                        xx, yy = np.meshgrid(np.arange(sample.shape[1]), np.arange(sample.shape[0]))
                        sample_cood = np.vstack((yy.ravel(), xx.ravel(), sample.ravel())).T
                        yield sample_cood
                else:
                    yield sample
                starter_ind = (starter_ind + offset) % data.shape[1]

    def get_tempogram(self):
        """
        This scales the original tempogram and then multiplies it with a multiplier
        :return: Manipulated Tempogram
        """
        tempogram = librosa.feature.tempogram(onset_envelope=self.oenv, sr=self.sample_rate,
                                              hop_length=self.hop_length, norm=np.inf)
        tempogram_scaled = (tempogram - np.min(tempogram)) / \
                           (np.max(tempogram) - np.min(tempogram))
        return tempogram_scaled * self.tempo_mult

    def get_spectrogram(self):
        """
        The original spectrogram is first mean filtered. Then it's dimensions are edited so as to match that of the
        tempogram. This is done using scaling and 2d interpolation. It is finally multiplied by a multiplier
        :return: Manipulated Spectrogram
        """
        spectrogram = librosa.amplitude_to_db(self.stft, amin=0.1, ref=np.max)  # Spectrogram
        spectrogram = librosa.decompose.nn_filter(spectrogram,
                                                  aggregate=np.mean)  # Spectrogram Mean Filtered
        freqs = librosa.fft_frequencies(sr=self.sample_rate)
        sampling_freqs = np.logspace(start=9 / 6,  # Originally "0"
                                     stop=np.log(freqs[-1]) / np.log(10),
                                     num=self.tempo_final.shape[0],  # Originally "0" (less wide)
                                     endpoint=True)
        sampling_freqs_indices = (sampling_freqs - np.min(sampling_freqs)) / \
                                 (np.max(sampling_freqs) - np.min(sampling_freqs)) * spectrogram.shape[0]
        freq_indices = np.arange(0, spectrogram.shape[0])
        time_indices = np.arange(0, spectrogram.shape[1])

        # Using RectBivariateSpline instead of interp2d
        f = interpolate.RectBivariateSpline(freq_indices, time_indices, spectrogram)
        spectrogram = np.array([f(idx, time_indices)[0] for idx in sampling_freqs_indices])

        spectrogram = (spectrogram - np.min(spectrogram)) / \
                      (np.max(spectrogram) - np.min(spectrogram))
        return spectrogram * self.spectr_mult

    def get_camera_x_position(self):
        """
        This gives us the x-coordinates of the camera position which follows the spectral centroid. The spectral
        centroid is smoothened (LOWESS) to avoid quick transitions which might be strenuous for the viewer
        :return: x-coordinates of the camera
        """
        cent_scaled = (self.spectral_centroid[0] - np.min(self.spectral_centroid[0])) / \
                      (np.max(self.spectral_centroid[0]) - np.min(self.spectral_centroid[0])) * \
                      self.tempo_final.shape[0]
        lowess = sm.nonparametric.lowess(cent_scaled,
                                         np.arange(0, len(cent_scaled)),
                                         frac=0.1, return_sorted=False)
        camera_x = (lowess - np.min(lowess)) / \
                   (np.max(lowess) - np.min(lowess)) * \
                   self.tempo_final.shape[0]
        return camera_x

    def get_chroma_tracer_z(self):
        """
        This provides the z coordinates of the chroma tracer. The chroma tracer flows over the tempogram with an offset
        :return: z coordinates of the chroma tracer
        """
        chroma_tracer_z = np.array([self.tempo_final[min(int(round(j)), self.tempo_final.shape[0] - 1), i]
                                    for i, j in enumerate(self.camera_x)])
        chroma_tracer_z_lowess = sm.nonparametric.lowess(chroma_tracer_z,
                                                         np.arange(0, len(chroma_tracer_z)),
                                                         frac=0.01, return_sorted=False)
        chroma_tracer_z = (chroma_tracer_z_lowess - np.min(chroma_tracer_z_lowess)) / \
                          (np.max(chroma_tracer_z_lowess) - np.min(chroma_tracer_z_lowess)) * self.tempo_mult
        return chroma_tracer_z + self.chroma_tracer_offset_height

    @staticmethod
    def tracker(frames):
        """
        Generator for tracking the frame number
        :param frames: No of frames before the codes stops
        :return: yields the current frame number
        """
        starter = 0
        while starter < frames:
            starter += 1
            yield starter
        sys.exit("Image generation complete!")

    @staticmethod
    def start():
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtWidgets.QApplication.instance().exec_()

    @staticmethod
    def points_in_circum(x, y, z, r, n=100):
        """
        Give center coordinates. Function will return pts corresponding to a circle in the XZ-Plane
        """
        return np.array([(math.cos(2 * pi / n * i) * r + x, y, math.sin(2 * pi / n * i) * r + z)
                         for i in range(0, n + 1)])

if __name__ == "__main__":
    try:
        # Create QApplication instance
        print("Starting application...")
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        print("Opening file selection dialog...")
        # Create temporary visualizer just to use the dialog
        temp_viz = Visualizer()
        audio_path, output_path = temp_viz.show_startup_dialog()

        if audio_path and output_path:
            print(f"Processing audio file: {audio_path}")
            print(f"Output will be saved to: {output_path}")

            # Create the visualizer with the selected audio file
            visualizer = Visualizer(audio_path)
            visualizer.capture_full_spectrogram(output_path)
        else:
            print("Operation cancelled by user.")
            sys.exit()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)