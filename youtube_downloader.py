import os
import sys
import yt_dlp
import re
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QProgressBar, QFileDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QFont

class DownloadThread(QThread):
    progress = pyqtSignal(float)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, url, output_path):
        super().__init__()
        self.url = url
        self.output_path = output_path

    def run(self):
        try:
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': os.path.join(self.output_path, '%(title)s.%(ext)s'),
                'progress_hooks': [self.progress_hook],
                'continuedl': True,
                'outtmpl_na_placeholder': '',
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.url])

            self.finished.emit("Download complete!")
        except Exception as e:
            self.error.emit(str(e))

    def progress_hook(self, d):
        if d['status'] == 'downloading':
            try:
                percent = d['_percent_str']
                percent = percent.replace('%','').strip()
                self.progress.emit(float(percent))
            except:
                pass  # Ignore progress errors
        if d['status'] == 'finished':
            self.progress.emit(100)

class YouTubeDownloaderGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('YouTube Downloader')
        self.setGeometry(300, 300, 500, 250)
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial;
            }
            QLineEdit, QPushButton {
                padding: 8px;
                font-size: 14px;
                border-radius: 4px;
            }
            QLineEdit {
                border: 1px solid #ccc;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QLabel {
                font-size: 14px;
            }
        """)

        layout = QVBoxLayout()

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter YouTube URL")
        layout.addWidget(self.url_input)

        self.download_btn = QPushButton('Download MP4', self)
        self.download_btn.clicked.connect(self.start_download)
        layout.addWidget(self.download_btn)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel('', self)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def start_download(self):
        url = self.url_input.text()
        if not url:
            self.status_label.setText("Please enter a valid YouTube URL")
            return

        cleaned_url = self.clean_url(url)
        if not cleaned_url:
            self.status_label.setText("Invalid YouTube URL")
            return

        download_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
        self.download_thread = DownloadThread(cleaned_url, download_dir)
        self.download_thread.progress.connect(self.update_progress)
        self.download_thread.finished.connect(self.download_finished)
        self.download_thread.error.connect(self.download_error)
        self.download_thread.start()

        self.download_btn.setEnabled(False)
        self.status_label.setText("Downloading MP4...")

    def clean_url(self, url):
        match = re.search(r'(?:youtu\.be/|youtube\.com/(?:embed/|v/|watch\?v=|watch\?.+&v=))([^?&]+)', url)
        if match:
            video_id = match.group(1)
            return f'https://www.youtube.com/watch?v={video_id}'
        else:
            return None

    def update_progress(self, percentage):
        self.progress_bar.setValue(int(percentage))

    def download_finished(self, message):
        self.status_label.setText(message)
        self.download_btn.setEnabled(True)
        self.progress_bar.setValue(100)

    def download_error(self, error_message):
        self.status_label.setText(f"Error: {error_message}")
        self.download_btn.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = YouTubeDownloaderGUI()
    ex.show()
    sys.exit(app.exec_())