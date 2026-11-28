"""Ventana principal de ZenTralAudio con exactamente 2 pestañas y reproductor inferior permanente."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.app_controller import AppController
from app.environment_tab import EnvironmentTab
from app.preview_player import PreviewPlayer
from app.restoration_tab import RestorationTab
from app.theme import apply_zentral_theme
from core.configuration import ConfigManager
from core.job_manager import JobManager
from core.paths import APP_NAME, APP_VERSION, AppPaths


class MainWindow(QMainWindow):
    """Ventana principal profesional de ZenTralAudio."""

    def __init__(
        self,
        *,
        paths: AppPaths,
        config_manager: ConfigManager,
        job_manager: JobManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.paths = paths
        self.config_manager = config_manager
        self.job_manager = job_manager

        app = QApplication.instance()
        if isinstance(app, QApplication):
            apply_zentral_theme(app, paths)

        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION} — Restauración IA y Ambientes Acústicos")
        self.resize(1240, 820)
        self.setMinimumSize(980, 680)
        # Inicia maximizada en Windows y macOS; el usuario puede restaurarla después.
        self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)

        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("HeaderBar")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 10, 16, 10)
        header_layout.setSpacing(10)

        brand_main = QLabel("ZenTral")
        brand_main.setObjectName("BrandTitle")
        brand_accent = QLabel("Audio")
        brand_accent.setObjectName("BrandAccent")

        self._device_badge = QLabel(self._detect_compute_badge())
        self._device_badge.setObjectName("StatusBadgeInfo")

        self._models_btn = QPushButton("Modelos IA")
        self._resources_btn = QPushButton("Recursos")
        self._settings_btn = QPushButton("Ajustes")
        self._export_btn = QPushButton("Exportar...")
        self._export_btn.setObjectName("PrimaryButton")

        header_layout.addWidget(brand_main)
        header_layout.addWidget(brand_accent)
        header_layout.addSpacing(12)
        header_layout.addWidget(self._device_badge)
        header_layout.addStretch(1)
        header_layout.addWidget(self._models_btn)
        header_layout.addWidget(self._resources_btn)
        header_layout.addWidget(self._settings_btn)
        header_layout.addWidget(self._export_btn)
        root_layout.addWidget(header)

        # Exactamente DOS pestañas principales
        self.tabs = QTabWidget()
        self.tabs.setObjectName("MainTabs")
        self.restoration_tab = RestorationTab()
        self.environment_tab = EnvironmentTab()
        self.tabs.addTab(self.restoration_tab, "RESTAURACIÓN IA")
        self.tabs.addTab(self.environment_tab, "AMBIENTES ACÚSTICOS")
        root_layout.addWidget(self.tabs, stretch=1)

        # Reproductor permanente inferior visible en ambas pestañas
        self.preview_player = PreviewPlayer()
        root_layout.addWidget(self.preview_player)

        self.setCentralWidget(central)

        self.controller = AppController(
            paths=paths,
            config_manager=config_manager,
            job_manager=job_manager,
            restoration_tab=self.restoration_tab,
            environment_tab=self.environment_tab,
            preview_player=self.preview_player,
            parent=self,
        )

        self._models_btn.clicked.connect(self.controller.open_model_manager)
        self._resources_btn.clicked.connect(self.controller.open_resource_manager)
        self._settings_btn.clicked.connect(self.controller.open_settings)
        self._export_btn.clicked.connect(self.controller.open_export_dialog)

    @staticmethod
    def _detect_compute_badge() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return f"GPU CUDA: {torch.cuda.get_device_name(0)}"
        except Exception:
            pass
        return "Modo CPU"

    def closeEvent(self, event: QCloseEvent) -> None:
        self.controller.shutdown()
        super().closeEvent(event)
