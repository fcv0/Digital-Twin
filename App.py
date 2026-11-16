import sys, os, threading, time
from Priv import _Vars
from Utils import utils, RingBuffer, stopFlusher, load_history_memmap, _BIN_DTYPE
import numpy as np


def _snap(seq):
    """Fast snapshot for plot code: numpy view for RingBuffer, np array
    otherwise. RingBuffer.snapshot() is a single contiguous memcpy — avoids
    the per-element Python loop that `list(rb)` would trigger."""
    if isinstance(seq, RingBuffer):
        return seq.snapshot()
    return np.asarray(seq)
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QFrame, QScrollArea, QSizePolicy,
    QTabWidget, QMenuBar, QFileDialog, QInputDialog,
)
from PyQt6.QtCore import QTimer, Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QPen, QAction
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Keep this in sync with FIRMWARE_VERSION in main.c
VERSION = "v1.02.06"

# ---- Palette (professional dark, single teal accent) ----------------------
BG        = "#1b1e23"
CARD      = "#242830"
CARD_EDGE = "#323843"
TEXT      = "#e6e9ef"    
MUTED     = "#8a93a3"
ACCENT    = "#1bb89a"
INPUT_BG  = "#2c313a"
OK        = "#3fb950"
WARN      = "#e0a13a"
DANGER    = "#e05a5a"
TARGET    = "#ff8c69"   # dashed setpoint lines on the plots
TEMP_TWIN = "#f76c5e"   # coral-red for avg_temp overlayed onto the OD history plot
# Distinct line colours for overlaying multiple reactors on the multiplexer tab
MULTI_COLORS = ["#1bb89a", "#e0a13a", "#9b8cff", "#ff8c69", "#3fb950", "#e05a5a",
                "#4db8ff", "#d46ad0", "#c9d14a", "#7ad1c0", "#ff6b9d", "#8a93a3"]

PILL_ACTIVE   = (f"background:{ACCENT}; color:#10141a; border:1px solid {ACCENT};"
                 " border-radius:8px; padding:8px 10px; font-weight:600;")
PILL_IDLE     = (f"background:{INPUT_BG}; color:{TEXT}; border:1px solid {CARD_EDGE};"
                 " border-radius:8px; padding:8px 10px;")
PILL_DISABLED = (f"background:#23272e; color:#5b626e; border:1px solid #2a2f37;"
                 " border-radius:8px; padding:8px 10px;")

APP_QSS = f"""
    QWidget {{ background:{BG}; color:{TEXT};
               font-family:'Segoe UI','SF Pro Display',Arial; font-size:13px; }}
    QFrame#card {{ background:{CARD}; border:1px solid {CARD_EDGE}; border-radius:12px; }}
    QFrame#header {{ background:{CARD}; border:1px solid {CARD_EDGE}; border-radius:12px; }}
    QLabel#cardTitle {{ color:{MUTED}; font-size:11px; font-weight:600; letter-spacing:1.5px; }}
    QLabel#bigValue  {{ color:{TEXT}; font-size:30px; font-weight:700; }}
    QLabel#subValue  {{ color:{MUTED}; font-size:12px; }}
    QLabel#headerTitle {{ font-size:18px; font-weight:700; }}
    QLabel#headerVersion {{ color:{MUTED}; font-size:11px; }}
    QLabel#runClock {{ color:{ACCENT}; font-size:17px; font-weight:700;
                       font-family:'Consolas','SF Mono',monospace; letter-spacing:1px;
                       background:{BG}; border:1px solid {CARD_EDGE}; border-radius:8px;
                       padding:3px 10px; }}
    QLabel#fieldLabel {{ color:{MUTED}; }}
    QScrollArea {{ border:none; background:{BG}; }}
    QScrollArea > QWidget > QWidget {{ background:{BG}; }}
    QLineEdit {{ background:{INPUT_BG}; border:1px solid {CARD_EDGE}; border-radius:8px;
                 padding:6px 8px; color:{TEXT}; }}
    QLineEdit:focus {{ border:1px solid {ACCENT}; }}
    QLineEdit:disabled {{ color:#5b626e; background:#23272e; border:1px solid #2a2f37; }}
    QComboBox {{ background:{INPUT_BG}; border:1px solid {CARD_EDGE}; border-radius:8px; padding:6px 10px; }}
    QComboBox:hover {{ border:1px solid {ACCENT}; }}
    QComboBox QAbstractItemView {{ background:{CARD}; selection-background-color:{ACCENT};
                                   selection-color:#10141a; border:1px solid {CARD_EDGE}; }}
    QPushButton {{ background:{INPUT_BG}; border:1px solid {CARD_EDGE}; border-radius:8px;
                   padding:8px 12px; color:{TEXT}; }}
    QPushButton:hover {{ background:#333a45; border:1px solid {ACCENT}; }}
    QPushButton:pressed {{ background:{ACCENT}; color:#10141a; }}
    QPushButton:disabled {{ background:#23272e; color:#5b626e; border:1px solid #2a2f37; }}
    QScrollBar:vertical {{ background:{BG}; width:10px; margin:0; }}
    QScrollBar::handle:vertical {{ background:{CARD_EDGE}; border-radius:5px; min-height:30px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height:0; }}
    QTabWidget::pane {{ border:1px solid {CARD_EDGE}; border-radius:12px; background:{BG}; top:-1px; }}
    QTabBar::tab {{ background:{INPUT_BG}; color:{MUTED}; padding:8px 18px; border:1px solid {CARD_EDGE};
                    border-bottom:none; border-top-left-radius:8px; border-top-right-radius:8px; margin-right:4px; }}
    QTabBar::tab:selected {{ background:{CARD}; color:{TEXT}; font-weight:600; }}
    QTableWidget {{ background:{CARD}; gridline-color:{CARD_EDGE}; border:1px solid {CARD_EDGE};
                    border-radius:8px; color:{TEXT}; }}
    QHeaderView::section {{ background:{INPUT_BG}; color:{MUTED}; border:none; padding:6px; font-weight:600; }}
    QTableWidget::item:selected {{ background:{ACCENT}; color:#10141a; }}
"""


class ToggleSwitch(QWidget):
    """Small pill switch with a sliding knob: green = on, red = off."""
    toggled = pyqtSignal(bool)

    def __init__(self, checked=False, parent=None):
        super().__init__(parent)
        self._checked = bool(checked)
        self.setFixedSize(54, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def isChecked(self):
        return self._checked

    def setChecked(self, value):
        value = bool(value)
        changed = value != self._checked
        self._checked = value
        self.update()
        if changed:
            self.toggled.emit(value)

    def mousePressEvent(self, event):
        self.setChecked(not self._checked)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        r = QRectF(0, 0, self.width() - 1, self.height() - 1)
        rad = r.height() / 2
        p.setBrush(QColor(OK if self._checked else DANGER))
        p.drawRoundedRect(r, rad, rad)
        d = r.height() - 6
        x = (r.width() - d - 3) if self._checked else 3
        p.setBrush(QColor("#ffffff"))
        p.drawEllipse(QRectF(x, 3, d, d))
        p.end()


class Spinner(QWidget):
    """Small indeterminate spinner (a rotating arc) for 'searching…' states.
    Cheap: one QTimer at ~16 FPS repainting a single arc; stopped when hidden."""
    def __init__(self, size=15, color=ACCENT, parent=None):
        super().__init__(parent)
        self._angle = 0
        self._color = color
        self.setFixedSize(size, size)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(60)

    def start(self):
        if not self._timer.isActive():
            self._timer.start()

    def stop(self):
        self._timer.stop()

    def _tick(self):
        self._angle = (self._angle - 30) % 360   # clockwise
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(self._color), 2.2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        m = 2
        rect = QRectF(m, m, self.width() - 2 * m, self.height() - 2 * m)
        p.drawArc(rect, self._angle * 16, 300 * 16)   # 300° arc → visible gap
        p.end()


class App(QWidget):
    def __init__(self):
        super().__init__()
        _Vars.initState = True
        self.initUI()

    # ---- small UI builders -------------------------------------------------
    def _card(self, title):
        card = QFrame(); card.setObjectName("card")
        v = QVBoxLayout(card); v.setContentsMargins(16, 14, 16, 14); v.setSpacing(10)
        if title:
            t = QLabel(title.upper()); t.setObjectName("cardTitle")
            v.addWidget(t)
        return card, v

    def _value_tile(self, title):
        card, v = self._card(title)
        big = QLabel("--");        big.setObjectName("bigValue")
        sub = QLabel("Target --"); sub.setObjectName("subValue")
        v.addWidget(big); v.addWidget(sub)
        return card, big, sub

    def _row(self, label, widget):
        row = QHBoxLayout()
        lab = QLabel(label); lab.setObjectName("fieldLabel")
        widget.setFixedWidth(120)
        row.addWidget(lab); row.addStretch(); row.addWidget(widget)
        return row

    def _pills(self, *buttons):
        row = QHBoxLayout(); row.setSpacing(8)
        for b in buttons:
            b.setStyleSheet(PILL_IDLE)
            row.addWidget(b)
        return row

    def _set_pills(self, buttons, active):
        for i, b in enumerate(buttons):
            if not b.isEnabled():
                b.setStyleSheet(PILL_DISABLED)
            else:
                b.setStyleSheet(PILL_ACTIVE if i == active else PILL_IDLE)

    def _flash(self, btn, ok):
        col = OK if ok else DANGER
        btn.setStyleSheet(f"background:{INPUT_BG}; border:1px solid {col}; color:{col};"
                          " border-radius:8px; padding:8px 12px; font-weight:600;")
        QTimer.singleShot(3000, lambda: btn.setStyleSheet(""))

    def _app_icon(self):
        pm = QPixmap(64, 64); pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(ACCENT));    p.drawRoundedRect(QRectF(6, 6, 52, 52), 16, 16)
        p.setBrush(QColor("#10141a")); p.drawEllipse(QRectF(19, 24, 26, 26))
        p.setBrush(QColor(ACCENT));    p.drawEllipse(QRectF(26, 14, 12, 12))
        p.end()
        return QIcon(pm)

    # ---- main build --------------------------------------------------------
    def initUI(self):
        self.setWindowTitle(f"Bioreactor Multiplexer {VERSION}")
        self.setWindowIcon(self._app_icon())
        self.resize(1500, 900)
        self.setStyleSheet(APP_QSS)

        # --- widgets (names preserved for existing signal wiring) ---
        self.portSelector = QComboBox(self)
        for p in _Vars.ports:
            self.portSelector.addItem(str(p))
        self.portSelector.currentIndexChanged.connect(self.changePort)
        # May start with zero devices — adopt the first one live once it connects (Upload IP + Rescan).
        self.currentPort = int(_Vars.ports[0]) if _Vars.ports else None

        self.saveData     = QPushButton("Save Data")
        self.resetData    = QPushButton("Reset Data")
        self.applyAll     = QPushButton("Apply to All Reactors")
        self.rescanDevices = QPushButton("Rescan Devices")
        self.uploadIP      = QPushButton("Upload IP")
        self.uploadIP.setToolTip("Push this host's IP to any plugged-in reactors over USB serial,\n"
                                 "then click Rescan Devices to capture them.")
        self._rescan_done = False
        self._upload_done = False
        self.targetTempBtn = QPushButton("Target Temp"); self.targetTempBtn.setCheckable(True)
        self.targetODBtn   = QPushButton("Target OD");   self.targetODBtn.setCheckable(True)
        self.showTargetTemp = False
        self.showTargetOD   = False
        self.pinchState = False
        self.cycleMotor   = QLineEdit(self)
        self.fanMotor     = QLineEdit(self)
        self.ch0Motor     = QLineEdit(self)     # aux CH0, always-on (packet byte 53)
        self.wasteMotor   = QLineEdit(self)
        self.freshMotor   = QLineEdit(self)
        self.tempTarg     = QLineEdit(self)
        self.tempHystVals = QLineEdit(self)
        self.tempbbDelay  = QLineEdit(self)
        # Temperature offset moved out of the Control Panel into the Config tab
        # (edited there + persisted to config.ini). No live widget on this panel.
        self.tempbangbang = QPushButton("Bang-Bang")
        self.tempHyst     = QPushButton("Hysteresis")
        self.tempPI       = QPushButton("PI")
        self.targOD       = QLineEdit(self)
        self.ODbbDelay    = QLineEdit(self)
        self.ODHystVal    = QLineEdit(self)
        # Bang-bang delay fields are kept (they still push bb_delay into the packet
        # via textChanged/updateValues) but the bang-bang mode is hidden in the
        # consumer build — hide the widgets so they don't float at the window's
        # top-left corner now that they're no longer in a card layout.
        self.tempbbDelay.hide()
        self.ODbbDelay.hide()
        self.ODbangbang   = QPushButton("Bang-Bang")
        self.ODHyst       = QPushButton("Hysteresis")
        self.ODClamp      = QPushButton("Clamp")
        self.pinch = QPushButton(f"Pinch State: {self.pinchState}")
        # Default: manual overrides OFF (auto pumps) when turbidostat is configured.
        self.overrideToggle = ToggleSwitch(checked=not _Vars.config.get("Turbidostat", False))

        # live value tiles
        temp_tile, self.currentTemp,  self.currentTempTarg = self._value_tile("Temperature")
        od_tile,   self.currentOD,    self.currentODTarg   = self._value_tile("Optical Density")

        # =================== HEADER ===================
        header = QFrame(); header.setObjectName("header")
        h = QHBoxLayout(header); h.setContentsMargins(18, 12, 18, 12); h.setSpacing(14)
        title = QLabel("BIOREACTOR  ·  MULTIPLEXER"); title.setObjectName("headerTitle")
        ver = QLabel(VERSION); ver.setObjectName("headerVersion")
        self.runClock = QLabel("RUN  00:00:00"); self.runClock.setObjectName("runClock")
        self.runClock.setToolTip("Run length (software timer) of the selected reactor — resets on device reset")
        h.addWidget(title); h.addWidget(ver); h.addSpacing(16); h.addWidget(self.runClock)
        # Strain of the selected reactor — set via Run > Set Strain.
        self.strainLabel = QLabel("STRAIN  —")
        self.strainLabel.setStyleSheet(f"color:{MUTED}; font-size:11px; font-weight:600;"
                                       " letter-spacing:1px;")
        self.strainLabel.setToolTip("Strain / culture label — Run > Set Strain")
        h.addSpacing(10); h.addWidget(self.strainLabel)
        # "Searching for reactors…" indicator — visible until at least one device
        # connects; the background auto-scan (see _auto_scan_worker) keeps looking.
        self.searchSpinner = Spinner(15)
        self.searchText = QLabel("Searching for reactors…")
        self.searchText.setStyleSheet(f"color:{MUTED}; font-size:12px;")
        self.searchBox = QWidget()
        sb = QHBoxLayout(self.searchBox); sb.setContentsMargins(12, 0, 0, 0); sb.setSpacing(8)
        sb.addWidget(self.searchSpinner); sb.addWidget(self.searchText)
        h.addWidget(self.searchBox)
        h.addStretch()
        self.portDots = {}
        self.dotsRow = QHBoxLayout(); self.dotsRow.setSpacing(12)
        for p in _Vars.ports:
            dot = QLabel(f"●  {p}"); dot.setStyleSheet(f"color:{MUTED}; font-size:12px;")
            self.portDots[p] = dot
            self.dotsRow.addWidget(dot)
        h.addLayout(self.dotsRow)
        self.lossLabel = QLabel("loss --"); self.lossLabel.setStyleSheet(f"color:{MUTED}; font-size:12px;")
        h.addWidget(self.lossLabel)
        h.addSpacing(8)
        h.addWidget(QLabel("Port"))
        self.portSelector.setFixedWidth(110)
        h.addWidget(self.portSelector)

        # =================== CONTROL COLUMN ===================
        # live tiles row
        tiles = QHBoxLayout(); tiles.setSpacing(12)
        tiles.addWidget(temp_tile); tiles.addWidget(od_tile)

        # temperature card — bang-bang mode + its delay field hidden in the
        # consumer build (users pick between hysteresis and PI). The widgets
        # themselves stay allocated so the pill-index code (self._set_pills)
        # can keep the same [bangbang, hyst, PI] index-order → firmware byte
        # mapping without a shim.
        temp_card, tv = self._card("Temperature Control")
        tv.addLayout(self._row("Target (°C)", self.tempTarg))
        tv.addLayout(self._row("Hysteresis band (+/- °C)", self.tempHystVals))
        tv.addLayout(self._pills(self.tempHyst, self.tempPI))

        # OD card — same treatment: bang-bang hidden; user chooses Hysteresis
        # or Clamp. The bang-bang widget instance stays alive for _set_pills.
        od_card, ov = self._card("Optical Density Control")
        ov.addLayout(self._row("Target OD", self.targOD))
        ov.addLayout(self._row("Hysteresis band (- OD)", self.ODHystVal))
        ov.addLayout(self._pills(self.ODHyst, self.ODClamp))

        # motors card
        motor_card, mv = self._card("Motors")
        mv.addLayout(self._row("Circulation motor (%)", self.cycleMotor))
        mv.addLayout(self._row("Aeration motor (%)",    self.fanMotor))
        mv.addLayout(self._row("Aux motor CH0 (0-255)", self.ch0Motor))
        # Manual overrides + pump 1/2 fields only make sense when turbidostat is
        # configured (otherwise there are no auto pumps to override). Wrapped in
        # a single QWidget so they can be hidden as a unit.
        self.overridesSection = QWidget()
        ov_l = QVBoxLayout(self.overridesSection); ov_l.setContentsMargins(0, 0, 0, 0); ov_l.setSpacing(10)
        orow = QHBoxLayout()
        olab = QLabel("Manual overrides"); olab.setObjectName("fieldLabel")
        orow.addWidget(olab); orow.addStretch(); orow.addWidget(self.overrideToggle)
        ov_l.addLayout(orow)
        ov_l.addLayout(self._row("Pump 1 | (0 - 255)", self.wasteMotor))
        ov_l.addLayout(self._row("Pump 2 | (0 - 255)", self.freshMotor))
        mv.addWidget(self.overridesSection)
        self.overridesSection.setVisible(bool(_Vars.config.get("Turbidostat", False)))

        # actions card
        act_card, av = self._card("Actions")
        srow = QHBoxLayout(); srow.setSpacing(8)
        srow.addWidget(self.saveData); srow.addWidget(self.resetData)
        av.addLayout(srow); av.addWidget(self.applyAll)
        # Target Temp / Target OD toggles live above the plots now — style them
        # here so they carry the pill look wherever they're mounted.
        self.targetTempBtn.setStyleSheet(PILL_IDLE)
        self.targetODBtn.setStyleSheet(PILL_IDLE)

        connect_card, av = self._card("Connection Finder")
        drow = QHBoxLayout(); drow.setSpacing(8)
        drow.addWidget(self.uploadIP); drow.addWidget(self.rescanDevices)
        av.addLayout(drow)

        pinch_card, av = self._card("Pinch Valve")
        drow= QHBoxLayout(); drow.setSpacing(8)
        drow.addWidget(self.pinch)
        av.addLayout(drow)

        control_col = QVBoxLayout(); control_col.setSpacing(14); control_col.setContentsMargins(0, 0, 0, 0)
        control_col.addLayout(tiles)
        control_col.addWidget(temp_card)
        control_col.addWidget(od_card)
        control_col.addWidget(motor_card)
        control_col.addWidget(act_card)
        control_col.addWidget(connect_card)
        control_col.addWidget(pinch_card)
        control_col.addStretch()
        control_holder = QWidget(); control_holder.setLayout(control_col)
        control_holder.setFixedWidth(430)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setWidget(control_holder)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(450)

        # =================== PLOTS COLUMN ===================
        self.graphOD    = PlotCanvas(width=8, height=4.0, title="Optical Density", xL="Time (s)", yL="OD")
        self.graphTemp  = PlotCanvas(width=8, height=4.0, title="Temperature", xL="Time (s)", yL="°C")

        def plot_card(canvas):
            card, v = self._card("")
            v.setContentsMargins(10, 10, 10, 10)
            v.addWidget(canvas)
            return card

        plots_col = QVBoxLayout(); plots_col.setSpacing(14)
        # Target-line toggles sit directly above their respective plots so the
        # operator can flip them without hunting through the Actions card.
        targets_row = QHBoxLayout(); targets_row.setSpacing(8)
        targets_row.addWidget(self.targetODBtn)     # order matches the plots below
        targets_row.addWidget(self.targetTempBtn)
        plots_col.addLayout(targets_row)
        plots_col.addWidget(plot_card(self.graphOD), 1)
        plots_col.addWidget(plot_card(self.graphTemp), 1)
        plots_holder = QWidget(); plots_holder.setLayout(plots_col)

        body = QHBoxLayout(); body.setSpacing(16)
        body.addWidget(scroll)
        body.addWidget(plots_holder, 1)
        control_tab = QWidget(); control_tab.setLayout(body)
        self.controlTab = control_tab
        self.multiTab   = self._build_multiplex_tab()
        self.debugTab   = self._build_debug_tab()
        self.historyTab = self._build_history_tab()

        self.tabs = QTabWidget()
        self.tabs.addTab(control_tab, "Control Panel")
        self.tabs.addTab(self.multiTab, "Multiplexer")
        self.tabs.addTab(self.debugTab, "Debug")
        self.tabs.addTab(self.historyTab, "History")
        self.tabs.addTab(self._build_config_tab(), "Config")
        # Refresh the history plot whenever the user lands on that tab. Cheap —
        # memmap load is O(1); the downsample is the real cost and is bounded
        # by target pixel count, not run length.
        self.tabs.currentChanged.connect(self._on_tab_changed)

        root = QVBoxLayout(self); root.setContentsMargins(16, 16, 16, 16); root.setSpacing(14)
        root.setMenuBar(self._build_menu_bar())
        root.addWidget(header)
        root.addWidget(self.tabs)
        self.setLayout(root)

        # =================== SIGNALS (unchanged wiring) ===================
        self.resetData.pressed.connect(lambda: self.resetDataClick(port=self.currentPort))
        self.cycleMotor.textChanged.connect(lambda: utils.changeMotorSpeeds(text=self.cycleMotor.text(), idx=[36, 37], sPosition=[0, 0], port=self.currentPort))
        self.fanMotor.textChanged.connect(lambda: utils.changeMotorSpeeds(text=self.fanMotor.text(), idx=[-1, 38], sPosition=[1, 1], port=self.currentPort))
        self.ch0Motor.textChanged.connect(self._ch0Changed)
        self.wasteMotor.textChanged.connect(lambda: utils.changeMotorSpeeds(text=self.wasteMotor.text(), idx=[39, 40], sPosition=[2, 2], port=self.currentPort))
        self.freshMotor.textChanged.connect(lambda: utils.changeMotorSpeeds(text=self.freshMotor.text(), idx=[41, 42], sPosition=[3, 3], port=self.currentPort))

        self.tempTarg.textChanged.connect(lambda: utils.changeControlValues(text=self.tempTarg.text(), idx=2, sPosition=0, controlSelection="Temperature", port=self.currentPort))
        self.tempHystVals.textChanged.connect(lambda: utils.changeControlValues(text=self.tempHystVals.text(), idx=6, sPosition=1, controlSelection="Temperature", port=self.currentPort))
        self.tempbbDelay.textChanged.connect(lambda: utils.changeControlValues(text=self.tempbbDelay.text(), idx=14, sPosition=2, controlSelection="Temperature", port=self.currentPort))
        self.tempbangbang.pressed.connect(lambda: self.changeTempMode(mode=0, port=self.currentPort))
        self.tempHyst.pressed.connect(lambda: self.changeTempMode(mode=1, port=self.currentPort))
        self.tempPI.pressed.connect(lambda: self.changeTempMode(mode=2, port=self.currentPort))

        self.targOD.textChanged.connect(lambda: utils.changeControlValues(text=self.targOD.text(), idx=20, sPosition=0, controlSelection="OD", port=self.currentPort))
        self.ODbbDelay.textChanged.connect(lambda: utils.changeControlValues(text=self.ODbbDelay.text(), idx=24, sPosition=1, controlSelection="OD", port=self.currentPort))
        self.ODHystVal.textChanged.connect(lambda: utils.changeControlValues(text=self.ODHystVal.text(), idx=28, sPosition=3, controlSelection="OD", port=self.currentPort))
        self.ODbangbang.pressed.connect(lambda: self.changeODState(state=0, port=self.currentPort))
        self.ODHyst.pressed.connect(lambda: self.changeODState(state=1, port=self.currentPort))
        self.ODClamp.pressed.connect(lambda: self.changeODState(state=2, port=self.currentPort))

        self.pinch.pressed.connect(lambda: self.togglePinchState(port=self.currentPort))

        self.saveData.pressed.connect(self.saveDataAllClick)
        self.applyAll.pressed.connect(lambda: self.applyToAllClick(port=self.currentPort))
        # Apply-to-All is only meaningful with >1 reactor — hidden otherwise
        # (kept up-to-date by _syncPorts as new devices come online).
        self.applyAll.setVisible(len(_Vars.ports) > 1)
        self.rescanDevices.pressed.connect(self.rescanDevicesClick)
        self.uploadIP.pressed.connect(self.uploadIPClick)
        self.targetTempBtn.toggled.connect(self.toggleTargetTemp)
        self.targetODBtn.toggled.connect(self.toggleTargetOD)
        self.overrideToggle.toggled.connect(self.setOverridesEnabled)

        turbidostat_on = _Vars.config.get("Turbidostat", False)
        for w in [self.ODbangbang, self.ODHyst, self.ODClamp, self.ODbbDelay, self.ODHystVal, self.targOD]:
            w.setEnabled(turbidostat_on)

        if self.currentPort is not None:
            # populate fields after signals so textChanged pushes config into the packet
            self.updateValues(port=self.currentPort)
            # initial pill highlight from the current packet modes
            msg = _Vars.activeDevices[self.currentPort]["OutgoingMessage"]
            self._set_pills([self.tempbangbang, self.tempHyst, self.tempPI], int(msg[18].item()))
            self._set_pills([self.ODbangbang, self.ODHyst, self.ODClamp], int(msg[32].item()))
            self._syncPinchButton(self.currentPort)
        self.setOverridesEnabled(self.overrideToggle.isChecked())
        self._syncModeMenu()
        self._refreshStrainLabel()

        # Background device discovery. The GUI now opens immediately (before any
        # reactor connects); this daemon keeps scanning so devices plugged in
        # after launch come online live, and drives the header "searching…"
        # spinner until the first one appears.
        self._msTabs = None
        if _Vars.config.get("MicroscopeEnabled", False):
            self.setMicroscopeEnabled(True)

        self._dbTabs = None
        if _Vars.config.get("DobotEnabled", False):
            self.setDobotEnabled(True)

        self._scan_lock = threading.Lock()
        self._autoscan_on = True
        self._update_search_indicator()
        threading.Thread(target=self._auto_scan_worker, daemon=True).start()

        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(lambda: self.updateGraphs(port=self.currentPort))
        time.sleep(.5)
        self.timer.start()

    def _update_search_indicator(self):
        """Show the header 'searching…' spinner only while no reactors are
        connected; hide it (and stop the animation) once at least one is live."""
        if not hasattr(self, "searchBox"):
            return
        if _Vars.ports:
            self.searchSpinner.stop()
            self.searchBox.setVisible(False)
        else:
            self.searchSpinner.start()
            self.searchBox.setVisible(True)

    def _do_scan(self):
        """Run one device scan, guarded so the background auto-scan and a manual
        Rescan Devices click never bind the same free port at once."""
        comm = getattr(_Vars, "comm", None)
        if comm is None or not hasattr(comm, "rescan_devices"):
            return []
        if not self._scan_lock.acquire(blocking=False):
            return []
        try:
            return comm.rescan_devices()
        finally:
            self._scan_lock.release()

    def _auto_scan_worker(self):
        """Poll for reactors in the background for the life of the window.
        Aggressive (~3 s) while none are connected so a freshly-plugged device
        appears quickly; relaxed (~15 s) once at least one is live, to keep
        picking up additional multiplexer reactors without churning sockets."""
        # Wait for the initial connect scan to finish so we don't double-bind ports.
        while not getattr(_Vars, "notConnected", False):
            time.sleep(0.1)
        if _Vars.config.get("Offline", False):
            return                        # simulator provides devices; no scanning
        while getattr(self, "_autoscan_on", False):
            try:
                self._do_scan()
            except Exception as e:
                print(f"[WARN] auto-scan: {e}")
            time.sleep(3.0 if not _Vars.ports else 15.0)

    def saveDataAllClick(self):
        """Flush every active reactor's run file — each port's streaming CSV
        gets any pending in-DRAM tail appended, and its XML sidecar rewritten
        with current metadata. Same semantics as the periodic background flush,
        just triggered on demand. Green flash if every port succeeded."""
        if not _Vars.ports:
            self._flash(self.saveData, False)
            print("[WARN] Save Data: no active reactors.")
            return
        ok = True
        for port in _Vars.ports:
            try:
                utils.dataSave(port=port)   # ignores *arrays kwarg — pulls from device dict
            except Exception as e:
                ok = False
                print(f"[ERROR] Save Data port {port}: {e}")
        self._flash(self.saveData, ok)

    def resetDataClick(self, port):
        try:
            utils.resetDevice(port=port)
            self._flash(self.resetData, True)
        except Exception as e:
            self._flash(self.resetData, False)
            print("[ERROR] | RESET FAILED...")
            print(e)

    def applyToAllClick(self, port):
        """Copy the current reactor's control settings to every other reactor on the line.
        Green for 3 s on success, red for 3 s on failure."""
        try:
            source = _Vars.activeDevices[port]
            for p in _Vars.ports:
                if p == port:
                    continue
                target = _Vars.activeDevices[p]
                dv_src = source["DeviceVariables"]
                dv_tgt = target["DeviceVariables"]
                target["OutgoingMessage"] = source["OutgoingMessage"].copy()
                dv_tgt["ControlValues"]["Temperature"] = list(dv_src["ControlValues"]["Temperature"])
                dv_tgt["ControlValues"]["OD"]          = list(dv_src["ControlValues"]["OD"])
                dv_tgt["MotorSpeeds"]                  = list(dv_src["MotorSpeeds"])
                dv_tgt["MotorDirections"]              = list(dv_src["MotorDirections"])
                dv_tgt["CH0Speed"]                     = dv_src.get("CH0Speed", 0)
                target["OutgoingMessage"][53]          = dv_src.get("CH0Speed", 0)
                dv_tgt["Turbidostat"]                  = dv_src["Turbidostat"]
                dv_tgt["TurbidostatMode"]              = dv_src["TurbidostatMode"]
                dv_tgt["HeaterMode"]                   = dv_src["HeaterMode"]
                target["Coefficients"]["TemperatureOffset"] = source["Coefficients"]["TemperatureOffset"]
                dv_tgt["bb_target_hit_time"]           = None
                dv_tgt["SendMessage"]                  = True
            self._flash(self.applyAll, True)
            print(f"[INFO] Applied port {port} settings to all reactors.")
        except Exception as e:
            self._flash(self.applyAll, False)
            print(f"[ERROR] | APPLY TO ALL FAILED: {e}")

    def rescanDevicesClick(self):
        """Background-scan for reactors that didn't connect at startup and add them live.
        New ports appear in the selector/header via _syncPorts on the next timer tick."""
        if getattr(_Vars, "comm", None) is None:
            print("[WARN] Rescan: no comms instance available.")
            return
        self.rescanDevices.setText("Scanning…")
        self.rescanDevices.setStyleSheet(
            f"background:{INPUT_BG}; border:1px solid {ACCENT}; color:{ACCENT};"
            " border-radius:8px; padding:8px 12px; font-weight:600;")
        self._rescan_done = False

        def worker():
            try:
                new = self._do_scan()          # shared guard with the auto-scan loop
                self._rescan_n, self._rescan_ok = len(new), True
            except Exception as e:
                self._rescan_n, self._rescan_ok = 0, False
                print(f"[ERROR] Rescan failed: {e}")
            self._rescan_done = True

        threading.Thread(target=worker, daemon=True).start()

    def uploadIPClick(self):
        """Push this host's IP to plugged-in reactors over USB serial (provision_over_serial),
        so a freshly-connected board learns where to send telemetry. Follow with Rescan Devices."""
        comm = getattr(_Vars, "comm", None)
        if comm is None:
            print("[WARN] Upload IP: no comms instance available.")
            return
        self.uploadIP.setText("Uploading…")
        self.uploadIP.setStyleSheet(
            f"background:{INPUT_BG}; border:1px solid {ACCENT}; color:{ACCENT};"
            " border-radius:8px; padding:8px 12px; font-weight:600;")
        self._upload_done = False

        def worker():
            try:
                comm.provision_over_serial()
                self._upload_ok = True
            except Exception as e:
                self._upload_ok = False
                print(f"[ERROR] Upload IP failed: {e}")
            self._upload_done = True

        threading.Thread(target=worker, daemon=True).start()

    def _adoptPort(self):
        """When the GUI started with no devices, adopt the first reactor that connects
        (e.g. after Upload IP + Rescan) so the controls bind to it."""
        if not _Vars.ports:
            return
        self._syncPorts()                      # make sure the selector/dots have the new port(s)
        self.currentPort = int(_Vars.ports[0])
        idx = self.portSelector.findText(str(self.currentPort))
        if idx >= 0:
            self.portSelector.blockSignals(True)
            self.portSelector.setCurrentIndex(idx)
            self.portSelector.blockSignals(False)
        self.updateValues(self.currentPort)
        try:
            msg = _Vars.activeDevices[self.currentPort]["OutgoingMessage"]
            self._set_pills([self.tempbangbang, self.tempHyst, self.tempPI], int(msg[18].item()))
            self._set_pills([self.ODbangbang, self.ODHyst, self.ODClamp], int(msg[32].item()))
        except Exception:
            pass
        self._syncPinchButton(self.currentPort)
        self._syncModeMenu()
        self._refreshStrainLabel()
        self._syncExperimentCard()
        print(f"[INFO] Adopted reactor on port {self.currentPort}.")

    def _syncPorts(self):
        """Keep the port selector + header dots in sync with _Vars.ports (handles
        devices added by a rescan) and flash the rescan button when it completes."""
        existing = {self.portSelector.itemText(i) for i in range(self.portSelector.count())}
        for p in _Vars.ports:
            if str(p) not in existing:
                self.portSelector.addItem(str(p))
                if p not in self.portDots:
                    dot = QLabel(f"●  {p}"); dot.setStyleSheet(f"color:{MUTED}; font-size:12px;")
                    self.portDots[p] = dot
                    self.dotsRow.addWidget(dot)
        if getattr(self, "_rescan_done", False):
            self._rescan_done = False
            self.rescanDevices.setText("Rescan Devices")
            self._flash(self.rescanDevices, getattr(self, "_rescan_ok", False))
            print(f"[INFO] Rescan added {getattr(self, '_rescan_n', 0)} device(s).")
        if getattr(self, "_upload_done", False):
            self._upload_done = False
            self.uploadIP.setText("Upload IP")
            self._flash(self.uploadIP, getattr(self, "_upload_ok", False))
        # Apply-to-All only makes sense with more than one reactor connected.
        # Re-evaluated every tick so it appears the moment a rescan brings a
        # second device online.
        self.applyAll.setVisible(len(_Vars.ports) > 1)
        # Hide the "searching…" spinner once a reactor is live (and re-show it if
        # every reactor later drops off).
        self._update_search_indicator()

    def changePort(self, index) -> None:
        txt = self.portSelector.currentText()
        if not txt:
            return
        self.currentPort = int(txt)
        print(f"[INFO] Switched control to port {self.currentPort}.")
        self.updateValues(self.currentPort)
        msg = _Vars.activeDevices[self.currentPort]["OutgoingMessage"]
        heater_mode = int(msg[18].item())
        od_mode     = int(msg[32].item())
        self._set_pills([self.tempbangbang, self.tempHyst, self.tempPI], heater_mode)
        self._set_pills([self.ODbangbang, self.ODHyst, self.ODClamp], od_mode)
        # Sync field enable/disable to THIS reactor's modes (fixes stale disabled
        # fields when switching between reactors with different control profiles).
        self.tempbbDelay.setDisabled(heater_mode != 0)
        self.tempHystVals.setDisabled(heater_mode != 1)
        turb_on = _Vars.config.get("Turbidostat", False)
        self.ODbbDelay.setDisabled((not turb_on) or od_mode != 0)
        self.ODHystVal.setDisabled((not turb_on) or od_mode != 1)
        # Reflect THIS reactor's override state on the toggle without writing to it.
        # Toggling the switch only changes the selected reactor; use Apply to All to propagate.
        manual = (int(msg[34].item()) == 0)
        self.overrideToggle.blockSignals(True)
        self.overrideToggle.setChecked(manual)
        self.overrideToggle.blockSignals(False)
        for w in (self.wasteMotor, self.freshMotor):
            w.setEnabled(manual)
        self._syncPinchButton(self.currentPort)   # reflect THIS reactor's byte 35
        self._syncModeMenu()            # tick the Modes menu from THIS reactor
        self._refreshStrainLabel()      # header strain follows the selection
        self._syncExperimentCard()      # reflect THIS reactor's experiment state

    def _syncPinchButton(self, port) -> None:
        """Point the Pinch Valve button at the selected reactor by reading ITS
        byte 35, instead of carrying the previously-selected reactor's state
        over (which would make the next press write a stale value)."""
        try:
            state = bool(int(_Vars.activeDevices[port]["OutgoingMessage"][35].item()))
        except Exception:
            state = False               # no reactor / unreadable — show rest state
        self.pinchState = state
        self.pinch.setStyleSheet(PILL_ACTIVE if state else PILL_IDLE)
        self.pinch.setText(f"Pinch State: {state}")
        self._msSyncReactorControls()

    def updateValues(self, port) -> None:
        if port is None or port not in _Vars.activeDevices:
            return
        if _Vars.activeDevices[self.currentPort]['DeviceVariables']['MotorDirections'][0] == 1:
            self.cycleMotor.setText(f"{-1*_Vars.activeDevices[port]['DeviceVariables']['MotorSpeeds'][0]}")
        else:
            self.cycleMotor.setText(f"{_Vars.activeDevices[port]['DeviceVariables']['MotorSpeeds'][0]}")
        self.fanMotor.setText(f"{_Vars.activeDevices[port]['DeviceVariables']['MotorSpeeds'][1]}")
        self.ch0Motor.setText(f"{_Vars.activeDevices[port]['DeviceVariables'].get('CH0Speed', 0)}")
        self._msSyncReactorControls()
        self.wasteMotor.setText(f"{_Vars.activeDevices[port]['DeviceVariables']['MotorSpeeds'][2]}")
        self.freshMotor.setText(f"{_Vars.activeDevices[port]['DeviceVariables']['MotorSpeeds'][3]}")
        self.tempTarg.setText(f"{_Vars.activeDevices[port]['DeviceVariables']['ControlValues']['Temperature'][0]}")
        self.tempHystVals.setText(f"{_Vars.activeDevices[port]['DeviceVariables']['ControlValues']['Temperature'][1]}")
        self.tempbbDelay.setText(f"{_Vars.activeDevices[port]['DeviceVariables']['ControlValues']['Temperature'][2]}")
        self.targOD.setText(f"{_Vars.activeDevices[port]['DeviceVariables']['ControlValues']['OD'][0]}")
        self.ODbbDelay.setText(f"{_Vars.activeDevices[port]['DeviceVariables']['ControlValues']['OD'][1]}")
        self.ODHystVal.setText(f"{_Vars.activeDevices[port]['DeviceVariables']['ControlValues']['OD'][3]}")

    def changeTempMode(self, mode, port) -> None:
        _Vars.activeDevices[port]["OutgoingMessage"][18] = mode
        _Vars.activeDevices[port]["DeviceVariables"]["HeaterMode"] = ("bangbang", "hyst", "pi")[mode]
        self.tempbbDelay.setDisabled(mode != 0)
        self.tempHystVals.setDisabled(mode != 1)
        self._set_pills([self.tempbangbang, self.tempHyst, self.tempPI], mode)
        _Vars.activeDevices[port]["DeviceVariables"]["SendMessage"] = True

    def changeODState(self, state, port) -> None:
        _Vars.activeDevices[port]["OutgoingMessage"][32] = state
        self.ODbbDelay.setDisabled(state != 0)
        self.ODHystVal.setDisabled(state != 1)
        self._set_pills([self.ODbangbang, self.ODHyst, self.ODClamp], state)
        _Vars.activeDevices[port]["DeviceVariables"]["SendMessage"] = True

    # =================== CONFIG TAB ===================
    # Known enumerated / boolean config keys, so the editor renders a dropdown
    # instead of a free-text box (and Save writes a valid value).
    _CFG_CHOICES = {
        ("DeviceVariables", "TurbidostatMode"): ["bangbang", "hyst", "clampod"],
        ("DeviceVariables", "HeaterMode"):      ["bangbang", "hyst", "pi"],
        ("Kalman",          "Model"):           ["scalar", "cv"],
        ("Calibration",     "BetaSource"):      ["auto", "predefined"],
    }

    # =================== MENU BAR ===================
    def _build_menu_bar(self):
        """Menu bar: File (save location), Modes (checkable live toggles with a
        tick reflecting the selected reactor's real state), Run (strain info).

        The Modes actions drive the SAME handlers as the on-panel widgets rather
        than writing packet bytes themselves, so the two views can't disagree.
        _syncModeMenu() pushes live state back into the ticks."""
        bar = QMenuBar(self)
        bar.setStyleSheet(
            f"QMenuBar {{ background:{CARD}; color:{TEXT}; border:1px solid {CARD_EDGE};"
            f" border-radius:8px; padding:2px; }}"
            f"QMenuBar::item {{ background:transparent; padding:6px 12px; border-radius:6px; }}"
            f"QMenuBar::item:selected {{ background:{INPUT_BG}; }}"
            f"QMenu {{ background:{CARD}; color:{TEXT}; border:1px solid {CARD_EDGE}; padding:6px; }}"
            f"QMenu::item {{ padding:6px 28px 6px 24px; border-radius:6px; }}"
            f"QMenu::item:selected {{ background:{INPUT_BG}; }}"
            f"QMenu::item:disabled {{ color:{MUTED}; }}"
            f"QMenu::separator {{ height:1px; background:{CARD_EDGE}; margin:6px 4px; }}"
        )
        self._modeActions = {}

        def add_check(menu, key, text, slot, tip=""):
            act = QAction(text, self); act.setCheckable(True)
            if tip:
                act.setToolTip(tip)
            act.toggled.connect(slot)
            menu.addAction(act)
            self._modeActions[key] = act
            return act

        # ---- File ----
        m_file = bar.addMenu("File")
        a = QAction("Set Save Location…", self); a.triggered.connect(self.setSaveLocation)
        m_file.addAction(a)
        a = QAction("Open Save Location", self); a.triggered.connect(self.openSaveLocation)
        m_file.addAction(a)
        m_file.addSeparator()
        self.saveLocAction = QAction("", self); self.saveLocAction.setEnabled(False)
        m_file.addAction(self.saveLocAction)

        # ---- Modes ---- (ticked = currently on for the selected reactor)
        m_modes = bar.addMenu("Modes")
        add_check(m_modes, "turbidostat", "Turbidostat", self.setTurbidostat,
                  "Auto pump control owns the dilution pumps (packet byte 34).")
        add_check(m_modes, "overrides", "Manual Pump Overrides",
                  lambda on: self.overrideToggle.setChecked(on),
                  "Pause auto pumps so the manual pump 1/2 fields drive CH3/CH4.")
        add_check(m_modes, "pinch", "Pinch Valve",
                  lambda on: self.setPinchState(on, self.currentPort),
                  "High-side CH2/CH3 complementary pinch pair (packet byte 35).")
        m_modes.addSeparator()
        add_check(m_modes, "microscope", "Microscope",
                  self.setMicroscopeEnabled,
                  "Show the Microscope + Microscope Optics tabs.")
        add_check(m_modes, "dobot", "Dobot Arm",
                  self.setDobotEnabled,
                  "Show the Dobot Magician sampling-arm tab.")
        m_modes.addSeparator()
        add_check(m_modes, "modled", "Modulated LED",
                  lambda on: self.modLedBtn.setChecked(on))
        add_check(m_modes, "mcuavg", "MCU Averaging",
                  lambda on: self.mcuAvgBtn.setChecked(on))
        add_check(m_modes, "cvkalman", "CV Kalman (2-state)",
                  lambda on: self.cvKalBtn.setChecked(on))
        m_modes.addSeparator()
        add_check(m_modes, "ttemp", "Show Target Temp",
                  lambda on: self.targetTempBtn.setChecked(on))
        add_check(m_modes, "tod", "Show Target OD",
                  lambda on: self.targetODBtn.setChecked(on))

        # ---- Run ----
        m_run = bar.addMenu("Run")
        a = QAction("Set Strain…", self); a.triggered.connect(self.setStrainDialog)
        m_run.addAction(a)
        m_run.addSeparator()
        self.strainAction = QAction("", self); self.strainAction.setEnabled(False)
        m_run.addAction(self.strainAction)

        self._refreshSaveLocationLabel()
        return bar

    def _syncModeMenu(self) -> None:
        """Push the selected reactor's live state into the menu ticks. Signals are
        blocked while setting so syncing never re-fires the handlers."""
        acts = getattr(self, "_modeActions", None)
        if not acts:
            return
        dev = _Vars.activeDevices.get(self.currentPort) if self.currentPort is not None else None
        msg = dev["OutgoingMessage"] if dev is not None else None

        def b(i):
            try:    return int(msg[i].item())
            except Exception: return 0

        kal = str(_Vars.config.get("KalmanModel", "scalar")).strip().lower()
        states = {
            "turbidostat": bool(dev["DeviceVariables"]["Turbidostat"]) if dev is not None
                           else bool(_Vars.config.get("Turbidostat", False)),
            "overrides":   self.overrideToggle.isChecked(),
            "pinch":       bool(b(35)) if msg is not None else self.pinchState,
            "modled":      bool(b(0))  if msg is not None else self.modLedBtn.isChecked(),
            "mcuavg":      (b(52) > 1) if msg is not None else self.mcuAvgBtn.isChecked(),
            "cvkalman":    kal in ("cv", "2state", "2-state"),
            "ttemp":       self.showTargetTemp,
            "tod":         self.showTargetOD,
            # Not derived from a device — it is a view toggle. Must be listed or
            # states.get(key, False) below would silently UNCHECK it every sync
            # (signals are blocked there, so the tabs would stay while the menu
            # said off, and the next click would be a no-op).
            "microscope":  getattr(self, "_msTabs", None) is not None,
            "dobot":       getattr(self, "_dbTabs", None) is not None,
        }
        for key, act in acts.items():
            act.blockSignals(True)
            act.setChecked(bool(states.get(key, False)))
            act.blockSignals(False)
        # Actions that write to a reactor are meaningless with none connected.
        for key in ("turbidostat", "overrides", "pinch", "modled", "mcuavg"):
            if key in acts:
                acts[key].setEnabled(dev is not None)
        if hasattr(self, "strainAction"):
            self.strainAction.setText(f"Strain: {utils.strain_for(self.currentPort)}"
                                      if dev is not None else "Strain: (no reactor)")

    def setTurbidostat(self, on) -> None:
        """Turn turbidostat (auto pump control) on/off across every reactor, and
        mirror it into the OD control enables + config so it survives a restart."""
        on = bool(on)
        _Vars.config["Turbidostat"] = on
        manual = self.overrideToggle.isChecked()
        for p in _Vars.ports:
            dev = _Vars.activeDevices.get(p)
            if dev is None:
                continue
            dev["DeviceVariables"]["Turbidostat"] = on
            # byte 34: auto owns the pumps only when turbidostat is on AND the
            # manual overrides aren't holding them.
            dev["OutgoingMessage"][34] = 0 if manual else (1 if on else 0)
            dev["DeviceVariables"]["SendMessage"] = True
        for w in (self.ODbangbang, self.ODHyst, self.ODClamp,
                  self.ODbbDelay, self.ODHystVal, self.targOD):
            w.setEnabled(on)
        if hasattr(self, "overridesSection"):
            self.overridesSection.setVisible(on)
        self._cfg_set("DeviceVariables", "Turbidostat", "True" if on else "False")
        self._syncModeMenu()

    # ---- File menu ----
    def _refreshSaveLocationLabel(self) -> None:
        if hasattr(self, "saveLocAction"):
            self.saveLocAction.setText(f"Saving to: {utils.log_root()}")

    def setSaveLocation(self) -> None:
        """Pick where run logs are written; persisted to config.ini [Storage].
        Runs already streaming keep their current file — the new location applies
        to run files opened from here on (after a Reset, or the next launch)."""
        d = QFileDialog.getExistingDirectory(self, "Choose save location for run logs",
                                             utils.log_root())
        if not d:
            return
        try:
            _Vars.config["SaveDirectory"] = d
            utils.saveConfigValues({("Storage", "SaveDirectory"): d})
            self._cfg_set("Storage", "SaveDirectory", d)
            self._refreshSaveLocationLabel()
            print(f"[INFO] Save location -> {d} (applies to new run files)")
        except Exception as e:
            print(f"[ERROR] Could not set save location: {e}")

    def openSaveLocation(self) -> None:
        path = utils.log_root()
        try:
            os.makedirs(path, exist_ok=True)
            if sys.platform.startswith("win"):
                os.startfile(path)                                  # noqa: S606
            else:
                import subprocess
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])
        except Exception as e:
            print(f"[WARN] Could not open {path}: {e}")

    # ---- Run menu ----
    def setStrainDialog(self) -> None:
        """Set the strain/culture label for the selected reactor. Lands in the XML
        sidecar on the next save; the CSV filename picks it up from the next run
        file onward (the current file keeps the name it was opened with)."""
        port = self.currentPort
        current = utils.strain_for(port) if port in _Vars.activeDevices else str(_Vars.config.get("Strain", ""))
        text, ok = QInputDialog.getText(
            self, "Strain info",
            "Strain / culture label"
            + (f" for reactor {port}:" if port is not None else " (default for new reactors):"),
            text=current)
        if not ok:
            return
        text = text.strip()
        if not text:
            return
        if port is not None and port in _Vars.activeDevices:
            _Vars.activeDevices[port]["Strain"] = text
        _Vars.config["Strain"] = text          # default for reactors that connect later
        _Vars.strain = text                    # legacy fallback
        try:
            utils.saveConfigValues({("Run", "Strain"): text})
            self._cfg_set("Run", "Strain", text)
        except Exception as e:
            print(f"[WARN] Could not persist strain: {e}")
        self._syncModeMenu()
        self._refreshStrainLabel()
        print(f"[INFO] Strain for port {port} -> {text!r} "
              f"(XML sidecar updates on next save; filename applies to new run files)")

    def _refreshStrainLabel(self) -> None:
        if hasattr(self, "strainLabel"):
            s = utils.strain_for(self.currentPort) if self.currentPort in _Vars.activeDevices \
                else str(_Vars.config.get("Strain", "—"))
            self.strainLabel.setText(f"STRAIN  {s}")

    def _build_config_editor(self):
        """Editable view of every config variable + Save Config / Sync from Live.
        Fields are generated from the files so they always map 1:1 back, and each
        save routes to the file that owns the section (reactor settings ->
        config.ini, simulator/twin -> simulator.ini). Sync from Live pulls the
        current reactor's running control state into the fields so you can
        snapshot a tuned run for next time."""
        card, cv = self._card("Configuration")
        card.setMinimumWidth(460)     # grows to fill the tab (was a fixed narrow column)
        self._cfg_fields = {}     # (section, key) -> (kind, widget)

        body = QWidget()
        form = QVBoxLayout(body); form.setSpacing(8); form.setContentsMargins(0, 0, 0, 0)
        last_section = None
        last_src = None
        for src, section, key, value in utils.readConfigItems():
            if src != last_src:
                # File banner — makes it obvious which settings are simulator-only.
                if last_src is not None:
                    form.addSpacing(12)
                fhdr = QLabel(f"— {src} —")
                fhdr.setStyleSheet(f"color:{ACCENT}; font-size:11px; font-weight:700;"
                                   " letter-spacing:1px;")
                form.addWidget(fhdr)
                last_src = src
                last_section = None
            if section != last_section:
                if last_section is not None:
                    form.addSpacing(6)
                hdr = QLabel(section.upper()); hdr.setObjectName("cardTitle")
                form.addWidget(hdr)
                last_section = section
            low = value.strip().lower()
            if (section, key) in self._CFG_CHOICES:
                w = QComboBox(); w.addItems(self._CFG_CHOICES[(section, key)])
                i = w.findText(value.strip())
                if i >= 0: w.setCurrentIndex(i)
                kind = "choice"
            elif low in ("true", "false"):
                w = QComboBox(); w.addItems(["True", "False"])
                w.setCurrentIndex(0 if low == "true" else 1)
                kind = "bool"
            else:
                w = QLineEdit(value); kind = "text"
            self._cfg_fields[(section, key)] = (kind, w)
            form.addLayout(self._row(key, w))
        form.addStretch()

        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(body)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        cv.addWidget(scroll, 1)

        self.cfgSync = QPushButton("Sync from Live")
        self.cfgSync.setToolTip("Pull the selected reactor's current control values "
                                "into these fields, then Save Config to persist them.")
        self.cfgSave = QPushButton("Save Config")
        brow = QHBoxLayout(); brow.setSpacing(8)
        brow.addWidget(self.cfgSync); brow.addWidget(self.cfgSave)
        cv.addLayout(brow)
        self.cfgStatus = QLabel("Edit values, then Save Config to persist. Each value is "
                                "written back to the file it came from (applied on next startup).")
        self.cfgStatus.setObjectName("subValue"); self.cfgStatus.setWordWrap(True)
        cv.addWidget(self.cfgStatus)

        self.cfgSync.clicked.connect(self.syncConfigFromLive)
        self.cfgSave.clicked.connect(self.saveConfigClick)
        return card

    def _cfg_value(self, kind, w):
        return w.currentText() if kind in ("bool", "choice") else w.text()

    def _cfg_set(self, section, key, value):
        f = self._cfg_fields.get((section, key))
        if not f:
            return
        kind, w = f
        if kind in ("bool", "choice"):
            i = w.findText(str(value))
            if i >= 0:
                w.setCurrentIndex(i)
        else:
            w.setText(str(value))

    def saveConfigClick(self):
        """Write every editor field back to config.ini (comment-preserving)."""
        try:
            updates = {sk: self._cfg_value(kind, w)
                       for sk, (kind, w) in self._cfg_fields.items()}
            n = utils.saveConfigValues(updates)
            self.cfgStatus.setText(
                f"Saved {n} values to config.ini. Restart to apply "
                "(control targets also take effect for new reactors).")
            self._flash(self.cfgSave, True)
        except Exception as e:
            self.cfgStatus.setText(f"Save failed: {e}")
            self._flash(self.cfgSave, False)

    def syncConfigFromLive(self):
        """Populate the editor fields from the selected reactor's live state so a
        tuned run can be snapshotted into config.ini for next startup."""
        port = self.currentPort
        if port is None or port not in _Vars.activeDevices:
            self.cfgStatus.setText("No active reactor to sync from.")
            return
        dev = _Vars.activeDevices[port]
        dv  = dev["DeviceVariables"]; cvv = dv["ControlValues"]; co = dev["Coefficients"]
        msg = dev["OutgoingMessage"]
        HEATER = {0: "bangbang", 1: "hyst", 2: "pi"}
        ODMODE = {0: "bangbang", 1: "hyst", 2: "clampod"}
        self._cfg_set("ControlValues", "Temperature",
                      ", ".join(str(x) for x in cvv.get("Temperature", [])))
        self._cfg_set("ControlValues", "OD",
                      ", ".join(str(x) for x in cvv.get("OD", [])))
        self._cfg_set("Offsets", "TemperatureOffset", co.get("TemperatureOffset", ""))
        self._cfg_set("DeviceVariables", "MotorSpeeds",
                      ", ".join(str(x) for x in dv.get("MotorSpeeds", [])))
        self._cfg_set("DeviceVariables", "MotorDirections",
                      ", ".join(str(x) for x in dv.get("MotorDirections", [])))
        self._cfg_set("DeviceVariables", "Turbidostat",
                      "True" if dv.get("Turbidostat") else "False")
        try:    self._cfg_set("DeviceVariables", "HeaterMode", HEATER.get(int(msg[18].item())))
        except Exception: pass
        try:    self._cfg_set("DeviceVariables", "TurbidostatMode", ODMODE.get(int(msg[32].item())))
        except Exception: pass
        self._cfg_set("Kalman", "Model", _Vars.config.get("KalmanModel", "scalar"))
        try:
            k = dev["ML"]["tempPI"]
            self._cfg_set("TempPI", "Kp", k.Kp); self._cfg_set("TempPI", "Ki", k.Ki)
        except Exception: pass
        self.cfgStatus.setText(
            f"Synced fields from live reactor on port {port}. Review, then Save Config.")


    # =================== MICROSCOPE TAB ===================
    def _build_microscope_tabs(self):
        """Embed the FULL microscope application as a tab.

        MicroscopeFull.py is an auto-generated PyQt6 port of
        Desktop/pycromanager/microscope_control.py (that file is PyQt5, which
        cannot share a process with PyQt6). Regenerate it with
        scratchpad/port_pyqt6.py if the upstream file changes — do not hand-edit.

        Everything the standalone app has comes across intact: connection, stage
        jog + go-to, camera/exposure/binning, illumination channels and intensity
        sliders, autofocus with its focus curve, focus stacks, the full
        acquisition planner (timing, channel/periodic/position tables, output)
        and the log panel — as its own Control / Acquisition / Logs sub-tabs.

        The pinch valve and aux motor CH0 are injected into its Control tab as a
        REACTOR CONTROLS panel, wired through MicroscopeFull.REACTOR_HOOKS."""
        import MicroscopeFull as MF

        # Hooks let the microscope panel drive this application's reactor state
        # without either module importing the other.
        MF.REACTOR_HOOKS.update({
            "pinch_get":    lambda: self.pinchState,
            "pinch_toggle": lambda: self.togglePinchState(port=self.currentPort),
            "ch0_get":      lambda: self.ch0Motor.text(),
            "ch0_set":      self._msCh0Set,
        })

        self._msWindow = MF.MainWindow()
        # takeCentralWidget() detaches and TRANSFERS OWNERSHIP. Do not use
        # setCentralWidget(QWidget()) here: QMainWindow DELETES the widget it is
        # replacing, so the panel we are about to embed gets destroyed. That
        # deletion is deferred, so it only shows up once an event loop runs —
        # the tab appears empty/missing in the real app but fine in a headless test.
        central = self._msWindow.takeCentralWidget()
        self._msPanel = getattr(self._msWindow, "reactor_panel", None)
        self._msSyncReactorControls()
        return (central,)

    def _ch0Changed(self):
        """Aux motor edited on the Control Panel -> push to the packet, then
        refresh the microscope panel so both views agree."""
        utils.changeCH0Speed(text=self.ch0Motor.text(), port=self.currentPort)
        self._msSyncReactorControls()

    def _msCh0Set(self, text):
        """Aux motor edited on the microscope panel -> drive the packet, then
        mirror onto the Control Panel field (signals blocked: no ping-pong)."""
        utils.changeCH0Speed(text=text, port=self.currentPort)
        if hasattr(self, "ch0Motor") and self.ch0Motor.text() != text:
            self.ch0Motor.blockSignals(True)
            self.ch0Motor.setText(text)
            self.ch0Motor.blockSignals(False)

    def _msSyncReactorControls(self):
        """Push reactor state into the microscope panel (no-op when it is absent)."""
        panel = getattr(self, "_msPanel", None)
        if panel is not None:
            try:    panel.refresh()
            except Exception: pass

    def _msStopLive(self):
        """Stop the microscope's own live acquisition / pollers on teardown."""
        w = getattr(self, "_msWindow", None)
        if w is None:
            return
        for attr in ("_live_acq", "_poller"):
            t = getattr(w, attr, None)
            if t is not None:
                try:    t.stop()
                except Exception: pass
                setattr(w, attr, None)

    # ---- enable / disable the whole feature ----
    def setMicroscopeEnabled(self, on):
        """Show or hide the microscope tab live, and persist the choice."""
        on = bool(on)
        _Vars.config["MicroscopeEnabled"] = on
        has = getattr(self, "_msTabs", None) is not None
        if on and not has:
            try:
                self._msTabs = self._build_microscope_tabs()
                for w in self._msTabs:
                    self.tabs.addTab(w, "Microscope")
            except Exception as e:
                self._msTabs = None
                print("[ERROR] Microscope tab unavailable: " + str(e))
                _Vars.config["MicroscopeEnabled"] = False
                on = False
        elif not on and has:
            self._msStopLive()
            for w in self._msTabs:
                i = self.tabs.indexOf(w)
                if i >= 0:
                    self.tabs.removeTab(i)
                w.deleteLater()
            self._msTabs = None
            self._msWindow = None
            self._msPanel = None
        try:
            utils.saveConfigValues({("Microscope", "Enabled"): "True" if on else "False"})
            self._cfg_set("Microscope", "Enabled", "True" if on else "False")
        except Exception:
            pass
        act = getattr(self, "_modeActions", {}).get("microscope")
        if act is not None and act.isChecked() != on:
            act.blockSignals(True); act.setChecked(on); act.blockSignals(False)

    # =========================== DOBOT ARM ==================================
    def _build_dobot_tab(self):
        """Embed the Dobot Magician sampling bench as a tab.

        DobotBench.build_window() returns a standalone QMainWindow plus its Arm.
        We keep the window alive (its QTimer pose-poller is parented to it) and
        embed only its central widget via takeCentralWidget() — the same
        ownership transfer used for the microscope. The arm never auto-connects:
        it only touches COM3 when the user clicks Connect inside the tab, so
        building the tab moves nothing."""
        import DobotBench as DB
        self._dbWindow, self._dbArm = DB.build_window()
        central = self._dbWindow.takeCentralWidget()
        return (central,)

    def _dbStopLive(self):
        """Stop the Dobot pose-poll timer and disconnect the arm on teardown."""
        w = getattr(self, "_dbWindow", None)
        if w is not None:
            t = getattr(w, "_timer", None)
            if t is not None:
                try:    t.stop()
                except Exception: pass
        a = getattr(self, "_dbArm", None)
        if a is not None:
            try:    a.disconnect()
            except Exception: pass

    def setDobotEnabled(self, on):
        """Show or hide the Dobot arm tab live, and persist the choice."""
        on = bool(on)
        _Vars.config["DobotEnabled"] = on
        has = getattr(self, "_dbTabs", None) is not None
        if on and not has:
            try:
                self._dbTabs = self._build_dobot_tab()
                for w in self._dbTabs:
                    self.tabs.addTab(w, "Dobot Arm")
            except Exception as e:
                self._dbTabs = None
                print("[ERROR] Dobot tab unavailable: " + str(e))
                _Vars.config["DobotEnabled"] = False
                on = False
        elif not on and has:
            self._dbStopLive()
            for w in self._dbTabs:
                i = self.tabs.indexOf(w)
                if i >= 0:
                    self.tabs.removeTab(i)
                w.deleteLater()
            self._dbTabs = None
            self._dbWindow = None
            self._dbArm = None
        try:
            utils.saveConfigValues({("Dobot", "Enabled"): "True" if on else "False"})
            self._cfg_set("Dobot", "Enabled", "True" if on else "False")
        except Exception:
            pass
        act = getattr(self, "_modeActions", {}).get("dobot")
        if act is not None and act.isChecked() != on:
            act.blockSignals(True); act.setChecked(on); act.blockSignals(False)

    # =================== BETA (auto output vs predefined) ===================
    def _set_beta_source_pills(self, source):
        self._set_pills([self.betaSrcAuto, self.betaSrcPredef],
                        1 if source == "predefined" else 0)

    def setBetaSource(self, source):
        """Choose which beta the OD calc uses (auto = learned live this run;
        predefined = the fixed configured value). Applies to every active reactor
        and the in-memory config; persist via Save Config / Save Auto → Predefined."""
        source = "predefined" if source == "predefined" else "auto"
        _Vars.config["BetaSource"] = source
        for p in _Vars.ports:
            _Vars.activeDevices[p]["Coefficients"]["BetaSource"] = source
        self._set_beta_source_pills(source)
        self._cfg_set("Calibration", "BetaSource", source)   # sync the Config editor field
        self.betaStatus2.setText(
            f"OD now uses the {source} β on all reactors. "
            "Save Config (or Save Auto → Predefined) to persist for next startup.")

    def onBetaPredefChanged(self):
        """Live-apply an edited predefined β to every reactor + in-memory config."""
        try:
            v = float(self.betaPredef.text())
        except ValueError:
            return
        for p in _Vars.ports:
            _Vars.activeDevices[p]["Coefficients"]["BetaPredefined"] = v
        _Vars.config["Beta"] = v
        self._cfg_set("Calibration", "Beta", self.betaPredef.text())

    def saveAutoAsPredefined(self):
        """Snapshot the current auto-detected β into the predefined value + config,
        and switch the OD source to predefined (the common calibrate-then-run flow)."""
        port = self.currentPort
        if port is None or port not in _Vars.activeDevices:
            self.betaStatus2.setText("No active reactor.")
            return
        auto = utils.betaAuto(port)
        if not (auto == auto) or auto <= 0:
            self.betaStatus2.setText("No auto β yet — let the blank window (BetaTime) accumulate first.")
            return
        self.betaPredef.setText(f"{auto:.6f}")     # fires onBetaPredefChanged → applies live
        try:
            utils.saveConfigValues({("Calibration", "Beta"): f"{auto:.6f}",
                                    ("Calibration", "BetaSource"): "predefined"})
            self.setBetaSource("predefined")
            self.betaStatus2.setText(
                f"Saved auto β = {auto:.5f} as predefined (config) and switched OD to "
                "predefined for all reactors.")
            self._flash(self.betaSaveAuto, True)
        except Exception as e:
            self.betaStatus2.setText(f"Save failed: {e}")
            self._flash(self.betaSaveAuto, False)

    def _refresh_beta_value(self, port):
        """Keep the auto-β readout current: value, whether it's latched vs still
        learning, and whether it's the one currently driving OD."""
        if not hasattr(self, "betaAutoVal"):
            return
        try:
            auto = utils.betaAuto(port)
            co = _Vars.activeDevices[port]["Coefficients"]
            if auto == auto:
                tag = "latched" if co["BetaFinal"] else "learning…"
                use = "  ← in use" if co.get("BetaSource", "auto") != "predefined" else ""
                self.betaAutoVal.setText(f"auto β {auto:.5f} ({tag}){use}")
            else:
                self.betaAutoVal.setText("auto β —")
        except Exception:
            self.betaAutoVal.setText("auto β —")

    # =================== CONFIG TAB (config editor + tuning) ===================
    def _build_config_tab(self):
        tab = QWidget()
        outer = QHBoxLayout(tab); outer.setContentsMargins(0, 10, 0, 0); outer.setSpacing(16)

        # --- Beta / tuning card ---
        beta, bv = self._card("Beta / Tuning")
        beta.setFixedWidth(300)
        self.modLedBtn = QPushButton("Modulated LED"); self.modLedBtn.setCheckable(True)
        self.modLedBtn.setStyleSheet(PILL_IDLE)
        bv.addWidget(self.modLedBtn)
        self.mcuAvgN = QLineEdit("4")
        bv.addLayout(self._row("Samples (N)", self.mcuAvgN))
        self.mcuAvgBtn = QPushButton("MCU Averaging"); self.mcuAvgBtn.setCheckable(True)
        self.mcuAvgBtn.setStyleSheet(PILL_IDLE)
        bv.addWidget(self.mcuAvgBtn)
        self.kalQ = QLineEdit("1e-5"); self.kalR = QLineEdit("5e-4")
        bv.addLayout(self._row("Kalman Q", self.kalQ))
        bv.addLayout(self._row("Kalman R", self.kalR))
        self.kalApply = QPushButton("Apply Kalman")
        bv.addWidget(self.kalApply)
        self.cvKalBtn = QPushButton("CV Kalman (2-state)"); self.cvKalBtn.setCheckable(True)
        self.cvKalBtn.setStyleSheet(PILL_IDLE)
        bv.addWidget(self.cvKalBtn)
        self.betaStatus = QLabel(""); self.betaStatus.setObjectName("subValue"); self.betaStatus.setWordWrap(True)
        bv.addWidget(self.betaStatus)

        # --- Experiment conditions card ---
        expt, ev = self._card("Experiment Conditions")
        self.exptBtn      = QPushButton("Hysteresis → Clamp")
        self.exptBtn.setStyleSheet(PILL_IDLE)
        self.exptDuration = QLineEdit("400"); self.exptDuration.setFixedWidth(70)
        self.exptDuration.setToolTip("Seconds of OD hysteresis before auto-switching to Clamp.")
        erow = QHBoxLayout(); erow.setSpacing(8)
        erow.addWidget(self.exptBtn, 1); erow.addWidget(self.exptDuration); erow.addWidget(QLabel("s"))
        ev.addLayout(erow)
        self.exptStatus = QLabel("Runs OD hysteresis with the Control Panel target/band, then switches to "
                                 "Clamp after the set delay. Click again to cancel.")
        self.exptStatus.setObjectName("subValue"); self.exptStatus.setWordWrap(True)
        ev.addWidget(self.exptStatus)

        # --- Beta card: auto-detected output + predefined value + which to use ---
        betacal, bcv = self._card("Beta")
        self.betaAutoVal = QLabel("auto β —"); self.betaAutoVal.setObjectName("bigValue")
        bcv.addWidget(self.betaAutoVal)
        self.betaPredef = QLineEdit(f"{_Vars.config.get('Beta', 0.0)}")
        bcv.addLayout(self._row("Predefined β", self.betaPredef))
        self.betaSrcAuto   = QPushButton("Use Auto")
        self.betaSrcPredef = QPushButton("Use Predefined")
        bcv.addLayout(self._pills(self.betaSrcAuto, self.betaSrcPredef))
        self.betaSaveAuto = QPushButton("Save Auto → Predefined")
        bcv.addWidget(self.betaSaveAuto)
        self.betaStatus2 = QLabel("Auto β is learned live from the first BetaTime seconds (blank) and "
                                  "latched — always shown as an output. Switch to Predefined for runs "
                                  "inoculated from t=0. Save Auto → Predefined stores the current auto β.")
        self.betaStatus2.setObjectName("subValue"); self.betaStatus2.setWordWrap(True)
        bcv.addWidget(self.betaStatus2)

        mid = QVBoxLayout(); mid.setSpacing(14)
        mid.addWidget(beta); mid.addWidget(betacal); mid.addWidget(expt); mid.addStretch()
        mid_holder = QWidget(); mid_holder.setLayout(mid); mid_holder.setFixedWidth(300)

        # Config editor is the primary content now — give it the freed space.
        outer.addWidget(self._build_config_editor(), 1)
        outer.addWidget(mid_holder)

        self.modLedBtn.toggled.connect(self.toggleModLED)
        self.mcuAvgBtn.toggled.connect(self.toggleMcuAvg)
        self.kalApply.clicked.connect(self.applyKalman)
        self.cvKalBtn.toggled.connect(self.toggleCVKalman)
        self.exptBtn.clicked.connect(self.startExperiment)
        self.betaSrcAuto.clicked.connect(lambda: self.setBetaSource("auto"))
        self.betaSrcPredef.clicked.connect(lambda: self.setBetaSource("predefined"))
        self.betaPredef.textChanged.connect(self.onBetaPredefChanged)
        self.betaSaveAuto.clicked.connect(self.saveAutoAsPredefined)
        self._set_beta_source_pills(_Vars.config.get("BetaSource", "auto"))
        self._expt = {}            # port -> {"timer": QTimer, "remaining": int} (independent per reactor)
        self._expt_last = {}       # port -> last done/cancelled status message
        self._expt_help = ("Runs OD hysteresis with the Control Panel target/band, then switches to "
                           "Clamp after the set delay. Per-port; click again to cancel this reactor.")
        # reflect the configured model on the pill without firing a rebuild for the default
        if _Vars.config.get("KalmanModel", "scalar") == "cv":
            self.cvKalBtn.setChecked(True)
        return tab

    def toggleModLED(self, checked) -> None:
        self.modLedBtn.setStyleSheet(PILL_ACTIVE if checked else PILL_IDLE)
        self.modLedBtn.setText("Modulated LED: On" if checked else "Modulated LED")
        dev = _Vars.activeDevices.get(self.currentPort)
        if dev is None:                       # no reactor yet — UI state only
            return
        dev["OutgoingMessage"][0] = 1 if checked else 0   # LED_mode: modulated dark-subtract
        dev["DeviceVariables"]["SendMessage"] = True
        self._syncModeMenu()

    def toggleMcuAvg(self, checked) -> None:
        try:    n = max(1, min(255, int(float(self.mcuAvgN.text()))))
        except Exception: n = 1
        self.mcuAvgBtn.setStyleSheet(PILL_ACTIVE if checked else PILL_IDLE)
        self.mcuAvgBtn.setText(f"MCU Averaging: {n}" if checked else "MCU Averaging")
        dev = _Vars.activeDevices.get(self.currentPort)
        if dev is None:
            return
        dev["OutgoingMessage"][52] = n if checked else 1   # byte 52: N-averaging
        dev["DeviceVariables"]["SendMessage"] = True
        self._syncModeMenu()

    def applyKalman(self) -> None:
        try:
            Q = float(self.kalQ.text()); R = float(self.kalR.text())
            k = _Vars.activeDevices[self.currentPort]["ML"]["Kalman"]
            k.Q = Q; k.R = R
            self.betaStatus.setText(f"Kalman Q={Q:g} R={R:g} applied to port {self.currentPort}")
        except Exception as e:
            self.betaStatus.setText(f"Bad Kalman value: {e}")

    def toggleCVKalman(self, checked) -> None:
        """Swap the OD filter between the scalar random-walk and the 2-state
        constant-velocity model on every port (rebuilds the per-device filters)."""
        self.cvKalBtn.setStyleSheet(PILL_ACTIVE if checked else PILL_IDLE)
        self.cvKalBtn.setText("CV Kalman: On" if checked else "CV Kalman (2-state)")
        try:    Q = float(self.kalQ.text())
        except Exception: Q = None
        try:    R = float(self.kalR.text())
        except Exception: R = None
        comm = getattr(_Vars, "comm", None)
        if comm is None or not hasattr(comm, "set_kalman_model"):
            self.betaStatus.setText("No comms instance — cannot switch filter model.")
            return
        comm.set_kalman_model("cv" if checked else "scalar", Q=Q, R=R)
        self.betaStatus.setText(
            f"OD filter: {'CV (2-state)' if checked else 'scalar random-walk'} — all ports rebuilt.")
        self._syncModeMenu()

    def startExperiment(self) -> None:
        """Experiment Conditions (per-port): switch the SELECTED reactor's OD control
        to Hysteresis now (using its Control Panel OD target/band already in
        ControlValues). The experiment first WAITS until current OD ≥ target OD, then
        runs the entered countdown and auto-switches that reactor to Clamp. Each port
        runs its own independent experiment; clicking again on a port that is already
        running cancels just that one (leaving it in Hysteresis)."""
        port = self.currentPort
        if port is None or port not in _Vars.activeDevices:
            self.exptStatus.setText("No active reactor to run the experiment on.")
            return
        if port in self._expt:                       # already running on THIS port = cancel it
            self._cancelExperiment(port)
            return
        try:
            dur = max(1, int(float(self.exptDuration.text())))
        except Exception:
            dur = 400
            self.exptDuration.setText("400")
        # Hysteresis now (control-panel target/band drive utils.ODHyst on the comms side).
        self.changeODState(state=1, port=port)
        timer = QTimer(self)
        timer.setInterval(1000)
        timer.timeout.connect(lambda p=port: self._expt_tick(p))   # bind the port into the callback
        # armed=False -> still waiting for OD to reach target; remaining starts once armed.
        self._expt[port] = {"timer": timer, "remaining": dur, "duration": dur, "armed": False}
        self._expt_last.pop(port, None)
        timer.start()
        self._syncExperimentCard()

    def _expt_tick(self, port) -> None:
        st = self._expt.get(port)
        if st is None:
            return
        dev = _Vars.activeDevices.get(port)
        if dev is None:                              # reactor vanished — abort cleanly
            st["timer"].stop(); self._expt.pop(port, None)
            self._expt_last[port] = f"Reactor {port} disconnected — experiment aborted."
            if port == self.currentPort:
                self._syncExperimentCard()
            return
        if not st["armed"]:
            # Arming gate: hold in hysteresis until current OD reaches the target.
            od_list = dev["OpticalDensity"]["OD"]
            tgt = dev["DeviceVariables"]["ControlValues"]["OD"][0]
            cur = od_list[-1] if od_list else None
            if cur is not None and cur >= tgt:
                st["armed"] = True
                st["remaining"] = st["duration"]     # start the countdown now
        else:
            st["remaining"] -= 1
            if st["remaining"] <= 0:
                st["timer"].stop()
                self._expt.pop(port, None)
                if port == self.currentPort:
                    self.changeODState(state=2, port=port)        # updates pills too
                else:                                             # not selected — write directly
                    dev["OutgoingMessage"][32] = 2
                    dev["DeviceVariables"]["SendMessage"] = True
                self._expt_last[port] = f"Switched to Clamp OD on port {port}."
        if port == self.currentPort:
            self._syncExperimentCard()

    def _cancelExperiment(self, port) -> None:
        st = self._expt.pop(port, None)
        if st is not None:
            st["timer"].stop()
        self._expt_last[port] = f"Experiment cancelled — port {port} left in Hysteresis."
        if port == self.currentPort:
            self._syncExperimentCard()

    def _syncExperimentCard(self) -> None:
        """Point the Experiment Conditions card at the currently-selected reactor:
        show its live countdown if running, else its last result / the default help."""
        if not hasattr(self, "_expt"):     # search tab not built yet
            return
        port = self.currentPort
        st = self._expt.get(port) if port is not None else None
        if st is not None:
            self.exptBtn.setStyleSheet(PILL_ACTIVE)
            self.exptBtn.setText("Cancel (Hyst→Clamp)")
            dev = _Vars.activeDevices.get(port, {})
            try:
                manual = int(dev["OutgoingMessage"][34].item()) == 0
            except Exception:
                manual = False
            warn = "  ⚠ auto pumps paused (Manual overrides on)" if manual else ""
            if not st["armed"]:
                try:
                    od_list = dev["OpticalDensity"]["OD"]
                    cur = od_list[-1] if od_list else float("nan")
                    tgt = dev["DeviceVariables"]["ControlValues"]["OD"][0]
                    self.exptStatus.setText(
                        f"Hysteresis on port {port} — waiting for OD ≥ target "
                        f"(cur {cur:.3f} / tgt {tgt:.3f}); Clamp {st['duration']}s after reached.{warn}")
                except Exception:
                    self.exptStatus.setText(
                        f"Hysteresis on port {port} — waiting for OD to reach target.{warn}")
            else:
                self.exptStatus.setText(
                    f"OD target reached on port {port} — switching to Clamp in {st['remaining']}s.{warn}")
        else:
            self.exptBtn.setStyleSheet(PILL_IDLE)
            self.exptBtn.setText("Hysteresis → Clamp")
            self.exptStatus.setText(self._expt_last.get(port, self._expt_help))

    def setOverridesEnabled(self, on) -> None:
        """Manual waste/fresh override switch.
        Green (on): pause the MCU's auto pump control (byte 34 = 0) so the overrides
                    actually drive CH3/CH4, and enable the input fields.
        Red (off):  hand the pumps back to turbidostat auto control (byte 34 = 1 if
                    turbidostat is configured), and lock the fields."""
        on = bool(on)
        for w in (self.wasteMotor, self.freshMotor):
            w.setEnabled(on)
        # No reactor yet (GUI opened before any connected) — nothing to write.
        port = self.currentPort
        if port is None or port not in _Vars.activeDevices:
            return
        dev = _Vars.activeDevices[port]
        dev["OutgoingMessage"][34] = 0 if on else (1 if _Vars.config.get("Turbidostat", False) else 0)
        dev["DeviceVariables"]["SendMessage"] = True
        self._syncModeMenu()

    def toggleTargetTemp(self, checked) -> None:
        self.showTargetTemp = checked
        self.targetTempBtn.setStyleSheet(PILL_ACTIVE if checked else PILL_IDLE)
        self.targetTempBtn.setText("Target Temp: On" if checked else "Target Temp")
        self._syncModeMenu()
        self.updateGraphs(self.currentPort)

    def toggleTargetOD(self, checked) -> None:
        self.showTargetOD = checked
        self.targetODBtn.setStyleSheet(PILL_ACTIVE if checked else PILL_IDLE)
        self.targetODBtn.setText("Target OD: On" if checked else "Target OD")
        self._syncModeMenu()
        self.updateGraphs(self.currentPort)

    def setPinchState(self, state, port) -> None:
        """Set the pinch valve pair on the selected reactor (packet byte 35).
        The MCU drives high-side CH2/CH3 complementarily from this byte:
        0 = CH2 on / CH3 off, 1 = CH2 off / CH3 on."""
        self.pinchState = bool(state)
        self.pinch.setStyleSheet(PILL_ACTIVE if self.pinchState else PILL_IDLE)
        self.pinch.setText(f"Pinch State: {self.pinchState}")
        self._msSyncReactorControls()
        if port is not None and port in _Vars.activeDevices:
            _Vars.activeDevices[port]["OutgoingMessage"][35] = 1 if self.pinchState else 0
            _Vars.activeDevices[port]["DeviceVariables"]["SendMessage"] = True
        # Sync AFTER the write: _syncModeMenu reads byte 35 back, so running it
        # first would re-tick from the stale value and make the next toggle a no-op.
        self._syncModeMenu()

    def togglePinchState(self, port) -> None:
        """Flip the pinch pair (the Control Panel button; the Modes menu sets it
        to an explicit value instead)."""
        self.setPinchState(not self.pinchState, port)


    # ---- extra tabs ----
    def _plot_card(self, canvas):
        card, cl = self._card("")
        cl.setContentsMargins(10, 10, 10, 10)
        cl.addWidget(canvas)
        return card

    def _build_debug_tab(self):
        """Per-reactor diagnostics: main+ref (twin axis), scatter ratios, raw ref."""
        self.dbgMainRef = PlotCanvas(width=8, height=3.0, title="Main / Ref", xL="Time (s)", yL="Main")
        self.dbgScatter = PlotCanvas(width=8, height=3.0, title="Scatter",     xL="Time (s)", yL="Scatter")
        self.dbgRawRef  = PlotCanvas(width=8, height=3.0, title="Raw Ref",      xL="Time (s)", yL="Ref (raw)")
        # Simulator-only: the same raw signals reconstructed through three filtering
        # domains, against the twin's ground truth. Empty on real hardware.
        self.dbgFilters = PlotCanvas(width=8, height=3.0, title="Filter comparison",
                                     xL="Time (s)", yL="OD")
        tab = QWidget(); v = QVBoxLayout(tab); v.setSpacing(14)
        for cv in (self.dbgMainRef, self.dbgScatter, self.dbgRawRef, self.dbgFilters):
            v.addWidget(self._plot_card(cv))
        return tab

    def _build_multiplex_tab(self):
        """All reactors overlaid: OD vs time and Temperature vs time."""
        self.multiOD   = PlotCanvas(width=8, height=4.0, title="OD — all reactors",          xL="Time (s)", yL="OD")
        self.multiTemp = PlotCanvas(width=8, height=4.0, title="Temperature — all reactors", xL="Time (s)", yL="°C")
        tab = QWidget(); v = QVBoxLayout(tab); v.setSpacing(14)
        for cv in (self.multiOD, self.multiTemp):
            v.addWidget(self._plot_card(cv))
        return tab

    def _build_history_tab(self):
        """Full-run history view backed by np.memmap over the per-port .bin file.
        The live plots only show the DRAM rolling window (WindowSeconds); this
        tab shows the ENTIRE run, downsampled via min/max stride so a multi-day
        run still renders in ms even though it never touches RAM.

        Two panels:
          1. OD + Kalman on left axis, avg_temp on the right axis (coral).
             One glance shows OD trajectory vs how tightly the heater held setpoint.
          2. Main + Ref·β on left axis, Scatter on the right axis (mirrors the
             Debug tab's Main/Ref diagnostic — useful for post-hoc calibration checks).
        """
        self.histOD      = PlotCanvas(width=8, height=4.0, title="OD — full run",
                                       xL="Time (s)", yL="OD")
        self.histMainRef = PlotCanvas(width=8, height=4.0, title="Main / Ref·β — full run",
                                       xL="Time (s)", yL="Main · Ref·β")
        tab = QWidget()
        v = QVBoxLayout(tab); v.setSpacing(14)
        # Header row: port selector, sample count, refresh button.
        row = QHBoxLayout(); row.setSpacing(8)
        row.addWidget(QLabel("Reactor"))
        self.histPortSelector = QComboBox(); self.histPortSelector.setFixedWidth(110)
        for p in _Vars.ports:
            self.histPortSelector.addItem(str(p))
        row.addWidget(self.histPortSelector)
        self.histStatus = QLabel("(no data yet)"); self.histStatus.setObjectName("subValue")
        row.addStretch(); row.addWidget(self.histStatus)
        self.histRefresh = QPushButton("Refresh"); self.histRefresh.setFixedWidth(100)
        row.addWidget(self.histRefresh)
        v.addLayout(row)
        for cv in (self.histOD, self.histMainRef):
            v.addWidget(self._plot_card(cv))
        self.histPortSelector.currentIndexChanged.connect(self._refresh_history)
        self.histRefresh.clicked.connect(self._refresh_history)
        return tab

    def _on_tab_changed(self, index):
        """Auto-refresh the History plot whenever the operator lands on it."""
        if self.tabs.widget(index) is getattr(self, "historyTab", None):
            self._refresh_history()

    def _refresh_history(self):
        """Load the current port's history .bin via np.memmap and redraw both
        history plots. Downsampled to ~5000 min/max pairs per line so an
        arbitrarily long run stays snappy."""
        try:
            # Keep the port selector in sync with newly-added devices.
            existing = {self.histPortSelector.itemText(i) for i in range(self.histPortSelector.count())}
            for p in _Vars.ports:
                if str(p) not in existing:
                    self.histPortSelector.addItem(str(p))
            txt = self.histPortSelector.currentText()
            if not txt:
                self.histStatus.setText("(no reactor)")
                return
            port = int(txt)
            mm = load_history_memmap(port)
            if mm is None or len(mm) == 0:
                self.histStatus.setText(
                    "(no history yet — flush hasn't run, or Reset Data cleared it)")
                self.histOD.update_lines([], title="OD — full run", yl="OD")
                self.histMainRef.update_lines([], title="Main / Ref·β — full run", yl="Main · Ref·β")
                return

            n_rows = len(mm)
            t     = np.asarray(mm["time_sw"])
            od    = np.asarray(mm["od"])
            kod   = np.asarray(mm["kod"])
            tmp_a = np.asarray(mm["avg_temp"])
            main  = np.asarray(mm["avg_main_1"])
            ref   = np.asarray(mm["avg_ref_1"])
            scat  = np.asarray(mm["avg_scatter_1"])

            # β for Ref·β overlay: BetaFinal if latched, else mean of the live
            # Beta window (same rule the Debug tab uses).
            dev = _Vars.activeDevices.get(port, {})
            beta_final = dev.get("Coefficients", {}).get("BetaFinal", 0.0)
            beta_list  = dev.get("Coefficients", {}).get("Beta", [])
            beta = beta_final if beta_final else (float(np.mean(beta_list)) if beta_list else 1.0)
            refb = ref * beta

            # Envelope downsample to ~5000 min/max pairs per line. Every series
            # runs off the same t so the twin-axis series and the main-axis
            # series stay time-aligned.
            xs, ys_od      = self._envelope(t, od,    target=5000)
            _,  ys_kod     = self._envelope(t, kod,   target=5000)
            _,  ys_temp    = self._envelope(t, tmp_a, target=5000)
            xs2, ys_main   = self._envelope(t, main,  target=5000)
            _,   ys_refb   = self._envelope(t, refb,  target=5000)
            _,   ys_scat   = self._envelope(t, scat,  target=5000)

            self.histOD.update_lines(
                [(xs, ys_od,  ACCENT,    "OD"),
                 (xs, ys_kod, "#e0a13a", "Kalman")],
                title="OD — full run", yl="OD",
                twin={"x": xs, "y": ys_temp, "color": TEMP_TWIN,
                      "label": "avg_temp", "ylabel": "°C"},
            )
            self.histMainRef.update_lines(
                [(xs2, ys_main, ACCENT,    "Main (0x45)"),
                 (xs2, ys_refb, "#e0a13a", "Ref·β")],
                title="Main / Ref·β — full run", yl="Main · Ref·β",
                twin={"x": xs2, "y": ys_scat, "color": "#9b8cff",
                      "label": "Scatter", "ylabel": "Scatter"},
            )

            bytes_used = n_rows * _BIN_DTYPE.itemsize
            self.histStatus.setText(
                f"port {port} · {n_rows:,} samples · {t[-1]-t[0]:.0f} s · "
                f"{bytes_used/1024/1024:.1f} MiB (memmapped, not loaded)  ·  β={beta:.4f}")
        except Exception as e:
            self.histStatus.setText(f"history load failed: {e}")

    @staticmethod
    def _envelope(x, y, target=5000):
        """Min/max stride downsample. For each of `target` buckets, emit the
        bucket's min and its max at the bucket's start x. Preserves visual
        peaks that a plain decimation would drop, without needing per-pixel
        knowledge of the plot width."""
        n = len(y)
        if n == 0:
            return x, y
        if n <= target * 2:
            return x, y     # already small — no downsample
        stride = max(1, n // target)
        m = (n // stride) * stride
        xs = x[:m:stride]
        yc = y[:m].reshape(-1, stride)
        mn = yc.min(axis=1); mx = yc.max(axis=1)
        xs_pairs = np.repeat(xs, 2)
        ys_pairs = np.column_stack([mn, mx]).ravel()
        return xs_pairs, ys_pairs

    def _xy(self, port, vals):
        """Snapshot (time, vals) for a port and trim to a common length (thread-safe).
        Uses _snap → RingBuffer.snapshot() is a single numpy memcpy; list(rb) would
        iterate every element in Python (much slower for 21600-sample windows)."""
        t = _snap(_Vars.activeDevices[port]["Time"]["Software"])
        y = _snap(vals)
        n = min(len(t), len(y))
        return t[:n], y[:n]

    def updateGraphs(self, port) -> None:
        # Started with no devices? Adopt the first one that connects, otherwise
        # just keep the selector/header in sync and wait (UI stays open and live).
        if port is None or port not in _Vars.activeDevices:
            if _Vars.ports:
                self._adoptPort()
                port = self.currentPort
            if port is None or port not in _Vars.activeDevices:
                self._syncPorts()
                return
        dev = _Vars.activeDevices[port]
        # live value tiles
        try:
            if dev["SensorsAvg"]["Temp"]:
                cur = dev["SensorsAvg"]["Temp"][-1]
                tgt = dev["DeviceVariables"]["ControlValues"]["Temperature"][0]
                off = dev["Coefficients"]["TemperatureOffset"]
                col = OK if abs(cur - tgt) <= 0.5 else TEXT
                self.currentTemp.setStyleSheet(f"font-size:30px; font-weight:700; color:{col};")
                self.currentTemp.setText(f"{cur:.2f} °C")
                self.currentTempTarg.setText(f"Target {tgt:.1f} °C   ·   offset {off:+.1f}")
        except Exception:
            pass
        try:
            if dev["OpticalDensity"]["OD"]:
                cur = dev["OpticalDensity"]["OD"][-1]
                tgt = dev["DeviceVariables"]["ControlValues"]["OD"][0]
                col = OK if abs(cur - tgt) <= 0.05 else TEXT
                self.currentOD.setStyleSheet(f"font-size:30px; font-weight:700; color:{col};")
                self.currentOD.setText(f"{cur:.3f}")
                self.currentODTarg.setText(f"Target {tgt:.2f}")
        except Exception:
            pass
        # run-length clock (always visible) — last software-time sample of the selected reactor.
        try:
            sw = dev["Time"]["Software"]
            secs = int(sw[-1]) if sw else 0
            hh, rem = divmod(max(secs, 0), 3600); mm, ss = divmod(rem, 60)
            self.runClock.setText(f"RUN  {hh:02d}:{mm:02d}:{ss:02d}")
        except Exception:
            pass
        # connection status header — colour by data liveness, not just loss%.
        # Grey = never received, Red = no data for >STALE_S, Green = receiving.
        STALE_S = 20.0
        now = time.time()
        for p, dot in self.portDots.items():
            d = _Vars.activeDevices.get(p, {})
            last = d.get("_lastRx", 0)
            if not last:
                col = MUTED                       # never connected
            elif (now - last) > STALE_S:
                col = DANGER                      # stale — no packets recently
            else:
                col = OK                          # live
            dot.setStyleSheet(f"color:{col}; font-size:12px;")
        try:
            rx, lost = dev["PacketLoss"]
            self.lossLabel.setText(f"loss {utils.packet_loss(rx, lost):.1f}%")
        except Exception:
            pass

        # Only redraw the plots on the currently-visible tab (saves CPU).
        w = self.tabs.currentWidget()
        if w is self.controlTab:
            tgt_temp = dev["DeviceVariables"]["ControlValues"]["Temperature"][0]
            tgt_od   = dev["DeviceVariables"]["ControlValues"]["OD"][0]
            self.graphTemp.update_plot(idx=0, port=port, show_target=self.showTargetTemp, target=tgt_temp,
                                       twin={"seq": dev["DeviceVariables"]["RelayState"], "color": "#9b8cff",
                                             "label": "Relay", "ylabel": "Relay (heater on/off)"})
            self.graphOD.update_plot(idx=1, port=port, show_target=self.showTargetOD, target=tgt_od,
                                     twin={"seq": dev["DeviceVariables"]["MotorState"], "color": "#9b8cff",
                                           "label": "Motor", "ylabel": "Motor state (0-255)"})
        elif w is self.debugTab:
            # β = final blank ratio once latched, else running mean of the beta window.
            beta_final = dev["Coefficients"]["BetaFinal"]
            beta_list  = dev["Coefficients"]["Beta"]
            beta = beta_final if beta_final else (float(np.mean(beta_list)) if beta_list else 1.0)
            # Element-wise aligned snapshot of the three signals + time (thread-safe).
            # _snap → numpy array; slicing and array-math below are then vectorised.
            t    = _snap(dev["Time"]["Software"])
            main = _snap(dev["SensorsAvg"]["Main"][1])
            ref  = _snap(dev["SensorsAvg"]["Ref"][1])
            scat = _snap(dev["SensorsAvg"]["Scatter"][1])
            n = min(len(t), len(main), len(ref), len(scat))
            t, main, ref, scat = t[:n], main[:n], ref[:n], scat[:n]
            # Top plot: Main and Ref×β share the left axis (similar magnitude); Scatter twin.
            refb = ref * beta   # numpy vectorised, was: [r * beta for r in ref]
            self.dbgMainRef.update_lines(
                [(t, main, ACCENT, "Main (0x45)"), (t, refb, "#e0a13a", "Ref×β")],
                "Main / Ref×β  (Scatter twin)", "Main · Ref×β",
                twin={"x": t, "y": scat, "color": "#9b8cff", "label": "Scatter", "ylabel": "Scatter"})
            # Bottom plot: (β·Ref)/Main on the left axis, Scatter/Ref on the twin.
            # np.divide's `where` keeps the /0 case NaN without a Python loop.
            with np.errstate(divide="ignore", invalid="ignore"):
                od_ratio = np.where(main != 0, beta * ref / main, np.nan)
                scat_ref = np.where(ref  != 0, scat / ref,          np.nan)
            self.dbgScatter.update_lines(
                [(t, od_ratio, ACCENT, "β·Ref/Main")],
                "β·Ref/Main  (Scatter/Ref twin)", "β · Ref / Main",
                twin={"x": t, "y": scat_ref, "color": "#e0a13a", "label": "Scatter/Ref", "ylabel": "Scatter / Ref"})
            # Raw (un-averaged) reference photodiode reading vs time.
            xrr, yrr = self._xy(port, dev["SensorsRaw"]["Ref"][1])
            self.dbgRawRef.update_lines([(xrr, yrr, ACCENT, "Ref raw (0x45)")], "Raw Ref", "Ref (raw)")
            # Filter-comparison (twin only): OD reconstructed three ways vs truth.
            fc = dev.get("FilterCompare")
            if fc and len(fc["A"]):
                series = []
                xt, yt = self._xy(port, fc["True"])
                if len(yt):
                    series.append((xt, yt, TEXT, "TRUTH (twin)"))
                for key, col, lab in (("A", ACCENT,    "A: filter OD ratio"),
                                      ("C", "#e0a13a", "C: filter MAIN only"),
                                      ("D", "#9b8cff", "D: filter MAIN+REF"),
                                      ("E", "#3fb950", "E: coupled MAIN+SCATTER")):
                    xa, ya = self._xy(port, fc[key])
                    series.append((xa, ya, col, lab))
                self.dbgFilters.update_lines(series, "Filter comparison — OD reconstructed 3 ways",
                                             "OD", alpha=0.6)
            else:
                self.dbgFilters.update_lines([], "Filter comparison (simulator/twin only)", "OD",
                                             alpha=0.6)
        elif w is self.multiTab:
            od_series, temp_series = [], []
            for i, p in enumerate(_Vars.ports):
                d = _Vars.activeDevices[p]
                c = MULTI_COLORS[i % len(MULTI_COLORS)]
                xo, yo = self._xy(p, d["OpticalDensity"]["OD"]);   od_series.append((xo, yo, c, str(p)))
                xt, yt = self._xy(p, d["SensorsAvg"]["Temp"]);     temp_series.append((xt, yt, c, str(p)))
            self.multiOD.update_lines(od_series, "OD — all reactors", "OD")
            self.multiTemp.update_lines(temp_series, "Temperature — all reactors", "°C")
        # Keep the auto-β readout current for the selected reactor.
        self._refresh_beta_value(port)
        self._syncPorts()

    def closeEvent(self, event) -> None:
        self._autoscan_on = False          # let the background discovery thread exit
        self._msStopLive()                 # stop the live-view thread
        self._dbStopLive()                 # stop the Dobot poller + disconnect the arm
        for port in _Vars.ports:
            device = _Vars.activeDevices[port]
            try:
                utils.dataSave(
                    device["SensorsRaw"]["Main"][0],    device["SensorsRaw"]["Main"][1],    device["SensorsRaw"]["Main"][2],
                    device["SensorsRaw"]["Ref"][0],     device["SensorsRaw"]["Ref"][1],     device["SensorsRaw"]["Ref"][2],
                    device["SensorsRaw"]["Scatter"][0], device["SensorsRaw"]["Scatter"][1], device["SensorsRaw"]["Scatter"][2],
                    device["SensorsAvg"]["Main"][0],    device["SensorsAvg"]["Main"][1],    device["SensorsAvg"]["Main"][2],
                    device["SensorsAvg"]["Ref"][0],     device["SensorsAvg"]["Ref"][1],     device["SensorsAvg"]["Ref"][2],
                    device["SensorsAvg"]["Scatter"][0], device["SensorsAvg"]["Scatter"][1], device["SensorsAvg"]["Scatter"][2],
                    device["OpticalDensity"]["OD"],     device["OpticalDensity"]["KOD"],
                    device["MLPredictions"]["RefPred"],  device["MLPredictions"]["OD"],
                    device["SensorsRaw"]["Temp"],       device["SensorsAvg"]["Temp"],
                    device["Coefficients"]["Beta"],     device["DeviceVariables"]["RelayState"],
                    device["DeviceVariables"]["MotorState"],
                    device["Time"]["Hardware"],         device["Time"]["Software"],
                    port=port,
                )
            except Exception as e:
                print(f"[ERROR] UI CLOSING ERROR: UNSAVED DATA | PORT: {port}: {e}")
        # Drain the async CSV queue synchronously before the process exits, so
        # any flush the worker hadn't picked up yet still hits disk.
        try:
            stopFlusher(drain=True)
        except Exception as e:
            print(f"[ERROR] flusher shutdown: {e}")


class PlotCanvas(FigureCanvas):
    def __init__(self, width=8, height=5, dpi=100, title="", xL="", yL=""):
        self.title, self.xL, self.yL = title, xL, yL
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor(CARD)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor(BG)
        self.ax2 = None        # optional twin (right) axis
        self._follow = True    # autoscale to latest data until the user zooms; double-click resumes
        self._annotations = [] # operator-placed markers: list of (x, y) in self.axes data coords
        self._annot_artists = []
        # Incremental-redraw cache — a per-tick tick would otherwise clear the axes
        # and re-plot the full arrays, which becomes the dominant CPU cost in long
        # runs. We only rebuild the axes when the STRUCTURE of what's being drawn
        # changes (idx switch, twin toggled, series set changes); otherwise the
        # existing Line2D artists get set_data() and matplotlib repaints in place.
        self._render_key   = None   # structural fingerprint of the last render
        self._line_cache   = {}     # id -> Line2D on the main axis
        self._twin_line    = None   # Line2D on ax2 (or None if no twin)
        self._target_line  = None   # axhline artist for the target (or None)
        FigureCanvas.__init__(self, self.fig)
        # Interaction: scroll wheel = cursor-centred zoom, double-click = back to live,
        # left-click = drop a red marker on the nearest data point, 'd'/Del = remove last, 'c' = clear.
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("key_press_event", self._on_key)
        self.plot()

    def _on_scroll(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        ax = event.inaxes
        self._follow = False                      # stop snapping back on the next redraw
        f = 0.8 if event.button == "up" else 1.25  # wheel up = zoom in
        x, y = event.xdata, event.ydata
        xlo, xhi = ax.get_xlim(); ylo, yhi = ax.get_ylim()
        ax.set_xlim(x - (x - xlo) * f, x + (xhi - x) * f)
        ax.set_ylim(y - (y - ylo) * f, y + (yhi - y) * f)
        self.draw_idle()

    def _on_press(self, event):
        if getattr(event, "dblclick", False):     # double-click = resume live autoscale
            # the first click of this double-click already dropped a marker — undo it.
            if self._annotations:
                self._annotations.pop()
                self._draw_annotations()
            self._follow = True
            self.axes.autoscale()
            if self.ax2 is not None:
                self.ax2.autoscale()
            self.draw_idle()
            return
        if event.button == 1 and event.inaxes is not None and event.x is not None:
            pt = self._nearest_point(event)       # snap to the closest plotted data point
            if pt is not None:
                self._annotations.append(pt)
                self._draw_annotations()
                self.draw_idle()

    def _nearest_point(self, event):
        """Closest vertex (in pixels) among the data lines on EITHER axis to the click.
        Returns (x, y, axkey) so the marker is drawn on the axis it belongs to —
        otherwise a twin-axis point would land at the wrong height on the left axis."""
        best, bestd, bestax = None, None, None
        for axkey, ax in (("axes", self.axes), ("ax2", self.ax2)):
            if ax is None:
                continue
            for line in ax.get_lines():
                lbl = line.get_label()
                if lbl.startswith("_") or lbl == "Target":   # skip markers / target rule
                    continue
                xd = np.asarray(line.get_xdata(), dtype=float)
                yd = np.asarray(line.get_ydata(), dtype=float)
                if xd.size == 0:
                    continue
                ok = np.isfinite(xd) & np.isfinite(yd)
                if not ok.any():
                    continue
                xd, yd = xd[ok], yd[ok]
                disp = ax.transData.transform(np.column_stack([xd, yd]))
                d2 = (disp[:, 0] - event.x) ** 2 + (disp[:, 1] - event.y) ** 2
                i = int(np.argmin(d2))
                if bestd is None or d2[i] < bestd:
                    bestd, best, bestax = d2[i], (float(xd[i]), float(yd[i])), axkey
        if best is None:
            return None
        return (best[0], best[1], bestax)

    def _on_key(self, event):
        if event.key in ("d", "delete", "backspace"):     # remove most recent marker
            if self._annotations:
                self._annotations.pop()
                self._draw_annotations()
                self.draw_idle()
        elif event.key in ("c", "escape"):                # clear all markers
            if self._annotations:
                self._annotations.clear()
                self._draw_annotations()
                self.draw_idle()

    def _draw_annotations(self):
        """Re-render the operator markers. Called after every redraw (axes get cleared)
        and on add/remove, so the artist list always matches self._annotations."""
        for a in self._annot_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._annot_artists = []
        for ann in self._annotations:
            x, y = ann[0], ann[1]
            axkey = ann[2] if len(ann) > 2 else "axes"
            ax = self.ax2 if axkey == "ax2" else self.axes
            if ax is None:        # twin gone (e.g. plot has no twin this redraw) — skip
                continue
            m, = ax.plot([x], [y], marker="o", markersize=6, linestyle="None",
                         markerfacecolor="#ff4d4d", markeredgecolor="white",
                         markeredgewidth=0.6, zorder=6, label="_annot")
            txt = ax.annotate(f"({x:.1f}, {y:.3g})", (x, y),
                              textcoords="offset points", xytext=(6, 6),
                              fontsize=7, fontweight="bold", color="#ff4d4d", zorder=7)
            self._annot_artists += [m, txt]

    def _style_axes(self, title, yl):
        self.axes.clear()
        self.fig.patch.set_facecolor(CARD)
        self.axes.set_facecolor(BG)
        if title:
            self.axes.set_title(title, color=TEXT, fontsize=11, fontweight="bold", loc="left")
        self.axes.set_xlabel(self.xL, color=MUTED, fontsize=9)
        self.axes.set_ylabel(yl, color=MUTED, fontsize=9)
        self.axes.tick_params(axis='x', colors=MUTED, labelsize=8)
        self.axes.tick_params(axis='y', colors=MUTED, labelsize=8)
        for s in self.axes.spines.values():
            s.set_color(CARD_EDGE)
        self.axes.grid(True, color=CARD_EDGE, alpha=0.4, linewidth=0.6)

    def plot(self):
        self._style_axes(self.title, self.yL)
        self.fig.tight_layout()
        self.draw()

    def _render(self, spec, target=None, twin=None, title=None, yl=None, legend_ncol=1):
        """Draw a set of lines with an incremental fast path.

        spec        list of dicts: {id, x, y, color, alpha, linewidth, label}
        target      float or None — horizontal setpoint line on the main axis
        twin        dict {x, y, color, label, ylabel} or None — right-axis line
        title, yl   axes labels (only applied when the axes are (re)built)

        Fast path: when the STRUCTURE (which lines, twin on/off, target on/off,
        title/ylabel) hasn't changed, we skip axes.clear() / tight_layout() /
        legend() / new-Line2D allocation, and just push new arrays into the
        cached Line2D artists via set_data(). That drops the per-frame cost
        from O(clear+plot+layout) to O(set_data+draw)."""
        keep = not self._follow
        if keep:
            xlim, ylim = self.axes.get_xlim(), self.axes.get_ylim()
            ax2lim = self.ax2.get_ylim() if self.ax2 is not None else None

        fp = (
            title or self.title,
            yl or self.yL,
            tuple((s["id"], s["color"], s.get("alpha", 0.9),
                   s.get("linewidth", 1.6), s.get("label", ""))
                  for s in spec),
            bool(twin),
            (twin["color"], twin.get("label", ""), twin.get("ylabel", "")) if twin else None,
            target is not None,
            legend_ncol,
        )

        if fp != self._render_key:
            # Structural change → full rebuild. This is the slow path but only
            # runs when the caller changes what's being plotted (idx switch,
            # twin toggle, series set change), not every tick.
            self._style_axes(title or self.title, yl or self.yL)
            self._line_cache = {}
            self._twin_line = None
            self._target_line = None
            if self.ax2 is not None:
                self.ax2.remove()
                self.ax2 = None
            for s in spec:
                (line,) = self.axes.plot(
                    [], [], color=s["color"], alpha=s.get("alpha", 0.9),
                    linewidth=s.get("linewidth", 1.6), label=s.get("label", ""),
                )
                self._line_cache[s["id"]] = line
            if target is not None:
                self._target_line = self.axes.axhline(
                    target, color=TARGET, linestyle="--", linewidth=1.4,
                    alpha=0.9, label="Target",
                )
            if twin is not None:
                self.ax2 = self.axes.twinx()
                self.ax2.set_facecolor("none")
                (self._twin_line,) = self.ax2.plot(
                    [], [], color=twin["color"], alpha=0.9, linewidth=1.6,
                    label=twin.get("label", ""),
                )
                self.ax2.set_ylabel(twin.get("ylabel", ""), color=twin["color"], fontsize=9)
                self.ax2.tick_params(axis="y", colors=twin["color"], labelsize=8)
            has_label = (any(bool(s.get("label")) for s in spec)
                         or (twin is not None) or (target is not None))
            if has_label:
                self.axes.legend(loc="upper left", fontsize=8, frameon=False,
                                 labelcolor=MUTED, ncol=legend_ncol)
            self._render_key = fp
            self.fig.tight_layout()

        # Fast path — push data into the cached artists.
        for s in spec:
            line = self._line_cache.get(s["id"])
            if line is not None:
                line.set_data(s["x"], s["y"])
        if self._target_line is not None and target is not None:
            self._target_line.set_ydata([target, target])
        if self._twin_line is not None and twin is not None:
            self._twin_line.set_data(twin["x"], twin["y"])

        if keep:
            self.axes.set_xlim(xlim); self.axes.set_ylim(ylim)
            if self.ax2 is not None and ax2lim is not None:
                self.ax2.set_ylim(ax2lim)
        else:
            # Recompute limits from the current line data (fast — data is
            # already bounded by the DRAM rolling window).
            self.axes.relim()
            self.axes.autoscale_view()
            if self.ax2 is not None:
                self.ax2.relim()
                self.ax2.autoscale_view()

        self._draw_annotations()
        self.draw_idle()

    def update_plot(self, idx, port, show_target=False, target=None, twin=None):
        """Primary plot. twin = optional dict {seq, color, label, ylabel} drawn on a right axis."""
        try:
            dev = _Vars.activeDevices[port]
            # Snapshot the shared buffers — the comms thread appends to them concurrently,
            # so take a numpy copy and trim to a common length to avoid mid-append races.
            # _snap → RingBuffer.snapshot() (contiguous memcpy) vs list() iteration.
            t = _snap(dev["Time"]["Software"])
            def paired(seq):
                y = _snap(seq)
                n = min(len(t), len(y))
                return t[:n], y[:n]
            if idx == 0:
                title, yl = "Temperature", "°C"
                xr, yr = paired(dev["SensorsRaw"]["Temp"])
                xa, ya = paired(dev["SensorsAvg"]["Temp"])
                spec = [
                    {"id": "raw", "x": xr, "y": yr, "color": ACCENT,    "alpha": 0.5,  "linewidth": 1.2, "label": "Raw"},
                    {"id": "avg", "x": xa, "y": ya, "color": "#e0a13a", "alpha": 0.7, "linewidth": 2,   "label": "Avg"},
                ]
            else:
                title, yl = "Optical Density", "OD"
                xo, yo = paired(dev["OpticalDensity"]["OD"])
                xk, yk = paired(dev["OpticalDensity"]["KOD"])
                spec = [
                    {"id": "od",     "x": xo, "y": yo, "color": ACCENT,    "alpha": 0.7, "linewidth": 2,   "label": "OD"},
                    {"id": "kalman", "x": xk, "y": yk, "color": "#e0a13a", "alpha": 0.7, "linewidth": 1.6, "label": "Kalman"},
                ]
            twin_data = None
            if twin is not None:
                tx, ty = paired(twin["seq"])
                twin_data = {"x": tx, "y": ty, "color": twin["color"],
                             "label": twin.get("label", ""), "ylabel": twin.get("ylabel", "")}
            self._render(spec, target=(target if show_target else None),
                         twin=twin_data, title=title, yl=yl, legend_ncol=1)
        except Exception as e:
            print(f"[ERROR] | PLOTTING ERROR idx={idx}: {e}")

    def update_lines(self, series, title="", yl="", twin=None, alpha=0.9):
        """Generic multi-line plot. series = list of (x, y, color, label).
        twin = optional dict {x, y, color, label, ylabel} drawn on a right axis.
        alpha = trace opacity; lower it when overlaying traces that sit on top of
        each other (e.g. the filter-comparison plot) so all of them stay readable."""
        try:
            # id includes color+label so a re-order or palette change is picked
            # up as a structural change; index alone would silently rewire.
            spec = [{"id": f"{i}:{color}:{label}", "x": x, "y": y, "color": color,
                     "alpha": alpha, "linewidth": 1.6, "label": label}
                    for i, (x, y, color, label) in enumerate(series)]
            twin_data = None
            if twin is not None:
                twin_data = {"x": twin["x"], "y": twin["y"], "color": twin["color"],
                             "label": twin.get("label", ""), "ylabel": twin.get("ylabel", "")}
            self._render(spec, target=None, twin=twin_data, title=title, yl=yl, legend_ncol=2)
        except Exception as e:
            print(f"[ERROR] | PLOT update_lines ({title}): {e}")


if __name__ == '__main__':
    utils.readConfig()
    from Comms import communications
    comm = communications()
    _Vars.comm = comm                       # expose for the GUI's Rescan Devices button
    _Vars.version = VERSION                  # expose for the run-metadata XML sidecar
    threading.Thread(target=comm.initStart, daemon=True).start()
    _Vars.initState = True
    # Open the GUI immediately — don't block on the first connect scan. The
    # window shows a "searching for reactors…" spinner and its background
    # auto-scan brings devices online live as they connect (adopting the first
    # one automatically). Feels responsive instead of a blank wait on launch.
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())
