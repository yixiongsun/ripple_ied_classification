import sys
import os

import pandas as pd
import qdarkstyle
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


REQUIRED_COLUMNS = {"subject_id", "session_id", "image_path"}


class ImageLabeler(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stratified Image Labeling Tool")
        self.resize(1100, 600)
        self.label_filter = "All"
        self.data = []
        self.filtered_data = []
        self.labels = {}
        self.classes = []
        self.missing_images = set()
        self.undo_stack = []
        self.is_dirty = False
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()

        # Left: list
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list_widget.itemSelectionChanged.connect(self.show_preview)
        self.list_widget.installEventFilter(self)
        # Right panel
        right_layout = QVBoxLayout()

        self.image_label = QLabel("Preview")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(450, 450)

        self.progress_label = QLabel("Progress: 0 / 0")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Stratified filters
        filter_layout = QHBoxLayout()

        self.subject_filter = QComboBox()
        self.subject_filter.currentTextChanged.connect(self.apply_filters)

        self.session_filter = QComboBox()
        self.session_filter.currentTextChanged.connect(self.apply_filters)

        filter_layout.addWidget(QLabel("Subject:"))
        filter_layout.addWidget(self.subject_filter)
        filter_layout.addWidget(QLabel("Session:"))
        filter_layout.addWidget(self.session_filter)

        self.label_filter_box = QComboBox()
        self.label_filter_box.currentTextChanged.connect(self.apply_filters)

        filter_layout.addWidget(QLabel("Label:"))
        filter_layout.addWidget(self.label_filter_box)

        # Buttons
        btn_layout = QHBoxLayout()

        load_btn = QPushButton("Load CSV")
        load_btn.clicked.connect(self.load_csv)

        class_btn = QPushButton("Set Classes")
        class_btn.clicked.connect(self.set_classes)

        next_btn = QPushButton("Next Unlabeled")
        next_btn.clicked.connect(self.go_to_next_unlabeled)

        save_btn = QPushButton("Save CSV")
        save_btn.clicked.connect(self.save_csv)

        btn_layout.addWidget(load_btn)
        btn_layout.addWidget(class_btn)
        btn_layout.addWidget(next_btn)
        btn_layout.addWidget(save_btn)

        right_layout.addLayout(filter_layout)
        right_layout.addWidget(self.image_label)
        right_layout.addWidget(self.progress_label)
        right_layout.addLayout(btn_layout)

        layout.addWidget(self.list_widget, 2)
        layout.addLayout(right_layout, 3)

        self.setLayout(layout)

    def eventFilter(self, source, event):
        if (
            source == self.list_widget
            and event.type() == QEvent.Type.KeyPress
            and self.handle_shortcut(event)
        ):
            return True
        return super().eventFilter(source, event)

    def handle_shortcut(self, event):
        key = event.key()
        if (
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
            and key == Qt.Key.Key_Z
        ):
            self.undo_last()
            return True
        if key == Qt.Key.Key_R:
            self.relabel_current()
            return True
        if self.classes and Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            index = key - Qt.Key.Key_1
            if index < len(self.classes):
                self.assign_label_keyboard(self.classes[index])
                return True
        return False

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        if not self.confirm_discard_changes():
            return

        try:
            df = pd.read_csv(path)
        except Exception as error:
            QMessageBox.critical(self, "Error", f"Could not read CSV:\n{error}")
            return

        missing_columns = REQUIRED_COLUMNS - set(df.columns)
        if missing_columns:
            names = ", ".join(sorted(missing_columns))
            QMessageBox.critical(self, "Error", f"CSV is missing columns: {names}")
            return

        has_labels = "label" in df.columns

        self.data = []
        self.labels = {}
        self.missing_images = set()
        self.undo_stack.clear()

        for _, row in df.iterrows():
            subject_id = str(row["subject_id"])
            session_id = str(row["session_id"])
            image_path = str(row["image_path"])

            if not os.path.exists(image_path):
                self.missing_images.add(image_path)

            entry = {
                "subject_id": subject_id,
                "session_id": session_id,
                "image_path": image_path,
            }
            self.data.append(entry)

            if has_labels and pd.notna(row["label"]):
                label = str(row["label"]).strip()
                if label:
                    self.labels[image_path] = label

        self.setup_filters()
        self.apply_filters()
        self.is_dirty = False

        available = len(self.data) - len(self.missing_images)
        msg = f"{available} images available from {len(self.data)} CSV rows."
        if self.missing_images:
            msg += f"\n{len(self.missing_images)} missing-image rows were retained but hidden."
        if has_labels:
            msg += f"\n{len(self.labels)} labels restored."

        QMessageBox.information(self, "Loaded", msg)

    def setup_filters(self):
        subjects = sorted(set(d["subject_id"] for d in self.data))
        sessions = sorted(set(d["session_id"] for d in self.data))
        labels = sorted(set(self.labels.values()))

        filters = (self.subject_filter, self.session_filter, self.label_filter_box)
        for filter_box in filters:
            filter_box.blockSignals(True)
        try:
            self.subject_filter.clear()
            self.subject_filter.addItem("All")
            self.subject_filter.addItems(subjects)

            self.session_filter.clear()
            self.session_filter.addItem("All")
            self.session_filter.addItems(sessions)

            self.label_filter_box.clear()
            self.label_filter_box.addItem("All")
            self.label_filter_box.addItem("Unlabeled")
            self.label_filter_box.addItems(labels)
        finally:
            for filter_box in filters:
                filter_box.blockSignals(False)

    def apply_filters(self):
        subject = self.subject_filter.currentText()
        session = self.session_filter.currentText()
        label_filter = self.label_filter_box.currentText()

        self.filtered_data = []

        for d in self.data:
            if d["image_path"] in self.missing_images:
                continue
            if subject != "All" and d["subject_id"] != subject:
                continue
            if session != "All" and d["session_id"] != session:
                continue

            image_path = d["image_path"]
            label = self.labels.get(image_path, "")

            if label_filter == "All":
                pass
            elif label_filter == "Unlabeled":
                if image_path in self.labels:
                    continue
            else:
                if label != label_filter:
                    continue

            self.filtered_data.append(d)

        self.refresh_list()
        self.go_to_first_unlabeled()

    def relabel_current(self):
        row = self.list_widget.currentRow()

        if row < 0 or not self.classes:
            return

        state = self.get_filter_state()

        entry = self.filtered_data[row]
        image_path = entry["image_path"]

        current_label = self.labels.get(image_path, "")

        label, ok = QInputDialog.getItem(
            self,
            "Relabel Image",
            f"Select new label (current: {current_label})",
            self.classes,
            editable=False
        )

        if not ok:
            return

        prev_label = self.labels.get(image_path)
        if label == prev_label:
            return
        self.undo_stack.append((image_path, prev_label))
        self.labels[image_path] = label
        self.is_dirty = True

        self.rebuild_view(state)

        for i, d in enumerate(self.filtered_data):
            if d["image_path"] == image_path:
                self.list_widget.setCurrentRow(i)
                break

    def refresh_list(self):
        self.list_widget.clear()

        for d in self.filtered_data:
            label = self.labels.get(d["image_path"], "")
            text = f"{d['subject_id']} | {d['session_id']} | {os.path.basename(d['image_path'])}"
            if label:
                text += f"  [{label}]"

            self.list_widget.addItem(QListWidgetItem(text))

        if self.filtered_data:
            self.list_widget.setCurrentRow(0)
        else:
            self.image_label.clear()
            self.image_label.setText("No images in this view")

        self.update_progress()

    def set_classes(self):
        text, ok = QInputDialog.getText(
            self, "Set Classes",
            "Enter classes (comma separated). Keys 1–9 map to them:"
        )
        if ok and text:
            classes = (item.strip() for item in text.split(","))
            self.classes = list(dict.fromkeys(item for item in classes if item))

    def keyPressEvent(self, event: QKeyEvent):
        if not self.handle_shortcut(event):
            super().keyPressEvent(event)

    def assign_label_keyboard(self, label):
        row = self.list_widget.currentRow()
        if row < 0:
            return

        state = self.get_filter_state()

        entry = self.filtered_data[row]
        image_path = entry["image_path"]

        prev_label = self.labels.get(image_path)
        if label == prev_label:
            return
        self.undo_stack.append((image_path, prev_label))
        self.labels[image_path] = label
        self.is_dirty = True

        self.rebuild_view(state)

        if self.filtered_data:
            current_row = next(
                (
                    index
                    for index, item in enumerate(self.filtered_data)
                    if item["image_path"] == image_path
                ),
                None,
            )
            next_row = row if current_row is None else current_row + 1
            next_row = min(next_row, len(self.filtered_data) - 1)
            self.list_widget.setCurrentRow(next_row)

    def undo_last(self):
        if not self.undo_stack:
            return

        state = self.get_filter_state()
        image_path, prev_label = self.undo_stack.pop()

        if prev_label is None:
            self.labels.pop(image_path, None)
        else:
            self.labels[image_path] = prev_label
        self.is_dirty = True

        self.rebuild_view(state)

        for i, d in enumerate(self.filtered_data):
            if d["image_path"] == image_path:
                self.list_widget.setCurrentRow(i)
                break

    def go_to_first_unlabeled(self):
        if not self.filtered_data:
            return

        for i, d in enumerate(self.filtered_data):
            if d["image_path"] not in self.labels:
                self.list_widget.setCurrentRow(i)
                return

        # If all labeled → go to first safely
        self.list_widget.setCurrentRow(0)

    def go_to_next_unlabeled(self):
        if not self.filtered_data:
            return
        current = self.list_widget.currentRow()
        order = list(range(current + 1, len(self.filtered_data))) + list(
            range(0, current + 1)
        )
        for i in order:
            d = self.filtered_data[i]
            if d["image_path"] not in self.labels:
                self.list_widget.setCurrentRow(i)
                return

        QMessageBox.information(self, "Done", "All images labeled in this view!")

    def show_preview(self):
        row = self.list_widget.currentRow()

        if row < 0 or row >= len(self.filtered_data):
            self.image_label.clear()
            return

        image_path = self.filtered_data[row]["image_path"]

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_label.setText("Failed to load image")
            return

        pixmap = pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        )

        self.image_label.setPixmap(pixmap)

    def get_filter_state(self):
        return {
            "subject": self.subject_filter.currentText(),
            "session": self.session_filter.currentText(),
            "label": self.label_filter_box.currentText()
        }

    def rebuild_view(self, state=None):
        self.setup_filters()
        if state is not None:
            self.restore_filter_state(state)
        self.apply_filters()

    def restore_filter_state(self, state):
        self.subject_filter.blockSignals(True)
        self.session_filter.blockSignals(True)
        self.label_filter_box.blockSignals(True)

        self.subject_filter.setCurrentText(state["subject"])
        self.session_filter.setCurrentText(state["session"])
        self.label_filter_box.setCurrentText(state["label"])

        self.subject_filter.blockSignals(False)
        self.session_filter.blockSignals(False)
        self.label_filter_box.blockSignals(False)

    def update_progress(self):
        total = len(self.filtered_data)
        labeled = sum(1 for d in self.filtered_data if d["image_path"] in self.labels)
        self.progress_label.setText(f"Progress (view): {labeled} / {total}")

    def save_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if not path:
            return

        rows = []
        for d in self.data:
            rows.append({
                "subject_id": d["subject_id"],
                "session_id": d["session_id"],
                "image_path": d["image_path"],
                "label": self.labels.get(d["image_path"], "")
            })

        try:
            df = pd.DataFrame(
                rows,
                columns=("subject_id", "session_id", "image_path", "label"),
            )
            df.to_csv(path, index=False)
        except Exception as error:
            QMessageBox.critical(self, "Error", f"Could not save CSV:\n{error}")
            return

        self.is_dirty = False
        QMessageBox.information(self, "Saved", f"Progress saved to {path}")

    def confirm_discard_changes(self):
        if not self.is_dirty:
            return True
        choice = QMessageBox.question(
            self,
            "Unsaved labels",
            "Discard unsaved label changes?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return choice == QMessageBox.StandardButton.Discard

    def closeEvent(self, event):
        if self.confirm_discard_changes():
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageLabeler()
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt6'))
    window.show()
    sys.exit(app.exec())
