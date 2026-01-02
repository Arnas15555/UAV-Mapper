import sys
from PySide6.QtWidgets import QApplication, QLabel

def main():
    app = QApplication(sys.argv)
    label = QLabel("UAV Mapper GUI OK")
    label.resize(300, 100)
    label.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
