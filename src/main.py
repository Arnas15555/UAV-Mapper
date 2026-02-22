from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from gui.app_window import AppWindow


def main() -> int:
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("UAV Mapper")
    app.setOrganizationName("Group10")
    app.setOrganizationDomain("local")

    w = AppWindow()
    w.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())