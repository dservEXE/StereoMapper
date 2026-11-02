"""Inventory and explicit deletion of application results, never program files."""
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime,timezone,timedelta
import os
import shutil
import stat


@dataclass
class Research:
    path: Path
    kind: str
    size: int
    changed: str
    blocked: str = ""


def reparse(path):
    info=path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info,"st_file_attributes",0)&getattr(stat,"FILE_ATTRIBUTE_REPARSE_POINT",0))


def size_text(size):
    for unit in ("Б","КБ","МБ","ГБ","ТБ"):
        if size<1024 or unit=="ТБ":return f"{size:.1f} {unit}"
        size/=1024


class ResearchCatalog:
    def __init__(self,workspace):
        self.workspace=Path(workspace).resolve()

    def approved(self,path):
        path=Path(os.path.abspath(path))
        parent=path.parent
        allowed=(parent==self.workspace/"output" or parent==self.workspace/"data/maps" or
                 (parent==self.workspace/"data" and path.name.startswith("calibration_")))
        if not allowed:raise ValueError("Можно удалять только отдельные сеансы из папок приложения.")
        # Validate every ancestor as well as the target before walking/deleting.
        for component in (path,*path.parents):
            if component==self.workspace:break
            if reparse(component):raise ValueError("Папки со ссылками или перенаправлением нельзя удалять здесь.")
        if not path.resolve().is_relative_to(self.workspace):
            raise ValueError("Папка находится за пределами приложения.")
        if not path.is_dir():raise ValueError("Выберите папку отдельного исследования.")
        return path

    @staticmethod
    def protected(path,protected):
        for other in protected:
            other=Path(other).resolve()
            if path==other or path.is_relative_to(other) or other.is_relative_to(path):return True
        return False

    @staticmethod
    def measure(path):
        total=0
        stack=[path]
        while stack:
            directory=stack.pop()
            for child in directory.iterdir():
                if reparse(child):raise ValueError("Внутри есть ссылка; автоматическое удаление запрещено.")
                if child.is_dir():stack.append(child)
                else:total+=child.stat().st_size
        return total

    def scan(self,protected=()):
        candidates=[];times={}
        for root in (self.workspace/"output",self.workspace/"data/maps",self.workspace/"data"):
            if not root.exists() or reparse(root):continue
            for path in root.iterdir():
                if root==self.workspace/"data" and not path.name.startswith("calibration_"):continue
                if not path.is_dir():continue
                try:modified=path.stat().st_mtime
                except OSError:continue
                times[path]=modified
                kind=("Калибровочные кадры" if root==self.workspace/"data" else "Архив 3D-карты" if root.name=="maps"
                      else "Снимок листа" if path.name.startswith("precision_") else "Глубина по фото" if path.name.startswith("photo_")
                      else "Проверка приложения" if path.name.startswith("quality_") else "Снимок и 3D-модели")
                blocked=""
                total=0
                try:
                    self.approved(path)
                    total=self.measure(path)
                    if self.protected(path.resolve(),protected):blocked="Открытый или текущий сеанс"
                except (OSError,ValueError) as error:blocked=str(error)
                changed=datetime.fromtimestamp(modified,timezone(timedelta(hours=3))).strftime("%d.%m.%Y %H:%M")
                candidates.append(Research(path,kind,total,changed,blocked))
        return sorted(candidates,key=lambda item:times[item.path],reverse=True)

    def delete(self,path,protected=()):
        path=self.approved(path)
        if self.protected(path.resolve(),protected):raise ValueError("Этот сеанс сейчас открыт или используется.")
        total=self.measure(path)  # Preflight the complete tree before the first removal.
        shutil.rmtree(path)
        return total
