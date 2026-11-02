"""Full-cloud editing, camera-space volume selection and rigid scan alignment."""
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
from .cloud_document import CloudDocument, load_cloud
from .cloud_geometry import cloud_bounds, project_points, render_points, selection_mask, box_mask, pick_point
from .trackball import arcball_rotation, axis_rotation
from .registration import rigid_transform, refine_icp
from .cloud_orientation import (canonical_view, frame_from_plane, rotation_matrix,
                                view_state, views_path, save_views, load_views)
from .ui import apply_theme, ScrolledFrame

FILE_TYPES = [("Облако или проект", "*.ply *.sqlite *.cloudproj"), ("PLY", "*.ply"),
              ("Архив карты", "*.sqlite"), ("Проект 3D", "*.cloudproj")]


class CloudViewer:
    def __init__(self, parent, points, colors, units="metres", title="Редактор 3D-карты",
                 total_count=None, max_points=250000, archive_path=None, source_path=None):
        apply_theme(parent)
        self.window = tk.Toplevel(parent)
        self.window.title(title); self.window.geometry("1240x860"); self.window.minsize(1000,700)
        self.document = CloudDocument(points, colors, units, source=archive_path or source_path)
        self.max_points = max(1000,int(max_points))
        self.archive_path, self.project_path = archive_path, None
        self.complete = not (archive_path or source_path) and (total_count is None or total_count <= len(points))
        self.busy = self.closed = self.pending = self.changed = False
        self.draw_id = self.drag = self.photo = self.landmark_task = None
        self.stroke, self.events = [], Queue()
        self.brush_pointer=None
        self.view_layer = None
        self.landmarks = {"reference":[],"moving":[]}
        self.landmark_pair = None
        self.scene_frame = np.eye(3)
        self.bookmarks = {}
        self.plane_points = None
        self.projection = tk.StringVar(value="perspective")
        self.view_name = tk.StringVar(value="Мой вид")
        self.rotation = np.eye(3)
        self.yaw,self.pitch,self.zoom,self.pan = 0.,0.,1.,[0.,0.]
        self.pan_viewport=(50,50)
        self.fit_on_first_viewport=True
        self.tool = tk.StringVar(value="rotate")
        self.operation = tk.StringVar(value="Новое")
        self.visible_only = tk.BooleanVar(value=False)
        self.depth_enabled = tk.BooleanVar(value=False)
        self.depth_near,self.depth_far = tk.DoubleVar(value=0),tk.DoubleVar(value=100)
        self.brush_radius,self.point_size = tk.StringVar(value="30"),tk.IntVar(value=2)
        self.active_layer,self.reference_layer,self.moving_layer = tk.StringVar(),tk.StringVar(),tk.StringVar()
        self.selection_scope = tk.StringVar(value="Все облака")
        self.message,self.help_text,self.stats = tk.StringVar(),tk.StringVar(),tk.StringVar()
        self.merge_text = tk.StringVar(value="Добавьте второй снимок и укажите три общие детали на двух облаках.")
        self._buttons,self._combos = [],[]
        self._make_ui()
        self.window.storage_paths = self.storage_paths
        self.window.storage_busy = lambda:self.busy
        self.window.protocol("WM_DELETE_WINDOW",self.close)
        self.window.bind("<Destroy>",self.on_destroy)
        self.canvas.bind("<Configure>",self.viewport_changed)
        for button,mode in ((1,"left"),(2,"pan"),(3,"pan")):
            self.canvas.bind(f"<ButtonPress-{button}>",lambda event,m=mode:self.start_drag(event,m))
            self.canvas.bind(f"<B{button}-Motion>",self.move)
            self.canvas.bind(f"<ButtonRelease-{button}>",self.release)
        # Tk resolves the native Alt modifier. On Windows Mod1 (0x0008) is
        # Num Lock, so testing it as Alt would force every drag into rotation.
        self.canvas.bind("<Alt-ButtonPress-1>",lambda event:self.start_drag(event,"rotate"))
        self.canvas.bind("<MouseWheel>",self.wheel)
        self.canvas.bind("<Motion>",self.brush_hover)
        self.canvas.bind("<Leave>",self.brush_leave)
        for key,fn in (("<Control-z>",self.undo),("<Control-y>",self.redo),("<Delete>",self.delete_selection),
                       ("<Escape>",self.cancel),("<Home>",self.reset)):
            self.canvas.bind(key,lambda event,f=fn:(f(),"break")[1])
        self.refresh(fit=True)
        self.message.set("Перетаскивайте модель мышью. Для очистки выберите прямоугольник, лассо или кисть.")
        self.poll_id=self.window.after(80,self.poll)
        if archive_path or source_path:self.load_path(archive_path or source_path)
        elif not self.complete:self.message.set("Это уменьшенный просмотр. Откройте исходный PLY или SQLite для редактирования всех точек.")

    def button(self,parent,text,command):
        item=ttk.Button(parent,text=text,command=command); item.pack(side="left",padx=3,pady=2)
        self._buttons.append(item); return item

    def combo(self,parent,var,values,width=20):
        item=ttk.Combobox(parent,textvariable=var,values=values,width=width,state="readonly")
        item.pack(fill="x",pady=3); self._combos.append(item); return item

    def row(self,parent):
        row=ttk.Frame(parent); row.pack(fill="x",pady=2); return row

    def _make_ui(self):
        from .viewer_ui import build
        build(self)

    def storage_paths(self):
        paths=[Path(layer.source).parent for layer in self.document.layers if layer.source]
        if self.project_path:paths.append(Path(self.project_path).parent)
        return paths

    def _orientation_ui(self,parent):
        pane=ScrolledFrame(parent); pane.pack(fill="both",expand=True)
        parent=pane.content
        ttk.Label(parent,text="Верх и стороны карты",font=("Segoe UI",11,"bold")).pack(anchor="w")
        ttk.Label(parent,text="Поверните карту как нужно, затем\nназначьте этот ракурс видом спереди.\nВерх экрана станет верхом карты.",foreground="#5b6975").pack(anchor="w",pady=7)
        self.button(self.row(parent),"Текущий вид = спереди",self.set_front)
        self.button(self.row(parent),"Горизонт по 3 точкам…",self.start_plane)
        self.button(self.row(parent),"Перевернуть верх / низ",self.flip_up)
        self.button(self.row(parent),"Вернуть оси исходной карты",self.reset_frame)
        ttk.Label(parent,text="Точные направления").pack(anchor="w",pady=(10,3))
        for pair in ((("Спереди","front"),("Сзади","back")),
                     (("Сверху","top"),("Снизу","bottom")),
                     (("Слева","left"),("Справа","right"))):
            row=self.row(parent)
            for text,name in pair:self.button(row,text,lambda v=name:self.preset(v))
        ttk.Separator(parent).pack(fill="x",pady=8)
        for text,value in (("Перспектива","perspective"),("Ортографическая проекция","orthographic")):
            ttk.Radiobutton(parent,text=text,variable=self.projection,value=value,command=self.request_draw).pack(anchor="w")
        ttk.Label(parent,text="Сохранённые положения",font=("Segoe UI",10,"bold")).pack(anchor="w",pady=(12,4))
        self.bookmark_combo=ttk.Combobox(parent,textvariable=self.view_name,width=23)
        self.bookmark_combo.pack(fill="x")
        row=self.row(parent); self.button(row,"Запомнить",self.remember_view); self.button(row,"Перейти",self.recall_view)
        self.button(self.row(parent),"Удалить положение",self.remove_view)
        row=self.row(parent); self.button(row,"Сохранить виды…",self.export_views); self.button(row,"Загрузить…",self.import_views)
        ttk.Label(parent,text="Проект хранит оси и виды внутри себя.\nПри экспорте PLY рядом сохраняется\nфайл .views.json — переносите оба.\nОриентация вида не двигает точки\nи не заменяет сшивку снимков.",foreground="#5b6975").pack(anchor="w",pady=10)
        pane.attach_wheel()

    def capture_camera(self):
        w,h=self.viewport()
        return dict(rotation=self.rotation.tolist(),center=self.center.tolist(),extent=self.extent,
                    zoom=self.zoom,pan=[self.pan[0]/w,self.pan[1]/h],projection=self.projection.get())

    def capture_views(self):
        return view_state(dict(version=1,frame=self.scene_frame.tolist(),current=self.capture_camera(),bookmarks=self.bookmarks))

    def apply_camera(self,state):
        self.drag=None; self.stroke=[]
        self.rotation=np.array(state["rotation"],float); self.center=np.array(state["center"],float)
        self.extent,self.zoom=state["extent"],state["zoom"]
        w,h=self.viewport(); self.pan=[state["pan"][0]*w,state["pan"][1]*h]
        self.pan_viewport=(w,h); self.fit_on_first_viewport=False
        self.projection.set(state["projection"]); self.request_draw()

    def apply_views(self,state):
        state=view_state(state)
        if state is None:return
        self.scene_frame=np.array(state["frame"],float); self.bookmarks=state["bookmarks"]
        self.bookmark_combo.configure(values=list(self.bookmarks))
        if self.bookmarks:self.view_name.set(next(iter(self.bookmarks)))
        self.apply_camera(state["current"])

    def set_front(self):
        if self.busy:return
        self.plane_points=None
        self.scene_frame=rotation_matrix(self.rotation); self.changed=True
        self.message.set("Оси заданы: верх экрана — верх карты, справа — правая сторона. Сохраните проект или виды.")
        self.request_draw()

    def flip_up(self):
        if self.busy:return
        self.scene_frame=axis_rotation(2,np.pi)@self.scene_frame; self.changed=True
        self.preset("front"); self.message.set("Верх и низ перевёрнуты. Сохраните проект или виды.")

    def reset_frame(self):
        if self.busy:return
        self.scene_frame=np.eye(3); self.changed=True; self.plane_points=None
        self.preset("front"); self.message.set("Возвращены оси исходной карты: X вправо, Y вниз, Z вперёд.")

    def remember_view(self):
        if self.busy:return
        name=self.view_name.get().strip()
        if not name or len(name)>80:self.message.set("Введите название положения длиной от 1 до 80 символов."); return
        if name not in self.bookmarks and len(self.bookmarks)>=100:self.message.set("Уже сохранено 100 положений."); return
        self.bookmarks[name]=self.capture_camera(); self.view_name.set(name)
        self.bookmark_combo.configure(values=list(self.bookmarks)); self.changed=True
        self.message.set(f"Положение «{name}» запомнено. Сохраните проект или виды для следующего запуска.")

    def recall_view(self):
        if self.busy:return
        state=self.bookmarks.get(self.view_name.get())
        if state is None:self.message.set("Выберите запомненное положение из списка."); return
        self.plane_points=None; self.cancel_landmarks(silent=True); self.apply_camera(state)
        self.message.set(f"Восстановлено положение «{self.view_name.get()}».")

    def remove_view(self):
        if not self.busy and self.view_name.get() in self.bookmarks:
            del self.bookmarks[self.view_name.get()]; self.changed=True
            self.bookmark_combo.configure(values=list(self.bookmarks)); self.message.set("Положение удалено из рабочего проекта.")

    def export_views(self):
        if self.busy:return
        source=self.project_path or next((layer.source for layer in self.document.layers if layer.source),None)
        path=filedialog.asksaveasfilename(parent=self.window,title="Сохранить оси и положения вида",
            initialdir=str(Path(source).parent) if source else None,
            initialfile=views_path(Path(source).name).name if source else "cloud.views.json",
            defaultextension=".views.json",filetypes=[("Настройки вида","*.views.json")])
        if path:
            try:
                self.document._protect_sources(path)
                if self.project_path and Path(path).resolve()==Path(self.project_path).resolve():
                    raise ValueError("Выберите отдельный файл настроек; нельзя заменить проект.")
                save_views(path,self.capture_views()); self.message.set(f"Оси и положения сохранены: {path}")
            except (OSError,ValueError) as error:self.message.set(f"Ошибка сохранения видов: {error}")

    def import_views(self):
        if self.busy:return
        path=filedialog.askopenfilename(parent=self.window,title="Загрузить оси и положения вида",filetypes=[("Настройки вида","*.views.json")])
        if path:
            try:self.apply_views(load_views(path)); self.changed=True; self.message.set("Оси и положения восстановлены. Координаты точек сохранены.")
            except (OSError,ValueError,KeyError) as error:self.message.set(f"Ошибка загрузки видов: {error}")

    def start_plane(self):
        if self.busy or not self.require_complete():return
        self.cancel_landmarks(silent=True); self.tool.set("rotate"); self.tool_changed()
        self.plane_points=[]
        self.message.set("Горизонт 1/3: щёлкните точку стола или пола. Выберите три разнесённые точки одной плоскости. Esc — отмена.")

    def pick_plane(self,x,y):
        points=self.points[self.displayed()]; w,h=self.viewport()
        index=pick_point(points,x,y,w,h,**self.camera(),radius=12,point_size=self.point_size.get())
        if index is None:self.message.set("Рядом нет видимой точки поверхности. Увеличьте деталь."); return
        self.plane_points.append(points[index].copy())
        if len(self.plane_points)==3:
            try:
                self.scene_frame=frame_from_plane(self.plane_points,self.scene_frame)
                self.changed=True; self.preset("front")
                self.message.set("Горизонт задан по плоскости. Если верх направлен неверно, нажмите «Перевернуть верх / низ». Сохраните проект или виды.")
            except ValueError as error:self.message.set(str(error)+" Нажмите «Горизонт по 3 точкам» для повтора.")
            self.plane_points=None
        else:self.message.set(f"Горизонт {len(self.plane_points)+1}/3: щёлкните следующую точку той же горизонтальной плоскости.")
        self.request_draw()

    def run_job(self,text,function,callback):
        if self.busy:return
        self.busy=True; self.message.set(text); self.sync_controls()
        def run():
            try:self.events.put((True,function(),callback))
            except Exception as error:self.events.put((False,str(error),None))
        Thread(target=run,daemon=True).start()

    def poll(self):
        self.poll_id=None
        if self.closed:return
        try:
            while True:
                ok,value,callback=self.events.get_nowait(); self.busy=False
                if ok:
                    try:callback(value)
                    except Exception as error:self.message.set(f"Ошибка: {error}")
                else:self.message.set(f"Ошибка: {value}")
                if self.closed:return
                self.sync_controls()
        except Empty:pass
        if not self.closed:self.poll_id=self.window.after(80,self.poll)

    def sync_controls(self):
        for item in self._buttons:item.configure(state="disabled" if self.busy else "normal")
        for item in self._combos:item.configure(state="disabled" if self.busy else "readonly")
        self.bookmark_combo.configure(state="disabled" if self.busy else "normal")
        if not self.busy:
            for button in (self.project_button,self.export_button):button.configure(state="normal" if self.complete else "disabled")
            for button in self.merge_buttons:button.configure(state="normal" if self.complete and len(self.document.layers)>1 else "disabled")
            self.undo_button.configure(state="normal" if self.document.can_undo else "disabled")
            self.redo_button.configure(state="normal" if self.document.can_redo else "disabled")
            state="normal" if self.complete and self.document.selection_count else "disabled"
            self.delete_button.configure(state=state); self.keep_button.configure(state=state)

    def require_complete(self):
        if self.complete:return True
        self.message.set("Откройте исходный PLY или SQLite: сейчас доступна только выборка для просмотра."); return False

    def load_path(self,path):
        path=Path(path)
        def load():
            if path.suffix.lower() in (".cloudproj",".npz"):return CloudDocument.load_project(path)
            p,c,u,s=load_cloud(path); return CloudDocument(p,c,u,source=s)
        def done(document):
            self.document=document; self.complete=True; self.changed=False
            self.project_path=str(path) if path.suffix.lower() in (".cloudproj",".npz") else None
            self.archive_path=str(path) if path.suffix.lower()==".sqlite" else None
            self.plane_points=None; self.scene_frame=np.eye(3); self.bookmarks={}
            self.bookmark_combo.configure(values=()); self.view_name.set("Мой вид")
            self.rotation=np.eye(3); self.projection.set("perspective")
            self.cancel_landmarks(silent=True); self.refresh(fit=True)
            self.message.set(f"Загружено {document.count:,} точек. Выделение и сохранение работают со всем облаком.")
            try:
                state=document.view_state
                if state is None and views_path(path).exists():state=load_views(views_path(path))
                if state is not None:
                    self.apply_views(state); self.message.set(f"Загружено {document.count:,} точек; оси и сохранённый вид восстановлены.")
            except (OSError,ValueError,KeyError) as error:
                self.message.set(f"Карта загружена; настройки вида прочитать не удалось: {error}")
        self.run_job("Загружаю все точки исходной карты…",load,done)

    def open_file(self):
        if self.busy:return
        if self.changed and not messagebox.askyesno("Открыть карту","Открыть другую карту без сохранения текущих изменений?",parent=self.window):return
        path=filedialog.askopenfilename(parent=self.window,filetypes=FILE_TYPES)
        if path:self.load_path(path)

    def add_file(self):
        if self.busy or not self.require_complete():return
        path=filedialog.askopenfilename(parent=self.window,title="Добавить снимок",filetypes=FILE_TYPES)
        if not path:return
        def done(result):
            p,c,u,s=result; index=self.document.add_cloud(p,c,u,source=s)
            self.changed=True; self.cancel_landmarks(silent=True); self.refresh(fit=True)
            self.moving_layer.set(self.layer_names[index]); self.active_layer.set(self.layer_names[index])
            self.notebook.select(1)
            self.message.set("Второе облако добавлено в исходных координатах. Укажите общие детали для совмещения.")
        self.run_job("Загружаю добавляемое облако полностью…",lambda:load_cloud(path),done)

    def save_ply(self):
        if self.busy or not self.require_complete():return
        if not self.document.count:self.message.set("Нет точек для сохранения. Можно отменить удаление."); return
        path=filedialog.asksaveasfilename(parent=self.window,title="Сохранить обработанное общее облако",
            initialfile="edited_cloud.ply",defaultextension=".ply",filetypes=[("Облако PLY","*.ply")])
        if path:
            state=self.capture_views()
            def export():
                count=self.document.export_ply(path)
                save_views(views_path(path),state)
                return count
            self.run_job("Сохраняю все оставленные точки и настройки вида…",export,
                lambda count:self.message.set(f"Сохранено {count:,} точек: {path}; оси и виды — в соседнем .views.json. Для сшивки сохраните также проект."))

    def save_project(self,close_after=False):
        if self.busy or not self.require_complete():return
        path=filedialog.asksaveasfilename(parent=self.window,title="Сохранить проект очистки и сшивки",
            initialfile=Path(self.project_path).name if self.project_path else "cloud_project.cloudproj",
            defaultextension=".cloudproj",filetypes=[("Проект 3D","*.cloudproj")])
        if path:
            self.document.view_state=self.capture_views()
            def done(value):
                self.project_path=path; self.changed=False; self.message.set(f"Проект сохранён: {path}")
                if close_after:self.closed=True; self.window.destroy()
            self.run_job("Сохраняю проект с точками, масками и положениями облаков…",lambda:self.document.save_project(path),done)

    def export_archive(self):self.save_ply()

    def refresh(self,fit=False):
        self.points,self.colors,self.ids=self.document.arrays(include_ids=True)
        self.units,self.total_count=self.document.units,self.document.count
        self.layer_names=[f"{i+1}. {layer.name}" for i,layer in enumerate(self.document.layers)]
        for combo in (self.layer_combo,self.reference_combo,self.moving_combo):combo.configure(values=self.layer_names)
        for var,index in ((self.active_layer,len(self.layer_names)-1),(self.reference_layer,0),(self.moving_layer,len(self.layer_names)-1)):
            if var.get() not in self.layer_names:var.set(self.layer_names[index] if self.layer_names else "")
        if self.view_layer is not None and self.view_layer>=len(self.layer_names):self.view_layer=None
        if fit:self.fit()
        self.sync_controls(); self.tool_changed(); self.request_draw()

    def camera(self):return dict(rotation=self.rotation,zoom=self.zoom,pan=self.pan,center=self.center,extent=self.extent,projection=self.projection.get())
    def viewport(self):return max(50,self.canvas.winfo_width()),max(50,self.canvas.winfo_height())
    def displayed(self):return np.ones(len(self.points),bool) if self.view_layer is None else self.ids[:,0]==self.view_layer

    def viewport_changed(self,event):
        w,h=self.viewport(); old_w,old_h=self.pan_viewport
        self.pan=[self.pan[0]*w/old_w,self.pan[1]*h/old_h]; self.pan_viewport=(w,h)
        if self.fit_on_first_viewport and hasattr(self,"points"):
            self.fit_on_first_viewport=False; self.fit()
        self.request_draw()

    def fit(self):
        points=self.points[self.displayed()]
        self.center,self.extent=cloud_bounds(points)
        self.zoom,self.pan=1.,[0.,0.]
        self.pan_viewport=self.viewport()
        if len(points):
            w,h=self.viewport()
            xy,_,front=project_points(points,w,h,**self.camera())
            if front.any():
                span=np.max(np.abs(xy[front]-[w/2,h/2]),axis=0)
                half=np.maximum(np.array([w,h])/2-12,1)
                ratios=half/np.maximum(span,1e-12)
                self.zoom=float(np.clip(min(ratios)*.95,.05,20)) if span.max()>1e-10 else 1.
        self.request_draw()

    def show_help(self):
        dialog=tk.Toplevel(self.window); dialog.title("Как пользоваться редактором 3D"); dialog.geometry("780x640")
        frame=ttk.Frame(dialog,padding=12); frame.pack(fill="both",expand=True)
        text=tk.Text(frame,wrap="word",font=("Segoe UI",11),padx=10,pady=10)
        scroll=ttk.Scrollbar(frame,command=text.yview); text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right",fill="y"); text.pack(fill="both",expand=True)
        path=Path(__file__).resolve().parents[1]/"3D_EDITOR_GUIDE.md"
        content=path.read_text(encoding="utf-8") if path.exists() else "Вращать: ЛКМ, сдвиг: ПКМ, масштаб: колесо. Выделите ненужные точки и удалите либо оставьте нужную область. Ctrl+Z отменяет действие. Для сшивки добавьте второе облако и укажите 3 пары общих точек."
        text.insert("1.0",content.replace("**","")); text.configure(state="disabled")

    def reset(self):
        self.preset("front")

    def preset(self,view):
        if self.busy:return
        self.drag=None; self.stroke=[]
        self.rotation=canonical_view(view,self.scene_frame); self.fit()

    def tool_changed(self):
        self.drag=None; self.stroke=[]
        self.brush_pointer=None
        brush=self.tool.get()=="brush"
        self.operation_combo.configure(values=("Добавить","Вычесть") if brush else ("Новое","Добавить","Вычесть","Пересечение"))
        if brush and self.operation.get() not in ("Добавить","Вычесть"):self.operation.set("Добавить")
        if self.tool.get()!="rotate" and self.landmark_task:
            self.cancel_landmarks(silent=True)
            self.merge_text.set("Выбор меток остановлен. Для сшивки снова нажмите «Указать 3 пары точек».")
        if self.tool.get()!="rotate" and self.plane_points is not None:
            self.plane_points=None; self.message.set("Настройка горизонта остановлена при смене инструмента.")
        text={"rotate":"ЛКМ — вращать: центр наклоняет, края поворачивают вокруг взгляда",
              "rectangle":"ЛКМ — протянуть прямоугольник","lasso":"ЛКМ — обвести область лассо","brush":"Кисть: ЛКМ сразу добавляет точки · Ctrl — вычитать · Esc — снять выделение"}[self.tool.get()]
        self.help_text.set(text+" · Alt+ЛКМ — вращать · ПКМ/СКМ — сдвиг · колесо — масштаб")
        self.canvas.configure(cursor="fleur" if self.tool.get()=="rotate" else "crosshair")
        self.request_draw()

    def start_drag(self,event,mode):
        if self.busy:return
        self.canvas.focus_set()
        if mode=="left":
            mode=self.tool.get()
        if mode=="rotate" and event.state&1:mode="roll"
        self.drag=dict(start=(event.x,event.y),last=(event.x,event.y),mode=mode,state=event.state,distance=0.)
        self.stroke=[(event.x,event.y)]
        if mode=="brush":
            if not self.require_complete():self.drag=None; return
            self.paint_brush((event.x,event.y),(event.x,event.y),event.state)

    def move(self,event):
        if self.busy or not self.drag:return
        x,y=self.drag["last"]; mode=self.drag["mode"]; w,h=self.viewport()
        self.drag["distance"]+=float(np.hypot(event.x-x,event.y-y))
        if mode=="rotate":self.rotation=arcball_rotation((x,y),(event.x,event.y),w,h)@self.rotation
        elif mode=="roll":
            center=np.array([w/2,h/2]); a=np.array([x,y])-center; b=np.array([event.x,event.y])-center
            angle=np.arctan2(a[0]*b[1]-a[1]*b[0],np.dot(a,b)); self.rotation=axis_rotation(2,angle)@self.rotation
        elif mode=="pan":self.pan[0]+=event.x-x; self.pan[1]+=event.y-y
        elif mode=="brush":self.paint_brush((x,y),(event.x,event.y),event.state)
        elif mode=="lasso" and np.hypot(event.x-self.stroke[-1][0],event.y-self.stroke[-1][1])>=3:self.stroke.append((event.x,event.y))
        self.drag["last"]=(event.x,event.y); self.request_draw()

    def release(self,event):
        if not self.drag:return
        if self.drag["mode"]=="brush" and not self.busy:
            self.paint_brush(self.drag["last"],(event.x,event.y),event.state)
        drag,self.drag=self.drag,None
        if self.busy:return
        if self.plane_points is not None and drag["mode"] in ("rotate","roll") and drag["distance"]<4:self.pick_plane(event.x,event.y)
        elif self.landmark_task and drag["mode"] in ("rotate","roll") and drag["distance"]<4:self.pick_landmark(event.x,event.y)
        elif drag["mode"] in ("rectangle","lasso") and self.require_complete():
            try:
                w,h=self.viewport(); kwargs=self.camera()
                kwargs.update(visible_only=self.visible_only.get(),point_size=self.point_size.get(),depth_range=self.selection_depth())
                if drag["mode"]=="rectangle":
                    x,y=drag["start"]; u,v=event.x,event.y; polygon=[(x,y),(u,y),(u,v),(x,v)]
                    mask=selection_mask(self.points,polygon,w,h,**kwargs)
                elif drag["mode"]=="lasso":
                    polygon=self.stroke+[(event.x,event.y)]
                    if len(polygon)<3:return
                    mask=selection_mask(self.points,polygon,w,h,**kwargs)
                self.apply_selection(mask,drag["state"])
            except (ValueError,MemoryError) as error:self.message.set(str(error))
        self.stroke=[]; self.request_draw()

    def paint_brush(self,first,second,state=0):
        """Apply an additive/eraser stamp now, using the full-cloud projection.

        Camera/surface eligibility is cached only for the current stroke; mouse
        motion tests the swept circle so fast movements cannot leave gaps.
        """
        if not self.drag or self.busy:return
        w,h=self.viewport()
        signature=(id(self.points),w,h,tuple(self.rotation.ravel()),tuple(self.center),self.extent,
                   self.zoom,tuple(self.pan),self.projection.get(),self.visible_only.get(),self.point_size.get(),
                   self.depth_enabled.get(),self.depth_near.get(),self.depth_far.get(),
                   self.view_layer,self.selection_scope.get(),self.active_layer.get())
        cache=self.drag.get("brush_cache")
        if cache is None or cache[0]!=signature:
            xy,depth,front=project_points(self.points,w,h,**self.camera())
            eligible=front&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)&self.edit_scope()
            interval=self.selection_depth()
            if interval:eligible&=(depth>=interval[0])&(depth<=interval[1])
            if self.visible_only.get():
                eligible&=selection_mask(self.points,[(0,0),(w,0),(w,h),(0,h)],w,h,**self.camera(),
                                        visible_only=True,point_size=self.point_size.get())
            indices=np.flatnonzero(eligible)
            cache=(signature,indices,xy[indices]); self.drag["brush_cache"]=cache
        _,indices,xy=cache
        a,b=np.asarray(first,float),np.asarray(second,float); segment=b-a
        t=np.clip((xy-a)@segment/max(float(segment@segment),1e-12),0,1)
        inside=np.sum((xy-a-t[:,None]*segment)**2,axis=1)<=float(self.brush_radius.get())**2
        mask=np.zeros(len(self.points),bool); mask[indices[inside]]=True
        erase=bool(state&4) or self.operation.get()=="Вычесть"
        self.document.select(mask,"subtract" if erase else "add")
        self.brush_pointer=tuple(second)
        self.message.set(f"Кисть {'вычитает' if erase else 'добавляет'} · выбрано {self.document.selection_count:,} точек · Esc или «Снять выделение» — сброс")
        self.sync_controls(); self.request_draw()

    def brush_hover(self,event):
        self.brush_pointer=(event.x,event.y)
        self.draw_brush_cursor()

    def brush_leave(self,event):
        self.brush_pointer=None; self.canvas.delete("brush_cursor")

    def draw_brush_cursor(self):
        self.canvas.delete("brush_cursor")
        if self.tool.get()=="brush" and self.brush_pointer is not None and not self.busy:
            x,y=self.brush_pointer; r=float(self.brush_radius.get())
            self.canvas.create_oval(x-r,y-r,x+r,y+r,outline="#ffda45",width=2,tags="brush_cursor")

    def selection_depth(self):
        if not self.depth_enabled.get() or not len(self.points):return None
        depth=(self.points@self.rotation[2]-self.center@self.rotation[2])/self.extent+2
        depth=depth[(depth>0)&self.displayed()]
        if not len(depth):return None
        a,b=sorted((self.depth_near.get()/100,self.depth_far.get()/100)); low,high=float(depth.min()),float(depth.max())
        return low+(high-low)*a,low+(high-low)*b

    def apply_selection(self,mask,state=0):
        mask=np.asarray(mask,bool)&self.edit_scope()
        operation="Добавить" if state&1 else "Вычесть" if state&4 else self.operation.get()
        if operation=="Пересечение":mask&=self.document.selection; operation="Новое"
        self.document.select(mask,{"Новое":"replace","Добавить":"add","Вычесть":"subtract"}[operation])
        self.message.set(f"Выбрано {self.document.selection_count:,} точек в 3D. Поверните модель для проверки границ.")
        self.sync_controls(); self.request_draw()

    def clear_selection(self):
        if not self.busy:
            self.drag=None; self.stroke=[]
            self.document.clear_selection(); self.sync_controls(); self.request_draw()

    def invert_selection(self):
        if self.busy or not self.require_complete():return
        mask=self.edit_scope()
        self.document.select(mask&~self.document.selection); self.sync_controls(); self.request_draw()

    def edit_scope(self):
        mask=self.displayed()
        if self.selection_scope.get()=="Выбранное облако":mask&=self.ids[:,0]==self.layer_index(self.active_layer)
        return mask

    def delete_selection(self):
        if self.busy or not self.require_complete():return
        scope=self.edit_scope(); count=int(np.count_nonzero(self.document.selection&scope))
        if not count:self.message.set("Сначала выделите точки прямоугольником, лассо или кистью."); return
        self.document.delete_selected(scope); self.changed=True; self.cancel_landmarks(silent=True); self.refresh()
        self.message.set(f"Убрано {count:,} точек из рабочей копии. Ctrl+Z — вернуть.")

    def keep_selection(self):
        if self.busy or not self.require_complete():return
        scope=self.edit_scope(); count=int(np.count_nonzero(self.document.selection&scope))
        if not count:self.message.set("Сначала выделите область, которую нужно оставить."); return
        removed=self.document.keep_selected(scope); self.changed=True; self.cancel_landmarks(silent=True); self.refresh()
        self.message.set(f"В выбранной области действия оставлено {count:,} точек, убрано {removed:,}. Ctrl+Z — вернуть.")

    def undo(self):
        if not self.busy and self.document.undo():
            self.changed=True; self.cancel_landmarks(silent=True); self.refresh(); self.message.set("Действие отменено.")

    def redo(self):
        if not self.busy and self.document.redo():
            self.changed=True; self.cancel_landmarks(silent=True); self.refresh(); self.message.set("Действие повторено.")

    def select_box(self):
        if self.busy or not self.require_complete() or not len(self.points):return
        dialog=tk.Toplevel(self.window); dialog.title("Пространственный объём XYZ"); dialog.transient(self.window)
        low,high=self.points.min(axis=0),self.points.max(axis=0); variables=[]
        ttk.Label(dialog,text="Границы в метрах" if self.units=="metres" else "Границы в единицах облака").grid(row=0,column=0,columnspan=3,padx=12,pady=10)
        for i,axis in enumerate("XYZ"):
            ttk.Label(dialog,text=axis).grid(row=i+1,column=0,padx=10); row=[]
            for j,value in enumerate((low[i],high[i])):
                var=tk.StringVar(value=repr(float(value))); row.append(var)
                ttk.Entry(dialog,textvariable=var,width=16).grid(row=i+1,column=j+1,padx=5,pady=4)
            variables.append(row)
        def apply():
            try:
                bounds=np.array([[float(v.get().replace(",",".")) for v in row] for row in variables])
                self.apply_selection(box_mask(self.points,bounds[:,0],bounds[:,1])); dialog.destroy()
            except ValueError as error:messagebox.showerror("Объём XYZ",str(error),parent=dialog)
        ttk.Button(dialog,text="Выделить объём",command=apply).grid(row=4,column=1,columnspan=2,pady=12)

    def wheel(self,event):
        if not self.busy:self.zoom=min(20,max(.05,self.zoom*(1.12 if event.delta>0 else 1/1.12))); self.request_draw()
        return "break"

    def layer_index(self,var):
        if var.get() not in self.layer_names:raise ValueError("Выберите облако из списка.")
        return self.layer_names.index(var.get())

    def layer_pair(self):
        reference,moving=self.layer_index(self.reference_layer),self.layer_index(self.moving_layer)
        if reference==moving:raise ValueError("Выберите два разных облака: основное и добавленное.")
        return reference,moving

    def start_landmarks(self):
        if self.busy or not self.require_complete():return
        try:reference,moving=self.layer_pair()
        except ValueError as error:self.message.set(str(error)); return
        if not len(self.document.layer_points(reference)[0]) or not len(self.document.layer_points(moving)[0]):
            self.message.set("В выбранном облаке нет оставленных точек."); return
        self.plane_points=None
        self.landmarks={"reference":[],"moving":[]}; self.landmark_pair=(reference,moving)
        self.landmark_task=(reference,moving,"reference"); self.view_layer=reference
        self.tool.set("rotate"); self.fit(); self.tool_changed(); self.landmark_prompt()

    def landmark_prompt(self):
        kind=self.landmark_task[2]; number=len(self.landmarks["moving"])+1
        text=f"Пара {number}/3: щёлкните {'деталь на основном облаке' if kind=='reference' else 'ту же деталь на добавленном облаке'}. Перетаскивание вращает."
        self.merge_text.set(text); self.message.set(text)

    def pick_landmark(self,x,y):
        reference,moving,kind=self.landmark_task; points=self.points[self.displayed()]; w,h=self.viewport()
        index=pick_point(points,x,y,w,h,**self.camera(),radius=12,point_size=self.point_size.get())
        if index is None:self.message.set("Рядом нет видимой точки. Увеличьте нужную деталь."); return
        self.landmarks[kind].append(points[index].copy())
        if kind=="reference":self.landmark_task=(reference,moving,"moving"); self.view_layer=moving
        elif len(self.landmarks["moving"])>=3:
            self.landmark_task=None; self.view_layer=None
            self.merge_text.set("Три пары готовы. Нажмите «Совместить по меткам», затем при необходимости уточните ICP.")
            self.message.set("Метки готовы. Совместите облака по меткам.")
        else:self.landmark_task=(reference,moving,"reference"); self.view_layer=reference
        self.fit()
        if self.landmark_task:self.landmark_prompt()

    def cancel_landmarks(self,silent=False):
        self.landmark_task=None; self.landmark_pair=None; self.landmarks={"reference":[],"moving":[]}; self.view_layer=None
        if not silent:self.merge_text.set("Метки сброшены. Укажите три пары общих деталей.")
        self.request_draw()

    def show_all(self):
        if not self.busy:self.landmark_task=None; self.view_layer=None; self.fit()

    def align_landmarks(self):
        if self.busy or not self.require_complete():return
        try:
            reference,moving=self.layer_pair()
            if self.landmark_pair!=(reference,moving) or min(map(len,self.landmarks.values()))<3:
                raise ValueError("Сначала укажите три пары соответствующих точек для выбранных облаков.")
            transform=rigid_transform(np.asarray(self.landmarks["moving"]),np.asarray(self.landmarks["reference"]))
            self.document.apply_transform(moving,transform@self.document.layers[moving].transform)
            self.changed=True; self.cancel_landmarks(silent=True); self.refresh(fit=True)
            self.merge_text.set("Облака совмещены по меткам. Проверьте их с разных сторон и уточните ICP.")
            self.message.set("Совмещение применено. Ctrl+Z — отменить.")
        except ValueError as error:self.message.set(str(error))

    def align_icp(self):
        if self.busy or not self.require_complete():return
        try:
            reference,moving=self.layer_pair(); source=self.document.layer_points(moving)[0].copy()
            target=self.document.layer_points(reference)[0].copy(); original=self.document.layers[moving].transform.copy()
        except ValueError as error:self.message.set(str(error)); return
        def done(result):
            if not result.usable:
                reasons={"insufficient_overlap":"слишком мало общей поверхности", "degenerate_geometry":"недостаточно пространственных деталей",
                         "degenerate_correspondences":"не найдены устойчивые соответствия", "no_improvement":"уточнение не уменьшило расхождение"}
                self.message.set(f"Положение не изменено: {reasons.get(result.reason,'не удалось уточнить совпадение')}. Сначала укажите общие точки."); return
            self.document.apply_transform(moving,result.transform@original)
            self.changed=True; self.cancel_landmarks(silent=True); self.refresh()
            factor,unit=(1000,"мм") if self.units=="metres" else (1,"ед.")
            self.merge_text.set(f"ICP: ошибка совпадений {result.before_rmse*factor:.3f} → {result.after_rmse*factor:.3f} {unit}; пар {result.correspondences:,}. Это согласие облаков, а не точность камеры.")
            self.message.set("Положение уточнено. Проверьте поверхность; Ctrl+Z — отменить.")
        self.run_job("Уточняю положение облаков по общей поверхности…",lambda:refine_icp(source,target),done)

    def cancel(self):
        self.drag=None; self.stroke=[]
        if self.plane_points is not None:
            self.plane_points=None; self.message.set("Настройка горизонта отменена.")
        elif self.landmark_task:self.cancel_landmarks()
        else:self.clear_selection()
        self.request_draw()

    def request_draw(self):
        if not self.closed and not self.pending:self.pending=True; self.draw_id=self.window.after(25,self.draw)

    def draw(self):
        self.pending=False; self.draw_id=None
        if self.closed:return
        w,h=self.viewport(); mask=self.displayed(); selected=self.document.selection
        indices=np.flatnonzero(mask)
        budget=min(self.max_points,60000) if self.drag else self.max_points
        if len(indices)>budget:
            # Sampling only affects the display. Edits and exports use the full document.
            selected_ids=indices[selected[indices]]
            extra=min(len(selected_ids),budget//3)
            normal_budget=max(1,budget-extra)
            sample=indices[::max(1,int(np.ceil(len(indices)/normal_budget)))]
            if extra:sample=np.unique(np.concatenate((sample,selected_ids[::max(1,int(np.ceil(len(selected_ids)/extra)))])))
            indices=sample[:budget]
        display_colors=self.colors[indices]
        if self.depth_enabled.get() and len(indices):
            interval=self.selection_depth()
            if interval:
                depth=(self.points[indices]@self.rotation[2]-self.center@self.rotation[2])/self.extent+2
                outside=(depth<interval[0])|(depth>interval[1])
                display_colors=display_colors.copy(); display_colors[outside]=np.uint8(display_colors[outside]*.2)
        bgr=render_points(self.points[indices],display_colors,w,h,**self.camera(),selection=selected[indices],
                          max_points=budget,point_size=self.point_size.get(),axes=False)
        self.photo=ImageTk.PhotoImage(Image.fromarray(bgr[:,:,::-1]))
        self.canvas.delete("all"); self.canvas.create_image(0,0,anchor="nw",image=self.photo)
        # Compass follows the user-defined scene basis, not the raw file axes.
        basis=self.rotation@self.scene_frame.T; origin=np.array([65.,h-65.])
        for vector,label,color in ((-basis[:,1],"Верх","#6dde8b"),(basis[:,0],"Право","#ff8585"),(basis[:,2],"Вперёд","#7cbbff")):
            end=origin+vector[:2]*40
            self.canvas.create_line(*origin,*end,fill=color,width=2,arrow="last")
            self.canvas.create_text(* (end+[5,-9]),text=label,fill=color,anchor="w")
        self.canvas.create_text(w-12,12,anchor="ne",fill="#adc3d1",text="Ортографическая проекция" if self.projection.get()=="orthographic" else "Перспектива")
        if self.plane_points:
            xy,_,front=project_points(self.plane_points,w,h,**self.camera())
            for index,((x,y),visible) in enumerate(zip(xy,front),1):
                if visible:
                    self.canvas.create_oval(x-6,y-6,x+6,y+6,outline="#ffda45",width=2)
                    self.canvas.create_text(x+12,y-12,text=str(index),fill="#ffda45")
        if self.drag and self.drag["mode"]=="rotate":
            radius=min(w,h)/2
            self.canvas.create_oval(w/2-radius,h/2-radius,w/2+radius,h/2+radius,outline="#536576",dash=(4,6))
        if self.drag and self.drag["mode"]=="rectangle":self.canvas.create_rectangle(*self.drag["start"],*self.drag["last"],outline="#ffda45",width=2,dash=(5,3))
        elif self.drag and self.drag["mode"]=="lasso" and len(self.stroke)>1:self.canvas.create_line(*np.asarray(self.stroke).ravel().tolist(),fill="#ffda45",width=2)
        self.draw_brush_cursor()
        label={"metres":"метры","millimetres":"миллиметры","relative":"условный масштаб","unknown":"масштаб не указан"}.get(self.units,self.units)
        self.caption.configure(text=f"{self.document.count:,} точек · {label}")
        self.stats.set(f"Облаков: {len(self.document.layers)} · оставлено: {self.document.count:,} · выбрано: {self.document.selection_count:,} · на экране до {len(indices):,}")
        if self.depth_enabled.get():
            a,b=sorted((self.depth_near.get(),self.depth_far.get()))
            self.canvas.create_text(12,12,anchor="nw",fill="#ffda45",text=f"Слой выделения: {a:.0f}–{b:.0f}% глубины вида")
        self.sync_controls()

    def close(self):
        if self.busy:self.message.set("Дождитесь завершения операции…"); return
        if self.changed:
            answer=messagebox.askyesnocancel("Сохранить проект?","Сохранить очистку и совмещение перед закрытием?",parent=self.window)
            if answer is None:return
            if answer:self.save_project(close_after=True); return
        self.closed=True; self.window.destroy()

    def on_destroy(self,event):
        if event.widget is self.window:
            self.closed=True
            if self.poll_id:
                try:self.window.after_cancel(self.poll_id)
                except tk.TclError:pass
                self.poll_id=None
            if self.draw_id:
                try:self.window.after_cancel(self.draw_id)
                except tk.TclError:pass
                self.draw_id=None
