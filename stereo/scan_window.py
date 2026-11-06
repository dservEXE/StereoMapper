from pathlib import Path
from queue import Queue,Empty
from threading import Thread
import tkinter as tk
from tkinter import ttk,messagebox
import cv2
import numpy as np
from . import ui
from .help_ui import help_icon


class ObjectScanWindow:
    def __init__(self,app):
        from .app import ImagePanel
        self.app=app
        self.window=tk.Toplevel(app.root)
        self.window.title("Объёмный точный снимок · накопление и уточнение")
        self.window.geometry("1120x800")
        self.active=self.loading=False
        self.directory=None
        self.anchor=None
        self.region=None
        self.corner=None
        self.events=Queue()
        self.window.storage_busy=lambda:self.active or self.loading
        self.window.storage_paths=lambda:[self.directory] if self.directory else []
        from .scan_ui import build
        build(self,app)
        self.window.protocol('WM_DELETE_WINDOW',self.close)
        self.poll_id=self.window.after(100,self.poll)
        self.window.bind('<Destroy>',self.on_destroy,add='+')

    def pick(self,x,y):
        if self.active or self.anchor is None:return
        h,w=self.anchor.shape[:2]
        if self.corner is None:
            self.corner=(x/w,y/h)
            self.region_text.set("Теперь щёлкните противоположный угол области предмета.")
        else:
            a,b=self.corner;self.corner=None
            self.region=(min(a,x/w),min(b,y/h),max(a,x/w),max(b,y/h))
            self.region_text.set("Покрытие считается внутри выбранного прямоугольника; сохраняется вся измеренная сцена.")
            image=self.anchor.copy();x0,y0,x1,y1=self.region
            cv2.rectangle(image,(int(x0*w),int(y0*h)),(int(x1*w),int(y1*h)),(0,210,255),2)
            self.left.set(image)

    def clear_region(self):
        if self.active:return
        self.region=self.corner=None
        self.region_text.set("Область контроля: весь выровненный кадр.")
        if self.anchor is not None:self.left.set(self.anchor)

    def set_active(self,value):
        self.active=value
        self.start_button.configure(state='disabled' if value else 'normal')
        for button in (self.stop_button,self.extend_button):button.configure(state='normal' if value else 'disabled')
        for control in self.controls:control.configure(state='disabled' if value else 'readonly')
        self.auto_control.configure(state='disabled' if value else 'normal')
        self.phase.set("Снимаем · медленно меняйте ракурс" if value else "Съёмка завершена")
        self.app.sync_ui(force=True)

    def start(self):
        if self.loading:return
        if not self.app.connected or not self.app.worker.engine:
            messagebox.showerror('Снимок','Подключите камеры и загрузите калибровку этого разрешения.',parent=self.window);return
        try:
            maximum=float(self.maximum.get());voxel=float(self.voxel.get())
            max_depth=float(self.max_depth.get());near_minimum=float(self.near_minimum.get())/100
            if max_depth<=.05 or not .05<=near_minimum<=.5:
                raise ValueError("Предел глубины больше 5 см; ближний поиск от 5 до 50 см.")
            if maximum!=0 and maximum<5:raise ValueError("Предел времени — от 5 секунд или 0 без предела.")
            if self.region and (self.region[2]-self.region[0]<.02 or self.region[3]-self.region[1]<.02):
                raise ValueError("Выберите более широкую область предмета.")
            options=dict(width=None if self.width.get()=='Оригинал' else int(self.width.get()),
                voxel_mm=voxel,maximum_seconds=maximum,automatic=self.auto.get(),region=self.region,
                max_depth=max_depth,near_minimum=near_minimum,
                ghost_filter={'Обычная':'normal','Строгая':'strict'}[self.ghost_filter.get()])
        except ValueError as error:
            messagebox.showerror('Настройки',str(error),parent=self.window);return
        self.anchor=None
        self.directory=None
        self.coverage.configure(value=0); self.coverage_text.set("Покрытие первого ракурса: 0%")
        self.set_active(True)
        self.view_button.configure(state='disabled',style="TButton");self.start_button.configure(style="Accent.TButton")
        self.message.set("Подготовка. Первые точки появятся после трёх согласованных наблюдений…")
        self.app.send('scan_start',options)

    def update(self,kind,value):
        if not self.window.winfo_exists():return
        if kind=='scan_started':
            self.directory=Path(value)
        elif kind=='scan_progress':
            if self.anchor is None:self.anchor=value['image'].copy()
            self.left.title_label.configure(text='Текущая глубина · независимые измерения')
            self.left.set(value['depth_image'])
            known=value['preview']
            image=self.anchor.copy()
            image[known]=np.uint8(image[known].astype(np.float32)*.45+np.array([40,130,35]))
            if self.region:
                h,w=known.shape;x0,y0,x1,y1=self.region
                cv2.rectangle(image,(int(x0*w),int(y0*h)),(int(x1*w),int(y1*h)),(0,210,255),2)
            self.right.set(image)
            self.summary.set(f"{value['points']:,} точек · {value['seconds']:.0f} сек · добавлено кадров: {value['integrated']}")
            ui.show_progress(self.coverage,value['coverage']*100)
            self.coverage_text.set(f"Покрытие первого ракурса: {value['coverage']:.0%}")
            self.message.set(value['tracking'])
        elif kind=='scan_finished':
            self.directory=Path(value['directory']);self.set_active(False)
            self.view_button.configure(state='normal' if value.get('points',0)>0 else 'disabled')
            self.phase.set("Скан сохранён" if value.get('points',0)>0 else "Скан завершён без точек")
            self.summary.set(f"{value.get('points',0):,} точек. "+("Откройте результат в 3D, чтобы убрать фон и сохранить нужную область." if value.get('points',0)>0 else "Проверьте глубину и освещение перед повторной съёмкой."))
            # One main button: opening the result takes over from starting a scan.
            if value.get('points',0)>0:
                self.view_button.configure(style="Accent.TButton");self.start_button.configure(style="TButton")
            self.message.set(value['reason'])
            self.left.title_label.configure(text='Последняя измеренная глубина')

    def view(self):
        if not self.directory or self.loading:return
        self.loading=True
        def load():
            from .map_store import VoxelMap
            try:
                store=VoxelMap(path=self.directory/'measured.sqlite',readonly=True)
                try:self.events.put(('model',(*store.arrays(500000),len(store))))
                finally:store.close()
            except Exception as error:self.events.put(('error',str(error)))
        Thread(target=load,daemon=True).start()

    def poll(self):
        self.poll_id=None
        if not self.window.winfo_exists():return
        try:
            kind,value=self.events.get_nowait();self.loading=False
            if kind=='model':
                from .viewer import CloudViewer
                points,colors,total=value
                CloudViewer(self.app.root,points,colors,title='Результат 3D-скана',total_count=total,
                    max_points=500000,archive_path=str(self.directory/'measured.sqlite'))
            else:messagebox.showerror('3D',value,parent=self.window)
        except Empty:pass
        self.poll_id=self.window.after(100,self.poll)

    def on_destroy(self,event):
        if event.widget is self.window and self.poll_id:
            try:self.window.after_cancel(self.poll_id)
            except tk.TclError:pass
            self.poll_id=None

    def close(self):
        if self.active:self.app.send('scan_stop')
        self.window.destroy()
