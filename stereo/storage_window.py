import tkinter as tk
from tkinter import ttk,messagebox
from queue import Queue,Empty
from threading import Thread
import os
from pathlib import Path
from .storage import ResearchCatalog,size_text
from . import ui


def session_cloud(directory):
    """Choose an editable project, otherwise the measured cloud of a session."""
    directory=Path(directory)
    try:
        projects=sorted(directory.glob('*.cloudproj'),key=lambda p:p.stat().st_mtime,reverse=True)
        for path in (*projects,*(directory/name for name in ('measured.sqlite','scan.ply','accumulated_cloud.ply','visible_cloud.ply','cloud.ply'))):
            if path.is_file():return path
        return next((p for p in sorted(directory.glob('*.ply')) if 'estimat' not in p.stem.lower()),None)
    except OSError:return None


class StorageWindow:
    def __init__(self,parent,workspace,protected=lambda:[],busy=lambda:False,on_open_cloud=None):
        ui.apply_theme(parent)
        self.on_open_cloud=on_open_cloud
        self.catalog=ResearchCatalog(workspace)
        self.protected,self.external_busy=protected,busy
        self.window=tk.Toplevel(parent)
        ui.style_window(self.window)
        self.window.title("Сохранённые сеансы · StereoMapper")
        self.window.geometry("1100x620")
        self.window.minsize(850,460)
        self.events=Queue();self.items={};self.working=False
        self.window.storage_busy=lambda:self.working
        self.summary=tk.StringVar(value="Подсчитываю размеры…")
        self.selection=tk.StringVar(value="Выберите сеанс, чтобы открыть результат")
        outer=ttk.Frame(self.window,padding=16);outer.pack(fill="both",expand=True)
        ttk.Label(outer,text="Сохранённые сеансы",style="Heading.TLabel").pack(anchor="w")
        ttk.Label(outer,textvariable=self.summary,style="CardTitle.TLabel").pack(anchor="w",pady=(ui.SPACE_XS,ui.SPACE_S))
        ttk.Label(outer,text="Выберите сеанс и откройте его карту в 3D. Двойной щелчок открывает карту; папку с остальными файлами можно открыть отдельно.",wraplength=1000,style="Muted.TLabel").pack(anchor="w",pady=(0,ui.SPACE_M))
        frame=ttk.Frame(outer);frame.pack(fill="both",expand=True)
        self.table=ttk.Treeview(frame,columns=("date","kind","size","name","state"),show="headings",selectmode="extended")
        for key,title,width in (("date","Изменено (МСК)",135),("kind","Тип",165),("size","Размер",85),("name","Сеанс",245),("state","Состояние",200)):
            self.table.heading(key,text=title,anchor="w");self.table.column(key,width=width,minwidth=65)
        scroll=ttk.Scrollbar(frame,orient="vertical",command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set);self.table.pack(side="left",fill="both",expand=True);scroll.pack(side="right",fill="y")
        self.table.bind("<<TreeviewSelect>>",lambda event:self.selected())
        self.table.bind("<Double-1>",lambda event:self.open_cloud())
        ttk.Label(outer,textvariable=self.selection,style="Muted.TLabel").pack(anchor="w",pady=ui.SPACE_S)
        bar=ttk.Frame(outer);bar.pack(fill="x")
        self.refresh_button=ttk.Button(bar,text="Обновить список",command=self.refresh);self.refresh_button.pack(side="left")
        self.open_button=ttk.Button(bar,text="Открыть в 3D",command=self.open_cloud,style="Accent.TButton",state="disabled");self.open_button.pack(side="left",padx=(ui.SPACE_S,0))
        ttk.Button(bar,text="Открыть выбранную папку",command=self.open_folder).pack(side="left",padx=(ui.SPACE_S,0))
        self.remove_button=ttk.Button(bar,text="Удалить выбранные…",command=self.remove,style="Danger.TButton");self.remove_button.pack(side="right")
        ttk.Label(outer,text="Удаление убирает папку сеанса целиком. Открытые сеансы защищены; перед очисткой остановите камеры.",style="CardInfo.TLabel",wraplength=1000).pack(anchor="w",pady=(ui.SPACE_M,0))
        self.window.protocol("WM_DELETE_WINDOW",self.close)
        self.refresh();self.poll_id=self.window.after(100,self.poll)
        self.window.bind('<Destroy>',self.on_destroy,add='+')

    def set_busy(self,value):
        self.working=value
        state="disabled" if value else "normal"
        self.remove_button.configure(state=state);self.refresh_button.configure(state=state)
        self.selected()

    def refresh(self):
        if self.working:return
        protected=list(self.protected())
        self.set_busy(True);self.summary.set("Подсчитываю размеры сохранённых исследований…")
        def job():
            try:self.events.put(("scanned",self.catalog.scan(protected)))
            except Exception as error:self.events.put(("error",str(error)))
        Thread(target=job,daemon=True).start()

    def selected(self):
        chosen=[self.items[key] for key in self.table.selection() if key in self.items]
        self.selection.set(f"Выбрано: {len(chosen)} · {size_text(sum(x.size for x in chosen))}. Для нескольких строк используйте Ctrl или Shift.")
        can_open=not self.working and self.on_open_cloud and len(chosen)==1 and session_cloud(chosen[0].path) is not None
        self.open_button.configure(state="normal" if can_open else "disabled")

    def open_cloud(self):
        if self.working or not self.on_open_cloud:return
        selected=self.table.selection()
        if len(selected)!=1 or selected[0] not in self.items:return
        path=session_cloud(self.items[selected[0]].path)
        if path is not None:self.on_open_cloud(path)

    def open_folder(self):
        selected=self.table.selection()
        if selected and selected[0] in self.items:os.startfile(self.items[selected[0]].path)

    def remove(self):
        if self.working:return
        if self.external_busy():
            messagebox.showinfo("Исследования","Для удаления остановите камеры и дождитесь окончания сохранения в других окнах.",parent=self.window);return
        protected=list(self.protected())
        chosen=[self.items[key] for key in self.table.selection() if key in self.items]
        if not chosen:return
        if any(x.blocked or self.catalog.protected(x.path.resolve(),protected) for x in chosen):
            messagebox.showinfo("Защищённый сеанс","Среди выбранных есть открытый или защищённый сеанс. Снимите с него выделение.",parent=self.window);return
        names="\n".join(f"• {x.kind}: {x.path.name}" for x in chosen[:8])
        if len(chosen)>8:names+=f"\n… и ещё {len(chosen)-8}"
        question=f"Удалить безвозвратно {len(chosen)} исследований ({size_text(sum(x.size for x in chosen))})?\n\n{names}\n\nПапки и все файлы внутри будут удалены."
        if not messagebox.askyesno("Удаление исследований",question,parent=self.window,default=messagebox.NO):return
        self.set_busy(True)
        def job():
            removed=0;errors=[]
            for entry in chosen:
                try:removed+=self.catalog.delete(entry.path,protected)
                except Exception as error:errors.append(f"{entry.path.name}: {error}")
            self.events.put(("removed",(removed,errors)))
        Thread(target=job,daemon=True).start()

    def poll(self):
        self.poll_id=None
        if not self.window.winfo_exists():return
        try:
            while True:
                kind,value=self.events.get_nowait();self.set_busy(False)
                if kind=="scanned":
                    self.table.delete(*self.table.get_children());self.items={}
                    for i,item in enumerate(value):
                        key=str(i);self.items[key]=item
                        self.table.insert("", "end",iid=key,values=(item.changed,item.kind,size_text(item.size),item.path.name,item.blocked),tags=("locked",) if item.blocked else ())
                    self.table.tag_configure("locked",foreground=ui.palette().disabled)
                    self.summary.set(f"Сеансов: {len(value)} · всего {size_text(sum(x.size for x in value))}")
                    self.selected()
                elif kind=="removed":
                    removed,errors=value
                    message=f"Удалено файлов объёмом {size_text(removed)}."
                    if errors:message+="\n\nНе удалось удалить:\n"+"\n".join(errors)
                    messagebox.showinfo("Очистка завершена",message,parent=self.window);self.refresh()
                else:messagebox.showerror("Исследования",value,parent=self.window)
        except Empty:pass
        self.poll_id=self.window.after(100,self.poll)

    def on_destroy(self,event):
        if event.widget is self.window and self.poll_id:
            try:self.window.after_cancel(self.poll_id)
            except tk.TclError:pass
            self.poll_id=None

    def close(self):
        if self.working:
            self.selection.set("Дождитесь завершения операции…");return
        self.window.destroy()
