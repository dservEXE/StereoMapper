"""A scan session with clear preparation, progress and result states."""
import tkinter as tk
from tkinter import ttk
from . import ui
from .ui import ScrolledFrame, Foldout, note
from .help_ui import help_icon


def help_icon_width(parent):
    probe=help_icon(parent,"object_scan")
    width=probe.winfo_reqwidth()
    probe.destroy()
    return width


def build(scan,app):
    from .app import ImagePanel
    ui.style_window(scan.window)
    scan.window.title("Новый 3D-скан · StereoMapper")
    scan.window.geometry("1180x840"); scan.window.minsize(1000,680)
    header=ttk.Frame(scan.window,padding=(ui.SPACE_L,10,ui.SPACE_M,10)); header.pack(fill="x")
    ttk.Label(header,text="Новый 3D-скан",style="Heading.TLabel").pack(side="left")
    help_icon(header,"object_scan").pack(side="right")
    ttk.Label(header,text="Подготовка  →  Съёмка  →  Готовая карта",style="Muted.TLabel").pack(side="right",padx=ui.SPACE_M)
    scan.message=tk.StringVar(value="Результат сохраняется автоматически после завершения съёмки.")
    bottom=ttk.Frame(scan.window,padding=(ui.SPACE_L,ui.SPACE_S)); bottom.pack(side="bottom",fill="x")
    label=ttk.Label(bottom,textvariable=scan.message,wraplength=1100,style="Accent.TLabel"); label.pack(anchor="w")
    ui.fit_wrap(bottom,label,margin=ui.SPACE_L*2)
    body=ttk.Frame(scan.window,style="App.TFrame",padding=ui.SPACE_M); body.pack(fill="both",expand=True)
    sidebar=ttk.Frame(body,width=332,style="Card.TFrame",padding=ui.CARD_INSET); sidebar.pack(side="left",fill="y",padx=(0,ui.SPACE_M)); sidebar.pack_propagate(False)
    buttons=ttk.Frame(sidebar,padding=(ui.SPACE_L,ui.SPACE_S)); buttons.pack(side="bottom",fill="x")
    ttk.Separator(sidebar).pack(side="bottom",fill="x")
    pane=ScrolledFrame(sidebar); pane.pack(fill="both",expand=True); settings=pane.content
    ttk.Label(settings,text="Перед началом",style="Section.TLabel").pack(anchor="w")
    note(settings,"Предмет неподвижен, свет ровный. Держите стереопару горизонтально: левая камера слева по направлению взгляда.")
    scan.width=tk.StringVar(value="Оригинал"); scan.maximum=tk.StringVar(value="60"); scan.voxel=tk.StringVar(value="1")
    scan.auto=tk.BooleanVar(value=True)
    scan.max_depth=tk.StringVar(value=str(app.worker.max_depth if app.worker else 1.2))
    scan.near_minimum=tk.StringVar(value=f"{app.worker.near_minimum*100:g}" if app.worker else "10")
    scan.ghost_filter=tk.StringVar(value="Строгая" if app.worker and app.worker.ghost_filter=="strict" else "Обычная")
    scan.controls=[]
    def field(parent,label,var,values,key=None):
        row=ttk.Frame(parent); row.pack(fill="x",pady=ui.SPACE_XS)
        ttk.Label(row,text=label,wraplength=145).pack(side="left")
        # Rows without a hint keep the same right edge for their field.
        if key:help_icon(row,key).pack(side="right",padx=(ui.SPACE_S,0))
        else:ttk.Frame(row,width=help_icon_width(row)).pack(side="right",padx=(ui.SPACE_S,0))
        combo=ttk.Combobox(row,textvariable=var,values=values,width=9,state="readonly")
        combo.pack(side="right"); scan.controls.append(combo)
    field(settings,"Глубина до, м",scan.max_depth,("0.5","0.8","1.2","2","3","5","8"),"max_depth")
    field(settings,"Время, сек",scan.maximum,("30","60","120","0"))
    field(settings,"Шаг точек, мм",scan.voxel,("0.5","1","2"),"voxel_mm")
    note(settings,"0 убирает лимит времени. Автостоп может завершить скан раньше. Шаг точек задаёт плотность карты.")
    scan.auto_control=ttk.Checkbutton(settings,text="Завершать при замедлении прироста",variable=scan.auto)
    scan.auto_control.pack(anchor="w",pady=ui.SPACE_XS)
    note(settings,"Автостоп работает после 5 секунд и 90% покрытия выбранной области первого ракурса.")
    more=Foldout(settings,"Дополнительные настройки"); more.pack(fill="x",pady=(0,ui.SPACE_XS))
    field(more.content,"Ширина расчёта",scan.width,("Оригинал","1280","960"),"processing_width")
    field(more.content,"Поиск от, см",scan.near_minimum,("5","10","15","20","30","50"),"near_minimum_cm")
    field(more.content,"Защита от выбросов",scan.ghost_filter,("Обычная","Строгая"),"ghost_filter")
    ttk.Button(more.content,text="Убрать область контроля",command=scan.clear_region).pack(fill="x",pady=ui.SPACE_S)
    scan.region_text=tk.StringVar(value="Необязательно: двумя щелчками на изображении задайте область контроля покрытия. Сохраняется вся измеренная сцена.")
    note(settings,variable=scan.region_text)
    scan.start_button=ttk.Button(buttons,text="Начать сканирование",command=scan.start,style="Accent.TButton")
    scan.start_button.pack(fill="x",pady=ui.SPACE_XS)
    scan.stop_button=ttk.Button(buttons,text="Завершить и сохранить",command=lambda:app.send("scan_stop"),state="disabled")
    scan.stop_button.pack(fill="x",pady=ui.SPACE_XS)
    scan.extend_button=ttk.Button(buttons,text="Продолжать без автостопа",command=lambda:app.send("scan_extend"),state="disabled")
    scan.extend_button.pack(fill="x",pady=ui.SPACE_XS)
    scan.view_button=ttk.Button(buttons,text="Открыть результат в 3D",command=scan.view,state="disabled")
    scan.view_button.pack(fill="x",pady=ui.SPACE_XS)
    pane.attach_wheel()
    workspace=ttk.Frame(body,style="App.TFrame"); workspace.pack(side="left",fill="both",expand=True)
    scan.phase=tk.StringVar(value="Готовы к съёмке")
    scan.summary=tk.StringVar(value="Нажмите «Начать сканирование», затем медленно меняйте ракурс камеры.")
    ttk.Label(workspace,textvariable=scan.phase,style="AppSection.TLabel").pack(anchor="w")
    summary=note(workspace,variable=scan.summary,width=800,style="App.TLabel")
    panels=ttk.Frame(workspace,style="App.TFrame"); panels.pack(fill="both",expand=True)
    panels.columnconfigure((0,1),weight=1,uniform="scan"); panels.rowconfigure(0,weight=1)
    scan.left=ImagePanel(panels,"Сцена и измеренная глубина",scan.pick,empty="Проверьте сцену",hint="После начала здесь будет показана глубина")
    scan.right=ImagePanel(panels,"Где уже есть измерения",empty="Покрытие появится во время съёмки",hint="Зелёным отмечаются измеренные участки первого ракурса")
    scan.left.grid(row=0,column=0,sticky="nsew",padx=(0,ui.SPACE_M//2)); scan.right.grid(row=0,column=1,sticky="nsew",padx=(ui.SPACE_M//2,0))
    scan.coverage=ttk.Progressbar(workspace,maximum=100,style="App.Horizontal.TProgressbar"); scan.coverage.pack(fill="x",pady=(ui.SPACE_M,ui.SPACE_XS))
    scan.coverage_text=tk.StringVar(value="Покрытие первого ракурса: 0%")
    ttk.Label(workspace,textvariable=scan.coverage_text,style="AppSection.TLabel").pack(anchor="w")
    explain=note(workspace,"Чёрное — достоверной глубины нет. Новые ракурсы могут заполнить пропуски. Процент относится к первому ракурсу, а не ко всем сторонам предмета.",width=800,style="App.TLabel")
    ui.fit_wrap(workspace,summary,explain,margin=ui.SPACE_S)
    worker=app.worker
    if worker and worker.engine and worker.raw:
        import cv2
        scan.anchor=cv2.remap(worker.raw[0],*worker.engine.maps[0],cv2.INTER_LINEAR)
        scan.left.set(scan.anchor)
