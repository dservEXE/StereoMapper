"""A quiet editing workspace with progressive disclosure of volume controls."""
import tkinter as tk
from tkinter import ttk
from .ui import ScrolledFrame, Foldout, note, NIGHT


def build(v):
    header=ttk.Frame(v.window,padding=(16,12,16,6)); header.pack(fill="x")
    ttk.Label(header,text="Редактор 3D",style="Heading.TLabel").pack(side="left")
    v.button(header,"Помощь  ?",v.show_help).pack_configure(side="right")
    v.caption=ttk.Label(header,style="Muted.TLabel"); v.caption.pack(side="right",padx=12)
    files=ttk.Frame(v.window,padding=(12,0)); files.pack(fill="x")
    v.button(files,"Открыть…",v.open_file)
    v.button(files,"+ Добавить снимок…",v.add_file)
    v.undo_button=v.button(files,"Отменить",v.undo)
    v.redo_button=v.button(files,"Повторить",v.redo)
    v.project_button=v.button(files,"Сохранить проект…",v.save_project)
    v.project_button.configure(style="Accent.TButton"); v.project_button.pack_configure(side="right")
    v.export_button=v.button(files,"Экспорт PLY…",v.save_ply); v.export_button.pack_configure(side="right")
    tools=ttk.Frame(v.window,padding=(16,10,12,0)); tools.pack(fill="x")
    for text,name in (("Вращать","rotate"),("Прямоугольник","rectangle"),("Лассо","lasso"),("Кисть","brush")):
        ttk.Radiobutton(tools,text=text,variable=v.tool,value=name,command=v.tool_changed).pack(side="left",padx=(0,12))
    v.button(tools,"Вписать",v.fit)
    for text,name in (("Спереди","front"),("Сверху","top"),("Справа","right")):
        v.button(tools,text,lambda n=name:v.preset(n))
    v.button(tools,"Сброс вида",v.reset)
    help_label=ttk.Label(v.window,textvariable=v.help_text,style="Muted.TLabel",wraplength=1150)
    help_label.pack(fill="x",padx=18,pady=(8,12))
    body=ttk.Frame(v.window,padding=(12,0)); body.pack(fill="both",expand=True)
    side=ttk.Frame(body,width=335); side.pack(side="right",fill="y",padx=(12,0)); side.pack_propagate(False)
    v.canvas=tk.Canvas(body,background=NIGHT,highlightthickness=0,takefocus=True)
    v.canvas.pack(side="left",fill="both",expand=True)
    book=ttk.Notebook(side,style="Inner.TNotebook"); book.pack(fill="both",expand=True); v.notebook=book
    edit_pane=ScrolledFrame(book); merge_pane=ScrolledFrame(book); orientation=ttk.Frame(book,padding=8)
    book.add(edit_pane,text="Очистка"); book.add(merge_pane,text="Сшивка"); book.add(orientation,text="Ориентация")
    edit,merge=edit_pane.content,merge_pane.content
    v._orientation_ui(orientation)
    ttk.Label(edit,text="Оставьте нужное",style="Section.TLabel").pack(anchor="w")
    note(edit,"Выберите инструмент над картой и обведите область. Жёлтые точки выделены.")
    v.delete_button=v.button(v.row(edit),"Удалить выбранное",v.delete_selection)
    v.delete_button.configure(style="Danger.TButton")
    v.keep_button=v.button(v.row(edit),"Оставить только выбранное",v.keep_selection)
    v.keep_button.configure(style="Accent.TButton")
    v.button(v.row(edit),"Снять выделение  ·  Esc",v.clear_selection)
    note(edit,"Ctrl+Z возвращает удалённое. Исходный скан сохраняется.")
    ttk.Separator(edit).pack(fill="x",pady=8)
    ttk.Label(edit,text="Как объединять выделения",style="Section.TLabel").pack(anchor="w")
    v.operation_combo=v.combo(edit,v.operation,("Новое","Добавить","Вычесть","Пересечение"))
    row=v.row(edit); ttk.Label(row,text="Радиус кисти, px").pack(side="left")
    brush=ttk.Combobox(row,textvariable=v.brush_radius,values=(10,20,30,50,80),width=5,state="readonly")
    brush.pack(side="right"); v._combos.append(brush)
    note(edit,"Кисть сразу добавляет точки. Ctrl вычитает. Каждый новый мазок сохраняет предыдущие.")
    for text,value in (("Насквозь, включая задние точки",False),("Только видимая поверхность",True)):
        ttk.Radiobutton(edit,text=text,variable=v.visible_only,value=value).pack(anchor="w",pady=2)
    volume=Foldout(edit,"Глубина, слои и объём XYZ"); volume.pack(fill="x",pady=(10,0)); advanced=volume.content
    ttk.Label(advanced,text="Область действия").pack(anchor="w")
    v.combo(advanced,v.selection_scope,("Все облака","Выбранное облако"))
    v.layer_combo=v.combo(advanced,v.active_layer,())
    ttk.Checkbutton(advanced,text="Ограничить глубину выделения",variable=v.depth_enabled,command=v.request_draw).pack(anchor="w",pady=8)
    ttk.Label(advanced,text="Ближняя и дальняя границы").pack(anchor="w")
    for var in (v.depth_near,v.depth_far):
        ttk.Scale(advanced,from_=0,to=100,variable=var,command=lambda n:v.request_draw()).pack(fill="x",pady=3)
    note(advanced,"Глубина отсчитывается вдоль взгляда. Поверните карту и проверьте границы.")
    row=v.row(advanced); v.button(row,"Объём XYZ…",v.select_box); v.button(row,"Инверсия",v.invert_selection)
    ttk.Label(advanced,text="Размер точек на экране").pack(anchor="w",pady=(12,0))
    v.combo(advanced,v.point_size,(1,2,3,4),5).bind("<<ComboboxSelected>>",lambda e:v.request_draw())
    ttk.Label(merge,text="Сшейте несколько снимков",style="Section.TLabel").pack(anchor="w")
    note(merge,"1. Добавьте снимок с общей поверхностью.\n2. Укажите одинаковые детали на обоих.\n3. Совместите и проверьте результат.")
    v.button(v.row(merge),"+ Добавить снимок…",v.add_file)
    ttk.Label(merge,text="Основной снимок",style="Section.TLabel").pack(anchor="w",pady=(12,0))
    v.reference_combo=v.combo(merge,v.reference_layer,())
    ttk.Label(merge,text="Добавленный снимок").pack(anchor="w",pady=(7,0))
    v.moving_combo=v.combo(merge,v.moving_layer,())
    for control in (v.reference_combo,v.moving_combo):control.bind("<<ComboboxSelected>>",lambda e:v.cancel_landmarks())
    v.merge_buttons=[]
    for text,fn in (("1. Указать 3 пары точек",v.start_landmarks),("2. Совместить по меткам",v.align_landmarks),("3. Уточнить совпадение (ICP)",v.align_icp)):
        button=v.button(v.row(merge),text,fn); v.merge_buttons.append(button)
    v.merge_buttons[1].configure(style="Accent.TButton")
    row=v.row(merge); v.button(row,"Показать оба",v.show_all); v.button(row,"Сброс меток",v.cancel_landmarks)
    note(merge,variable=v.merge_text)
    note(merge,"Выбирайте разнесённые детали в общей части. Снимки должны иметь одинаковый масштаб. После сшивки сохраните проект для дальнейшего редактирования.")
    edit_pane.attach_wheel(); merge_pane.attach_wheel()
    bottom=ttk.Frame(v.window,padding=(16,10)); bottom.pack(fill="x")
    stats_label=ttk.Label(bottom,textvariable=v.stats,style="CardInfo.TLabel",wraplength=1150); stats_label.pack(anchor="w")
    message=ttk.Label(bottom,textvariable=v.message,wraplength=1150,foreground="#087e80"); message.pack(anchor="w",pady=(5,0))
    def resize(event):
        width=max(400,event.width-36)
        help_label.configure(wraplength=width); stats_label.configure(wraplength=width); message.configure(wraplength=width)
    v.window.bind("<Configure>",lambda e:resize(e) if e.widget is v.window else None,add="+")
