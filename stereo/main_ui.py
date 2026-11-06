"""Task-oriented application layout; numerical processing stays in app/worker."""
import tkinter as tk
from tkinter import ttk
from .help_ui import help_icon
from .maixsense import TOF_LABEL
from .presets import PRESETS
from .ui import apply_theme, note, ScrolledFrame, Foldout, quick_guide, BG


def build(app):
    from .app import ImagePanel
    apply_theme(app.root)
    from .version import VERSION
    app.root.title("StereoMapper · 3D-сканирование · версия " + VERSION)
    app.root.geometry(f"{min(1440,app.root.winfo_screenwidth()-50)}x{min(930,app.root.winfo_screenheight()-80)}")
    app.root.minsize(1000,680)
    app.ui_actions=[]; app.ui_signature=None
    app.connection_pending=False; app.calibration_busy=False; app.calibration_pairs=0
    app.connection_badge=tk.StringVar(value="Камеры не подключены")
    app.calibration_badge=tk.StringVar(value="Калибровка не загружена")
    app.next_hint=tk.StringVar(value="Начните с подключения камер или откройте уже сохранённую 3D-карту.")
    app.scan_hint=tk.StringVar()
    outer=ttk.Frame(app.root,padding=(24,18,24,12),style="App.TFrame"); outer.pack(fill="both",expand=True)
    header=ttk.Frame(outer,style="App.TFrame"); header.pack(fill="x",pady=(0,16))
    brand=ttk.Frame(header,style="App.TFrame"); brand.pack(side="left")
    ttk.Label(brand,text="StereoMapper",style="Title.TLabel").pack(anchor="w")
    ttk.Label(brand,text="От двух камер — к вашей 3D-карте",style="App.TLabel").pack(anchor="w",pady=(2,0))
    ttk.Button(header,text="Как начать  ?",command=lambda:quick_guide(app.root)).pack(side="right",padx=(8,0))
    ttk.Button(header,text="Сохранённые сеансы",command=app.open_storage).pack(side="right",padx=(8,0))
    app.open_model_button=ttk.Button(header,text="Открыть 3D-карту…",command=app.open_model)
    app.open_model_button.pack(side="right")
    badges=ttk.Frame(outer,style="App.TFrame"); badges.pack(fill="x",pady=(0,10))
    app.connection_label=ttk.Label(badges,textvariable=app.connection_badge,style="Badge.TLabel")
    app.connection_label.pack(side="left",padx=(0,8))
    app.calibration_label=ttk.Label(badges,textvariable=app.calibration_badge,style="Badge.TLabel")
    app.calibration_label.pack(side="left")
    app.workflow=ttk.Notebook(outer); app.workflow.pack(fill="both",expand=True)
    from .update_ui import UpdateController
    app.updates = UpdateController(app, outer)
    app.workflow_pages={}; app.ui_panes=[]; app.ui_footers={}

    def page(key,title):
        page=ttk.Frame(app.workflow,style="App.TFrame"); app.workflow.add(page,text=title)
        app.workflow_pages[key]=page
        host=ttk.Frame(page,style="Card.TFrame",width=330)
        host.pack(side="left",fill="y",padx=(0,12)); host.pack_propagate(False)
        footer=ttk.Frame(host,padding=(12,8)); footer.pack(side="bottom",fill="x"); app.ui_footers[key]=footer
        sidebar=ScrolledFrame(host)
        sidebar.pack(fill="both",expand=True)
        view=ttk.Frame(page,style="App.TFrame"); view.pack(side="left",fill="both",expand=True)
        app.ui_panes.append(sidebar)
        return sidebar.content,view

    def heading(parent,title,description):
        ttk.Label(parent,text=title,style="Heading.TLabel",wraplength=280).pack(anchor="w")
        note(parent,description)

    def action(parent,text,fn,capability=None,key=None,primary=False):
        control=app.button(parent,text,fn,key,accent=primary)
        if capability:app.ui_actions.append((control,capability))
        return control

    def fields(parent,descriptions):
        frame=ttk.Frame(parent); frame.pack(fill="x")
        for row,entry in enumerate(descriptions):app.entry(frame,*entry[:2],row,*entry[2:])

    def line(parent):ttk.Separator(parent).pack(fill="x",pady=12)

    camera,views=page("cameras","1  Камеры")
    heading(camera,"Подключим камеры","Проверьте два изображения. Для начала обычно подходят индексы 0 и 1.")
    ttk.Label(camera,text="Тип подключения").pack(anchor="w",pady=(0,3))
    mode_row=ttk.Frame(camera); mode_row.pack(fill="x",pady=(0,7))
    help_icon(mode_row,"mode").pack(side="right",padx=(7,0))
    mode=ttk.Combobox(mode_row,textvariable=app.vars["mode"],values=("Две камеры","Два вида рядом",TOF_LABEL),state="readonly")
    mode.pack(side="left",fill="x",expand=True); app.settings_controls.append((mode,"readonly"))
    fields(camera,[("Левая камера","left"),("Правая камера","right"),("Адрес MaixSense","tof_address")])
    ttk.Label(camera,text="Что вы снимаете?",style="Section.TLabel").pack(anchor="w",pady=(10,4))
    preset=ttk.Combobox(camera,textvariable=app.vars["profile"],values=("Свои настройки",*PRESETS),state="readonly")
    preset.pack(fill="x"); preset.bind("<<ComboboxSelected>>",lambda e:app.apply_preset())
    app.settings_controls.append((preset,"readonly")); app.profile_combo=preset
    note(camera,"Профиль подставит начальные настройки. Его можно выбрать до подключения.")
    app.connect_button=action(app.ui_footers["cameras"],"Подключить камеры",app.toggle_connect,key="backend",primary=True)
    details=Foldout(camera,"Параметры подключения"); details.pack(fill="x",pady=5)
    fields(details.content,[("Ширина кадра","width",("640","800","960","1080","1280","1920")),
                           ("Высота кадра","height",("480","540","600","720","1080")),
                           ("Кадров / сек","fps",("10","15","25","30","50","60")),
                           ("Драйвер Windows","backend",("AUTO","DSHOW","MSMF"))])
    app.check(details.content,"Поменять левую и правую",app.vars["swap"],"swap",setting=True)
    app.check(details.content,"Линии выравнивания",app.guides,"guides")
    line(camera)
    note(camera,"Прикройте левый объектив рукой: должно закрыться левое изображение. Порядок важен для глубины.")
    action(camera,"Далее: калибровка →",lambda:app.show_page("calibration"))
    views.rowconfigure(0,weight=1); views.columnconfigure((0,1),weight=1,uniform="camera")
    left=ImagePanel(views,"Левая камера",app.pick_live,empty="Здесь появится левое изображение",hint="Подключите камеры на панели слева")
    right=ImagePanel(views,"Правая камера",empty="Здесь появится правое изображение",hint="Оба объектива должны видеть одну сцену")
    left.grid(row=0,column=0,sticky="nsew",padx=(0,6)); right.grid(row=0,column=1,sticky="nsew",padx=(6,0))
    app.camera_measurement=ttk.Label(views,textvariable=app.measurement,style="App.TLabel",wraplength=700)
    app.camera_measurement.grid(row=1,column=0,columnspan=2,sticky="w",pady=(8,0)); app.camera_measurement.grid_remove()

    calibration,views=page("calibration","2  Калибровка")
    heading(calibration,"Свяжем пиксели с метрами","Калибровку выполняют один раз для пары камер и разрешения. Готовый файл можно загрузить.")
    action(calibration,"Загрузить калибровку…",app.load_calibration,"connected")
    line(calibration)
    ttk.Label(calibration,text="Или снимите доску",style="Section.TLabel").pack(anchor="w")
    note(calibration,"Сделайте 20–30 снимков: меняйте положение и наклон доски. Перед снимком замирайте.")
    action(calibration,"Показать / напечатать доску…",app.make_board,key="board")
    action(calibration,"Примеры правильных ракурсов",app.open_calibration_help,key="calibration_help")
    calibration_actions=app.ui_footers["calibration"]
    ttk.Label(calibration_actions,textvariable=app.count,style="Section.TLabel").pack(anchor="w",pady=(4,4))
    app.calibration_progress=ttk.Progressbar(calibration_actions,maximum=15); app.calibration_progress.pack(fill="x",pady=(0,5))
    app.capture_button=action(calibration_actions,"Снять пару · пробел / S",app.capture_calibration,"capture",key="capture",primary=True)
    note(calibration,variable=app.calibration_hint)
    app.calibrate_button=action(calibration_actions,"Рассчитать калибровку",lambda:app.send("calibrate"),"calibrate",key="calibrate")
    board_options=Foldout(calibration,"Доска и сохранённые серии"); board_options.pack(fill="x",pady=5)
    fields(board_options.content,[("Углов по горизонтали","columns"),("Углов по вертикали","rows"),("Клетка, мм","square_mm")])
    note(board_options.content,"На экранной доске миниатюра камер скрывается на секунду при съёмке.")
    action(board_options.content,"Начать новую серию",lambda:app.send("new_series"),"connected")
    action(board_options.content,"Продолжить сохранённую серию…",app.restore_series,"connected")
    action(calibration,"Далее: сканирование →",lambda:app.show_page("scan"))
    app.calibration_panel=ImagePanel(views,"Предпросмотр калибровки",empty="Покажите доску обоим объективам",hint="После подключения здесь появится изображение левой камеры")
    app.calibration_panel.pack(fill="both",expand=True)
    ttk.Label(app.calibration_panel,textvariable=app.calibration_text,style="CardInfo.TLabel",wraplength=650).pack(fill="x",pady=(8,0))

    scan,views=page("scan","3  Сканирование")
    heading(scan,"Создадим 3D-карту","Сначала проверьте глубину, затем запустите скан. Держите пару ровно в начале и двигайте её медленно.")
    note(scan,variable=app.scan_hint)
    app.depth_button=action(app.ui_footers["scan"],"Включить глубину",app.toggle_depth,"calibrated",key="depth")
    app.scan_button=action(app.ui_footers["scan"],"Новый 3D-скан…",app.open_object_scan,"calibrated",key="object_scan",primary=True)
    note(scan,"Скан сам сохранит результат. В 3D-редакторе можно убрать фон, выровнять верх и сшить ракурсы.")
    line(scan)
    ttk.Label(scan,text="Непрерывная карта",style="Section.TLabel").pack(anchor="w")
    mapping=app.check(scan,"Накапливать при движении",app.mapping,"mapping",lambda:app.send("mapping",app.mapping.get()))
    app.ui_actions.append((mapping,"calibrated"))
    action(scan,"Открыть текущую карту в 3D",lambda:app.send("view"),"frame")
    action(scan,"Сохранить кадр и карту…",lambda:app.send("export"),"frame",key="export")
    more_scan=Foldout(scan,"Настройки и действия карты"); more_scan.pack(fill="x",pady=6)
    fields(more_scan.content,[("Дальняя граница, м","max_depth"),("Ближний поиск, см","near_minimum_cm",("5","10","15","20","30"))])
    note(more_scan.content,"Параметры применяются при подключении. Для изменения отключите камеры.")
    action(more_scan.content,"Начать карту заново",lambda:app.send("reset"),"calibrated")
    action(more_scan.content,"Открыть архив карты…",app.open_map_archive,key="archive")
    action(more_scan.content,"Открыть папку результатов",app.open_output)
    app.scan_views=ttk.Notebook(views,style="Inner.TNotebook"); app.scan_views.pack(fill="both",expand=True)
    depth_host=ttk.Frame(app.scan_views); map_host=ttk.Frame(app.scan_views)
    app.scan_views.add(depth_host,text="Глубина"); app.scan_views.add(map_host,text="Карта сверху")
    depth=ImagePanel(depth_host,"Глубина сцены",app.pick,empty="Проверим, что камера видит в 3D",hint="Подключите камеры, загрузите калибровку и включите глубину")
    depth.pack(fill="both",expand=True)
    app.depth_legend_label=ttk.Label(depth,textvariable=app.depth_legend,style="CardInfo.TLabel",wraplength=650)
    app.depth_legend_label.pack(fill="x",pady=(8,0))
    ttk.Label(depth,textvariable=app.measurement,wraplength=650,foreground="#087e80").pack(fill="x",pady=(4,0))
    overview=ImagePanel(map_host,"Карта сверху",empty="Здесь появятся измеренные поверхности",hint="Включите глубину; для накопления отметьте «Накапливать при движении»")
    overview.pack(fill="both",expand=True)
    ttk.Label(overview,textvariable=app.tracking,style="CardInfo.TLabel",wraplength=650).pack(fill="x",pady=(8,0))
    app.panels=[left,right,depth,overview]

    advanced=ttk.Frame(app.workflow,style="App.TFrame"); app.workflow.add(advanced,text="Дополнительно")
    app.workflow_pages["advanced"]=advanced
    tools=ScrolledFrame(advanced,style="Card.TFrame",width=350); tools.pack(side="left",fill="y",padx=(0,12)); tools.pack_propagate(False)
    settings=ScrolledFrame(advanced,style="Card.TFrame"); settings.pack(side="left",fill="both",expand=True)
    app.ui_panes.extend((tools,settings)); tools,settings=tools.content,settings.content
    heading(tools,"Специальные инструменты","Для отдельных задач. Основное сканирование находится на предыдущей вкладке.")
    app.precision_button=action(tools,"Измерить деталь на снимке…",lambda:app.send("precision"),"calibrated",key="precision")
    note(tools,"Подробный неподвижный снимок для размеров листа или небольшой детали. Не нужен для обычной 3D-карты.")
    action(tools,"Глубина по фотографии…",app.open_photos)
    note(tools,"Оценка глубины одного изображения имеет условный масштаб.")
    action(tools,"3D с оценёнными поверхностями",lambda:app.send("view",True),"frame",key="smooth")
    line(tools)
    ttk.Label(tools,text="Диагностика",style="Section.TLabel").pack(anchor="w")
    action(tools,"Проверить рабочую дистанцию",lambda:app.send("distance_report"),"calibrated",key="range")
    action(tools,"Скорость и нагрузка",app.show_performance)
    action(tools,"Архивы карт…",app.open_map_archive)
    action(tools,"Сеансы и место на диске",app.open_storage)
    heading(settings,"Параметры расчёта","Изменяются до подключения камер. Если вы не уверены, начните с готового профиля на вкладке «Камеры».")
    fields(settings,[("Фильтрация","quality",("Быстрая","Сбалансированная","Подробная")),
                     ("Ширина расчёта","processing_width",("640","960","1280","Оригинал")),
                     ("Диапазон поиска, px","disparities",("64","128","192","256","384","512")),
                     ("Вычисления","acceleration",("Авто · NVIDIA / CPU","Процессор","NVIDIA CUDA")),
                     ("Защита от выбросов","ghost_filter",("Обычная","Строгая","Выключена")),
                     ("Шаг точек, мм","voxel_mm",("0.5","1","2","5","10","20","50")),
                     ("Пиксели в карту: шаг","map_stride",("1","2","3","4")),
                     ("Точек для просмотра","preview_points",("100000","250000","500000","1000000"))])
    for text,key in (("Искать близкие предметы","near_priority"),("Сохранять широкое поле зрения","wide_view"),("Стабилизировать неподвижный кадр","temporal")):
        app.check(settings,text,app.vars[key],key,setting=True)
    app.check(settings,"Оценивать гладкие поверхности",app.smooth_depth,"smooth",app.update_surface_mode)
    diagnostics=Foldout(settings,"Технические сведения"); diagnostics.pack(fill="x",pady=10)
    for variable in (app.calibration_text,app.range_text,app.stats,app.map_stats):note(diagnostics.content,variable=variable,width=650)
    for pane in app.ui_panes:pane.attach_wheel()
    footer=ttk.Frame(outer,style="App.TFrame"); footer.pack(fill="x",pady=(12,0))
    ttk.Label(footer,textvariable=app.next_hint,style="App.TLabel",wraplength=1200).pack(anchor="w")
    status=ttk.Label(footer,textvariable=app.status,style="Status.TLabel",wraplength=1200)
    status.pack(anchor="w",pady=(4,0))
    footer.bind("<Configure>",lambda e:[child.configure(wraplength=max(400,e.width-12)) for child in footer.winfo_children()])
    app.workflow.bind("<<NotebookTabChanged>>",lambda e:app.sync_ui(force=True))
    app.root.bind("<F1>",lambda e:quick_guide(app.root))
    app.sync_ui(force=True)
