"""Task-oriented application layout; numerical processing stays in app/worker."""
import tkinter as tk
from tkinter import ttk
from .accelerator_ui import AcceleratorPanel
from .help_ui import help_icon
from .maixsense import TOF_LABEL
from .presets import PRESETS
from . import ui
from .ui import note, ScrolledFrame, Foldout, quick_guide

SIDEBAR_WIDTH = 332
PAGE_PADDING = ui.SPACE_M
# The settings page uses three columns from this content width, two below it.
WIDE_SETTINGS = 1150


def build(app):
    from .app import ImagePanel
    ui.use_theme(app.vars["theme"].get())
    app.vars["theme"].set(ui.THEME_LABELS[ui.theme_name(app.vars["theme"].get())])
    ui.style_window(app.root)
    ui.set_animations(app.root, app.vars["animations"].get())
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
    outer=ttk.Frame(app.root,style="App.TFrame"); outer.pack(fill="both",expand=True)
    header=ttk.Frame(outer,padding=(ui.SPACE_L,10,ui.SPACE_M,4)); header.pack(fill="x")
    ttk.Label(header,text="StereoMapper",style="Title.TLabel").pack(side="left",padx=(0,ui.SPACE_L))
    # Created in reading order for the Tab key, packed from the right edge.
    app.open_model_button=ttk.Button(header,text="Открыть 3D-карту…",style="Toolbar.TButton",command=app.open_model)
    sessions=ttk.Button(header,text="Сохранённые сеансы",style="Toolbar.TButton",command=app.open_storage)
    guide=ttk.Button(header,text="Как начать  ?",style="Toolbar.TButton",command=lambda:quick_guide(app.root))
    guide.pack(side="right",padx=(ui.SPACE_S,0)); sessions.pack(side="right",padx=(ui.SPACE_S,0)); app.open_model_button.pack(side="right")
    app.connection_label=ttk.Label(header,textvariable=app.connection_badge,style="Badge.TLabel")
    app.connection_label.pack(side="left",padx=(0,ui.SPACE_S))
    app.calibration_label=ttk.Label(header,textvariable=app.calibration_badge,style="Badge.TLabel")
    app.calibration_label.pack(side="left")
    status_bar=ttk.Frame(outer,padding=(ui.SPACE_L,6,ui.SPACE_M,7)); status_bar.pack(side="bottom",fill="x")
    app.workflow=ttk.Notebook(outer); app.workflow.pack(fill="both",expand=True)
    from .update_ui import UpdateController
    app.updates = UpdateController(app, outer, status_bar)
    app.workflow_pages={}; app.ui_panes=[]; app.ui_footers={}

    def page(key,title):
        page=ttk.Frame(app.workflow,style="App.TFrame",padding=PAGE_PADDING); app.workflow.add(page,text=title)
        app.workflow_pages[key]=page
        host=ttk.Frame(page,style="Card.TFrame",width=SIDEBAR_WIDTH,padding=ui.CARD_INSET)
        host.pack(side="left",fill="y",padx=(0,PAGE_PADDING)); host.pack_propagate(False)
        footer=ttk.Frame(host,padding=(ui.SPACE_L,6,ui.SPACE_L,8)); footer.pack(side="bottom",fill="x"); app.ui_footers[key]=footer
        ttk.Separator(host).pack(side="bottom",fill="x")
        sidebar=ScrolledFrame(host)
        sidebar.pack(fill="both",expand=True)
        view=ttk.Frame(page,style="App.TFrame"); view.pack(side="left",fill="both",expand=True)
        app.ui_panes.append(sidebar)
        return sidebar.content,view

    def heading(parent,title,description):
        ttk.Label(parent,text=title,style="Heading.TLabel",wraplength=280).pack(anchor="w")
        note(parent,description)

    def section(parent,title,top=ui.SPACE_S):
        ttk.Label(parent,text=title,style="Section.TLabel").pack(anchor="w",pady=(top,ui.SPACE_XS))

    def action(parent,text,fn,capability=None,key=None,primary=False):
        control=app.button(parent,text,fn,key,accent=primary)
        if capability:app.ui_actions.append((control,capability))
        return control

    def fields(parent,descriptions,stretch=False):
        frame=ttk.Frame(parent); frame.pack(fill="x")
        for row,entry in enumerate(descriptions):app.entry(frame,*entry[:2],row,*entry[2:],stretch=stretch)

    def line(parent):ttk.Separator(parent).pack(fill="x",pady=ui.SPACE_M)

    camera,views=page("cameras","1  Камеры")
    heading(camera,"Подключим камеры","Проверьте два изображения. Для начала обычно подходят индексы 0 и 1.")
    ttk.Label(camera,text="Тип подключения").pack(anchor="w",pady=(ui.SPACE_XS,ui.SPACE_XS))
    mode_row=ttk.Frame(camera); mode_row.pack(fill="x",pady=(0,ui.SPACE_XS))
    # The list comes first in Tab order, so switching to this page does not pop up a hint.
    mode=ttk.Combobox(mode_row,textvariable=app.vars["mode"],values=("Две камеры","Два вида рядом",TOF_LABEL),state="readonly")
    help_icon(mode_row,"mode").pack(side="right",padx=(ui.SPACE_S,0))
    mode.pack(side="left",fill="x",expand=True); app.settings_controls.append((mode,"readonly"))
    fields(camera,[("Левая камера","left"),("Правая камера","right"),("Адрес MaixSense","tof_address")])
    app.check(camera,"MaixSense: отсекать ложную глубину",app.vars["tof_signal_filter"],"tof_signal_filter",command=app.update_signal_filter)
    line(camera)
    section(camera,"Что вы снимаете?",0)
    preset=ttk.Combobox(camera,textvariable=app.vars["profile"],values=("Свои настройки",*PRESETS),state="readonly")
    preset.pack(fill="x"); preset.bind("<<ComboboxSelected>>",lambda e:app.apply_preset())
    app.settings_controls.append((preset,"readonly")); app.profile_combo=preset
    note(camera,"Профиль подставит начальные настройки. Его можно выбрать до подключения.")
    app.connect_button=action(app.ui_footers["cameras"],"Подключить камеры",app.toggle_connect,key="backend",primary=True)
    details=Foldout(camera,"Параметры подключения"); details.pack(fill="x",pady=(0,ui.SPACE_XS))
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
    left.grid(row=0,column=0,sticky="nsew",padx=(0,PAGE_PADDING//2)); right.grid(row=0,column=1,sticky="nsew",padx=(PAGE_PADDING//2,0))
    app.camera_measurement=ttk.Label(views,textvariable=app.measurement,style="App.TLabel",wraplength=700)
    app.camera_measurement.grid(row=1,column=0,columnspan=2,sticky="w",pady=(ui.SPACE_S,0)); app.camera_measurement.grid_remove()

    calibration,views=page("calibration","2  Калибровка")
    heading(calibration,"Свяжем пиксели с метрами","Калибровку выполняют один раз для пары камер и разрешения. Готовый файл можно загрузить.")
    action(calibration,"Загрузить калибровку…",app.load_calibration,"connected")
    line(calibration)
    section(calibration,"Или снимите доску",0)
    note(calibration,"Сделайте 20–30 снимков: меняйте положение и наклон доски. Перед снимком замирайте.")
    action(calibration,"Показать / напечатать доску…",app.make_board,key="board")
    action(calibration,"Примеры правильных ракурсов",app.open_calibration_help,key="calibration_help")
    calibration_actions=app.ui_footers["calibration"]
    ttk.Label(calibration_actions,textvariable=app.count,style="CardTitle.TLabel").pack(anchor="w",pady=(ui.SPACE_XS,ui.SPACE_XS))
    app.calibration_progress=ttk.Progressbar(calibration_actions,maximum=15); app.calibration_progress.pack(fill="x",pady=(0,ui.SPACE_XS))
    app.capture_button=action(calibration_actions,"Снять пару · пробел / S",app.capture_calibration,"capture",key="capture",primary=True)
    note(calibration,variable=app.calibration_hint)
    app.calibrate_button=action(calibration_actions,"Рассчитать калибровку",lambda:app.send("calibrate"),"calibrate",key="calibrate")
    board_options=Foldout(calibration,"Доска и сохранённые серии"); board_options.pack(fill="x",pady=(0,ui.SPACE_XS))
    fields(board_options.content,[("Углов по горизонтали","columns"),("Углов по вертикали","rows"),("Клетка, мм","square_mm")])
    note(board_options.content,"На экранной доске миниатюра камер скрывается на секунду при съёмке.")
    action(board_options.content,"Начать новую серию",lambda:app.send("new_series"),"connected")
    action(board_options.content,"Продолжить сохранённую серию…",app.restore_series,"connected")
    action(calibration,"Далее: сканирование →",lambda:app.show_page("scan"))
    app.calibration_panel=ImagePanel(views,"Предпросмотр калибровки",empty="Покажите доску обоим объективам",hint="После подключения здесь появится изображение левой камеры")
    app.calibration_panel.pack(fill="both",expand=True)
    ttk.Label(app.calibration_panel,textvariable=app.calibration_text,style="CardInfo.TLabel",wraplength=650).pack(fill="x",pady=(ui.SPACE_S,0))

    scan,views=page("scan","3  Сканирование")
    heading(scan,"Создадим 3D-карту","Сначала проверьте глубину, затем запустите скан. Держите пару ровно в начале и двигайте её медленно.")
    note(scan,variable=app.scan_hint,style="TLabel")
    app.depth_button=action(app.ui_footers["scan"],"Включить глубину",app.toggle_depth,"calibrated",key="depth")
    app.scan_button=action(app.ui_footers["scan"],"Новый 3D-скан…",app.open_object_scan,"calibrated",key="object_scan",primary=True)
    note(scan,"Скан сам сохранит результат. В 3D-редакторе можно убрать фон, выровнять верх и сшить ракурсы.")
    line(scan)
    section(scan,"Непрерывная карта",0)
    mapping=app.check(scan,"Накапливать при движении",app.mapping,"mapping",lambda:app.send("mapping",app.mapping.get()))
    app.ui_actions.append((mapping,"calibrated"))
    action(scan,"Открыть текущую карту в 3D",lambda:app.send("view"),"frame")
    action(scan,"Сохранить кадр и карту…",lambda:app.send("export"),"frame",key="export")
    more_scan=Foldout(scan,"Настройки и действия карты"); more_scan.pack(fill="x",pady=(ui.SPACE_XS,0))
    fields(more_scan.content,[("Дальняя граница, м","max_depth"),("Ближний поиск, см","near_minimum_cm",("5","10","15","20","30"))])
    note(more_scan.content,"Параметры применяются при подключении. Для изменения отключите камеры.")
    action(more_scan.content,"Начать карту заново",lambda:app.send("reset"),"calibrated")
    action(more_scan.content,"Открыть архив карты…",app.open_map_archive,key="archive")
    action(more_scan.content,"Открыть папку результатов",app.open_output)
    app.scan_views=ttk.Notebook(views,style="Inner.TNotebook"); app.scan_views.pack(fill="both",expand=True)
    depth_host=ttk.Frame(app.scan_views,style="App.TFrame"); map_host=ttk.Frame(app.scan_views,style="App.TFrame")
    app.scan_views.add(depth_host,text="Глубина"); app.scan_views.add(map_host,text="Карта сверху")
    depth=ImagePanel(depth_host,"Глубина сцены",app.pick,empty="Проверим, что камера видит в 3D",hint="Подключите камеры, загрузите калибровку и включите глубину")
    depth.pack(fill="both",expand=True)
    app.depth_legend_label=ttk.Label(depth,textvariable=app.depth_legend,style="CardInfo.TLabel",wraplength=650)
    app.depth_legend_label.pack(fill="x",pady=(ui.SPACE_S,0))
    ttk.Label(depth,textvariable=app.measurement,wraplength=650,style="Accent.TLabel").pack(fill="x",pady=(ui.SPACE_XS,0))
    overview=ImagePanel(map_host,"Карта сверху",empty="Здесь появятся измеренные поверхности",hint="Включите глубину; для накопления отметьте «Накапливать при движении»")
    overview.pack(fill="both",expand=True)
    ttk.Label(overview,textvariable=app.tracking,style="CardInfo.TLabel",wraplength=650).pack(fill="x",pady=(ui.SPACE_S,0))
    app.panels=[left,right,depth,overview]

    weighing,views=page("weighing","4  Взвешивание")
    from .weighing_ui import WeighingTab
    app.weighing_tab=WeighingTab(app,app.workflow_pages["weighing"],weighing,views,app.ui_footers["weighing"],heading,action,ImagePanel)

    advanced=ttk.Frame(app.workflow,style="App.TFrame",padding=PAGE_PADDING); app.workflow.add(advanced,text="Дополнительно")
    app.workflow_pages["advanced"]=advanced
    tools=ScrolledFrame(advanced,style="Card.TFrame",width=SIDEBAR_WIDTH,padding=ui.CARD_INSET); tools.pack(side="left",fill="y",padx=(0,PAGE_PADDING)); tools.pack_propagate(False)
    settings=ScrolledFrame(advanced,style="Card.TFrame",padding=ui.CARD_INSET); settings.pack(side="left",fill="both",expand=True)
    app.ui_panes.extend((tools,settings)); tools,settings=tools.content,settings.content
    heading(tools,"Специальные инструменты","Для отдельных задач. Основное сканирование находится на предыдущей вкладке.")
    app.precision_button=action(tools,"Измерить деталь на снимке…",lambda:app.send("precision"),"calibrated",key="precision")
    note(tools,"Подробный неподвижный снимок для размеров листа или небольшой детали. Не нужен для обычной 3D-карты.")
    action(tools,"Глубина по фотографии…",app.open_photos)
    note(tools,"Оценка глубины одного изображения имеет условный масштаб.")
    action(tools,"3D с оценёнными поверхностями",lambda:app.send("view",True),"frame",key="smooth")
    line(tools)
    section(tools,"Ускорение на видеокарте",0)
    app.accelerator_panel=AcceleratorPanel(tools,app)
    line(tools)
    section(tools,"Диагностика",0)
    action(tools,"Проверить рабочую дистанцию",lambda:app.send("distance_report"),"calibrated",key="range")
    action(tools,"Скорость и нагрузка",app.show_performance)
    action(tools,"Архивы карт…",app.open_map_archive)
    action(tools,"Сеансы и место на диске",app.open_storage)
    calculation=ttk.Frame(settings); modes=ttk.Frame(settings); appearance=ttk.Frame(settings)
    def arrange_settings(wide):
        gap=(0,ui.SPACE_L+ui.SPACE_S)
        calculation.grid(row=0,column=0,sticky="new",padx=gap)
        if wide:
            modes.grid(row=0,column=1,rowspan=1,sticky="new",padx=gap)
            appearance.grid(row=0,column=2,rowspan=1,sticky="new",padx=(0,ui.SPACE_S))
            for column in (0,1,2):settings.columnconfigure(column,weight=1,uniform="settings")
        else:
            modes.grid(row=1,column=0,sticky="new",padx=gap,pady=(ui.SPACE_L,0))
            appearance.grid(row=0,column=1,rowspan=2,sticky="new",padx=(0,ui.SPACE_S))
            settings.columnconfigure(0,weight=3,uniform="settings")
            settings.columnconfigure(1,weight=2,uniform="settings")
            settings.columnconfigure(2,weight=0,uniform="")
    def reflow_settings(event,state={"wide":None}):
        wide=event.width>=WIDE_SETTINGS
        if wide!=state["wide"]:
            state["wide"]=wide
            arrange_settings(wide)
    arrange_settings(False)
    settings.bind("<Configure>",reflow_settings,add="+")
    heading(calculation,"Параметры расчёта","Изменяются до подключения камер. Если вы не уверены, начните с готового профиля на вкладке «Камеры».")
    fields(calculation,[("Фильтрация","quality",("Быстрая","Сбалансированная","Подробная")),
                     ("Ширина расчёта","processing_width",("640","960","1280","Оригинал")),
                     ("Диапазон поиска, px","disparities",("64","128","192","256","384","512")),
                     ("Вычисления","acceleration",("Авто · NVIDIA / CPU","Процессор","NVIDIA CUDA")),
                     ("Защита от выбросов","ghost_filter",("Обычная","Строгая","Выключена")),
                     ("Шаг точек, мм","voxel_mm",("0.5","1","2","5","10","20","50")),
                     ("Пиксели в карту: шаг","map_stride",("1","2","3","4")),
                     ("Точек для просмотра","preview_points",("100000","250000","500000","1000000"))],stretch=True)
    section(modes,"Режимы обработки",0)
    for text,key in (("Искать близкие предметы","near_priority"),("Сохранять широкое поле зрения","wide_view"),("Стабилизировать неподвижный кадр","temporal")):
        app.check(modes,text,app.vars[key],key,setting=True)
    app.check(modes,"Оценивать гладкие поверхности",app.smooth_depth,"smooth",app.update_surface_mode)
    ttk.Label(appearance,text="Оформление",style="Heading.TLabel").pack(anchor="w")
    appearance_note=note(appearance,"Как выглядит программа. На расчёты эти настройки не влияют.")
    theme=ttk.Frame(appearance); theme.pack(fill="x")
    theme.columnconfigure(1,weight=1); theme.columnconfigure(0,minsize=64)
    ttk.Label(theme,text="Тема").grid(row=0,column=0,sticky="w",pady=ui.SPACE_XS)
    theme_box=ttk.Combobox(theme,textvariable=app.vars["theme"],values=tuple(ui.THEME_LABELS.values()),state="readonly")
    theme_box.grid(row=0,column=1,sticky="ew",pady=ui.SPACE_XS)
    theme_box.bind("<<ComboboxSelected>>",lambda e:app.change_theme())
    help_icon(theme,"theme").grid(row=0,column=2,padx=(ui.SPACE_S,0))
    theme_note=note(appearance,"Новая тема применится при следующем запуске программы.")
    app.check(appearance,"Анимации интерфейса",app.vars["animations"],"animations",command=app.change_animations)
    line(appearance)
    ttk.Label(appearance,text="Технические сведения",style="Section.TLabel").pack(anchor="w")
    diagnostics=[note(appearance,variable=variable) for variable in (app.calibration_text,app.range_text,app.stats,app.map_stats)]
    ui.fit_wrap(appearance,appearance_note,theme_note,*diagnostics,margin=ui.SPACE_L)
    for pane in app.ui_panes:pane.attach_wheel()
    messages=ttk.Frame(status_bar); messages.pack(side="left",fill="x",expand=True)
    hint=ttk.Label(messages,textvariable=app.next_hint,style="CardInfo.TLabel",wraplength=1200,justify="left")
    hint.pack(anchor="w")
    status=ttk.Label(messages,textvariable=app.status,style="Status.TLabel",wraplength=1200,justify="left")
    status.pack(anchor="w")
    ui.fit_wrap(messages,hint,status,margin=ui.SPACE_L)
    app.workflow.bind("<<NotebookTabChanged>>",lambda e:app.sync_ui(force=True),add="+")
    app.root.bind("<F1>",lambda e:quick_guide(app.root))
    app.sync_ui(force=True)
