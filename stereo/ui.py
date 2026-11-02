"""Shared visual language and keyboard-accessible disclosure controls."""
import tkinter as tk
from tkinter import ttk

BG = "#f3f6fa"
SURFACE = "#ffffff"
INK = "#1c3048"
MUTED = "#64758b"
ACCENT = "#087e80"
BORDER = "#dce5ee"
NIGHT = "#122033"


def apply_theme(widget):
    root = widget._root()
    if getattr(root, "_stereo_theme", False): return
    root._stereo_theme = True
    root.configure(background=BG)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", font=("Segoe UI", 10), foreground=INK, background=SURFACE)
    style.configure("TFrame", background=SURFACE)
    style.configure("App.TFrame", background=BG)
    style.configure("TLabel", background=SURFACE, foreground=INK)
    style.configure("Muted.TLabel", foreground=MUTED)
    style.configure("App.TLabel", background=BG, foreground=MUTED)
    style.configure("Title.TLabel", font=("Segoe UI", 23, "bold"), background=BG, foreground=INK)
    style.configure("Heading.TLabel", font=("Segoe UI", 14, "bold"))
    style.configure("Section.TLabel", font=("Segoe UI", 11, "bold"))
    style.configure("Card.TFrame", background=SURFACE, bordercolor=BORDER, borderwidth=1, relief="solid")
    style.configure("CardTitle.TLabel", background=SURFACE, font=("Segoe UI", 11, "bold"), foreground=INK)
    style.configure("CardInfo.TLabel", background=SURFACE, font=("Segoe UI", 9), foreground=MUTED)
    style.configure("TButton", font=("Segoe UI", 10), padding=(12, 8), background=SURFACE,
                    foreground=INK, bordercolor=BORDER, borderwidth=1, focusthickness=1, focuscolor=ACCENT)
    style.map("TButton", background=[("disabled", "#f4f6f9"), ("pressed", "#dfebf1"), ("active", "#edf4f7")],
              foreground=[("disabled", "#98a4b3")], bordercolor=[("focus", ACCENT)])
    style.configure("Accent.TButton", background=ACCENT, foreground="white", bordercolor=ACCENT, font=("Segoe UI",10,"bold"))
    style.map("Accent.TButton", background=[("disabled", "#d5e5e7"),("pressed", "#075d63"),("active", "#076d72")],
              foreground=[("disabled", "#7d979b"),("!disabled", "white")], bordercolor=[("!disabled", ACCENT)])
    style.configure("Danger.TButton", foreground="#b43d48", background="#fff6f6", bordercolor="#f0d8da")
    style.map("Danger.TButton", background=[("active", "#fce8eb")], foreground=[("disabled", "#b4a2a5")])
    style.configure("Quiet.TButton", borderwidth=0, background=SURFACE, foreground=ACCENT, padding=(4, 6))
    for control in ("TCheckbutton", "TRadiobutton"):
        style.configure(control, background=SURFACE, padding=(0, 3))
        style.map(control, background=[("active", SURFACE)], foreground=[("disabled", "#98a4b3")])
    style.configure("TEntry", padding=(7,6), fieldbackground=SURFACE, bordercolor=BORDER, lightcolor=BORDER)
    style.configure("TCombobox", padding=(7,6), fieldbackground=SURFACE, background="#eef3f8", bordercolor=BORDER, arrowsize=13)
    style.map("TCombobox", fieldbackground=[("readonly", SURFACE),("disabled", "#f4f6f9")],
              foreground=[("disabled", "#98a4b3")])
    style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0,0,0,10))
    style.configure("TNotebook.Tab", padding=(20,11), background="#e9eff5", foreground=MUTED, font=("Segoe UI",10,"bold"))
    style.map("TNotebook.Tab", background=[("selected",SURFACE),("active","#e0eaf1")],
              foreground=[("selected",ACCENT)], lightcolor=[("selected",SURFACE)], bordercolor=[("selected", BORDER)])
    style.configure("Inner.TNotebook", background=SURFACE, tabmargins=(0,0,0,8))
    style.configure("Inner.TNotebook.Tab", padding=(12,7), font=("Segoe UI",9))
    style.configure("Help.TLabel", background="#e7f3f4", foreground=ACCENT, font=("Segoe UI",9,"bold"), padding=(3,2))
    style.configure("Badge.TLabel", background="#eaf3f6", foreground="#376779", padding=(10,5), font=("Segoe UI",9))
    style.configure("Success.TLabel", background="#e5f4ef", foreground="#246c54", padding=(10,5), font=("Segoe UI",9))
    style.configure("Status.TLabel", background=BG, foreground=INK, font=("Segoe UI",9))
    style.configure("TProgressbar", background=ACCENT, troughcolor="#e6edf3", borderwidth=0, thickness=5)
    style.configure("Treeview", rowheight=32, fieldbackground=SURFACE, background=SURFACE, bordercolor=BORDER)
    style.configure("Treeview.Heading", padding=(8,10), background="#edf2f7", font=("Segoe UI",10,"bold"))
    style.map("Treeview", background=[("selected", "#d8eef0")], foreground=[("selected",INK)])
    style.configure("Vertical.TScrollbar", background="#c8d6e2", troughcolor=BG, borderwidth=0, arrowsize=12)


def note(parent, text=None, variable=None, width=280, **kwargs):
    label=ttk.Label(parent, text=text, textvariable=variable, style="Muted.TLabel",
                    wraplength=width, justify="left", **kwargs)
    label.pack(fill="x", pady=(4,8))
    return label


class ScrolledFrame(ttk.Frame):
    """Wheel handling is local to this pane; never replaces global bindings."""
    def __init__(self,parent,**kwargs):
        super().__init__(parent,**kwargs)
        self.canvas=tk.Canvas(self,background=SURFACE,highlightthickness=0,width=310)
        self.scroll=ttk.Scrollbar(self,orient="vertical",command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.scroll.pack(side="right",fill="y"); self.canvas.pack(side="left",fill="both",expand=True)
        self.content=ttk.Frame(self.canvas,padding=(16,16,12,12))
        item=self.canvas.create_window(0,0,window=self.content,anchor="nw")
        self.content.bind("<Configure>",lambda e:self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>",lambda e:self.canvas.itemconfigure(item,width=e.width))
        # A unique bindtag adds local scrolling without stealing a 3D wheel or a
        # combobox's native choice handling in another window.
        self.tag=f"ScrollPane{id(self)}"
        self.canvas.bind_class(self.tag,"<MouseWheel>",self._wheel)
        self.bind("<Destroy>",self._destroy,add="+")

    def attach_wheel(self):
        def attach(widget):
            if not isinstance(widget,(ttk.Combobox,ttk.Entry,tk.Entry)):
                tags=widget.bindtags()
                if self.tag not in tags:widget.bindtags((tags[0],self.tag,*tags[1:]))
            for child in widget.winfo_children():attach(child)
        attach(self)

    def _wheel(self,event):
        self.canvas.yview_scroll(-int(event.delta/120),"units")
        return "break"

    def _destroy(self,event):
        if event.widget is self:self.unbind_class(self.tag,"<MouseWheel>")


class Foldout(ttk.Frame):
    def __init__(self,parent,title,opened=False):
        super().__init__(parent)
        self.title,self.opened=title,opened
        self.toggle_button=ttk.Button(self,style="Quiet.TButton",command=self.toggle)
        self.toggle_button.pack(fill="x")
        self.content=ttk.Frame(self,padding=(0,4,0,8))
        self._update()

    def toggle(self):
        self.opened=not self.opened; self._update()

    def _update(self):
        self.toggle_button.configure(text=("▾  " if self.opened else "▸  ")+self.title)
        if self.opened:self.content.pack(fill="x")
        else:self.content.pack_forget()


def quick_guide(parent):
    window=tk.Toplevel(parent); window.title("С чего начать · StereoMapper")
    window.geometry("640x560"); window.minsize(500,440)
    buttons=ttk.Frame(window,padding=(20,8,20,16)); buttons.pack(side="bottom",fill="x")
    ttk.Button(buttons,text="Понятно",style="Accent.TButton",command=window.destroy).pack(side="right")
    pane=ScrolledFrame(window); pane.pack(fill="both",expand=True); frame=pane.content
    ttk.Label(frame,text="Ваша первая 3D-карта",style="Heading.TLabel").pack(anchor="w",pady=(0,12))
    for title,text in (("1. Подключите камеры","На вкладке «Камеры» выберите подключение. Прикройте левый объектив: должно закрыться левое изображение."),
                       ("2. Подготовьте калибровку","Она связывает изображения с реальными метрами. Загрузите готовую для этой пары или снимите доску в 20–30 разных положениях. Подсказки находятся на вкладке «Калибровка»."),
                       ("3. Проверьте глубину","На вкладке «Сканирование» включите глубину. Чёрные участки означают, что измерения нет. Добавьте свет, детали или измените ракурс."),
                       ("4. Соберите и очистите карту","Нажмите «Новый 3D-скан», начните съёмку и медленно перемещайте камеры. Завершите скан и откройте 3D: там можно вращать, вырезать фон и сшивать снимки.")):
        ttk.Label(frame,text=title,style="Section.TLabel").pack(anchor="w",pady=(9,0))
        note(frame,text,width=560)
    note(frame,"Уже есть карта? «Открыть 3D-карту» работает без подключённых камер. Значок ? объясняет соседнюю настройку.",width=560)
    pane.attach_wheel()
