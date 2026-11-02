"""Plain-language calibration instructions and an illustrated pose plan."""
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image,ImageTk


class CalibrationHelpWindow:
    def __init__(self,parent,directory):
        self.window=tk.Toplevel(parent)
        self.window.help_view=self
        self.window.title('Калибровка · какие ракурсы снимать')
        self.window.geometry('1080x860')
        title=ttk.Frame(self.window,padding=14);title.pack(fill='x')
        ttk.Label(title,text='33 разных ракурса, а не 33 приближения',font=('Segoe UI',18,'bold')).pack(anchor='w')
        ttk.Label(title,text='Фокус и положение объективов фиксируются до съёмки. Доска целиком видна и резкая в обеих камерах.',wraplength=1020).pack(anchor='w',pady=(6,0))
        navigation=ttk.Frame(title);navigation.pack(anchor='w',pady=(8,0))
        host=ttk.Frame(self.window);host.pack(fill='both',expand=True)
        canvas=tk.Canvas(host,highlightthickness=0,background='#eef3f7')
        scroll=ttk.Scrollbar(host,orient='vertical',command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        self.canvas=canvas
        scroll.pack(side='right',fill='y');canvas.pack(side='left',fill='both',expand=True)
        body=ttk.Frame(canvas,padding=16)
        item=canvas.create_window(0,0,window=body,anchor='nw')
        body.bind('<Configure>',lambda event:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda event:canvas.itemconfigure(item,width=event.width))
        def wheel(event):
            canvas.yview_scroll(-int(event.delta/120),'units')
            return 'break'
        self.window.bind('<MouseWheel>',wheel)
        paragraphs=[
            ('1. Подключение и доска','Начните с 1280 × 720 / 30 или стабильного Full HD / 30. Подключите напрямую к компьютеру. 9 × 6 внутренних углов — это 10 × 7 клеток. Измерьте настоящую клетку линейкой. Матовая печать на жёсткой плоскости уменьшает блики и муар; экранную доску также проверяйте линейкой.'),
            ('2. Расстояние','Доска 10 × 7 клеток по 17 мм имеет размер около 170 × 119 мм. Она может не помещаться в общий вид на 15 см. Начните примерно с 30–70 см по реальной видимости и резкости. Если рабочий ближний фокус не даёт резкости там, уменьшите клетки, измерьте их и начните новую серию. Например, клетки по 10 мм дают рисунок 100 × 70 мм. После калибровки фокус не меняется.'),
            ('Если доска на мониторе','Можно оставлять монитор неподвижным и перемещать всю стереопару: смотреть на экран слева/справа, сверху/снизу, направляя камеру на доску. Так появляется наклон относительно плоскости экрана. Сдвиг только вперёд/назад его не создаёт. Объективы и плата должны сохранять взаимное положение.'),
            ('3. Положения — 9 пар','Центр, слева, справа, сверху, снизу и четыре угла. В каждом положении все углы видны обеим камерам. Простое приближение фронтальной доски не заменяет наклон.'),
            ('4. Наклоны — 16 пар','8 пар с разворотом влево/вправо на 20–35° и 8 пар с наклоном верхнего края к камере/от камеры. Меняйте знак наклона, положение и расстояние. Не используйте только один небольшой поворот.'),
            ('5. Сочетания и масштаб — 8 пар','4 пары: одновременно горизонтальный и вертикальный наклон примерно по 15–25°. Ещё 4 пары немного ближе и дальше с умеренным наклоном. Не обрезайте крайние углы доски.'),
            ('6. Каждый снимок','Переставьте доску или камеру, затем полностью остановите обе. Подождите 1–2 секунды и снимите пару. Не захватывайте доску во время движения. Избегайте бликов, смаза и муара экранных клеток.'),
            ('7. Расчёт и проверка','Программа требует минимум 15 пар и 5 заметно наклонных ракурсов. Лучше пройти весь план из 33. Маленький RMS не доказывает точность в миллиметрах: после принятия проверьте новые ракурсы по линиям выравнивания и несколько размеров известного эталона на рабочей дистанции 15/30 см. Повторите измерения 5–10 раз.'),
            ('8. Близкая глубина','До подключения выберите профиль «Зона 1×1 м · с 15 см»: ближний поиск включён, от 10 см, широкое поле зрения. Быстрый профиль с поиском 96 пикселей не охватывает 10–15 см при текущей калибровке. Шаг карты 1 мм задаёт плотность, а не удостоверяет точность.')]
        for heading,text in paragraphs:
            ttk.Label(body,text=heading,font=('Segoe UI',12,'bold')).pack(anchor='w',pady=(12,4))
            ttk.Label(body,text=text,wraplength=980,justify='left').pack(anchor='w',fill='x')
        image_path=Path(directory)/'calibration_poses.png'
        self.photo=None
        if image_path.exists():
            with Image.open(image_path) as source:
                source.thumbnail((980,1600))
                self.photo=ImageTk.PhotoImage(source.copy())
            self.illustration=ttk.Label(body,image=self.photo)
            self.illustration.pack(pady=16)
            def show_diagram():
                self.window.update_idletasks()
                canvas.yview_moveto(max(0,(self.illustration.winfo_y()-8)/max(1,body.winfo_height())))
            self.show_diagram=show_diagram
            ttk.Button(navigation,text='Схема наклонов',command=show_diagram).pack(side='left',padx=(0,8))
        ttk.Button(navigation,text='К инструкции',command=lambda:canvas.yview_moveto(0)).pack(side='left')
        ttk.Button(self.window,text='Закрыть',command=self.window.destroy).pack(anchor='e',padx=16,pady=10)
