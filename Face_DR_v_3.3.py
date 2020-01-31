# импортируем все необходимые модули
import cv2 as cv  # модуль для работы с компьютерным зрением
import numpy as np
import os
from PIL import Image, ImageTk
import time, threading
import tkinter as tk
from tkinter.ttk import Style
from tkinter import ttk, Entry, messagebox, filedialog, Radiobutton, IntVar, Menu

# -------------------------------------------------------------------------------------------------------------------- #

LARGE_FONT = ("Verdana", 12)
name = ""


# класс для графичского интерфейса программы
class Face_DR_GUI(tk.Tk):

    # инициализация, основные действия
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("Face recognition v2.0")  # заголовок окна
        tk.Tk.iconbitmap(self, default='logo1.ico')  # устанавливаем иконку окна
        self.orientation()
        self.resizable(width=0, height=0)  # запрещаем изменение размера окна приложения
        
        self.container = tk.Frame(self)  # переменная container нужна для хранения объекта Frame
        self.container.pack(fill="both", expand=1)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames_dict = {}  # объект словаря для хранения окон приложения
        for frame in (Frame1, Frame2, Frame3, Frame4):
            init_frame = frame(self.container, self)
            self.frames_dict[frame] = init_frame
            init_frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(Frame1)  # на данном этапе отображаем первое окно

        # настраиваем меню (делалось для демонстрационных целей)
        menu_bar = Menu()
        edit_menu = Menu()
        edit_menu.add_command(label="Изменение настроек алгоритмов",
                              command=lambda: messagebox.showinfo("", "Данная функция пока недоступна"))
        menu_bar.add_cascade(label="Файл")
        menu_bar.add_cascade(label="Редактировать", menu=edit_menu)
        menu_bar.add_cascade(label="Вид")
         
        self.config(menu=menu_bar)
        
    # метод, открывающий диалог, для выбора файла: изображение, либо видео
    def load_file(self, file_format):
        chosen_file = filedialog.Open(root, filetypes=[file_format]).show()
        return chosen_file

    # метод открывает отдельное окно для ввода имени папки, в которую будут сохранены изображения лица нового пользователя
    def new_face(self, message):
        popup = tk.Tk()
        popup.wm_title("Дабавление лица")
        label = ttk.Label(popup, text=message, font=("Verdana", 10))
        label.pack(side="top", padx=5, pady=5)
        name_entry = Entry(popup)
        name_entry.pack()
        button = ttk.Button(popup, text="Ok", command=lambda: new_recognizer_app.making_data(name_entry, popup))
        button.pack(side="top", padx=5, pady=5)
        w_width = 200
        w_height = 100
        scr_width = popup.winfo_screenwidth()
        scr_height = popup.winfo_screenheight()
        popup.geometry("%dx%d+%d+%d" % (w_width, w_height, (scr_width-w_width)/2, (scr_height-w_height)/2))
        popup.resizable(width=0, height=0)
        popup.mainloop()

    # метод, в котором происходит настройка размеров окна приложения, его ориентации на экране
    def orientation(self):
        w_width = 500
        w_height = 300
        scr_width = self.winfo_screenwidth()
        scr_height = self.winfo_screenheight()
        self.geometry("%dx%d+%d+%d" % (w_width, w_height, (scr_width-w_width)/2, (scr_height-w_height)/2))

    # метод для отображения выбранного окна
    def show_frame(self, source):
        frame = self.frames_dict[source]
        frame.tkraise()

# -------------------------------------------------------------------------------------------------------------------- #


# класс, содержащий инструменты для обнаружения и распознавания лиц
class PhotoVideoRecognizer():

    def __init__(self, *args, **kwargs):
        self.names = {}
        self.recognizer = []
        print("New photo/video recognizer object has been created")

    # метод для добавления 40 изображений нового лица в базу лиц
    def making_data (self, message, popup):
        storage = 'storage'  # имя хранилища папок с лицами
        person_name = message.get()  # получаем имя нового пользователя
        # если введённое имя уже есть в базе, сообщаем об этом
        if person_name in os.listdir(storage):
            messagebox.showinfo("Операция прервана", "Имя уже используется!")
            popup.attributes("-topmost", True)
            return
        # иначе создаём новую папку, куда сохраняем фрагменты лица с веб-камеры
        else:
            popup.destroy()
            data_path = storage + "/" + person_name
            os.makedirs(data_path)
            cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
            current_camera = cv.VideoCapture(0)  # инициализируем веб-камеру
            image_num = 1
            count = 0
            pause = 0
            total_images = 40
            while count < total_images:
                retval = False
                while not retval:
                    (retval, video_frame) = current_camera.read()  # считываем кадр
                    if not retval:
                        print("Failed to open webcam.")
                        return
                        break
                x1 = 0
                x2 = 10
                cv.putText(video_frame, 'Saving image ', (x1, x2), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))  # отображаем текст
                grey_frame = cv.cvtColor(video_frame, cv.COLOR_BGR2GRAY)  # преобразуем кадр в отенки серого
                found_faces = cascade.detectMultiScale(grey_frame, 1.2, 5)  # обнаруживаем лица с помощью каскадов Хаара
                for (x, y, h, w) in found_faces:
                    cv.rectangle(video_frame, (x, y), (x+w, y+h), (55, 125, 255), 2)  # обводим лицо рамкой
                    cv.putText(video_frame, person_name, (x, y), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                    face_fragment = grey_frame[y:y+h, x:x+w]  # вырезаем фрагмент с лицом
                    fragment_resized = cv.resize(face_fragment, (250, 250), interpolation=cv.INTER_AREA)  # приводим к одному размеру все фрагменты
                    if pause == 0:
                        print('Saving image ' + str(count+1))
                        cv.putText(video_frame, '%16s ' % str(count+1), (x1, x2), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                        cv.imwrite('%s/%s.jpeg' % (data_path, image_num), fragment_resized)  # записываем фрагмент в папку
                        image_num += 1
                        count += 1
                        pause = 1
                    if pause > 0:  # имитация паузы
                        pause = (pause + 1) % 6
                        print(pause)
                cv.imshow('Video captured', video_frame)  # отображаем видеопоток
                k = cv.waitKey(30) & 0xFF  # обработчик нажатия на кнопку Esc
                if k == 27:
                    break
            current_camera.release()  # отключаем камеру
            cv.destroyAllWindows() 
            messagebox.showinfo("Успешно!", "Лицо добавлено в базу")
            
    # ----------------------------------------------------------------------------------------------------- #
    # метод для обнаружения лиц
    def face_detection(self, image, classifier, operation):
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        cascade = cv.CascadeClassifier(classifier)
        found_faces = cascade.detectMultiScale(gray_image, 1.6, 3)
        if operation == "training":
            for(x, y, h, w) in found_faces:
                face_fragment = gray_image[y:y+h, x:x+w]
            face = cv.resize(face_fragment, (250, 250), interpolation=cv.INTER_AREA)
            return face
        elif operation == "detection":
            faces_fragment_list = []
            for(x, y, h, w) in found_faces:
                face_fragment = gray_image[y:y+h, x:x+w]
                faces_fragment_list.append(face_fragment)
            return np.array(faces_fragment_list), found_faces
            
    # ------------------------------------------------Training----------------------------------------------------- #
    # метод, с помощью которого проходим по всей базе с изображениями лиц и собираем данные для тренировки алгоритма
    def prepare_data(self, storage):
        images = []  # список для изображений
        labels = []  # список меток, каждое значение метки соответствет определённому имени, например 0 - Иван, 1 - Алексей и т.д. Метки для разных изображений лиц одного человека одинаковы
        names = {}  # словарь для имён людей / названий папок с изображениями лиц
        id = 0
        subdirs = os.listdir(storage)
        for subdir_name in subdirs:  # проходим в цикле по каждой папке хранилища
            names[id] = subdir_name  # добавляем в словарь имя
            person_path = storage + '/' + subdir_name  # путь до текущей папки с изображениями
            files = os.listdir(person_path) 
            for file_name in files:  # перебираем в цикле изображения из текущей папки
                left_part, right_part = os.path.splitext(file_name)
                if right_part.lower() not in ['.png', '.bmp', '.jpeg', '.gif', '.pgm']:  # проверяем, чтобы файл был допустимого формата
                    print(file_name + ' - wrong file type')
                    continue
                file_path = person_path + '/' + file_name
                label = id
                image = cv.imread(file_path, 0)  # считываем изображение
                (height, width) = image.shape
                if height > 250 and width > 250:  # проверяем размеры изображения (предполагалось, что все изображения лиц должны иметь размеры 250х250 пикселей,
                    # либо меньше, и если изображение имеет бОльшие размеры, значит оно необработано и помимо лица содержит другие части тела, объекты, соответственно,
                    # изображение необходимо обработать - обнаружить лицо)
                    image = self.face_detection(image, 'haarcascade_frontalface_default.xml', "training")
                elif height < 250 and width < 250:
                    image = cv.resize(image, (250, 250), interpolation=cv.INTER_AREA)

                if image is not None:
                    images.append(image)  # добавляем фрагмент лица в список лиц
                    labels.append(label)  # добавляем метку в списо
            id += 1
        return images, labels, names

    # метод запускающий тренировку алгоритма распознавания
    def data_training(self, parent):
        def data_training_inserted():
            images, labels, names = self.prepare_data('storage')  # метод, с помощью которого проходим по всей базе с изображениями лиц
            parent.progress_bar.start()  # запуск полосы прогресса
            self.t1 = time.time()  # замер времени
            recognizer = cv.face.LBPHFaceRecognizer_create(1, 8, 7, 7, 130)  # инициализируем алгоритм
            recognizer.train(images, np.array(labels))  # тренировка алгоритма
            self.t2 = time.time()
            self.training_time = self.t2 - self.t1 
            parent.progress_bar.stop()
            parent.show_time(self.training_time)
            self.names = names
            self.recognizer = recognizer
        threading.Thread(target=data_training_inserted).start()  # запускаем в отдельном потоке data_training_inserted
       
    # ----------------------------------------------------Drawing------------------------------------------------- #
    # метод, рисующий прямоугольник, охватывающий обнаруженное лицо
    def draw_rectangle(self, image, rectangle):
        (x, y, w, h) = rectangle
        cv.rectangle(image, (x, y), (x+w, y+h), (55, 125, 255), 5)
        
    #Метод, выводящий текст на изображение  
    def draw_text(self, image, text, x, y):
        def draw_text_inserted(image, text, x, y): 
            if text[1] < 28:  # если степень сходства меньше 28, тогда пишем имя и само значение степени сходства (считается, что, чем её значение меньше, тем алгоритм более уверен, что это тот или иной челове)
                if image.shape[0] > 500:  # простейшая регулировка размеров надписи
                    cv.putText(image, '%s - %.0f' % text, (x, y), cv.FONT_HERSHEY_DUPLEX, int(image.shape[0]/300), (0, 255, 0), int(image.shape[0]/300))
                else:
                    cv.putText(image, '%s - %.0f' % text, (x, y), cv.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 1)
            else:  # иначе пишем, что лицо не было распознано
                if image.shape[0] > 500:
                    cv.putText(image, 'Not recognized', (x, y), cv.FONT_HERSHEY_DUPLEX, int(image.shape[0]/600), (0, 0, 255), int(image.shape[0]/500))
                else:
                    cv.putText(image, 'Not recognized', (x, y), cv.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 1)
        t1 = threading.Thread(target=draw_text_inserted, args=(image, text, x, y))
        t1.start()
        t1.join()

    # ----------------------------------------------------Prediction------------------------------------------------- #
    # Метод для распознавания обнаруженных лиц
    def predict(self, image, face_fragments, found_faces, source, parent):
        if source == 'photo':
            text = str(len(face_fragments))
            parent.status_message("Total faces found: " + text, 240)
        for face_num in range(len(face_fragments)):  # проходим по всем найденным лицам
            current_face = face_fragments[face_num]
            current_face_resized = cv.resize(current_face, (250, 250), interpolation=cv.INTER_AREA)
            current_rectangle = found_faces[face_num]
            name_of_the_person, confidence = self.recognizer.predict(current_face_resized)  # распознаём лицо с помощью алгоритма LBPH, используя результаты тренировки
            label_text = (self.names[name_of_the_person], confidence)
            self.draw_rectangle(image, current_rectangle)  # вызываем метод, который рисует прямоугольник, охватывающий обнаруженное лицо
            (x, y, w, h) = current_rectangle  # получаем параметры прямоугольника
            self.draw_text(image, label_text, x, y)  # вызываем метод, который выводит на изображение тест, содержащий имя того человека, который был распознан, и степень сходства
        return image

    # ----------------------------------------------------Managers------------------------------------------------- #
    # метод "менеджер" по распознаванию лиц на фото
    def recognize_photo(self, parent):
        path_to_image = root.load_file(('*.jpg files', '.jpg'))  # вызов метода, открывающего диалог выбора файла-изображения
        if not path_to_image: return
        parent.status_message("Trying to recognize the face...", 220)
        image = cv.imread(path_to_image)  # считываем изображение
        face_fragments, found_faces = self.face_detection(image, 'haarcascade_frontalface_default.xml', "detection")  # вызываем метод, обнаруживающий лица
        predicted_image = self.predict(image, face_fragments, found_faces, 'photo', parent)  # вызываем метод для распознавания обнаруженных лиц
        if image.shape[0] > 500:  # в данном условном операторе проверяем размеры изображения, если его высота больше 500 пикселей, то уменьшаем его, сохраняя пропорции
            final_height = 500
            rate = image.shape[0]/final_height
            final_dimensions = (int(image.shape[1]/rate), final_height)
            cv.imshow('Complete', cv.resize(predicted_image, final_dimensions, interpolation=cv.INTER_AREA))
        else:  # либо просто выводим изображение на экран
            cv.imshow('Complete', predicted_image) 

        k = cv.waitKey(30) & 0xFF
        if k == 27:
            cv.destroyAllWindows()

    # метод-менеджер по распознаванию лиц на видео
    def recognize_video(self, source):  # принцип работы аналогичен предыдущему методу
        if source == 0:
            camera = cv.VideoCapture(source)
        else:
            source = root.load_file(('*.avi files', '.avi'))
            if not source: return
            camera = cv.VideoCapture(source)
        while(True):
            retval = False
            while(not retval):
                retval, video_stream = camera.read()
                if not retval:
                    print("Failed to open webcam.")
                    return
                    break
            face_fragments, found_faces = self.face_detection(video_stream, 'haarcascade_frontalface_default.xml', 'detection')
            predicted_image = self.predict(video_stream, face_fragments, found_faces, 'video', None)
            cv.namedWindow('Complete', cv.WINDOW_NORMAL)
            cv.imshow('Complete', predicted_image)
            k = cv.waitKey(30) & 0xFF
            if k == 27:
                break
        camera.release()
        cv.destroyAllWindows()


# GUI windows
"""Ниже представлены 4 класса, соответствующие 4-м окнам графического интерфейса программы. В классах настриваем
оформление, привязываем к кнопкам команды, а также определяем дополнительные методы."""
# ------------------------------------------------------Frame1-------------------------------------------------------- #
# первое окно приложения


class Frame1(tk.Frame):

    def __init__(self, container, parent):
        tk.Frame.__init__(self, container)
        # рамка для разграничения пространства в окнах
        self.helping_frame = tk.Frame(self, relief="ridge", borderwidth=1, background="#333")
        self.helping_frame.pack(fill="both", expand=1)

        label = ttk.Label(self.helping_frame, text="""
Это учебная версия программы для обнаружения/распоз-
навания лиц. На данный момент она позволяет обнару-
живать и распознавать лица на фото или видео и добав-
лять новые лица в базу данных. Программа использует
метод Viola-Jones для обнаружения и метод LBPH для
распознавания. Чтобы начать работу, нажмите "Next"
Чтобы выйти из программы, нажмите "Exit".""",
        background="#333", foreground="white", font=LARGE_FONT)
        label.place(x=5, y=-15)

        image = Image.open("map1.jpg")
        megapolis = ImageTk.PhotoImage(image)
        pic = ttk.Label(self.helping_frame, image=megapolis, background="#333")
        pic.image = megapolis
        pic.pack(side="bottom")

        button_exit = ttk.Button(self, text="Exit", command=parent.destroy)
        button_exit.pack(side="right", pady=5, padx=5)
        button_next = ttk.Button(self, text="Next", command=lambda: parent.show_frame(Frame2))
        button_next.pack(side="right", pady=5, padx=5)


# -------------------------------------------------------Frame2------------------------------------------------------- #
# второе окно приложения

class Frame2(tk.Frame):

    def __init__(self, container, parent):
        tk.Frame.__init__(self, container)
        # рамка для разграничения пространства в окнах
        self.helping_frame = tk.Frame(self, relief="ridge", borderwidth=1, background="#333")
        self.helping_frame.pack(fill="both", expand=1)

        Style().configure("TButton", background="#333")

        button_train = ttk.Button(self.helping_frame, text="Train", command=lambda: self.show_progress_bar())
        button_train.grid(row=0, column=1, padx=10, pady=10)

        label_train = ttk.Label(self.helping_frame, text="""Нажмите "Train", чтобы начать тренировку""",
                                background="#333", foreground="white", font=LARGE_FONT)
        label_train.grid(row=0, column=0, padx=10, pady=10)

        self.progress_bar = ttk.Progressbar(self.helping_frame, length=400, mode="indeterminate")
        self.progress_bar.place(x=50, y=50)

        self.button_photo = ttk.Button(self.helping_frame, text="Photo", command=lambda: parent.show_frame(Frame3))
        self.button_photo.place(x=410, y=110)
        self.button_photo['state'] = 'disabled'

        label_photo = ttk.Label(self.helping_frame, text="""Нажмите "Photo", чтобы распознать лица на фотографии""",
                                background="#333", foreground="white", font=("Verdana", 9))
        label_photo.place(x=10, y=110)

        self.button_video = ttk.Button(self.helping_frame, text="Video", command=lambda: parent.show_frame(Frame4))
        self.button_video.place(x=410, y=150)
        self.button_video['state'] = 'disabled'

        label_video = ttk.Label(self.helping_frame, text="""Нажмите "Video", чтобы распознать лица на видео""",
                                background="#333", foreground="white", font=("Verdana", 9))
        label_video.place(x=10, y=150)

        label_add= ttk.Label(self.helping_frame, text="""
Для добавления нового лица в тренировочную базу с помощью
веб-камеры нажмите кнопку "Add" на нижней панели""",
                             background="#333", foreground="white", font=("Verdana", 9))
        label_add.place(x=10, y=200)

        # bottom buttons
        button_exit = ttk.Button(self, text="Exit", command=parent.destroy)
        button_exit.pack(side="right", pady=5, padx=5)
        button_back = ttk.Button(self, text="Back", command=lambda: parent.show_frame(Frame1))
        button_back.pack(side="right", pady=5, padx=5)
        button_add = ttk.Button(self, text="Add", command=lambda: parent.new_face("Введите в поле имя:"))
        button_add.pack(side="right", pady=5, padx=5)

    def show_progress_bar(self):
        new_recognizer_app.data_training(self)

    def show_time(self, times):
        label_time = ttk.Label(self.helping_frame, text=("Время тренировки: %s" % str(times)),
                               background="#333", foreground="white", font=("Verdana", 8))
        label_time.place(x=50, y=80)
        self.button_photo['state'] = 'normal'
        self.button_video['state'] = 'normal'


# ------------------------------------------------------Frame3-------------------------------------------------------- #
# третье окно приложения

class Frame3(tk.Frame):

    def __init__(self, container, parent):
        tk.Frame.__init__(self, container)
        # рамка для разграничения пространства в окнах
        self.helping_frame = tk.Frame(self, relief="ridge", borderwidth=1, background="#333")
        self.helping_frame.pack(fill="both", expand=1)

        label = ttk.Label(self.helping_frame, text="""
Нажмите на кнопку "Выбрать фото" и выберите фотогра-
фию с помощью файлового проводника(файл с форматом
".jpeg"). После этого нажмите "Открыть", и распозна-
вание начнётся автоматически.
Примечание: важно, чтобы путь к файлу не содержал
символов кириллицы.""",
        background="#333", foreground="white", font=LARGE_FONT)
        label.place(x = 5, y = 10)

        self.button_recognize = ttk.Button(self.helping_frame, text="Выбрать фото",
                                           command=lambda: new_recognizer_app.recognize_photo(self))
        self.button_recognize.pack(side="bottom", pady=50)

        # bottom buttons
        self.button_back = ttk.Button(self, text="Back", command=lambda: parent.show_frame(Frame2))
        self.button_back.pack(side="right", pady=5, padx=90)

    def status_message(self, text_string, y_extern):
        x_extern = 10
        label_status = ttk.Label(self.helping_frame, text=("%s" % text_string),
                                 background="#333", foreground="white", font=("Verdana", 8))
        label_status.place(x=x_extern, y=y_extern)


# ------------------------------------------------------Frame4-------------------------------------------------------- #
# четвёртое окно приложения
class Frame4(tk.Frame):

    def __init__(self, container, parent):
        tk.Frame.__init__(self, container)
        # рамка для разграничения пространства в окнах
        self.helping_frame = tk.Frame(self, relief="ridge", borderwidth=1, background="#333")
        self.helping_frame.pack(fill="both", expand=1)

        label = ttk.Label(self.helping_frame, text="""
Ниже выберите источник видеоизображения: web-каме-
ра, либо существующий видеофайл. При выборе web-ка-
меры нажмите кнопку "Начать", а при выборе файла -
кнопку "Выбрать" и найдите видео с помощью файлового
проводника. После этого нажмите "Открыть", и распо-
знавание начнётся автоматически.
Примечание: важно, чтобы путь к файлу не содержал
символов кириллицы.""",
        background="#333", foreground="white", font=LARGE_FONT)
        label.place(x=5, y=10)

        self.button_start = ttk.Button(self.helping_frame, text="Начать",
                                       command=lambda: new_recognizer_app.recognize_video(0))
        self.button_start.place(x=250, y=200)
        self.button_start['state'] = 'disabled'

        self.button_choose = ttk.Button(self.helping_frame, text="Выбрать",
                                        command=lambda: new_recognizer_app.recognize_video(1))
        self.button_choose.place(x=350, y=200)
        self.button_choose['state'] = 'disabled'

        # bottom buttons
        button_back = ttk.Button(self, text="Back", command=lambda: parent.show_frame(Frame2))
        button_back.pack(side="right", pady=5, padx=90)

        items = [("web", 1), ("file", 2)]
        self.var = IntVar()
        x = 50
        for item, item_num in items:
            Radiobutton(self.helping_frame, text=item, value=item_num, bg="#333",
                        activebackground="#333",  fg="white", selectcolor="black",
                        variable=self.var, command=self.select).place(x=x, y=200)
            x = 150
         
        self.label_info = ttk.Label(self.helping_frame, background="#333", foreground="white")
        self.label_info.place(x=50, y=230)
         
    def select(self):
        value = self.var.get()
        if value == 1:
            self.label_info.config(text="Выбрана web-камера")
            self.button_start['state'] = 'normal'
            self.button_choose['state'] = 'disabled'
        elif value == 2:
            self.label_info.config(text="Выбран файл")
            self.button_choose['state'] = 'normal'
            self.button_start['state'] = 'disabled'


# ------------------------------------------------Main programm------------------------------------------------------- #


root = Face_DR_GUI()
new_recognizer_app = PhotoVideoRecognizer()
root.mainloop()
