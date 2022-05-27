# -*- coding: utf-8 -*-

import os, cv2, validators, yt_dlp, datetime
import numpy as np
import pandas as pd
import glob
import easyocr
import spacy
import ru_core_news_md
from spacy.lang.xx import MultiLanguage
from thefuzz import fuzz
import traceback

from pages import utils

#----------------------------------------------------------------------

class ParserException(RuntimeError):
    pass

class EmptyResultsException(ParserException):
    pass

#----------------------------------------------------------------------

spacy.prefer_gpu()

DATA_DIR = utils.absp('parser')

#----------------------------------------------------------------------

class Parser:
    """
    Класс для распознавания текста на серии изображений и выдачи результата
    в виде единого списка пар "тег - значение", где "тег" - это категория / класс,
    а "значение" - имя/фамилия человека. 

    Класс использует EasyOCR для нахождения текстовых блоков и распознавания текста,
    далее сортирует блоки по строкам и столбцам, определяет, в каких блоках
    содержатся имена/фамилии людей (при помощи SpaCy), и выдает список
    пар "тег - ФИО". Также сами выделенные блоки текста и исходные изображения
    можно обрабатывать в callback-процедуре (например, для формирования данных
    для собственной модели распознавания текста).
    """

    def __init__(self, images=None, langs=['ru', 'en'], on_parse=None,
                 optimize_duplicates=85, ocr_network=None,
                 min_confidence=0, mark_images=True, 
                 contrast_ths=0.4, adjust_contrast=0.7, 
                 width_ths=0.55, add_margin=0.02, mag_ratio=1.7, **kwargs):
        self.images = images
        self.min_confidence = min_confidence
        self.mark_images = mark_images
        self.on_parse = on_parse
        self.optimize_duplicates = optimize_duplicates
        self.ocr_kwargs = kwargs if kwargs else {}
        self.ocr_kwargs.update(dict(contrast_ths=contrast_ths, adjust_contrast=adjust_contrast, 
                                    width_ths=width_ths, add_margin=add_margin, mag_ratio=mag_ratio))
        self.ocr_engine = easyocr.Reader([l for l in langs] if langs else ['ru', 'en'], gpu=True, 
                                         recog_network=ocr_network or 'standard', # custom OCR neural network
                                         model_storage_directory=DATA_DIR, user_network_directory=DATA_DIR)
        if 'ru' in langs:
            self.nlp = ru_core_news_md.load() # Russian model (medium)
        else:
            self.nlp = MultiLanguage()
        
    def parse(self):
        """
        Основной метод класса - разбирает исходные изображения один за другим
        и выводит сформированный список пар "тег - ФИО".
        """
        if not self.images:
            raise ParserException('Нет изображений для обработки!')
        parsed = None
        imgs = []
        cnt_images = len(self.images)
        for i, imgfile in enumerate(self.images):
            try:
                blocks, img = self.ocr(imgfile, self.min_confidence, True, self.mark_images, **self.ocr_kwargs)
                imgs.append(img)
                parsed = self.parse_names(blocks, parsed)
                if self.on_parse:
                    self.on_parse(i, cnt_images, imgfile, img, blocks, parsed)
            except EmptyResultsException:
                print(f'Изображение {i+1} / {cnt_images}: текстовые данные не найдены!')
                continue
            except:
                traceback.print_exc()
                continue
        if self.optimize_duplicates >= 0 and self.optimize_duplicates < 100:
            parsed, opt_cnt = self.optimize_parsed_names(parsed, self.optimize_duplicates)
            if opt_cnt:
                print(f'ИСКЛЮЧЕНО {opt_cnt} ДУБЛИКАТОВ.')
        return (self.parsed_to_df(parsed), imgs)

    def make_training_dataset(self, dataset_name, save_dir=DATA_DIR, images=None):
        """
        Создает датасет из блоков текста (изображений) и таблицы соответствия файл / текст
        для возможности дальнейшего обучения собственной модели OCR, которая может
        скармливаться EasyOCR.

        Описание процесса - https://github.com/JaidedAI/EasyOCR/blob/master/custom_model.md
        """
        if images is None:
            images = self.images
        if images is None:
            raise ParserException('Нет изображений для обработки!')

        dspath = os.path.join(save_dir, dataset_name)
        if not os.path.exists(dspath):
            # создать папку, если ее нет
            os.makedirs(dspath)
        else:
            # удалить все файлы
            for f in glob.glob(os.path.join(dspath, '*')):
                try:
                    os.remove(f)
                except:
                    pass

        pairs = []
        counter = 0
        for imgfile in images:
            try:
                blocks, img = self.ocr(imgfile, self.min_confidence, False, False, **self.ocr_kwargs)
                for _, row in blocks.iterrows():
                    outfile = f'{counter:05}_{dataset_name}.jpg'
                    cropped = img[row.tl_y:row.bl_y, row.tl_x:row.tr_x]
                    cv2.imwrite(os.path.join(dspath, outfile), cropped)
                    pairs.append((outfile, row.text))
                    counter += 1
            except EmptyResultsException:
                print(f'Изображение "{imgfile}": текстовые данные не найдены!')
                continue
            except:
                traceback.print_exc()
                continue

        if not pairs:
            return (None, dspath)

        dfpairs = pd.DataFrame.from_records(pairs, columns=['filename', 'words'])
        dfpairs.to_csv(os.path.join(dspath, 'labels.csv'), index=False)
        return (dfpairs, dspath)

    def parsed_to_df(self, parsed):
        """
        Преобразует список пар "тег - значение" в DataFrame.
        """
        if not parsed:
            return None
        dfparsed = pd.DataFrame.from_records(parsed, columns=['tag', 'name']).drop_duplicates()
        return dfparsed

    def ocr_results_to_dataframe(self, results):
        """
        Преобразует список результатов OCR в DataFrame.
        Описание столбцов - см. https://www.jaided.ai/easyocr/tutorial/
        """
        if not results:
            return None
        flat_list = [[el for l1 in row[0] for el in l1] + list(row[1:]) for row in results]
        df = pd.DataFrame.from_records(flat_list, columns=['tl_x', 'tl_y', 'tr_x', 'tr_y', 'br_x', 'br_y', 'bl_x', 'bl_y', 'text', 'conf'])
        df.insert(8, 'ht', df['bl_y'] - df['tl_y'])
        df.insert(9, 'wd', df['tr_x'] - df['tl_x'])
        dtype = {c: np.int16 for c in df.columns[:10]}
        dtype.update({'text': str, 'conf': float})
        df = df.astype(dtype)
        df['text'] = df['text'].str.strip()
        return df

    def ocr(self, img, min_confidence=0, detect_names=True, mark_image=False, **kwargs):
        """
        Выполняет OCR (распознавание текста) на данном изображении,
        формируя на выходе DataFrame с блоками текста и их координатами / размерами.
        Блоки сортируются по положению: сверху вниз и слева направо.

        kwargs - see https://www.jaided.ai/easyocr/documentation/
        """
        if isinstance(img, str):
            img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not kwargs:
            kwargs = {}
        
        results = self.ocr_results_to_dataframe(self.ocr_engine.readtext(img, **kwargs))
        if results is None:
            raise EmptyResultsException()
        
        if min_confidence:
            results = results.loc[results['conf'] >= min_confidence].copy()

        if results is None:
            raise EmptyResultsException()

        lres = len(results)
        results['tag'] = [''] * lres

        # убрать лишние блоки
        results = self.clean_blocks_heur(results)
        if results is None:
            raise EmptyResultsException()        

        # объединить соседние блоки
        results = self.chain_blocks_heur(results)

        # кластеризация по "строкам", чтобы читать сверху вниз и слева направо
        results = self.cluster_blocks_heur(results)

        if detect_names:
            # определить тип текста: фамилия / имя или нечто другое
            lres = len(results)          
            for i in range(lres):
                txt = results.iat[i, 10]
                while '  ' in txt:
                    txt = txt.replace('  ', ' ')
                spl = txt.split(' ')
                if len(spl) > 1 and all(c[0] == c[0].upper() for c in spl if c):
                    results.iat[i, 12] = 'name'
                    continue
                doc = self.nlp(txt)
                for ent in doc.ents:
                    if ent.label_ == 'PER':
                        results.iat[i, 12] = 'name'
                        break
   
        img1 = img.copy()
        if mark_image:     
            img_shapes = img1.copy()
            alpha = 0.5
            for row in results.itertuples(False):
                color = (230, 230, 50) if row.tag == 'name' else (230, 50, 50)
                cv2.rectangle(img_shapes, (row.tl_x, row.tl_y), (row.br_x, row.br_y), color, -1)
            cv2.addWeighted(img_shapes, alpha, img1, 1 - alpha, 0, img1)

        return (results, img1)

    def clean_blocks_heur(self, blocks):
        """
        Удаление лишних блоков.
        """
        # Убираем блоки, не содержащие букв, а также длиной менее 2 символов
        blocks = blocks.loc[(blocks.text.str.contains(r'\w', False)) & (blocks.text.str.len() > 2)]
        if len(blocks) == 0: return None
        # Вычисляем наиболее частую высоту блока (строки) и удаляем блоки, превышающие 1.5 значения
        mode_ht = blocks.ht.mode().iat[0]
        blocks = blocks.loc[blocks.ht <= (1.5 * mode_ht)]

        return blocks.copy()

    def chain_blocks_heur(self, blocks, horizontal=0.5):
        """
        Сцепляет вместе рядом стоящие однородные блоки, которые не были объединены автоматически.
        """
        mode_ht = blocks.ht.mode().iat[0]
        cnt = len(blocks)
        # горизонтальное сцепление              
        max_dist = round(mode_ht * horizontal)
        i = 0
        while i < cnt:
            if blocks.iat[i, -1] != 'DEL':
                j = 0
                while j < cnt:                    
                    if all([
                        (i != j), # выбрать другой блок
                        (blocks.iat[j, 0] - blocks.iat[i, 2] > 0), # мин. расстояние по горизонтали между tr_x[1] и tl_x[2]
                        (blocks.iat[j, 0] - blocks.iat[i, 2] <= max_dist), # макс. расстояние по горизонтали между tr_x[1] и tl_x[2]
                        (abs(blocks.iat[j, 1] - blocks.iat[i, 1]) <= max_dist) # макс. расстояние по вертикали между tl_y[1] и tl_y[2]
                        ]):
                        # tr_x, tr_y, br_x, br_y
                        blocks.iloc[i, 2:6] = blocks.iloc[j, 2:6].copy()
                        # ht
                        blocks.iat[i, 8] = blocks.iat[i, 7] - blocks.iat[i, 1]
                        # wd
                        blocks.iat[i, 9] = blocks.iat[i, 2] - blocks.iat[i, 0]
                        # text
                        blocks.iat[i, 10] = ' '.join(blocks.iloc[[i, j], 10].to_list())
                        # conf
                        blocks.iat[i, 11] = blocks.iloc[[i, j], 11].mean()
                        # label - отмечаем старый блок для удаления (уже объединили)
                        blocks.iat[j, -1] = 'DEL'
                        i -= 1
                        break
                    j += 1
            i += 1
        blocks = blocks.loc[blocks.tag != 'DEL']

        return blocks.copy()

    def split_blocks_heur(self, blocks, vertical=2.0, horizontal=3.0):
        """
        Разбивает исходные данные (блоки) на группы блоков, разделенные по
        минимальному расстоянию по вертикали и/или горизонтали.
        Возвращает список датафреймов, отсортированный по координатам (tl_x, tl_y).
        """
        pass

    def cluster_blocks_heur(self, blocks, vertical=0.25, horizontal=0.2):
        """
        Выделение строк и столбцов среди блоков, определение выравнивания
        для дальнейшего решения о том, с какой стороны теги (роли), с какой - имена.
        """
        # обычная высота блока
        mode_ht = blocks.ht.mode().iat[0]
        cnt = len(blocks)

        # новый столбец для номеров строк
        blocks['row'] = np.full(len(blocks), -1, dtype=np.int16)

        # макс. отклонение по вертикали для поиска блоков в строке (25% от высоты блока)
        max_dist = round(mode_ht * vertical)

        # вычисляем номера строк для всех блоков
        i = 0
        c = -1
        while i < cnt:
            # пропускаем блоки с уже присвоенным номером строки    
            if blocks.iat[i, -1] == -1:
                c += 1
                j = 0
                while j < cnt:
                    # если встретили блок, чей нижний край отклоняется не более max_dist...
                    if (blocks.iat[j, -1] == -1) and (abs(blocks.iat[j, 5] - blocks.iat[i, 5]) <= max_dist):
                        # присваиваем номер строки
                        blocks.iat[j, -1] = c
                    j += 1                
            i += 1

        # теперь у нас есть блоки по строкам:
        # 0: [__________]   [_____]    [______]
        # 1:      [_________________]  [___]
        # 2:             [______]
        
        last_row = blocks.row.max()
        # сортируем данные по строкам и координате X (лев. верх. вершина)
        blocks.sort_values(['row', 'tl_x'], inplace=True)
        # а теперича создадим столбец для хранения позиции в строке (слева направо)
        blocks['hindex'] = np.full(len(blocks), -1, dtype=np.int16)
        # вычисляем позицию каждого блока в каждой строке - цикл по номерам строк
        for r in range(last_row + 1):
            # все блоки в данной строке
            l = len(blocks.loc[blocks.row == r])
            # просто присваиваем им номера по порядку начиная с 0
            blocks.loc[blocks['row'] == r, 'hindex'] = list(range(l))

        # теперь у нас есть позции блоков в каждой строке:
        # 0: [0_________]   [1____]    [2_____]
        # 1:      [0________________]  [1__]
        # 2:             [0_____]

        # сортируем по строкам и столбцам
        blocks.sort_values(['row', 'hindex'], inplace=True)
        
        return blocks.copy()

    def split_names(self, txt):
        """
        Служебный метод для разбиения строк, разделенных запятой.
        """
        spl = [t.strip() for t in txt.split(',')]
        return spl if len(spl) > 1 else spl[0]

    def parse_names(self, blocks, parsed=None):
        """
        После операции кластеризации (cluster_blocks_heur) проходит по строкам
        и выписывает все пары тег-имя в список.
        """
        if parsed is None:
            # список извлеченных пар тег-имя в формате: 
            # [ [tag, name], [tag, name], ... ]
            parsed = []

        # все блоки уже сортированы по порядку (сверху внизу и слева направо):
        # [1_______]   [2________]
        #     [3____________]
        #    [4_____]  [5____] [6_______]

        # применим простое правило: сверху/слева - тег, справа/снизу - имя
        tags = []
        for _, drow in blocks.iterrows():
            # если наши блок, определенный как ФИО
            if drow.tag == 'name':
                # если есть накопленные выше теги
                if tags:
                    # объединяем теги через пробел и добавляем ФИО
                    parsed.append([' '.join(tags), drow.text])
                    tags.clear()
                # тегов нет, надо посмотреть последний в списке
                else:
                    tag = None
                    # если список уже содержит пары тег-имя
                    if parsed:
                        # ищем первый тег в обратном порядке
                        for t, _ in parsed[::-1]:
                            if t:
                                tag = t
                                break
                    # если тег не нашли, будет None
                    parsed.append([tag, drow.text])
            else:
                tags.append(drow.text)       

        return parsed

    def optimize_parsed_names(self, parsed, match_cutoff=85):
        """
        Оптимизация финальной таблицы с тегами и именами:
        - удаление дублирующихся пар
        """
        if (match_cutoff is None) or (match_cutoff < 0) or (match_cutoff > 100):
            return (parsed, 0)
        parsed1 = [el + [True] for el in parsed]
        lp = len(parsed1)
        for i in range(lp):
            if not parsed1[i][-1]: continue
            for j in range(lp):
                if i == j or not parsed1[j][-1]: continue
                if fuzz.ratio(''.join(parsed1[i][:-1]), ''.join(parsed1[j][:-1])) > match_cutoff:
                    parsed1[j][-1] = False
        parsed2 = [el[:-1] for el in parsed1 if el[-1]]
        opt_count = lp - len(parsed2)
        return (parsed2, opt_count)

#----------------------------------------------------------------------

class Extractor:

    def isurl(s):
        res = validators.url(s.strip())
        try:
            return res
        except:
            return False

    def __init__(self, uri, datadir, parser=None, unique_id=None, 
                 on_download=None, on_extract=None, on_parse=None):
        self.videofile = uri
        self.datadir = datadir        
        self.id = unique_id if not unique_id is None else 'JOB__' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.on_download = on_download 
        self.on_extract = on_extract
        self.parser = parser or Parser(on_parse=on_parse)
        self._get_file()
        
        self.videocap = cv2.VideoCapture(self.videofile)
        self.get_video_properties()

    def _get_file(self):
        if os.path.isfile(self.videofile):
            return
        elif Extractor.isurl(self.videofile):
            try:
                fname = self.download(self.videofile, self.datadir)
                if fname:
                    self.videofile = fname
                else:
                    raise Exception(f'Unable to download URL {self.videofile} to file!')
            except Exception as err:
                raise ParserException(str(err))
        else:
            raise ParserException(f'URI {self.videofile} is not an existing file or a valid URL!')

    def download(self, url, savedir):
        def dhook(d):
            if not self.on_download:
                return
            if d['status'] != 'error':
                self.on_download(d['status'], d['filename'], d.get('downloaded_bytes', 0), d.get('total_bytes', 0), d.get('elapsed', 0), d.get('eta', 0))
            else:
                self.on_download(d['status'], d['filename'])
        ydl_opts = {'format': '(webm/mp4/flv/3gp)/bestvideo[height<=?1080]/best',
                    'progress_hooks': [dhook], 'outtmpl': os.path.join(savedir, '%(title)s.%(ext)s')}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, True)
            filename = ydl.prepare_filename(info)
        try:
            return filename
        except:
            return None

    def get_video_properties(self):
        self.frame_count = int(self.videocap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.videocap.get(cv2.CAP_PROP_FPS)
        self.frame_height = int(self.videocap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.videocap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.duration = self.frame_count / self.fps

    def extract_frames(self, time_start=None, time_end=None, sample_interval=2.0, on_get_frame=None):
        if time_start:
            self.videocap.set(cv2.CAP_PROP_POS_MSEC, time_start * 1000)
        if time_end:
            time_end *= 1000
        else:
            time_end = self.duration * 1000
        sample_interval *= 1000

        i = 0
        ok, img = self.videocap.read()
        while ok:
            i += 1
            cur_time = self.videocap.get(cv2.CAP_PROP_POS_MSEC)
            cur_frame = self.videocap.get(cv2.CAP_PROP_POS_FRAMES)
            if on_get_frame:
                on_get_frame(img, i, cur_time, cur_frame)

            new_time = cur_time + sample_interval
            if new_time > time_end:
                break
            self.videocap.set(cv2.CAP_PROP_POS_MSEC, new_time)
            ok, img = self.videocap.read() 

        # get last frame
        self.videocap.set(cv2.CAP_PROP_POS_MSEC, time_end - self.fps)
        ok, img = self.videocap.read() 
        if ok:
            cur_time = self.videocap.get(cv2.CAP_PROP_POS_MSEC)
            cur_frame = self.videocap.get(cv2.CAP_PROP_POS_FRAMES)
            if on_get_frame:
                on_get_frame(img, i + 1, cur_time, cur_frame)

    def extract_and_save_frames(self, time_start=None, time_end=None, sample_interval=2.0):
        images = []
        def callback(img, n, cur_time, cur_frame):
            nonlocal images
            imfile = os.path.join(self.datadir, f'{self.id}_{n:05}_frame_{cur_frame:.0f}_{cur_time:.0f}ms.jpg')
            cv2.imwrite(imfile, img)
            # print(f'EXTRACTED FRAME {n} ==> "{imfile}"')
            if self.on_extract:
                self.on_extract(img, n, cur_time, cur_frame)
            images.append(imfile)
        self.extract_frames(time_start, time_end, sample_interval, callback)
        return images

    def extract_and_get_frames(self, time_start=None, time_end=None, sample_interval=2.0):
        images = []
        def callback(img, n, cur_time, cur_frame):
            nonlocal images
            # print(f'EXTRACTED FRAME {n}')
            if self.on_extract:
                self.on_extract(img, n, cur_time, cur_frame)
            images.append(img)
        self.extract_frames(time_start, time_end, sample_interval, callback)
        return images

    def extract_and_parse(self, images=None, time_start=None, time_end=None, sample_interval=2.0, **kwargs):
        if not images:
            images = self.extract_and_save_frames(time_start, time_end, sample_interval)
        if not images:
            raise ParserException('Не удалось извлечь фреймы из видео!')
        self.parser.images = images
        if kwargs:
            self.parser.ocr_kwargs.update(kwargs)
        return self.parser.parse() # dfparsed, imgs