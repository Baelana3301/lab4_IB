import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import os
import struct
import numpy as np
from typing import List, Tuple


# Реализация функции хеширования MāHash8 из предыдущей лабораторной работы
class ParkMillerGenerator:
    def __init__(self, seed=1):
        self.a = 16807  # 7^5 = 16807
        self.m = 2 ** 31 - 1  # 2^31 - 1 = 2147483647
        self.state = seed

    def next_int(self):
        """Генерация следующего псевдослучайного числа"""
        self.state = (self.a * self.state) % self.m
        return self.state

    def next_byte(self):
        """Генерация следующего байта"""
        return self.next_int() % 256

    def next_bytes(self, n):
        """Генерация последовательности из n байтов"""
        return bytes([self.next_byte() for _ in range(n)])


# Таблица подстановки из MāHash7
sTable = [
    0xa3, 0xd7, 0x09, 0x83, 0xf8, 0x48, 0xf6, 0xf4, 0xb3, 0x21, 0x15, 0x78, 0x99, 0xb1, 0xaf, 0xf9,
    0xe7, 0x2d, 0x4d, 0x8a, 0xce, 0x4c, 0xca, 0x2e, 0x52, 0x95, 0xd9, 0x1e, 0x4e, 0x38, 0x44, 0x28,
    0x0a, 0xdf, 0x02, 0xa0, 0x17, 0xf1, 0x60, 0x68, 0x12, 0xb7, 0x7a, 0xc3, 0xe9, 0xfa, 0x3d, 0x53,
    0x96, 0x84, 0x6b, 0xba, 0xf2, 0x63, 0x9a, 0x19, 0x7c, 0xae, 0xe5, 0xf5, 0xf7, 0x16, 0x6a, 0xa2,
    0x39, 0xb6, 0x7b, 0x0f, 0xc1, 0x93, 0x81, 0x1b, 0xee, 0xb4, 0x1a, 0xea, 0xd0, 0x91, 0x2f, 0xb8,
    0x55, 0xb9, 0xda, 0x85, 0x3f, 0x41, 0xbf, 0xe0, 0x5a, 0x58, 0x80, 0x5f, 0x66, 0x0b, 0xd8, 0x90,
    0x35, 0xd5, 0xc0, 0xa7, 0x33, 0x06, 0x65, 0x69, 0x45, 0x00, 0x94, 0x56, 0x6d, 0x98, 0x9b, 0x76,
    0x97, 0xfc, 0xb2, 0xc2, 0xb0, 0xfe, 0xdb, 0x20, 0xe1, 0xeb, 0xd6, 0xe4, 0xd4, 0x47, 0x4a, 0x1d,
    0x42, 0xed, 0x9e, 0x6e, 0x49, 0x3c, 0xcd, 0x43, 0x27, 0xd2, 0x07, 0xd4, 0xde, 0xc7, 0x67, 0x18,
    0x89, 0xcb, 0x30, 0x1f, 0x8d, 0xc6, 0x8f, 0xaa, 0xc8, 0x74, 0xdc, 0xc9, 0x5d, 0x5c, 0x31, 0xa4,
    0x70, 0x88, 0x61, 0x2c, 0x9f, 0x0d, 0x2b, 0x87, 0x50, 0x82, 0x54, 0x64, 0x26, 0x7d, 0x03, 0x40,
    0x34, 0x4b, 0x1c, 0x73, 0xd1, 0xc4, 0xfd, 0x3b, 0xcc, 0xfb, 0x7f, 0xab, 0xe6, 0x3e, 0x5b, 0xa5,
    0xad, 0x04, 0x23, 0x9c, 0x14, 0x51, 0x22, 0xf0, 0x29, 0x79, 0x71, 0x7e, 0xff, 0x8c, 0x0e, 0xe2,
    0x0c, 0xef, 0xbc, 0x72, 0x75, 0x6f, 0x37, 0xa1, 0xec, 0xd3, 0x8e, 0x62, 0x8b, 0x86, 0x10, 0xe8,
    0x08, 0x77, 0x11, 0xbe, 0x92, 0x4f, 0x24, 0xc5, 0x32, 0x36, 0x9d, 0xcf, 0xf3, 0xa6, 0xbb, 0xac,
    0x5e, 0x6c, 0xa9, 0x13, 0x57, 0x25, 0xb5, 0xe3, 0xbd, 0xa8, 0x3a, 0x01, 0x05, 0x59, 0x2a, 0x46
]


def LROT14(x):
    """Циклический сдвиг влево на 14 бит"""
    return ((x << 14) | (x >> 18)) & 0xFFFFFFFF


def RROT14(x):
    """Циклический сдвиг вправо на 14 бит"""
    return ((x << 18) | (x >> 14)) & 0xFFFFFFFF


def Mahash8(data):
    """Функция хеширования MāHash8"""
    if isinstance(data, str):
        data = data.encode('utf-8')

    length = len(data)
    hash1 = length
    hash2 = length

    for i in range(length):
        byte_val = data[i]
        index = (byte_val + i) & 0xFF

        hash1 += sTable[index]
        hash1 = LROT14(hash1 + ((hash1 << 6) ^ (hash1 >> 11)))

        hash2 += sTable[index]
        hash2 = RROT14(hash2 + ((hash2 << 6) ^ (hash2 >> 11)))

    sh1 = hash1
    sh2 = hash2

    hash1 = ((sh1 >> 16) & 0xFFFF) | ((sh2 & 0xFFFF) << 16)
    hash2 = ((sh2 >> 16) & 0xFFFF) | ((sh1 & 0xFFFF) << 16)

    return (hash1 ^ hash2) & 0xFFFFFFFF


class MatrixCipher:
    """Класс для матричного шифрования с блоком 5 байт"""

    def __init__(self, key_generator):
        self.generator = key_generator
        self.block_size = 5  # Размер блока 5 байт

        # Генерируем набор матриц для каждого байта в блоке
        self.matrices = self._generate_matrices()
        self.inverse_matrices = [self._calculate_inverse_matrix(mat) for mat in self.matrices]

    def _generate_matrices(self):
        """Генерация набора матриц 3x3 на основе пароля"""
        matrices = []
        for _ in range(self.block_size):
            matrix = []
            for i in range(3):
                row = []
                for j in range(3):
                    # Генерируем значения от 1 до 255
                    value = (self.generator.next_int() % 254) + 1
                    row.append(value)
                matrix.append(row)
            matrices.append(np.array(matrix, dtype=np.int64))
        return matrices

    def _calculate_inverse_matrix(self, matrix):
        """Вычисление обратной матрицы по модулю 256"""
        try:
            # Вычисляем определитель
            det = int(np.round(np.linalg.det(matrix)))
            if det == 0:
                # Если определитель 0, используем единичную матрицу
                return np.eye(3, dtype=np.int64)

            # Находим обратный определитель по модулю 256
            det_mod = det % 256
            if det_mod == 0:
                det_mod = 256

            # Находим обратный элемент для определителя
            det_inv = None
            for i in range(1, 256):
                if (det_mod * i) % 256 == 1:
                    det_inv = i
                    break

            if det_inv is None:
                return np.eye(3, dtype=np.int64)

            # Вычисляем присоединенную матрицу
            adjugate = np.zeros((3, 3), dtype=np.int64)
            for i in range(3):
                for j in range(3):
                    # Минор
                    minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
                    minor_det = int(np.round(np.linalg.det(minor)))
                    # Алгебраическое дополнение
                    adjugate[j, i] = (-1) ** (i + j) * minor_det

            # Умножаем на обратный определитель по модулю 256
            inverse = (adjugate * det_inv) % 256
            # Обеспечиваем, чтобы значения были в диапазоне 0-255
            inverse = np.where(inverse < 0, inverse + 256, inverse)
            return inverse.astype(np.int64)

        except Exception:
            # В случае ошибки возвращаем единичную матрицу
            return np.eye(3, dtype=np.int64)

    def _process_block(self, block, matrices):
        """Обработка блока с использованием набора матриц"""
        if len(block) < self.block_size:
            # Дополняем блок нулями если он меньше
            block = block + b'\x00' * (self.block_size - len(block))

        result = bytearray()

        # Обрабатываем каждый байт в блоке
        for i, byte_val in enumerate(block):
            if i < len(matrices):
                # Создаем вектор из текущего байта и двух следующих
                vector = []
                for j in range(3):
                    idx = (i + j) % len(block)
                    vector.append(block[idx])

                vector = np.array(vector, dtype=np.int64)

                # Умножаем на матрицу
                result_vector = np.dot(matrices[i], vector) % 256

                # Берем только первый элемент результата для текущей позиции
                result.append(int(result_vector[0]) % 256)
            else:
                # Если матриц не хватило, просто копируем байт
                result.append(byte_val)

        return bytes(result)

    def encrypt_block(self, block):
        """Шифрование одного блока"""
        return self._process_block(block, self.matrices)

    def decrypt_block(self, block):
        """Дешифрование одного блока"""
        return self._process_block(block, self.inverse_matrices)


class MatrixCipherCBC:
    """Реализация матричного шифрования в режиме CBC"""

    def __init__(self, key_generator):
        self.cipher = MatrixCipher(key_generator)
        self.block_size = self.cipher.block_size

    def _generate_iv(self):
        """Генерация вектора инициализации"""
        return self.cipher.generator.next_bytes(self.block_size)

    def _xor_blocks(self, block1, block2):
        """Побитовое XOR двух блоков"""
        return bytes(a ^ b for a, b in zip(block1, block2))

    def _pad_data(self, data):
        """Дополнение данных до кратного размера блока"""
        padding_length = self.block_size - (len(data) % self.block_size)
        if padding_length == self.block_size:
            padding_length = 0
        return data + bytes([padding_length] * padding_length)

    def _unpad_data(self, data):
        """Удаление дополнения из данных"""
        if not data:
            return data
        padding_length = data[-1]
        if padding_length > 0 and padding_length <= self.block_size:
            # Проверяем, что все байты дополнения одинаковы
            if all(b == padding_length for b in data[-padding_length:]):
                return data[:-padding_length]
        return data

    def encrypt(self, data):
        """Шифрование данных в режиме CBC"""
        if not data:
            return b''

        # Генерируем IV
        iv = self._generate_iv()

        # Дополняем данные
        padded_data = self._pad_data(data)

        blocks = []
        previous_block = iv

        # Шифрование каждого блока
        for i in range(0, len(padded_data), self.block_size):
            block = padded_data[i:i + self.block_size]

            # XOR с предыдущим зашифрованным блоком
            xor_block = self._xor_blocks(block, previous_block)

            # Шифрование блока
            encrypted_block = self.cipher.encrypt_block(xor_block)

            blocks.append(encrypted_block)
            previous_block = encrypted_block

        # Объединяем IV и все блоки
        result = iv + b''.join(blocks)
        return result

    def decrypt(self, data):
        """Дешифрование данных в режиме CBC"""
        if len(data) < self.block_size:
            raise ValueError("Данные слишком короткие для CBC режима")

        # Извлекаем IV
        iv = data[:self.block_size]
        encrypted_data = data[self.block_size:]

        blocks = []
        previous_block = iv

        # Дешифрование каждого блока
        for i in range(0, len(encrypted_data), self.block_size):
            block = encrypted_data[i:i + self.block_size]

            # Дешифрование блока
            decrypted_block = self.cipher.decrypt_block(block)

            # XOR с предыдущим зашифрованным блоком
            xor_block = self._xor_blocks(decrypted_block, previous_block)

            blocks.append(xor_block)
            previous_block = block

        # Объединяем все блоки и удаляем дополнение
        result = b''.join(blocks)
        return self._unpad_data(result)


class BlockCipherApp:
    """Главный класс приложения для блочного шифрования"""

    def __init__(self, root):
        self.root = root
        self.root.title("Блочный шифр - Матричное шифрование + CBC - Вариант 4")
        self.root.geometry("900x800")

        # Инициализация компонентов
        self.generator = None
        self.current_file_content = None
        self.encrypted_content = None
        self.is_encrypted = False

        self.create_widgets()

    def create_widgets(self):
        """Создание элементов интерфейса"""
        # Заголовок
        title_label = tk.Label(self.root,
                               text="Блочный шифр: Матричное шифрование (5 байт)",
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=10)

        # Вкладки
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Вкладка шифрования
        encryption_frame = ttk.Frame(notebook)
        notebook.add(encryption_frame, text="Шифрование/Дешифрование")

        # Вкладка хеширования
        hashing_frame = ttk.Frame(notebook)
        notebook.add(hashing_frame, text="Хеширование")

        self.setup_encryption_tab(encryption_frame)
        self.setup_hashing_tab(hashing_frame)

        # Статус бар
        self.status_var = tk.StringVar()
        self.status_var.set("Готов к работе")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1,
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_encryption_tab(self, parent):
        """Настройка вкладки шифрования"""
        # Фрейм для пароля
        password_frame = ttk.LabelFrame(parent, text="Пароль для шифрования")
        password_frame.pack(fill='x', padx=10, pady=5)

        tk.Label(password_frame, text="Пароль:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.password_entry = tk.Entry(password_frame, show="*", width=30, bg='beige')
        self.password_entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        # Фрейм для выбора режима
        mode_frame = ttk.LabelFrame(parent, text="Режим шифрования")
        mode_frame.pack(fill='x', padx=10, pady=5)

        self.mode_var = tk.StringVar(value="ECB")

        tk.Radiobutton(mode_frame, text="Простое матричное шифрование (ECB)",
                       variable=self.mode_var, value="ECB").pack(anchor='w', padx=5, pady=2)
        tk.Radiobutton(mode_frame, text="Матричное шифрование + CBC",
                       variable=self.mode_var, value="CBC").pack(anchor='w', padx=5, pady=2)

        # Фрейм для информации о алгоритме
        info_frame = ttk.LabelFrame(parent, text="Информация об алгоритме")
        info_frame.pack(fill='x', padx=10, pady=5)

        info_text = """Алгоритм: Матричное шифрование
• Размер блока: 5 байт
• Используется 5 матриц 3x3 (по одной на каждый байт блока)
• Каждая матрица генерируется на основе пароля
• Режимы: ECB (простой) и CBC (сцепление блоков)
• Используется функция хеширования MāHash8 для инициализации генератора"""

        info_label = tk.Label(info_frame, text=info_text, justify=tk.LEFT, font=("Arial", 9))
        info_label.pack(padx=5, pady=5)

        # Фрейм для операций с файлами
        file_frame = ttk.LabelFrame(parent, text="Операции с файлами")
        file_frame.pack(fill='x', padx=10, pady=5)

        self.btn_load_file = tk.Button(file_frame, text="Загрузить файл",
                                       command=self.load_file, padx=10, pady=5, bg="lightblue")
        self.btn_load_file.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_encrypt = tk.Button(file_frame, text="Зашифровать",
                                     command=self.encrypt_file, padx=10, pady=5, bg="lightgreen")
        self.btn_encrypt.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_decrypt = tk.Button(file_frame, text="Расшифровать",
                                     command=self.decrypt_file, padx=10, pady=5, bg="lightyellow")
        self.btn_decrypt.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_save_result = tk.Button(file_frame, text="Сохранить результат",
                                         command=self.save_result, padx=10, pady=5, bg="orange")
        self.btn_save_result.pack(side=tk.LEFT, padx=5, pady=5)

        # Текстовое поле для вывода
        text_frame = ttk.LabelFrame(parent, text="Содержимое файла")
        text_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.file_text = scrolledtext.ScrolledText(text_frame, width=80, height=20, wrap=tk.WORD)
        self.file_text.pack(fill='both', expand=True, padx=5, pady=5)

        # Информация о файле
        self.file_info_var = tk.StringVar()
        self.file_info_var.set("Файл не загружен")
        file_info_label = tk.Label(parent, textvariable=self.file_info_var,
                                   font=("Arial", 10), fg="blue")
        file_info_label.pack(pady=5)

    def setup_hashing_tab(self, parent):
        """Настройка вкладки хеширования"""
        # Фрейм для ввода пароля
        hash_input_frame = ttk.LabelFrame(parent, text="Ввод пароля для хеширования")
        hash_input_frame.pack(fill='x', padx=10, pady=5)

        tk.Label(hash_input_frame, text="Пароль:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.hash_password_entry = tk.Entry(hash_input_frame, width=30, bg='beige')
        self.hash_password_entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        self.hash_password_entry.insert(0, "my_password")

        self.btn_hash = tk.Button(hash_input_frame, text="Вычислить хеш",
                                  command=self.calculate_hash, padx=10, pady=5, bg="lightgreen")
        self.btn_hash.grid(row=0, column=2, padx=5, pady=5)

        # Фрейм для результатов хеширования
        hash_result_frame = ttk.LabelFrame(parent, text="Результат хеширования")
        hash_result_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.hash_result_text = scrolledtext.ScrolledText(hash_result_frame, width=80, height=10,
                                                          wrap=tk.WORD, font=("Courier", 10))
        self.hash_result_text.pack(fill='both', expand=True, padx=5, pady=5)

        # Информация о функции хеширования
        info_frame = ttk.LabelFrame(parent, text="Информация о функции хеширования")
        info_frame.pack(fill='x', padx=10, pady=5)

        info_text = """Функция хеширования: MāHash8
• Использует таблицу подстановки sTable из алгоритма Skipjack
• Работает с двумя 32-разрядными значениями hash1 и hash2
• На каждом шаге выполняет циклические сдвиги и перемешивание битов
• Возвращает 32-битный хеш (8 шестнадцатеричных цифр)"""

        info_label = tk.Label(info_frame, text=info_text, justify=tk.LEFT, font=("Arial", 9))
        info_label.pack(padx=5, pady=5)

    def load_file(self):
        """Загрузка файла для шифрования/дешифрования"""
        filename = filedialog.askopenfilename(
            filetypes=[("All files", "*.*"), ("Text files", "*.txt"), ("Binary files", "*.bin")]
        )

        if filename:
            try:
                with open(filename, 'rb') as f:
                    self.current_file_content = f.read()

                # Сбрасываем флаг шифрования при загрузке нового файла
                self.is_encrypted = False
                self.encrypted_content = None

                # Пытаемся декодировать как текст
                try:
                    text_content = self.current_file_content.decode('utf-8')
                    preview = text_content[:500] + ("..." if len(text_content) > 500 else "")
                    self.file_text.delete(1.0, tk.END)
                    self.file_text.insert(tk.END, preview)
                    file_type = "Текстовый"
                    self.is_encrypted = self.looks_like_encrypted(text_content)
                except UnicodeDecodeError:
                    # Бинарный файл
                    hex_preview = self.current_file_content[:100].hex()
                    self.file_text.delete(1.0, tk.END)
                    self.file_text.insert(tk.END, f"Бинарные данные (первые 100 байт в HEX):\n{hex_preview}")
                    file_type = "Бинарный"
                    self.is_encrypted = True

                status_info = f"Загружен {file_type} файл: {os.path.basename(filename)} ({len(self.current_file_content)} байт)"
                if self.is_encrypted:
                    status_info += " [ВОЗМОЖНО ЗАШИФРОВАН]"

                self.file_info_var.set(status_info)
                self.status_var.set(f"Файл загружен: {os.path.basename(filename)}")

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при загрузке файла: {str(e)}")
                self.status_var.set("Ошибка загрузки файла")

    def looks_like_encrypted(self, text):
        """Пытается определить, является ли текст зашифрованным"""
        if len(text) == 0:
            return False

        printable_count = sum(1 for char in text if 32 <= ord(char) <= 126)
        printable_ratio = printable_count / len(text)

        # Если менее 70% печатных символов - считаем возможным зашифрованным
        return printable_ratio < 0.7

    def initialize_generator(self):
        """Инициализация генератора псевдослучайных чисел на основе пароля"""
        password = self.password_entry.get()
        if not password:
            messagebox.showwarning("Предупреждение", "Введите пароль для инициализации генератора")
            return False

        # Вычисляем хеш пароля для использования как seed
        password_hash = Mahash8(password)
        self.generator = ParkMillerGenerator(password_hash)

        self.status_var.set(f"Генератор инициализирован с хешем пароля: {password_hash:08X}")
        return True

    def encrypt_file(self):
        """Шифрование файла"""
        if self.current_file_content is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите файл")
            return

        if not self.initialize_generator():
            return

        try:
            self.status_var.set("Шифрование...")
            self.root.update()

            mode = self.mode_var.get()

            if mode == "CBC":
                # Используем режим CBC
                cipher = MatrixCipherCBC(self.generator)
                encrypted_data = cipher.encrypt(self.current_file_content)
            else:
                # Используем простой режим ECB
                cipher = MatrixCipher(self.generator)
                # Шифруем каждый блок отдельно
                blocks = []
                for i in range(0, len(self.current_file_content), cipher.block_size):
                    block = self.current_file_content[i:i + cipher.block_size]
                    encrypted_block = cipher.encrypt_block(block)
                    blocks.append(encrypted_block)
                encrypted_data = b''.join(blocks)

            self.encrypted_content = encrypted_data
            self.is_encrypted = True

            # Показываем превью зашифрованных данных
            try:
                text_preview = encrypted_data[:200].decode('utf-8', errors='ignore')
                preview = f"Зашифрованные данные (первые 200 символов):\n{text_preview}"
            except:
                hex_preview = encrypted_data[:100].hex()
                preview = f"Зашифрованные бинарные данные (первые 100 байт в HEX):\n{hex_preview}"

            self.file_text.delete(1.0, tk.END)
            self.file_text.insert(tk.END, preview)

            self.file_info_var.set(f"Файл зашифрован ({mode} режим). Размер: {len(encrypted_data)} байт [ЗАШИФРОВАН]")
            self.status_var.set(f"Шифрование завершено ({mode} режим)")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при шифровании: {str(e)}")
            self.status_var.set("Ошибка шифрования")

    def decrypt_file(self):
        """Дешифрование файла"""
        if self.current_file_content is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите файл")
            return

        if not self.initialize_generator():
            return

        try:
            self.status_var.set("Дешифрование...")
            self.root.update()

            mode = self.mode_var.get()

            # Используем encrypted_content если есть, иначе current_file_content
            data_to_decrypt = self.encrypted_content if self.encrypted_content is not None else self.current_file_content

            if mode == "CBC":
                # Используем режим CBC
                cipher = MatrixCipherCBC(self.generator)
                decrypted_data = cipher.decrypt(data_to_decrypt)
            else:
                # Используем простой режим ECB
                cipher = MatrixCipher(self.generator)
                # Дешифруем каждый блок отдельно
                blocks = []
                for i in range(0, len(data_to_decrypt), cipher.block_size):
                    block = data_to_decrypt[i:i + cipher.block_size]
                    decrypted_block = cipher.decrypt_block(block)
                    blocks.append(decrypted_block)
                decrypted_data = b''.join(blocks)

            # Показываем результат
            try:
                text_content = decrypted_data.decode('utf-8')
                preview = text_content[:500] + ("..." if len(text_content) > 500 else "")
                self.file_text.delete(1.0, tk.END)
                self.file_text.insert(tk.END, preview)
                file_type = "Текстовый"
            except UnicodeDecodeError:
                hex_preview = decrypted_data[:100].hex()
                self.file_text.delete(1.0, tk.END)
                self.file_text.insert(tk.END, f"Бинарные данные (первые 100 байт в HEX):\n{hex_preview}")
                file_type = "Бинарный"

            self.file_info_var.set(
                f"Файл расшифрован ({mode} режим). {file_type} файл, размер: {len(decrypted_data)} байт")
            self.current_file_content = decrypted_data
            self.is_encrypted = False
            self.encrypted_content = None
            self.status_var.set(f"Дешифрование завершено ({mode} режим)")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при дешифровании: {str(e)}")
            self.status_var.set("Ошибка дешифрования")

    def save_result(self):
        """Сохранение результата (зашифрованного или расшифрованного) в файл"""
        if self.current_file_content is None and self.encrypted_content is None:
            messagebox.showwarning("Предупреждение", "Нет данных для сохранения")
            return

        data_to_save = self.encrypted_content if self.encrypted_content is not None else self.current_file_content

        filename = filedialog.asksaveasfilename(
            defaultextension=".bin",
            filetypes=[("All files", "*.*"), ("Text files", "*.txt"), ("Binary files", "*.bin")]
        )

        if filename:
            try:
                with open(filename, 'wb') as f:
                    f.write(data_to_save)

                messagebox.showinfo("Успех", f"Данные сохранены в файл:\n{filename}")
                self.status_var.set(f"Данные сохранены: {os.path.basename(filename)}")

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при сохранении файла: {str(e)}")
                self.status_var.set("Ошибка сохранения")

    def calculate_hash(self):
        """Вычисление хеша пароля"""
        password = self.hash_password_entry.get()
        if not password:
            messagebox.showwarning("Предупреждение", "Введите пароль для хеширования")
            return

        try:
            hash_value = Mahash8(password)

            result_text = f"""Результат хеширования пароля:
Пароль: '{password}'
Хеш (десятичный): {hash_value}
Хеш (шестнадцатеричный): {hash_value:08X}
Хеш (бинарный): {hash_value:032b}

Детали вычисления:
• Длина пароля: {len(password)} символов
• Использована функция: MāHash8
• Размер хеша: 32 бита (4 байта)"""

            self.hash_result_text.delete(1.0, tk.END)
            self.hash_result_text.insert(tk.END, result_text)
            self.status_var.set(f"Вычислен хеш пароля: {hash_value:08X}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при вычислении хеша: {str(e)}")
            self.status_var.set("Ошибка вычисления хеша")


def main():
    """Главная функция приложения"""
    root = tk.Tk()
    app = BlockCipherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()