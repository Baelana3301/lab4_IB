import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import os
import struct
import numpy as np
from typing import List, Tuple

# Реализация хеш-функции MaHash8 из предыдущей лабораторной работы
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
    0x89, 0xcb, 0x30, 0x1f, 0x8d, 0xc6, 0x8f, 0xaa, 0xc8, 0x74, 0xdc, 0xc9, 0x5d, 0x5e, 0x31, 0xa4,
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


def MaHash8(data):
    """Реализация хеш-функции MaHash8"""
    if isinstance(data, str):
        data = data.encode('utf-8')

    length = len(data)
    hash1 = length
    hash2 = length

    for i in range(length):
        byte_val = data[i]
        index = (byte_val + i) & 0xFF

        # Обновление hash1
        hash1 += sTable[index]
        hash1 = LROT14(hash1 + ((hash1 << 6) ^ (hash1 >> 11))) & 0xFFFFFFFF

        # Обновление hash2
        hash2 += sTable[index]
        hash2 = RROT14(hash2 + ((hash2 << 6) ^ (hash2 >> 11))) & 0xFFFFFFFF

        # Перестановка частей хешей
        sh1 = hash1
        sh2 = hash2
        hash1 = ((sh1 >> 16) & 0xFFFF) | ((sh2 & 0xFFFF) << 16)
        hash2 = ((sh2 >> 16) & 0xFFFF) | ((sh1 & 0xFFFF) << 16)

    return hash1 ^ hash2


def ma_hash8_hex(data):
    """MaHash8 в шестнадцатеричном формате"""
    return format(MaHash8(data), '08x')


# Реализация генератора Парка-Миллера для создания псевдослучайных чисел
class ParkMillerGenerator:
    def __init__(self, seed=1):
        self.a = 16807  # γ5 = 16807
        self.m = 2 ** 31 - 1  # 2^31 - 1 = 2147483647
        self.state = seed

    def next_int(self):
        """Генерация следующего псевдослучайного числа"""
        self.state = (self.a * self.state) % self.m
        return self.state

    def next_float(self):
        """Генерация случайного числа в диапазоне [0, 1)"""
        return self.next_int() / self.m

    def next_int_range(self, min_val, max_val):
        """Генерация случайного числа в заданном диапазоне"""
        return min_val + int(self.next_float() * (max_val - min_val + 1))


# УПРОЩЕННЫЙ И ИСПРАВЛЕННЫЙ MatrixCipher
class MatrixCipher:
    def __init__(self, block_size=5):
        self.block_size = block_size
        # Используем простой модуль для избежания проблем с округлением
        self.modulus = 251  # Простое число меньше 256

    def mod_inverse(self, a, m):
        """Вычисление модульного обратного числа"""

        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y

        gcd, x, _ = extended_gcd(a % m, m)
        if gcd != 1:
            raise ValueError(f"Обратный элемент не существует для {a} mod {m}")
        return (x % m + m) % m

    def generate_key_matrix(self, seed):
        """Генерация обратимой матрицы ключа на основе seed"""
        gen = ParkMillerGenerator(seed)

        # Генерируем простую матрицу с небольшими числами
        while True:
            matrix = []
            for i in range(self.block_size):
                row = []
                for j in range(self.block_size):
                    # Генерируем числа от 1 до 10 для простоты
                    row.append(gen.next_int_range(1, 10))
                matrix.append(row)

            # Проверяем определитель вручную
            try:
                # Простая проверка для матрицы 5x5
                det = self.determinant_5x5(matrix)
                if det % self.modulus == 0:
                    continue

                # Если определитель не нулевой, вычисляем обратную матрицу
                inv_matrix = self.compute_modular_inverse(matrix)
                return matrix, inv_matrix
            except:
                continue

    def determinant_5x5(self, matrix):
        """Вычисление определителя для матрицы 5x5"""
        # Простая реализация для небольших матриц
        a, b, c, d, e = matrix[0]
        f, g, h, i, j = matrix[1]
        k, l, m, n, o = matrix[2]
        p, q, r, s, t = matrix[3]
        u, v, w, x, y = matrix[4]

        # Формула определителя 5x5 (упрощенная)
        det = (
                a * self.det4x4([[g, h, i, j], [l, m, n, o], [q, r, s, t], [v, w, x, y]]) -
                b * self.det4x4([[f, h, i, j], [k, m, n, o], [p, r, s, t], [u, w, x, y]]) +
                c * self.det4x4([[f, g, i, j], [k, l, n, o], [p, q, s, t], [u, v, x, y]]) -
                d * self.det4x4([[f, g, h, j], [k, l, m, o], [p, q, r, t], [u, v, w, y]]) +
                e * self.det4x4([[f, g, h, i], [k, l, m, n], [p, q, r, s], [u, v, w, x]])
        )
        return det % self.modulus

    def det4x4(self, matrix):
        """Вычисление определителя для матрицы 4x4"""
        a, b, c, d = matrix[0]
        e, f, g, h = matrix[1]
        i, j, k, l = matrix[2]
        m, n, o, p = matrix[3]

        return (
                a * self.det3x3([[f, g, h], [j, k, l], [n, o, p]]) -
                b * self.det3x3([[e, g, h], [i, k, l], [m, o, p]]) +
                c * self.det3x3([[e, f, h], [i, j, l], [m, n, p]]) -
                d * self.det3x3([[e, f, g], [i, j, k], [m, n, o]])
        ) % self.modulus

    def det3x3(self, matrix):
        """Вычисление определителя для матрицы 3x3"""
        a, b, c = matrix[0]
        d, e, f = matrix[1]
        g, h, i = matrix[2]

        return (a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)) % self.modulus

    def compute_modular_inverse(self, matrix):
        """Вычисление обратной матрицы в модульной арифметике"""
        n = self.block_size
        # Создаем расширенную матрицу [A|I]
        augmented = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(matrix[i][j] % self.modulus)
            for j in range(n):
                row.append(1 if i == j else 0)
            augmented.append(row)

        # Прямой ход метода Гаусса
        for i in range(n):
            # Ищем ненулевой элемент в столбце i
            pivot = i
            while pivot < n and augmented[pivot][i] == 0:
                pivot += 1
            if pivot == n:
                raise ValueError("Матрица не обратима")

            # Меняем строки местами
            augmented[i], augmented[pivot] = augmented[pivot], augmented[i]

            # Нормализуем ведущую строку
            pivot_val = augmented[i][i]
            inv_pivot = self.mod_inverse(pivot_val, self.modulus)

            for j in range(2 * n):
                augmented[i][j] = (augmented[i][j] * inv_pivot) % self.modulus

            # Обнуляем столбец в других строках
            for k in range(n):
                if k != i:
                    factor = augmented[k][i]
                    for j in range(2 * n):
                        augmented[k][j] = (augmented[k][j] - factor * augmented[i][j]) % self.modulus

        # Извлекаем обратную матрицу
        inv_matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(augmented[i][n + j])
            inv_matrix.append(row)

        return inv_matrix

    def encrypt_block(self, block, key_matrix):
        """Шифрование одного блока матричным методом"""
        # Дополняем блок если нужно
        if len(block) < self.block_size:
            padding = self.block_size - len(block)
            block = block + bytes([0] * padding)

        # Преобразуем блок в вектор
        vector = [byte % self.modulus for byte in block]

        # Умножаем матрицу на вектор
        result_vector = [0] * self.block_size
        for i in range(self.block_size):
            for j in range(self.block_size):
                result_vector[i] = (result_vector[i] + key_matrix[i][j] * vector[j]) % self.modulus

        # Преобразуем результат обратно в байты
        encrypted_block = bytes(x % 256 for x in result_vector)

        return encrypted_block

    def decrypt_block(self, block, inv_key_matrix):
        """Дешифрование одного блока матричным методом"""
        if len(block) < self.block_size:
            raise ValueError(f"Размер блока должен быть не менее {self.block_size} байт")

        # Преобразуем блок в вектор
        vector = [byte % self.modulus for byte in block]

        # Умножаем обратную матрицу на вектор
        result_vector = [0] * self.block_size
        for i in range(self.block_size):
            for j in range(self.block_size):
                result_vector[i] = (result_vector[i] + inv_key_matrix[i][j] * vector[j]) % self.modulus

        # Преобразуем результат обратно в байты
        decrypted_block = bytes(x % 256 for x in result_vector)

        return decrypted_block


# УПРОЩЕННЫЙ CBC режим
class CBCCipher:
    def __init__(self, block_cipher, block_size=5):
        self.block_cipher = block_cipher
        self.block_size = block_size

    def generate_iv(self, seed):
        """Генерация вектора инициализации (IV)"""
        gen = ParkMillerGenerator(seed)
        iv = bytes(gen.next_int_range(0, 255) for _ in range(self.block_size))
        return iv

    def pad_data(self, data):
        """Простое дополнение нулями"""
        padding_length = self.block_size - (len(data) % self.block_size)
        if padding_length == self.block_size:
            padding_length = 0
        return data + bytes([0] * padding_length)

    def unpad_data(self, data):
        """Удаление нулевого дополнения"""
        # Убираем нули с конца
        while data and data[-1] == 0:
            data = data[:-1]
        return data

    def encrypt(self, data, key_matrix, iv):
        """Шифрование в режиме CBC"""
        # Дополняем данные
        padded_data = self.pad_data(data)

        blocks = []
        for i in range(0, len(padded_data), self.block_size):
            block = padded_data[i:i + self.block_size]
            blocks.append(block)

        encrypted_blocks = []
        prev_block = iv

        for block in blocks:
            # XOR с предыдущим зашифрованным блоком
            xor_block = bytes(a ^ b for a, b in zip(block, prev_block))

            # Шифруем блок
            encrypted_block = self.block_cipher.encrypt_block(xor_block, key_matrix)
            encrypted_blocks.append(encrypted_block)
            prev_block = encrypted_block

        return iv + b''.join(encrypted_blocks)

    def decrypt(self, data, inv_key_matrix):
        """Дешифрование в режиме CBC"""
        if len(data) < self.block_size:
            raise ValueError("Данные слишком короткие для CBC режима")

        iv = data[:self.block_size]
        encrypted_data = data[self.block_size:]

        blocks = []
        for i in range(0, len(encrypted_data), self.block_size):
            block = encrypted_data[i:i + self.block_size]
            blocks.append(block)

        decrypted_blocks = []
        prev_block = iv

        for block in blocks:
            # Дешифруем блок
            decrypted_block = self.block_cipher.decrypt_block(block, inv_key_matrix)

            # XOR с предыдущим зашифрованным блоком
            xor_block = bytes(a ^ b for a, b in zip(decrypted_block, prev_block))
            decrypted_blocks.append(xor_block)
            prev_block = block

        # Объединяем и убираем дополнение
        result = b''.join(decrypted_blocks)
        return self.unpad_data(result)


# Остальной код приложения остается без изменений...
# [Здесь должен быть остальной код класса CryptoApp и main()]


# Основной класс приложения (остается без изменений)
class CryptoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Блочное шифрование - Вариант 4 (Матричное + CBC)")
        self.root.geometry("900x750")

        # Инициализация шифров
        self.block_size = 5  # Для варианта 4
        self.matrix_cipher = MatrixCipher(self.block_size)
        self.cbc_cipher = CBCCipher(self.matrix_cipher, self.block_size)

        # Переменные состояния
        self.current_file = None
        self.processed_data = None
        self.key_matrix = None
        self.inv_key_matrix = None
        self.iv = None

        self.create_widgets()

    def create_widgets(self):
        """Создание элементов интерфейса"""
        # Заголовок
        title_label = tk.Label(self.root, text="Блочное шифрование - Матричное шифрование + режим CBC",
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=10)

        # Создание вкладок
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Вкладка хеширования
        hash_frame = ttk.Frame(notebook)
        notebook.add(hash_frame, text="Хеширование паролей")

        # Вкладка шифрования
        crypto_frame = ttk.Frame(notebook)
        notebook.add(crypto_frame, text="Шифрование/Дешифрование")

        self.setup_hash_tab(hash_frame)
        self.setup_crypto_tab(crypto_frame)

        # Статус бар
        self.status_var = tk.StringVar()
        self.status_var.set("Готов к работе")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1,
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_hash_tab(self, parent):
        """Настройка вкладки хеширования"""
        # Пароль
        tk.Label(parent, text="Пароль:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.password_entry = tk.Entry(parent, width=50)
        self.password_entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        # Кнопки хеширования
        hash_buttons_frame = tk.Frame(parent)
        hash_buttons_frame.grid(row=1, column=0, columnspan=2, pady=10)

        self.btn_hash_mahash8 = tk.Button(hash_buttons_frame, text="Хеш (MaHash8)",
                                          command=self.hash_mahash8, padx=10, pady=5, bg="lightgreen")
        self.btn_hash_mahash8.pack(side=tk.LEFT, padx=5)

        # Результаты хеширования
        tk.Label(parent, text="Результаты хеширования:").grid(row=2, column=0, padx=5, pady=5, sticky='w')

        self.hash_results_text = scrolledtext.ScrolledText(parent, width=80, height=8)
        self.hash_results_text.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')

        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(3, weight=1)

    def setup_crypto_tab(self, parent):
        """Настройка вкладки шифрования"""
        # Пароль для шифрования
        tk.Label(parent, text="Пароль для шифрования:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.crypto_password_entry = tk.Entry(parent, width=50)
        self.crypto_password_entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        # Выбор режима шифрования
        tk.Label(parent, text="Режим шифрования:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.cipher_mode = tk.StringVar(value="matrix")
        mode_frame = tk.Frame(parent)
        mode_frame.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        tk.Radiobutton(mode_frame, text="Матричное шифрование", variable=self.cipher_mode,
                       value="matrix").pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Режим CBC", variable=self.cipher_mode,
                       value="cbc").pack(side=tk.LEFT)

        # Файловые операции
        file_buttons_frame = tk.Frame(parent)
        file_buttons_frame.grid(row=2, column=0, columnspan=2, pady=10)

        self.btn_select_file = tk.Button(file_buttons_frame, text="Выбрать файл",
                                         command=self.select_file, padx=10, pady=5, bg="lightyellow")
        self.btn_select_file.pack(side=tk.LEFT, padx=5)

        self.btn_encrypt = tk.Button(file_buttons_frame, text="Зашифровать",
                                     command=self.encrypt_file, padx=10, pady=5, bg="lightgreen")
        self.btn_encrypt.pack(side=tk.LEFT, padx=5)

        self.btn_decrypt = tk.Button(file_buttons_frame, text="Расшифровать",
                                     command=self.decrypt_file, padx=10, pady=5, bg="lightcoral")
        self.btn_decrypt.pack(side=tk.LEFT, padx=5)

        self.btn_save_result = tk.Button(file_buttons_frame, text="Сохранить результат",
                                         command=self.save_result, padx=10, pady=5, bg="lightblue")
        self.btn_save_result.pack(side=tk.LEFT, padx=5)

        # Информация о файле
        tk.Label(parent, text="Информация о файле и процессе:").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.file_info_text = scrolledtext.ScrolledText(parent, width=80, height=15)
        self.file_info_text.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')

        # Прогресс бар
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(parent, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky='ew')

        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(4, weight=1)

    def hash_mahash8(self):
        """Хеширование с помощью MaHash8"""
        password = self.password_entry.get()
        if not password:
            messagebox.showwarning("Предупреждение", "Введите пароль")
            return

        hash_result = ma_hash8_hex(password)
        numeric_hash = MaHash8(password)

        self.hash_results_text.delete(1.0, tk.END)
        self.hash_results_text.insert(tk.END, f"Пароль: {password}\n")
        self.hash_results_text.insert(tk.END, f"MaHash8 хеш (hex): {hash_result}\n")
        self.hash_results_text.insert(tk.END, f"MaHash8 хеш (число): {numeric_hash}\n")
        self.hash_results_text.insert(tk.END, f"Длина: {len(hash_result)} символов\n")

        self.status_var.set("MaHash8 хеш вычислен")

    def select_file(self):
        """Выбор файла для шифрования/дешифрования"""
        filename = filedialog.askopenfilename()
        if filename:
            self.current_file = filename
            file_size = os.path.getsize(filename)

            self.file_info_text.delete(1.0, tk.END)
            self.file_info_text.insert(tk.END, f"Выбран файл: {filename}\n")
            self.file_info_text.insert(tk.END, f"Размер: {file_size} байт\n")
            self.file_info_text.insert(tk.END, f"Размер блока: {self.block_size} байт\n")
            self.file_info_text.insert(tk.END,
                                       f"Количество блоков: {file_size // self.block_size + (1 if file_size % self.block_size else 0)}\n")

            # Показ превью для текстовых файлов
            if file_size < 1024:  # Показываем превью только для маленьких файлов
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        preview = f.read(200)
                        self.file_info_text.insert(tk.END, f"\nПревью:\n{preview}")
                        if file_size > 200:
                            self.file_info_text.insert(tk.END, "\n... (файл обрезан)")
                except:
                    try:
                        with open(filename, 'rb') as f:
                            preview = f.read(50)
                            hex_preview = ' '.join(f'{b:02x}' for b in preview)
                            self.file_info_text.insert(tk.END, f"\nПревью (hex):\n{hex_preview}")
                    except:
                        self.file_info_text.insert(tk.END, "\n(не удалось прочитать файл)")

            self.status_var.set(f"Выбран файл: {os.path.basename(filename)}")

    def init_cipher_from_password(self, password):
        """Инициализация шифра из пароля"""
        seed = MaHash8(password)
        self.key_matrix, self.inv_key_matrix = self.matrix_cipher.generate_key_matrix(seed)
        self.iv = self.cbc_cipher.generate_iv(seed + 1)

        # Добавляем отладочную информацию
        self.file_info_text.insert(tk.END, f"\nОтладочная информация:\n")
        self.file_info_text.insert(tk.END, f"Seed: {seed}\n")
        self.file_info_text.insert(tk.END, f"Ключевая матрица: {self.key_matrix}\n")
        self.file_info_text.insert(tk.END, f"Обратная матрица: {self.inv_key_matrix}\n")

        return seed

    def encrypt_file(self):
        """Шифрование файла"""
        if not self.current_file:
            messagebox.showwarning("Предупреждение", "Сначала выберите файл")
            return

        password = self.crypto_password_entry.get()
        if not password:
            messagebox.showwarning("Предупреждение", "Введите пароль для шифрования")
            return

        try:
            self.status_var.set("Шифрование...")
            self.progress_var.set(0)
            self.root.update()

            # Инициализация шифра из пароля
            seed = self.init_cipher_from_password(password)

            # Чтение файла
            with open(self.current_file, 'rb') as f:
                file_data = f.read()

            mode = self.cipher_mode.get()

            if mode == "matrix":
                # Матричное шифрование (простой режим) с сохранением длины
                original_length = len(file_data)
                length_info = original_length.to_bytes(4, 'big')
                data_with_length = length_info + file_data

                blocks = []
                for i in range(0, len(data_with_length), self.block_size):
                    block = data_with_length[i:i + self.block_size]
                    if len(block) < self.block_size:
                        padding = self.block_size - len(block)
                        block = block + bytes([0] * padding)
                    blocks.append(block)

                encrypted_blocks = []
                total_blocks = len(blocks)

                for i, block in enumerate(blocks):
                    encrypted_block = self.matrix_cipher.encrypt_block(block, self.key_matrix)
                    encrypted_blocks.append(encrypted_block)

                    # Обновление прогресса
                    progress = (i + 1) / total_blocks * 100
                    self.progress_var.set(progress)
                    self.root.update()

                self.processed_data = b''.join(encrypted_blocks)
                mode_name = "Матричное шифрование"

            else:  # CBC режим
                self.processed_data = self.cbc_cipher.encrypt(file_data, self.key_matrix, self.iv)
                mode_name = "Режим CBC"
                self.progress_var.set(100)

            # Показываем информацию о результате
            self.file_info_text.delete(1.0, tk.END)
            self.file_info_text.insert(tk.END, f"Файл зашифрован: {self.current_file}\n")
            self.file_info_text.insert(tk.END, f"Режим: {mode_name}\n")
            self.file_info_text.insert(tk.END, f"Исходный размер: {len(file_data)} байт\n")
            self.file_info_text.insert(tk.END, f"Зашифрованный размер: {len(self.processed_data)} байт\n")
            self.file_info_text.insert(tk.END, f"Размер блока: {self.block_size} байт\n")

            # Показываем превью зашифрованных данных
            hex_preview = self.processed_data[:100].hex()
            self.file_info_text.insert(tk.END, f"\nПревью (hex):\n{hex_preview}")
            if len(self.processed_data) > 100:
                self.file_info_text.insert(tk.END, "\n... (данные обрезаны)")

            self.status_var.set(f"Файл зашифрован ({mode_name})")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при шифровании: {str(e)}")
            self.status_var.set("Ошибка шифрования")

    def decrypt_file(self):
        """Дешифрование файла"""
        if not self.current_file:
            messagebox.showwarning("Предупреждение", "Сначала выберите файл")
            return

        password = self.crypto_password_entry.get()
        if not password:
            messagebox.showwarning("Предупреждение", "Введите пароль для дешифрования")
            return

        try:
            self.status_var.set("Дешифрование...")
            self.progress_var.set(0)
            self.root.update()

            # Инициализация шифра из пароля
            seed = self.init_cipher_from_password(password)

            # Чтение зашифрованного файла
            with open(self.current_file, 'rb') as f:
                encrypted_data = f.read()

            mode = self.cipher_mode.get()

            if mode == "matrix":
                # Матричное дешифрование (простой режим) с восстановлением длины
                blocks = []
                for i in range(0, len(encrypted_data), self.block_size):
                    block = encrypted_data[i:i + self.block_size]
                    blocks.append(block)

                decrypted_blocks = []
                total_blocks = len(blocks)

                for i, block in enumerate(blocks):
                    decrypted_block = self.matrix_cipher.decrypt_block(block, self.inv_key_matrix)
                    decrypted_blocks.append(decrypted_block)

                    # Обновление прогресса
                    progress = (i + 1) / total_blocks * 100
                    self.progress_var.set(progress)
                    self.root.update()

                # Объединяем все блоки
                all_data = b''.join(decrypted_blocks)

                # Извлекаем информацию о длине
                if len(all_data) >= 4:
                    original_length = int.from_bytes(all_data[:4], 'big')
                    self.processed_data = all_data[4:4 + original_length]
                else:
                    self.processed_data = all_data

                mode_name = "Матричное шифрование"

            else:  # CBC режим
                self.processed_data = self.cbc_cipher.decrypt(encrypted_data, self.inv_key_matrix)
                mode_name = "Режим CBC"
                self.progress_var.set(100)

            # Показываем информацию о результате
            self.file_info_text.delete(1.0, tk.END)
            self.file_info_text.insert(tk.END, f"Файл расшифрован: {self.current_file}\n")
            self.file_info_text.insert(tk.END, f"Режим: {mode_name}\n")
            self.file_info_text.insert(tk.END, f"Размер данных: {len(self.processed_data)} байт\n")

            # Попытка показать превью для текста
            try:
                text_preview = self.processed_data.decode('utf-8', errors='replace')
                self.file_info_text.insert(tk.END, f"\nПревью:\n{text_preview}")
                if len(text_preview) > 200:
                    self.file_info_text.insert(tk.END, "\n... (данные обрезаны)")
            except:
                hex_preview = self.processed_data[:100].hex()
                self.file_info_text.insert(tk.END, f"\nПревью (hex):\n{hex_preview}")
                if len(self.processed_data) > 100:
                    self.file_info_text.insert(tk.END, "\n... (данные обрезаны)")

            self.status_var.set(f"Файл расшифрован ({mode_name})")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при дешифровании: {str(e)}")
            self.status_var.set("Ошибка дешифрования")

    def save_result(self):
        """Сохранение результата шифрования/дешифрования"""
        if self.processed_data is None:
            messagebox.showwarning("Предупреждение", "Нет данных для сохранения")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".bin",
            filetypes=[("All files", "*.*"), ("Text files", "*.txt"), ("Binary files", "*.bin")]
        )

        if filename:
            try:
                with open(filename, 'wb') as f:
                    f.write(self.processed_data)

                messagebox.showinfo("Успех", f"Данные сохранены в файл:\n{filename}")
                self.status_var.set(f"Данные сохранены: {os.path.basename(filename)}")

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при сохранении: {str(e)}")
                self.status_var.set("Ошибка сохранения")


def main():
    root = tk.Tk()
    app = CryptoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()