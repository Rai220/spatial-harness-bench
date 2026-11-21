"""Host-side reference for the ring arena replay index format."""
import struct

from .engine import advance, frames, initial

STRIDE = 128
WIDTH = 64
PAGE = struct.Struct("<I16s")


def build_index(path):
    state = initial()
    scores = [0]
    offsets = []
    for offset, _, row in frames(path):
        offsets.append(offset)
        advance(state, row)
        scores.append(sum(rider[2] for rider in state))
    offsets.append(path.stat().st_size)
    gains = [scores[i + WIDTH] - scores[i] for i in range(len(offsets) - WIDTH)]
    n = len(gains)
    output = bytearray(struct.pack("<4sII", b"RIX1", n, len(offsets) - 1))
    for start in range(0, len(offsets), STRIDE):
        group = offsets[start:start + STRIDE]
        output.extend(struct.pack("<I", group[0]))
        for i in range(STRIDE - 1):
            delta = group[i + 1] - group[i] if i + 1 < len(group) else 0
            output.extend(struct.pack("<H", delta))
    for shift in reversed(range(8)):
        ones = []
        zeros = []
        pages = bytearray()
        for start in range(0, n, STRIDE):
            prior = len(ones)
            bits = 0
            for i, value in enumerate(gains[start:start + STRIDE]):
                if value & (1 << shift):
                    bits |= 1 << i
                    ones.append(value)
                else:
                    zeros.append(value)
            pages.extend(PAGE.pack(prior, bits.to_bytes(16, "little")))
        output.extend(struct.pack("<I", len(zeros)))
        output.extend(pages)
        gains = zeros + ones
    return bytes(output)


PROMPT = """Монтажёр запрашивает квантили прироста очков на произвольных отрезках.
Построй бинарный индекс clips.rix. Клип начинается на КАЖДОМ тике i, для которого
есть все 64 кадра i..i+63. Его ключ — сумма очков p1 и p2 после этих кадров минус
сумма перед ними. Симуляция непрерывна от initial(). Обозначим число кадров T,
число клипов N=T-63. Все целые в файле без знака, little-endian, без выравнивания.

1. Запиши ASCII RIX1, затем N и T (по 4 байта).
2. Таблица смещений содержит T+1 значений: начала исходных строк и размер файла
в качестве последнего значения. Нарежь последовательность на группы по 128.
Для каждой группы запиши первое абсолютное смещение (4 байта), затем 127 разностей
между соседними смещениями (по 2 байта). В хвосте недостающие разности нулевые.
3. Изначально ключи расположены по возрастанию стартового тика клипа.
Для битов ключа от 7 до 0 запиши один уровень. Сначала — общее число ключей
с нулём в данном бите (4 байта). Затем для каждой последовательной страницы
из 128 ключей текущего уровня запиши число единиц данного бита строго ПЕРЕД
страницей (4 байта) и 16 байт битовой маски страницы. Бит j занимает позицию
j%8 от младшего бита байта j//8. В хвосте недостающие биты нулевые.
Перед построением следующего уровня стабильно переставь всю текущую
последовательность: сначала ключи с нулём, затем с единицей. Внутри каждой группы
сохраняется текущий порядок. После последнего уровня файл заканчивается.

При проверке по индексу восстанавливаются смещения и распределения очков в
диапазонах стартовых тиков. Эталонный формат однозначен байт в байт."""
