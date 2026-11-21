# Spatial Harness Bench

A benchmark in development for measuring spatial reasoning in text-only language models.

The project follows the structure of [Harness Bench Fast](https://github.com/ai-forever/harness-bench-fast), with separate task definitions, verifiers, and runners.

## 05 · Монтажный индекс

В задаче `05-arcade-clips` харнес получает игровой движок и журнал матча на 30 584 тика. Для каждого из 30 521 перекрывающихся клипов длиной 64 кадра нужно вычислить прирост очков двух игроков и построить бинарный индекс `clips.rix`. Второй файл, `query.py`, находит клип заданного ранга в диапазоне стартовых тиков. При равном приросте выше стоит более ранний клип.

За точный индекс и правильные ответы на всех трёх реплеях начисляется 1 балл. Каждый запрос запускает `query.py` в отдельном процессе. Программа получает страницы индекса через stdin/stdout и должна вернуть стартовый тик и прирост очков за 96 чтений и 5 секунд.

Запуск на Linux с Bubblewrap и системным Python 3.12. Адаптер харнеса получает условие через stdin и работает с файлами текущего каталога. Для замеров изолируйте его от исходников бенчмарка и эталонного решения.

```sh
python3 -m spatial_harness_bench.arcade_clips --output runs/arcade-clips-01 -- \
  claude -p --bare --no-session-persistence --strict-mcp-config \
  --tools Read,Write,Bash --allowedTools Read,Write,Bash
```

В каталоге результата сохраняются сданные файлы и отчёт `result.json` с оценкой.
