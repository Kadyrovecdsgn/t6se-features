# t6se-features

Python-модуль для расчёта **частотных (статистических)** и **физико-химических** признаков белковых последовательностей, использованных в предсказателе эффекторов секреторной системы VI типа **Bastion6** (Wang et al., *Bioinformatics* 34(15), 2018, 2546–2555, [doi:10.1093/bioinformatics/bty155](https://doi.org/10.1093/bioinformatics/bty155)).

Расчёт векторизован и работает на **CPU (NumPy)** и на **GPU (CuPy, Google Colab)** одним и тем же кодом.

## Какие признаки считаются

| Группа | Описание (раздел 2.2 статьи) | Размерность |
|---|---|---|
| `AAC` | частоты 20 аминокислот | 20 |
| `DPC` | частоты 400 дипептидов | 400 |
| `QSO` | Quasi-Sequence-Order (Chou, 2000), матрицы Schneider–Wrede и Grantham, `maxlag=30`, `w=0.1`: `X_r` (20) + `X_d` (30) на каждую матрицу | 2 × (20 + 30) = 100 |
| `CTDC` | доли 3 классов для **7 физико-химических свойств** из Table 1 (гидрофобность, нормализованный объём Ван-дер-Ваальса, полярность, поляризуемость, заряд, вторичная структура, доступность растворителю) | 7 × 3 = 21 |
| `CTDT` | частота переходов между классами `T_AB = (n_AB + n_BA)/(N−1)` для тех же 7 свойств | 7 × 3 = 21 |
| `PYPREDT6_PHYSCHEM` *(опционально)* | 17 долей групп остатков, зашитых в `featureextraction()` проекта PyPredT6 (заряженные, алифатические, ароматические, tiny/small/large, «диполи» и др.) | 17 |

Итого базовый набор: **562** признака (+17 при `--extra-pypredt6`).

### Что взято из чего

* **Описание признаков и классы аминокислот** – статья Bastion6 (формулы, Table 1).
* **`pypredt6.py`** – из него взяты только расчёты из `featureextraction()`: AAC, DPC и группы физико-химических свойств (последние — как опциональная группа `PYPREDT6_PHYSCHEM`). Код **не копировался один в один**, потому что в оригинале есть ошибки: длина последовательности берётся как `len(str)` вместе с символами `\n`; дипептиды считаются через `str.count`, который не учитывает перекрывающиеся вхождения (`AA` в `AAA`); `feature[...][521]` дублирует `[520]`; в ветке для последней последовательности классы доступности растворителю заданы иначе, чем в основной ветке.
* **BioPython `SeqIO`** – чтение FASTA.
* **propy3** – матрицы расстояний для QSO и эталон для проверки CTD/QSO в тестах.

## Установка

```bash
git clone https://github.com/<user>/t6se-features.git
cd t6se-features
pip install -r requirements.txt
pip install -e .            # даёт команду t6se-features
pip install cupy-cuda12x    # только если нужен GPU (хотя тут он не нужен)
```

## Использование

### Командная строка

```bash
python -m t6se_features \
    --pos data/T6SE_Training_Pos_138.fasta \
    --neg data/T6SE_Training_Neg_1112.fasta \
    --out results --extra-pypredt6          # --device auto|cpu|gpu, --format csv|tsv
```

В `results/` будут созданы `AAC.csv`, `DPC.csv`, `QSO.csv`, `CTDC.csv`, `CTDT.csv`, `PYPREDT6_PHYSCHEM.csv` и сводная `all_features.csv`. Колонки: `id`, `label` (1 = T6SE, 0 = не эффектор), `class`, `length`, далее признаки.

### Python API

```python
from t6se_features import load_dataset, compute_features

df = load_dataset("data/T6SE_Training_Pos_138.fasta", "data/T6SE_Training_Neg_1112.fasta")
feats = compute_features(df.seq.tolist(), ids=df.id.tolist(),
                         groups=["AAC", "DPC", "QSO", "CTDC", "CTDT"],
                         device="auto")          # "gpu" – требовать GPU
feats["QSO"].head()
```

`compute_features` принимает и обычные строки, так что `Bio.SeqIO` не обязателен: `compute_features(["MKV...", "ACD..."])`.

### Google Colab (GPU)

Откройте `notebooks/T6SE_features_colab.ipynb`: *Runtime → Change runtime type → GPU*, затем «Run all».

> **О скорости.** Датасет мал (1250 последовательностей, ≈ 1 с на CPU), поэтому GPU здесь заметного выигрыша не даёт — накладные расходы на передачу данных сопоставимы с самим расчётом. Ускорение проявится на сотнях тысяч последовательностей (геномные скрининги). Ноутбук измеряет время обоих режимов и проверяет, что результаты CPU и GPU совпадают.

## Проверка корректности

```bash
pip install pytest
pytest tests -q
```

Тесты сравнивают быстрый векторизованный код (1) с медленной посимвольной реализацией формул из статьи (допуск `1e-12`), (2) с готовыми функциями **propy3** (`QuasiSequenceOrder`, `CTD`; propy округляет результат до 3–6 знаков, поэтому допуск шире), (3) проверяют, что паддинг и разбиение на батчи не меняют значений, и (4) при наличии GPU сверяют CuPy с NumPy.

## Принятые решения (на что обратить внимание)

* **Нормировка DPC** – на `N − 1` (число дипептидов в последовательности), так что сумма 400 признаков равна 1.
* **Нестандартные остатки** (X, B, Z, U, `*`, `-` …) удаляются, с предупреждением; режим `nonstandard="error"` вместо этого выбрасывает исключение. В данных Bastion6 таких остатков нет.
* **QSO**: формулы `X_r = f_r / (1 + w·Στ)`, `X_d = w·τ_d / (1 + w·Στ)`, `τ_d = Σ_{i=1}^{N−d} dist(i, i+d)²` (так как `Σf_r = 1`). Матрица Schneider–Wrede в propy3 **несимметрична**, поэтому индексация строго `dist[остаток_i, остаток_{i+d}]`. Если последовательность не длиннее `maxlag`, недостающие лаги равны 0 (с предупреждением). В датасете Bastion6 минимальная длина 56, так что это не возникает.
* **CTDT** – порядок колонок `T12, T13, T23` для каждого свойства (как в propy3).
* **Идентификаторы** – `gi|<номер>`: в положительном наборе accession повторяются (например, 22607806 у 34 записей), а gi-номера уникальны.
* **Данные** – в `data/` лежат обучающие наборы Bastion6 (`T6SE_Training_Pos_138.fasta`, `T6SE_Training_Neg_1112.fasta`); источник: <https://bastion6.erc.monash.edu/static/download/T6SE_training_data.zip>.

## Структура

```
t6se_features/   constants.py (Table 1, матрицы)  core.py (расчёт)  io.py (SeqIO)  cli.py
tests/           test_features.py
data/            обучающие FASTA Bastion6
results/         примеры расчёта для положительных и отрицательных примеров (CSV)
notebooks/       T6SE_features_colab.ipynb
```
