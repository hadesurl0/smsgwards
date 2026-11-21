#!/usr/bin/env python3
"""Генерация короткой презентации SMSGwards (8 слайдов, 16:9).

Запуск:  python3 docs/make_presentation.py
Результат: docs/SMSGwards_Presentation.pptx
"""

from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

# --------------------------------------------------------------------------- #
# Пути и константы
# --------------------------------------------------------------------------- #
DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
FIG = DOCS / "figures"
OUT = DOCS / "SMSGwards_Presentation.pptx"

W, H = 13.333, 7.5
MARGIN = 0.65
CONTENT_W = W - 2 * MARGIN  # 12.033

FONT = "Fira Sans"
MONO = "Fira Mono"

DARK = RGBColor(0x0B, 0x1F, 0x1C)
DARK_CARD = RGBColor(0x11, 0x2E, 0x29)
DARK_LINE = RGBColor(0x1F, 0x4B, 0x45)
ACCENT = RGBColor(0x0D, 0x94, 0x88)
ACCENT_LIGHT = RGBColor(0x14, 0xB8, 0xA6)
MINT = RGBColor(0x5E, 0xEA, 0xD4)
MINT_PALE = RGBColor(0xCC, 0xFB, 0xF1)
LIGHT = RGBColor(0xF8, 0xFA, 0xFC)
CARD = RGBColor(0xFF, 0xFF, 0xFF)
TEXT = RGBColor(0x0F, 0x17, 0x2A)
MUTED = RGBColor(0x47, 0x55, 0x69)
SOFT = RGBColor(0x64, 0x74, 0x8B)
BORDER = RGBColor(0xCB, 0xD5, 0xE1)
HAIRLINE = RGBColor(0xE2, 0xE8, 0xF0)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

AMBER = RGBColor(0xD9, 0x77, 0x06)
AMBER_BG = RGBColor(0xFE, 0xF3, 0xC7)
RED = RGBColor(0xDC, 0x26, 0x26)
RED_BG = RGBColor(0xFE, 0xE2, 0xE2)
SAFE_BG = RGBColor(0xE6, 0xF7, 0xF4)

TOTAL_SLIDES = 8


# --------------------------------------------------------------------------- #
# Базовые помощники
# --------------------------------------------------------------------------- #
def rect(slide, x, y, w, h, fill, line=None, line_w=0.75, radius=0.04,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    shp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.shadow.inherit = False
    # Убираем ссылку на тему оформления: она добавляет тень и лишние эффекты.
    style = shp._element.find(qn("p:style"))
    if style is not None:
        shp._element.remove(style)
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(line_w)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE and radius:
        shp.adjustments[0] = radius
    return shp


def frame(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP, pad=(0.0, 0.0, 0.0, 0.0)):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = anchor
    left, top, right, bottom = pad
    tf.margin_left = Inches(left)
    tf.margin_top = Inches(top)
    tf.margin_right = Inches(right)
    tf.margin_bottom = Inches(bottom)
    return tf


def _bullet(p, char, color, marL, indent):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(marL))))
    pPr.set("indent", str(int(Inches(indent))))
    for tag in ("a:buClr", "a:buFont", "a:buChar", "a:buNone", "a:buAutoNum"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    bu_clr = etree.SubElement(pPr, qn("a:buClr"))
    clr = etree.SubElement(bu_clr, qn("a:srgbClr"))
    clr.set("val", str(color))
    bu_font = etree.SubElement(pPr, qn("a:buFont"))
    bu_font.set("typeface", "Arial")
    bu_char = etree.SubElement(pPr, qn("a:buChar"))
    bu_char.set("char", char)


def fill(tf, paras):
    """Заполняет текстовый фрейм списком описаний абзацев."""
    for i, spec in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = spec.get("align", PP_ALIGN.LEFT)
        p.line_spacing = spec.get("line", 1.12)
        if "before" in spec:
            p.space_before = Pt(spec["before"])
        if "after" in spec:
            p.space_after = Pt(spec["after"])

        runs = spec.get("runs")
        if runs is None:
            runs = [(spec.get("text", ""), {})]
        for txt, style in runs:
            run = p.add_run()
            run.text = txt
            f = run.font
            f.name = style.get("font", spec.get("font", FONT))
            f.size = Pt(style.get("size", spec.get("size", 13)))
            f.bold = style.get("bold", spec.get("bold", False))
            f.italic = style.get("italic", spec.get("italic", False))
            f.color.rgb = style.get("color", spec.get("color", TEXT))
            if style.get("baseline"):
                f._rPr.set("baseline", str(style["baseline"]))

        if spec.get("bullet"):
            _bullet(
                p,
                spec.get("bullet_char", "•"),
                spec.get("bullet_color", ACCENT),
                spec.get("marL", 0.22),
                spec.get("indent", -0.22),
            )


def text(slide, x, y, w, h, spec, anchor=MSO_ANCHOR.TOP, pad=(0.0, 0.0, 0.0, 0.0)):
    tf = frame(slide, x, y, w, h, anchor=anchor, pad=pad)
    fill(tf, spec if isinstance(spec, list) else [spec])
    return tf


def bullets(slide, x, y, w, h, items, size=12.5, color=TEXT, gap=5, line=1.14,
            bullet_color=ACCENT):
    tf = frame(slide, x, y, w, h)
    fill(tf, [dict(text=it, size=size, color=color, bullet=True,
                   bullet_color=bullet_color, after=gap, line=line) for it in items])
    return tf


def card(slide, x, y, w, h, accent=True, fill_color=CARD, line_color=BORDER,
         radius=0.045, accent_color=ACCENT):
    rect(slide, x, y, w, h, fill_color, line_color, radius=radius)
    if accent:
        rect(slide, x, y + 0.16, 0.055, h - 0.32, accent_color,
             shape=MSO_SHAPE.RECTANGLE)


def label(slide, x, y, w, txt, color=ACCENT, size=8.5):
    text(slide, x, y, w, 0.24, dict(text=txt, size=size, bold=True, color=color))


def header(slide, title, num):
    rect(slide, 0, 0, W, 1.02, DARK, shape=MSO_SHAPE.RECTANGLE)
    rect(slide, 0, 1.02, W, 0.035, ACCENT, shape=MSO_SHAPE.RECTANGLE)
    rect(slide, MARGIN, 0.30, 0.075, 0.42, ACCENT_LIGHT, shape=MSO_SHAPE.RECTANGLE)
    text(slide, MARGIN + 0.23, 0.24, 9.4, 0.54,
         dict(text=title, size=24, bold=True, color=WHITE), anchor=MSO_ANCHOR.MIDDLE)
    text(slide, 10.45, 0.38, 2.23, 0.28,
         dict(text=f"{num:02d} / {TOTAL_SLIDES:02d}", size=10, bold=True,
              color=ACCENT_LIGHT, align=PP_ALIGN.RIGHT), anchor=MSO_ANCHOR.MIDDLE)


def footer(slide):
    rect(slide, MARGIN, 6.98, CONTENT_W, 0.012, HAIRLINE, shape=MSO_SHAPE.RECTANGLE)
    text(slide, MARGIN, 7.05, 9.0, 0.26,
         dict(text="SMSGwards · обнаружение мошеннических SMS и подозрительных ссылок",
              size=8.5, color=SOFT))
    text(slide, 11.0, 7.05, 1.68, 0.26,
         dict(text="2026", size=8.5, color=SOFT, align=PP_ALIGN.RIGHT))


def content_slide(prs, title, num):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    rect(slide, 0, 0, W, H, LIGHT, shape=MSO_SHAPE.RECTANGLE)
    header(slide, title, num)
    footer(slide)
    return slide


def metric_chip(slide, x, y, w, h, value, name, accent=ACCENT):
    rect(slide, x, y, w, h, CARD, BORDER, radius=0.06)
    rect(slide, x + 0.14, y + 0.14, 0.04, h - 0.28, accent, shape=MSO_SHAPE.RECTANGLE)
    text(slide, x + 0.27, y + 0.10, w - 0.35, 0.36,
         dict(text=value, size=17, bold=True, color=TEXT))
    text(slide, x + 0.27, y + 0.44, w - 0.35, 0.22,
         dict(text=name, size=8, color=SOFT))


def figure(slide, filename, x, y, w, caption):
    path = FIG / filename
    if not path.exists():
        return 0.0
    with Image.open(path) as im:
        iw, ih = im.size
    h = w * ih / iw
    pad = 0.12
    rect(slide, x - pad, y - pad, w + 2 * pad, h + 2 * pad, CARD, BORDER, radius=0.03)
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    text(slide, x - pad, y + h + pad + 0.06, w + 2 * pad, 0.24,
         dict(text=caption, size=9, color=SOFT, align=PP_ALIGN.CENTER))
    return h


def background_art(slide):
    """Мягкие декоративные окружности для тёмных слайдов."""
    circle = slide.shapes.add_shape(
        MSO_SHAPE.OVAL, Inches(9.4), Inches(-1.55), Inches(5.6), Inches(5.6))
    circle.shadow.inherit = False
    circle.fill.solid()
    circle.fill.fore_color.rgb = DARK_CARD
    circle.line.fill.background()

    ring = slide.shapes.add_shape(
        MSO_SHAPE.OVAL, Inches(10.7), Inches(3.9), Inches(3.9), Inches(3.9))
    ring.shadow.inherit = False
    ring.fill.background()
    ring.line.color.rgb = DARK_LINE
    ring.line.width = Pt(1.25)


# --------------------------------------------------------------------------- #
# Слайд 1 — титульный
# --------------------------------------------------------------------------- #
def slide_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, H, DARK, shape=MSO_SHAPE.RECTANGLE)
    background_art(s)

    text(s, 0.9, 1.55, 8.0, 0.3,
         dict(text="DJANGO  ·  MACHINE LEARNING  ·  БЕЗОПАСНОСТЬ",
              size=10, bold=True, color=ACCENT_LIGHT))
    text(s, 0.9, 1.95, 9.6, 1.0,
         dict(text="SMSGwards", size=58, bold=True, color=WHITE))
    text(s, 0.9, 3.05, 8.9, 1.15,
         dict(text="Интеллектуальная система обнаружения мошеннических SMS-сообщений "
                   "и подозрительных ссылок",
              size=21, color=MINT_PALE, line=1.25))
    rect(s, 0.9, 4.38, 1.5, 0.055, ACCENT, shape=MSO_SHAPE.RECTANGLE)

    text(s, 0.9, 4.68, 6.0, 0.26,
         dict(text="АВТОРЫ ПРОЕКТА", size=9.5, bold=True, color=ACCENT_LIGHT))
    text(s, 0.9, 4.96, 8.0, 0.42,
         dict(text="Мурат Ринат, Мирланов Айдархан", size=19, bold=True, color=WHITE))
    text(s, 0.9, 5.55, 8.0, 0.3,
         dict(text="Django 5  ·  DRF  ·  scikit-learn  ·  Bootstrap 5", size=10.5,
              color=RGBColor(0x8F, 0xB8, 0xB0)))

    text(s, 10.95, 0.55, 1.7, 0.3,
         dict(text="2026", size=13, bold=True, color=MINT, align=PP_ALIGN.RIGHT))
    return s


# --------------------------------------------------------------------------- #
# Слайд 2 — актуальность и проблема
# --------------------------------------------------------------------------- #
def slide_problem(prs):
    s = content_slide(prs, "Актуальность и проблема", 2)

    cards = [
        ("Смишинг и социальная инженерия", [
            "Давление и срочность: «карта заблокирована», «списание сейчас»",
            "Легенды: банк, Госуслуги, выигрыш, доставка",
            "Цель — код из SMS, CVV или перевод",
        ]),
        ("Подделка и маскировка URL", [
            "Punycode и домены-двойники: sberbank-online.ru",
            "Сокращатели скрывают настоящий адрес",
            "Кириллица и латиница в одном домене",
            "Бренд в поддомене: sberbank.ru.verify-login.com",
        ]),
        ("Почему анализ должен быть совместным", [
            "Чистый текст с фишинговой ссылкой — текстовой модели мало",
            "Легитимная ссылка в шаблонном SMS — правила дают ложный сигнал",
            "Текст + URL + эвристики = устойчивая оценка",
        ]),
    ]

    cw, gap = 3.82, 0.29
    y, ch = 1.42, 3.18
    for i, (title, items) in enumerate(cards):
        x = MARGIN + i * (cw + gap)
        card(s, x, y, cw, ch)
        text(s, x + 0.3, y + 0.26, cw - 0.55, 0.6,
             dict(text=title, size=14.5, bold=True, color=TEXT, line=1.05))
        bullets(s, x + 0.3, y + 1.02, cw - 0.58, ch - 1.15, items, size=11.5, gap=7)

    # Вывод
    cy = 4.95
    rect(s, MARGIN, cy, CONTENT_W, 1.45, SAFE_BG, radius=0.05)
    rect(s, MARGIN, cy + 0.22, 0.06, 1.0, ACCENT, shape=MSO_SHAPE.RECTANGLE)
    text(s, MARGIN + 0.35, cy + 0.24, 2.4, 0.26,
         dict(text="ВЫВОД", size=9, bold=True, color=ACCENT))
    text(s, MARGIN + 0.35, cy + 0.52, CONTENT_W - 0.75, 0.75,
         dict(text="Ни один сигнал не решает задачу в одиночку: система оценивает сообщение "
                   "и ссылку вместе — и объясняет, что именно вызвало подозрение.",
              size=13.5, color=TEXT, line=1.25))
    return s


# --------------------------------------------------------------------------- #
# Слайд 3 — архитектура
# --------------------------------------------------------------------------- #
def slide_architecture(prs):
    s = content_slide(prs, "Архитектура системы", 3)

    cards = [
        ("Веб-платформа", [
            "Django 5 + Django REST Framework",
            "OpenAPI / Swagger: /api/docs/",
            "Bootstrap 5 + Chart.js",
        ]),
        ("Модули", [
            "accounts — вход и роли",
            "analysis — проверка и история",
            "dashboard — аналитика и мониторинг",
        ]),
        ("Данные и ML", [
            "SQLite как хранилище",
            "ML-пайплайн: preprocessing, url_features",
            "rules и risk — единая логика оценки",
        ]),
    ]
    cw, gap = 3.82, 0.29
    y, ch = 1.28, 2.02
    for i, (title, items) in enumerate(cards):
        x = MARGIN + i * (cw + gap)
        card(s, x, y, cw, ch)
        text(s, x + 0.28, y + 0.22, cw - 0.5, 0.3,
             dict(text=title, size=13.5, bold=True, color=TEXT))
        bullets(s, x + 0.28, y + 0.68, cw - 0.52, ch - 0.8, items, size=11, gap=4)

    text(s, MARGIN, 3.52, 6.0, 0.26,
         dict(text="ПОТОК ОБРАБОТКИ ЗАПРОСА", size=9, bold=True, color=ACCENT))

    steps = [
        ("01  ВХОД", "SMS + URL"),
        ("02  АНАЛИЗ", "NLP-модель · URL-модель · правила СИ"),
        ("03  ОЦЕНКА", "Risk Score 0–100 и категория"),
        ("04  ОТЧЁТ", "Вердикт, причины, рекомендация"),
    ]
    bw, bh, by = 2.55, 1.32, 3.82
    bx0 = MARGIN
    step_gap = (CONTENT_W - 4 * bw) / 3
    for i, (cap, desc) in enumerate(steps):
        x = bx0 + i * (bw + step_gap)
        rect(s, x, by, bw, bh, CARD, BORDER, radius=0.05)
        rect(s, x, by, bw, 0.07, ACCENT if i != 3 else ACCENT_LIGHT,
             shape=MSO_SHAPE.RECTANGLE)
        text(s, x + 0.24, by + 0.22, bw - 0.45, 0.24,
             dict(text=cap, size=8.5, bold=True, color=ACCENT))
        text(s, x + 0.24, by + 0.52, bw - 0.45, 0.72,
             dict(text=desc, size=12, color=TEXT, line=1.2))

        if i < 3:
            ax = x + bw + (step_gap - 0.42) / 2
            arrow = s.shapes.add_shape(
                MSO_SHAPE.RIGHT_ARROW, Inches(ax), Inches(by + bh / 2 - 0.11),
                Inches(0.42), Inches(0.22))
            arrow.shadow.inherit = False
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = ACCENT_LIGHT
            arrow.line.fill.background()

    band_y = 5.42
    rect(s, MARGIN, band_y, CONTENT_W, 1.0, DARK, radius=0.05)
    text(s, MARGIN + 0.35, band_y + 0.16, CONTENT_W - 0.7, 0.7,
         dict(text="Один запрос — три независимых сигнала и прозрачное объяснение результата",
              size=14.5, bold=True, color=WHITE), anchor=MSO_ANCHOR.MIDDLE)
    return s


# --------------------------------------------------------------------------- #
# Слайд 4 — SMS-классификатор
# --------------------------------------------------------------------------- #
def slide_sms(prs):
    s = content_slide(prs, "Мультиязычный SMS-классификатор  v2.0.0", 4)
    lx, lw = MARGIN, 6.15
    rx, rw = 7.15, 5.25

    # Данные
    card(s, lx, 1.28, lw, 1.28)
    label(s, lx + 0.3, 1.42, lw - 0.5, "ДАТАСЕТ")
    text(s, lx + 0.3, 1.66, lw - 0.5, 0.34,
         dict(runs=[
             ("32 011", dict(size=15, bold=True, color=TEXT)),
             (" сырых  →  ", dict(size=12, color=MUTED)),
             ("28 736", dict(size=15, bold=True, color=ACCENT)),
             (" очищенных SMS", dict(size=12, color=MUTED)),
         ]))
    text(s, lx + 0.3, 2.03, lw - 0.5, 0.3,
         dict(text="UCI en · ruSpamModels ru/kk · kazspam-v1 kk · kz_transformers kk",
              size=9.5, color=SOFT))

    # Признаки и модель
    card(s, lx, 2.68, lw, 1.88)
    label(s, lx + 0.3, 2.82, lw - 0.5, "ПРИЗНАКИ И МОДЕЛЬ")
    bullets(s, lx + 0.3, 3.10, lw - 0.55, 1.35, [
        "TF-IDF: word(1–2) + char_wb(3–5)",
        "6 ручных признаков → 164 564 признака",
        "Logistic Regression, balanced, threshold = 0.29",
    ], size=12, gap=5)

    # Метрики
    text(s, lx, 4.70, 3.0, 0.24,
         dict(text="МЕТРИКИ НА ТЕСТЕ", size=9, bold=True, color=ACCENT))
    metrics = [
        ("0.9961", "ROC-AUC"), ("0.9925", "PR-AUC"), ("0.9532", "F1"),
        ("0.9766", "RECALL"), ("0.9309", "PRECISION"), ("0.9724", "ACCURACY"),
    ]
    cw = (lw - 2 * 0.14) / 3
    for i, (val, name) in enumerate(metrics):
        row, col = divmod(i, 3)
        metric_chip(s, lx + col * (cw + 0.14), 4.96 + row * 0.82, cw, 0.72, val, name)

    # График
    ih = figure(s, "03_sms_roc_pr_curves.png", rx, 1.28, rw,
                "ROC- и PR-кривые на тестовой выборке")

    # Гибридный режим
    hy = 1.28 + ih + 0.52
    card(s, rx, hy, rw + 0.26, 1.42)
    label(s, rx + 0.3, hy + 0.16, rw - 0.4, "ГИБРИДНЫЙ РЕЖИМ · RU / KK")
    text(s, rx + 0.3, hy + 0.42, rw - 0.4, 0.32,
         dict(text="ML + правила социальной инженерии", size=13.5, bold=True, color=TEXT))
    text(s, rx + 0.3, hy + 0.80, rw - 0.4, 0.3,
         dict(runs=[
             ("ROC-AUC 0.9823", dict(size=12, bold=True, color=ACCENT)),
             ("   ·   ", dict(size=12, color=BORDER)),
             ("Recall 0.9333", dict(size=12, bold=True, color=ACCENT)),
         ]))

    text(s, rx, hy + 1.62, rw + 0.2, 0.6,
         dict(text="Порог 0.29 выбран под высокий recall: пропустить мошенника дороже, "
                   "чем перепроверить сообщение.",
              size=10, color=SOFT, line=1.2))
    return s


# --------------------------------------------------------------------------- #
# Слайд 5 — анализатор URL
# --------------------------------------------------------------------------- #
def slide_url(prs):
    s = content_slide(prs, "Структурный анализатор URL  v1.0.0", 5)
    lx, lw = MARGIN, 6.15
    rx, rw = 7.15, 5.25

    card(s, lx, 1.28, lw, 2.06)
    label(s, lx + 0.3, 1.42, lw - 0.5, "22 СТРУКТУРНЫХ ПРИЗНАКА · БЕЗ СЕТЕВЫХ ЗАПРОСОВ")
    bullets(s, lx + 0.3, 1.72, lw - 0.55, 1.5, [
        "Энтропия Шеннона, длина, глубина, параметры",
        "Punycode, смешение алфавитов, IP вместо домена",
        "Сходство с брендом по расстоянию Левенштейна",
        "Сокращатели, подозрительные TLD, «@» в URL",
    ], size=11.5, gap=4)

    card(s, lx, 3.50, lw, 1.30)
    label(s, lx + 0.3, 3.62, lw - 0.5, "МОДЕЛЬ И ПОРОГИ")
    text(s, lx + 0.3, 3.86, lw - 0.5, 0.3,
         dict(text="HistGradientBoosting", size=14, bold=True, color=TEXT))
    text(s, lx + 0.3, 4.22, lw - 0.5, 0.28,
         dict(runs=[
             ("SAFE < 0.35", dict(size=10.5, bold=True, color=ACCENT)),
             ("  ·  SUSPICIOUS 0.35–0.50  ·  ", dict(size=10.5, color=MUTED)),
             ("DANGEROUS ≥ 0.50", dict(size=10.5, bold=True, color=RED)),
         ]))

    text(s, lx, 4.98, 3.0, 0.24,
         dict(text="МЕТРИКИ ТЕСТА", size=9, bold=True, color=ACCENT))
    metrics = [
        ("0.9679", "ROC-AUC"), ("0.9685", "PR-AUC"), ("0.9023", "F1"),
        ("0.9199", "PRECISION"), ("0.8854", "RECALL"), ("≈ 2,5 мс", "НА ОДИН URL"),
    ]
    cw = (lw - 2 * 0.14) / 3
    for i, (val, name) in enumerate(metrics):
        row, col = divmod(i, 3)
        metric_chip(s, lx + col * (cw + 0.14), 5.24 + row * 0.78, cw, 0.70, val, name)

    ih = figure(s, "05_url_feature_importance.png", rx, 1.28, rw,
                "Важность структурных признаков URL-модели")

    cy = 1.28 + ih + 0.52
    card(s, rx, cy, rw + 0.26, 2.25)
    label(s, rx + 0.3, cy + 0.18, rw - 0.4, "ПОЧЕМУ БЕЗ СЕТЕВЫХ ЗАПРОСОВ")
    bullets(s, rx + 0.3, cy + 0.50, rw - 0.42, 1.2, [
        "Приватно: URL не открывается и не запрашивается",
        "Быстро: около 2,5 мс на адрес",
        "Устойчиво к сокращателям и маскировке",
    ], size=11, gap=5)
    text(s, rx + 0.3, cy + 1.62, rw - 0.4, 0.4,
         dict(runs=[
             ("≈ 2,5 мс", dict(size=15, bold=True, color=ACCENT)),
             ("  на проверку адреса", dict(size=10.5, color=SOFT)),
         ]))
    return s


# --------------------------------------------------------------------------- #
# Слайд 6 — Risk Score и объяснимость
# --------------------------------------------------------------------------- #
def slide_risk(prs):
    s = content_slide(prs, "Совместный Risk Score и объяснимость", 6)
    lx, lw = MARGIN, 5.9
    rx, rw = 6.85, 5.85

    card(s, lx, 1.28, lw, 1.72)
    label(s, lx + 0.3, 1.42, lw - 0.5, "ФОРМУЛА")
    text(s, lx + 0.3, 1.68, lw - 0.5, 0.34,
         dict(runs=[
             ("Risk = ", dict(size=13.5, bold=True, color=TEXT)),
             ("0.5·P(SMS) + 0.3·P(URL) + 0.2·S", dict(size=13.5, bold=True, color=ACCENT)),
             ("se", dict(size=13.5, bold=True, color=ACCENT, baseline=-25000)),
         ]))
    text(s, lx + 0.3, 2.14, lw - 0.5, 0.62,
         dict(text="Усиление: запрос кода → ≥ 70; оба сигнала высоки → ≥ 85.",
              size=10.5, color=MUTED, line=1.25))

    card(s, lx, 3.14, lw, 3.05)
    label(s, lx + 0.3, 3.28, lw - 0.5, "КАТЕГОРИИ РИСКА")
    rows = [
        ("SAFE", "0–34", "признаки не обнаружены", SAFE_BG, ACCENT),
        ("SUSPICIOUS", "35–69", "подозрительные признаки", AMBER_BG, AMBER),
        ("FRAUDULENT", "70–100", "высокая вероятность мошенничества", RED_BG, RED),
    ]
    for i, (name, rng, desc, bg, fg) in enumerate(rows):
        ry = 3.62 + i * 0.76
        rect(s, lx + 0.28, ry, lw - 0.56, 0.68, bg, radius=0.08)
        rect(s, lx + 0.28, ry + 0.17, 0.05, 0.34, fg, shape=MSO_SHAPE.RECTANGLE)
        text(s, lx + 0.44, ry + 0.08, 2.3, 0.28,
             dict(text=name, size=11.5, bold=True, color=fg))
        text(s, lx + 0.44, ry + 0.36, 2.7, 0.26,
             dict(text=desc, size=8.5, color=MUTED))
        text(s, lx + lw - 1.5, ry + 0.19, 1.1, 0.3,
             dict(text=rng, size=12.5, bold=True, color=fg, align=PP_ALIGN.RIGHT))

    text(s, lx + 0.3, 5.92, lw - 0.55, 0.24,
         dict(text="Порог 70 активирует принудительные правила усиления",
              size=9, color=SOFT))

    # Объяснимость
    card(s, rx, 1.28, rw, 2.3)
    label(s, rx + 0.3, 1.42, rw - 0.4, "ОБЪЯСНИМОСТЬ")
    bullets(s, rx + 0.3, 1.72, rw - 0.42, 1.75, [
        "Причины на человеческом языке, а не только вероятность",
        "Например: «просит код из SMS», «домен похож на Сбербанк»",
        "Вклад каждого сигнала в итоговый балл",
        "Рекомендация под уровень риска",
    ], size=11.5, gap=5)

    # Безопасный показ URL
    card(s, rx, 3.70, rw, 1.42)
    label(s, rx + 0.3, 3.84, rw - 0.4, "БЕЗОПАСНЫЙ ПОКАЗ ССЫЛКИ")
    rect(s, rx + 0.3, 4.14, rw - 0.58, 0.58, LIGHT, BORDER, radius=0.06)
    text(s, rx + 0.45, 4.24, rw - 0.85, 0.4,
         dict(runs=[
             ("https://", dict(size=11.5, color=SOFT, font=MONO)),
             ("sberbank.ru", dict(size=11.5, bold=True, color=ACCENT, font=MONO)),
             (".verify-login.com/pay", dict(size=11.5, color=RED, font=MONO)),
         ]))
    text(s, rx + 0.3, 4.80, rw - 0.5, 0.26,
         dict(text="Домен подсвечен, переход по ссылке не выполняется",
              size=9.5, color=SOFT))

    rect(s, rx, 5.24, rw, 0.95, SAFE_BG, radius=0.05)
    text(s, rx + 0.28, 5.34, rw - 0.5, 0.78,
         dict(text="Оценка носит вероятностный характер и не гарантирует безопасность: "
                   "система предупреждает, а решение остаётся за человеком.",
              size=10.5, color=TEXT, line=1.25))
    return s


# --------------------------------------------------------------------------- #
# Слайд 7 — безопасность и приватность
# --------------------------------------------------------------------------- #
def slide_security(prs):
    s = content_slide(prs, "Безопасность и приватность", 7)
    cards = [
        ("Маскирование данных", [
            "Карты, телефоны, суммы → *** по умолчанию",
            "Маскирование выполняется до сохранения",
            "В историю попадает обезличенный текст",
        ]),
        ("Защита от XSS", [
            "Вывод только через textContent",
            "HTML из SMS не интерпретируется",
            "Нет inline-вставок пользовательских данных",
        ]),
        ("Контроль доступа", [
            "5 неудачных входов → блокировка на 60 с",
            "Rate-limit API: 30 запросов в минуту",
            "Роли: пользователь и администратор",
        ]),
        ("Приватность истории", [
            "История изолирована по пользователям",
            "Админ видит статистику, но не чужие SMS",
            "Доступ к чужим проверкам закрыт",
        ]),
    ]
    cw, gap = 5.88, 0.27
    ch = 2.28
    for i, (title, items) in enumerate(cards):
        row, col = divmod(i, 2)
        x = MARGIN + col * (cw + gap)
        y = 1.28 + row * (ch + 0.17)
        card(s, x, y, cw, ch)
        text(s, x + 0.32, y + 0.22, 0.9, 0.24,
             dict(text=f"0{i + 1}", size=9, bold=True, color=ACCENT))
        text(s, x + 0.32, y + 0.48, cw - 0.6, 0.34,
             dict(text=title, size=14.5, bold=True, color=TEXT))
        bullets(s, x + 0.32, y + 0.98, cw - 0.62, ch - 1.1, items, size=11.5, gap=4)

    by = 6.18
    rect(s, MARGIN, by, CONTENT_W, 0.6, SAFE_BG, radius=0.05)
    text(s, MARGIN + 0.3, by + 0.08, CONTENT_W - 0.6, 0.44,
         dict(text="Безопасность заложена в архитектуру: данные минимизируются на входе, "
                   "а доступ ограничен ролью.",
              size=11, color=TEXT), anchor=MSO_ANCHOR.MIDDLE)
    return s


# --------------------------------------------------------------------------- #
# Слайд 8 — итоги
# --------------------------------------------------------------------------- #
def slide_final(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, H, DARK, shape=MSO_SHAPE.RECTANGLE)
    background_art(s)

    text(s, 0.9, 1.15, 10.0, 0.3,
         dict(text="ИТОГИ И ДЕМОНСТРАЦИЯ", size=10, bold=True, color=ACCENT_LIGHT))
    text(s, 0.9, 1.52, 10.6, 0.85,
         dict(text="Спасибо за внимание!", size=42, bold=True, color=WHITE))
    text(s, 0.9, 2.42, 9.6, 0.6,
         dict(text="SMSGwards — обнаружение мошеннических SMS и подозрительных ссылок",
              size=15, color=MINT_PALE, line=1.2))
    rect(s, 0.9, 3.15, 1.5, 0.055, ACCENT, shape=MSO_SHAPE.RECTANGLE)

    stats = [
        ("234+", "автотестов"),
        ("REST API", "/api/v1/"),
        ("Swagger", "/api/docs/"),
        ("Мониторинг", "история проверок"),
    ]
    sw, sgap = 2.72, 0.28
    for i, (val, cap) in enumerate(stats):
        x = 0.9 + i * (sw + sgap)
        rect(s, x, 3.55, sw, 1.05, DARK_CARD, DARK_LINE, radius=0.07)
        rect(s, x, 3.55, 0.055, 1.05, ACCENT, shape=MSO_SHAPE.RECTANGLE)
        text(s, x + 0.28, 3.72, sw - 0.45, 0.34,
             dict(text=val, size=15, bold=True, color=WHITE))
        text(s, x + 0.28, 4.10, sw - 0.45, 0.3,
             dict(text=cap, size=9.5, color=RGBColor(0x8F, 0xB8, 0xB0)))

    text(s, 0.9, 5.05, 6.0, 0.26,
         dict(text="АВТОРЫ ПРОЕКТА", size=9.5, bold=True, color=ACCENT_LIGHT))
    text(s, 0.9, 5.32, 9.0, 0.42,
         dict(text="Мурат Ринат, Мирланов Айдархан", size=20, bold=True, color=WHITE))
    text(s, 0.9, 5.92, 9.0, 0.3,
         dict(text="Django 5 · DRF · scikit-learn · Bootstrap 5", size=10.5,
              color=RGBColor(0x8F, 0xB8, 0xB0)))

    text(s, 10.95, 0.55, 1.7, 0.3,
         dict(text="2026", size=13, bold=True, color=MINT, align=PP_ALIGN.RIGHT))
    return s


# --------------------------------------------------------------------------- #
# Сборка
# --------------------------------------------------------------------------- #
def main():
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)

    slide_title(prs)
    slide_problem(prs)
    slide_architecture(prs)
    slide_sms(prs)
    slide_url(prs)
    slide_risk(prs)
    slide_security(prs)
    slide_final(prs)

    cp = prs.core_properties
    cp.title = "SMSGwards — обнаружение мошеннических SMS и подозрительных ссылок"
    cp.author = "Мурат Ринат, Мирланов Айдархан"
    cp.subject = "Защита от смишинга и фишинговых ссылок"
    cp.comments = "8 слайдов, 16:9"

    prs.save(OUT)
    print(f"OK: {OUT} ({OUT.stat().st_size / 1024:.1f} КБ), слайдов: {len(prs.slides)}")


if __name__ == "__main__":
    main()
