# TGbot_X5check
Чат-бот в Telegram, который позволяет оцифровать информацию, которая хранится на чеке в магазинах X5 Group
port re
import csv
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
import easyocr
from PIL import Image

# ──────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ──────────────────────────────────────────────────────────

MIN_CONF = 25.0
CURRENCY = "₽"

STORE_ALIASES = {
    "Пятёрочка": ["ПЯТЕРОЧ", "ПЯТЁРОЧ", "ПЯТЕР", "ПЯТЁР", "ПЯТЬЕРОЧКА", "5", "ПЯТЕРОЧКА", "ПЯТЕРКА"],
    "Перекрёсток": ["ПЕРЕКРЕСТ", "ПЕРЕКРЁСТОК", "PEREKRESTOK"],
    "Чижик": ["ЧИЖИК", "CHIZHIK", "ЧИЖИКЪ"],
}

IGNORE_KEYWORDS = [
    "НДС", "СКИДКА", "СДАЧА", "НАЛОГ", "РЕКЛАМА", "СПАСИБО",
    "ОКРУГЛЕНИЕ", "НАЛИЧНЫМИ", "ЭЛЕКТРОННЫМИ", "ПОДЫТОГ", "ПРИНЯТО", "САЙТ", "ООО"
]

ITEM_PATTERNS = [
    # Самый важный — цена*кол-во (ваш лог: "99.99*1")
    r"^(?P<name>.+?)\s+(?P<price>\d+[\.,]\d{2})\s*\*\s*(?P<qty>\d+)$",

    # Фрагменты с *1/*2
    r"^(?P<name>.+?)\s*(?P<qty>\d+)\s*\*\s*1\s*(?P<price>\d+[\.,]\d{2})$",

    # Цена + кол-во + итого
    r"^(?P<name>.+?)\s+(?P<price>\d+[\.,]\d{2})\s*(?P<qty>\*\d+|\d+)\s*(?P<total>\d+[\.,]\d{2})?$",

    # Название + цена (qty=1)
    r"^(?P<name>.+?)\s+(?P<price>\d+[\.,]\d{2})\s*(₽|руб)?$",

    # Классика с ×
    r"^(?P<name>.+?)\s+(?P<qty>\d+[.,]?\d*)\s*[xх*×]\s*(?P<price>[\d\s.,]+)$",

    #Специально под наш OCR
    r"^(?P<price>\d+[\.,]\d{2})\s*\*\s*(?P<qty>\d+)$",
]

# Ленивая инициализация EasyOCR
reader = None

def get_reader():
    global reader
    if reader is None:
        print("Инициализация EasyOCR...")
        reader = easyocr.Reader(['ru', 'en'], gpu=False, download_enabled=False)
        print("EasyOCR готов")
    return reader

# ──────────────────────────────────────────────────────────
# ФУНКЦИИ
# ──────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.upper().replace('Ё', 'Е').strip())

def detect_store(full_text: str) -> str:
    t = normalize(full_text)
    for store, aliases in STORE_ALIASES.items():
        for alias in aliases:
            if alias in t:
                return store
    return "Other"

def find_date(full_text: str) -> Optional[str]:
    patterns = [
        r"(\d{2})\.(\d{2})\.(\d{2})\b",                         
        r"(\d{2})\.(\d{2})\.(\d{2})\s+\d{2}:\d{2}",                
        r"\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2})\b",              
        r"(\d{2})\.(\d{2})\.(\d{4})\b",                                ]
    for pat in patterns:
        m = re.search(pat, full_text, re.IGNORECASE)
        if m:
            d, mo, y = m.groups()
            if len(y) == 2:
                y = "20" + y
            try:
                return f"{int(d):02d}.{int(mo):02d}.{int(y):04d}"
            except:
                pass
    # Fallback — ищем любые похожие даты
    m = re.search(r"\d{2}\.\d{2}\.\d{2}", full_text)
    if m:
        return m.group() + " (примерно)"
    return None

def _normalize_number(s: str) -> str:
    if not s:
        return ""
    s = s.replace(" ", "").replace(",", ".").replace("\u00A0", "")
    s = re.sub(r"[^\d.-]", "", s)
    return s

def find_total(full_text: str) -> Optional[str]:
    money_pat = re.compile(r"\d+[\.,]\d{2}")
    lines = normalize(full_text).splitlines()[-35:]  # последние строки
    
    found_final = False
    candidates = []
    
    for line in reversed(lines):
        norm_line = normalize(line)
        
        if any(kw in norm_line for kw in ["ИТОГ С УЧЕТОМ СКИДОК", "ИТОГО", "ИТОГ:", "К ОПЛАТЕ", "СУММА К ОПЛАТЕ"]):
            m = money_pat.search(line)
            if m:
                return _normalize_number(m.group())
            found_final = True
        
        if found_final:
            candidates.extend(money_pat.findall(line))
    
    if candidates:
        return max(candidates, key=lambda x: float(_normalize_number(x)))
    
    return None

def preprocess_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Не удалось загрузить изображение")
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_enh = clahe.apply(l)
    enhanced = cv2.merge((l_enh, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced  # цветное — лучше для EasyOCR

def ocr_easyocr(image_path: str) -> Tuple[str, List[Tuple[str, float]]]:
    processed = preprocess_image(image_path)
    pil_img = Image.fromarray(processed)
    
    r = get_reader()
    
    result = r.readtext(np.array(pil_img), detail=1, paragraph=False,
                        min_size=5,
                        contrast_ths=0.01,
                        text_threshold=0.4,
                        low_text=0.2,
                        mag_ratio=3.0)
    
    lines = []
    full_text_parts = []
    
    for detection in result:
        bbox, text, conf = detection
        text = text.strip()
        if text and conf * 100 >= MIN_CONF:
            lines.append((text, conf * 100))
            full_text_parts.append(text)
    
    lines.sort(key=lambda x: min(p[1] for p in bbox))
    
    return "\n".join(full_text_parts), lines

def extract_items(lines: List[Tuple[str, float]]) -> List[Dict]:
    items = []
    in_items = False
    current_name = ""

    print("DEBUG: распознано строк:", len(lines))
    print("DEBUG: первые 15 строк (полные):")
    for t, c in lines[:15]:
        print(f"  {t} (conf: {c:.1f})")

    for text, conf in lines:
        norm = normalize(text)

        if any(x in norm for x in ["КАССОВЫЙ ЧЕК", "ТОВАР", "ЦЕНА", "КОЛ-ВО", "СУММА", "КОЛИЧЕСТВО"]):
            in_items = True
            continue
        
        if any(x in norm for x in ["ИТОГО", "ИТОГ С УЧЕТОМ", "К ОПЛАТЕ", "ПОДЫТОГ", "НДС", "СКИДКА:"]):
            in_items = False
            continue
            
        if not in_items:
            continue

        # 1. Полная строка "Название цена*кол-во"
        m = re.search(r"^(?P<name>.+?)\s+(?P<price>\d+[\.,]\d{2})\s*\*\s*(?P<qty>\d+)$", text.strip(), re.IGNORECASE)
        if m:
            name = m.group("name").strip()
            qty = int(m.group("qty"))
            price = _normalize_number(m.group("price"))
            items.append({
                "name": name,
                "qty": qty,
                "unit_price": price,
                "total": price,
                "discount": None,
                "conf": round(conf, 1)
            })
            current_name = ""
            continue

        # 2. Фрагмент цены или количества после имени
        price_m = re.search(r'(\d+[\.,]\d{2})', text)
        qty_m = re.search(r'\*(\d+)', text) or re.search(r'\b(\d+)\b', text)

        if price_m and current_name:
            price = _normalize_number(price_m.group(1))
            qty = int(qty_m.group(1)) if qty_m else 1
            items.append({
                "name": current_name,
                "qty": qty,
                "unit_price": price,
                "total": price,
                "discount": None,
                "conf": round(conf, 1)
            })
            current_name = ""
            continue

        # 3. Начинаем новое имя товара (если строка начинается с буквы)
        if re.match(r'^[А-ЯA-Z0-9].+', text) and len(text) > 4:
            current_name = text.strip()

    return items

def export_csv(data: Dict, filename: str):
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Магазин", "Дата", "Товар", "Кол-во", "Цена за ед.", 
                         "Сумма", "Скидка", "Уверенность %", "Общий итог"])
        
        store = data.get("store", "—")
        date = data.get("date", "—")
        total = data.get("total", "—")
        
        if not data.get("items"):
            writer.writerow([store, date, "", "", "", "", "", "", total])
            return
            
        for item in data["items"]:
            writer.writerow([
                store, date,
                item["name"],
                item["qty"],
                item.get("unit_price", "—"),
                item["total"],
                item.get("discount", "—"),
                item["conf"],
                total
            ])
