from __future__ import annotations

from collections import Counter
from typing import Mapping

import pandas as pd


ROBUSTNESS_CASES = [
    ("Hom nay toi di hoc va gap ban be.", "CLEAN", "neutral"),
    ("Bai viet nay phan tich hien tuong ky thi tren mang.", "CLEAN", "neutral"),
    ("Chung ta nen ton trong su khac biet cua moi nguoi.", "CLEAN", "neutral"),
    ("Moi nguoi deu co quyen duoc doi xu cong bang.", "CLEAN", "neutral"),
    ("Toi khong ghet ai ca.", "CLEAN", "negation"),
    ("Khong ai dang bi xuc pham trong cau nay.", "CLEAN", "negation"),
    ("Khong nen cong kich nguoi khac vi gioi tinh hay que quan.", "CLEAN", "negation"),
    ("Dung su dung ngon tu thu ghet de tan cong bat ky nhom nao.", "CLEAN", "negation"),
    ("Chung ta khong nen ky thi nguoi khac.", "CLEAN", "negation"),
    ("Can ngan chan loi noi thu han tren mang xa hoi.", "CLEAN", "keyword_context"),
    ("Tu ngu do rat xuc pham, nhung toi dang trich dan de phan tich.", "CLEAN", "keyword_context"),
    ("Bao cao nay noi ve viec phat hien ngon tu thu ghet.", "CLEAN", "keyword_context"),
    ("Bai nghien cuu nay phan tich cac cau noi phan biet doi xu.", "CLEAN", "keyword_context"),
    ("Cau noi tren la vi du ve loi le doc hai trong du lieu.", "CLEAN", "keyword_context"),
    ("May noi chuyen ngu that.", "OFFENSIVE", "direct_insult"),
    ("Thang nay dung la do vo hoc.", "OFFENSIVE", "direct_insult"),
    ("Con nho do noi nang mat day.", "OFFENSIVE", "direct_insult"),
    ("Bai viet cua may nham nhi va rac ruoi.", "OFFENSIVE", "direct_insult"),
    ("Do la mot thang ngu ngoc.", "OFFENSIVE", "direct_insult"),
    ("May dung la ke vo dung.", "OFFENSIVE", "direct_insult"),
    ("No noi chuyen cuc ky thieu van hoa.", "OFFENSIVE", "direct_insult"),
    ("Cai do cua may that rac ruoi.", "OFFENSIVE", "direct_insult"),
    ("Dung la do an hai.", "OFFENSIVE", "direct_insult"),
    ("Lu do khong nen duoc song o day.", "HATE", "group_attack"),
    ("Duoi het bon do ra khoi dat nuoc nay.", "HATE", "group_attack"),
    ("Nhung nguoi nhu vay khong xung dang co quyen loi.", "HATE", "group_attack"),
    ("Cam cho cai nhom do di hoc va di lam.", "HATE", "group_attack"),
    ("Khong cho bon do song chung voi chung ta.", "HATE", "group_attack"),
    ("Loai het cai nhom do ra khoi cong dong.", "HATE", "group_attack"),
    ("Bon do khong duoc phep o lai khu nay.", "HATE", "group_attack"),
    ("Tui no la mam benh cua xa hoi.", "HATE", "dehumanization"),
    ("Loai nguoi do chi lam ban cong dong.", "HATE", "dehumanization"),
    ("Bon do la rac ruoi cua xa hoi.", "HATE", "dehumanization"),
    ("Cai nhom do la moi nguy hai can bi loai bo.", "HATE", "dehumanization"),
    ("Khong cho bon do xuat hien o khu nay nua.", "HATE", "exclusion"),
    ("Cam nhom do vao truong hoc va noi lam viec.", "HATE", "exclusion"),
    ("Dung de nhung nguoi do co bat ky quyen loi nao.", "HATE", "exclusion"),
]


def build_robustness_frame(label2id: Mapping[str, int]) -> tuple[pd.DataFrame, dict]:
    rows = []
    category_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    for text, label, category in ROBUSTNESS_CASES:
        if label not in label2id:
            raise ValueError(f"Unknown robustness label: {label}")
        rows.append({"text": text, "label": int(label2id[label])})
        category_counts[category] += 1
        label_counts[label] += 1
    summary = {
        "base_examples": len(rows),
        "label_counts": dict(label_counts),
        "category_counts": dict(category_counts),
    }
    return pd.DataFrame(rows), summary
