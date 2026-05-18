from __future__ import annotations

from collections import Counter
from typing import Mapping


# Train-only hard cases. Keep these separate from ROBUSTNESS_HOLDOUT_CASES so
# manual robustness testing does not accidentally test memorized sentences.
ROBUSTNESS_TRAIN_CASES = [
    ("Hom nay toi di hoc va gap ban be.", "CLEAN", "neutral"),
    ("Bua nay quan ca phe dong hon moi ngay.", "CLEAN", "neutral"),
    ("Bai viet nay phan tich hien tuong ky thi tren mang.", "CLEAN", "neutral"),
    ("Chung ta nen ton trong su khac biet cua moi nguoi.", "CLEAN", "neutral"),
    ("Moi nguoi deu co quyen duoc doi xu cong bang.", "CLEAN", "neutral"),
    ("Binh luan nay chi dang hoi them thong tin.", "CLEAN", "neutral"),
    ("Toi khong ghet ai ca.", "CLEAN", "negation"),
    ("Khong ai dang bi xuc pham trong cau nay.", "CLEAN", "negation"),
    ("Khong nen cong kich nguoi khac vi gioi tinh hay que quan.", "CLEAN", "negation"),
    ("Dung su dung ngon tu thu ghet de tan cong bat ky nhom nao.", "CLEAN", "negation"),
    ("Chung ta khong nen ky thi nguoi khac.", "CLEAN", "negation"),
    ("Khong duoc ha thap bat cu ai vi noi sinh cua ho.", "CLEAN", "negation"),
    ("Can ngan chan loi noi thu han tren mang xa hoi.", "CLEAN", "keyword_context"),
    ("Tu ngu do rat xuc pham, nhung toi dang trich dan de phan tich.", "CLEAN", "keyword_context"),
    ("Bao cao nay noi ve viec phat hien ngon tu thu ghet.", "CLEAN", "keyword_context"),
    ("Bai nghien cuu nay phan tich cac cau noi phan biet doi xu.", "CLEAN", "keyword_context"),
    ("Cau noi tren la vi du ve loi le doc hai trong du lieu.", "CLEAN", "keyword_context"),
    ("Nhom nghien cuu dang danh gia cac binh luan doc hai.", "CLEAN", "keyword_context"),
    ("May noi chuyen ngu that.", "OFFENSIVE", "direct_insult"),
    ("Thang nay dung la do vo hoc.", "OFFENSIVE", "direct_insult"),
    ("Con nho do noi nang mat day.", "OFFENSIVE", "direct_insult"),
    ("Bai viet cua may nham nhi va rac ruoi.", "OFFENSIVE", "direct_insult"),
    ("Do la mot thang ngu ngoc.", "OFFENSIVE", "direct_insult"),
    ("May dung la ke vo dung.", "OFFENSIVE", "direct_insult"),
    ("No noi chuyen cuc ky thieu van hoa.", "OFFENSIVE", "direct_insult"),
    ("Cai do cua may that rac ruoi.", "OFFENSIVE", "direct_insult"),
    ("Dung la do an hai.", "OFFENSIVE", "direct_insult"),
    ("Binh luan cua may nghe that kem coi.", "OFFENSIVE", "direct_insult"),
    ("No lam viec nhu mot ke bat tai.", "OFFENSIVE", "direct_insult"),
    ("May noi cau nao cung vo duyen.", "OFFENSIVE", "direct_insult"),
    ("Cai y kien nay qua ngu xuan.", "OFFENSIVE", "direct_insult"),
    ("Do la mot bai dang rac ruoi.", "OFFENSIVE", "direct_insult"),
    ("May noi nghe chan that.", "OFFENSIVE", "direct_insult"),
    ("No dung la ke chang biet dieu.", "OFFENSIVE", "direct_insult"),
    ("Cau tra loi cua may that ngu ngoc.", "OFFENSIVE", "direct_insult"),
    ("Nguoi nay hanh xu qua te hai.", "OFFENSIVE", "direct_insult"),
    ("Binh luan do nghe cuc ky tho lo.", "OFFENSIVE", "direct_insult"),
    ("May viet bai nay qua kem.", "OFFENSIVE", "direct_insult"),
    ("No la dua an noi bo la.", "OFFENSIVE", "direct_insult"),
    ("Cach lap luan cua may that tham hai.", "OFFENSIVE", "direct_insult"),
    ("Cai y tuong nay dung la nham nhi.", "OFFENSIVE", "direct_insult"),
    ("May lam toi thay buc minh vi qua vo ly.", "OFFENSIVE", "direct_insult"),
    ("Bai dang nay qua kem chat luong va vo ich.", "OFFENSIVE", "direct_insult"),
    ("No cu xu nhu mot ke thieu y thuc.", "OFFENSIVE", "direct_insult"),
    ("May an noi chang co chut lich su nao.", "OFFENSIVE", "direct_insult"),
    ("Cau noi cua no nghe rat mat day.", "OFFENSIVE", "direct_insult"),
    ("Thang do lam viec qua au.", "OFFENSIVE", "direct_insult"),
    ("Lu do khong nen duoc song o day.", "HATE", "group_attack"),
    ("Duoi het bon do ra khoi dat nuoc nay.", "HATE", "group_attack"),
    ("Nhung nguoi nhu vay khong xung dang co quyen loi.", "HATE", "group_attack"),
    ("Cam cho cai nhom do di hoc va di lam.", "HATE", "group_attack"),
    ("Khong cho bon do song chung voi chung ta.", "HATE", "group_attack"),
    ("Loai het cai nhom do ra khoi cong dong.", "HATE", "group_attack"),
    ("Bon do khong duoc phep o lai khu nay.", "HATE", "group_attack"),
    ("Nhom do khong nen duoc tham gia xa hoi.", "HATE", "group_attack"),
    ("Tat ca bon do phai bien mat khoi khu pho nay.", "HATE", "group_attack"),
    ("Tui no la mam benh cua xa hoi.", "HATE", "dehumanization"),
    ("Loai nguoi do chi lam ban cong dong.", "HATE", "dehumanization"),
    ("Bon do la rac ruoi cua xa hoi.", "HATE", "dehumanization"),
    ("Cai nhom do la moi nguy hai can bi loai bo.", "HATE", "dehumanization"),
    ("Bon do khong khac gi sau bo.", "HATE", "dehumanization"),
    ("Nhom nguoi do la dich benh cua dat nuoc.", "HATE", "dehumanization"),
    ("Khong cho bon do xuat hien o khu nay nua.", "HATE", "exclusion"),
    ("Cam nhom do vao truong hoc va noi lam viec.", "HATE", "exclusion"),
    ("Dung de nhung nguoi do co bat ky quyen loi nao.", "HATE", "exclusion"),
    ("Khong cho cai nhom do thue nha o day.", "HATE", "exclusion"),
]


# Evaluation-only holdout cases. These should not be appended to the training
# frame. They intentionally cover the same categories with different wording.
ROBUSTNESS_HOLDOUT_CASES = [
    ("Sang nay toi doc tin tuc roi di lam.", "CLEAN", "neutral"),
    ("Cuoc hop da ket thuc som hon du kien.", "CLEAN", "neutral"),
    ("Moi nguoi nen duoc lang nghe mot cach cong bang.", "CLEAN", "neutral"),
    ("Binh luan nay khong cong kich bat ky ai.", "CLEAN", "neutral"),
    ("Toi phan doi moi hinh thuc ky thi.", "CLEAN", "negation"),
    ("Khong nen dung loi le thieu ton trong voi nguoi khac.", "CLEAN", "negation"),
    ("Dung tan cong mot nguoi chi vi ho khac minh.", "CLEAN", "negation"),
    ("Khong co ly do gi de ha thap mot cong dong.", "CLEAN", "negation"),
    ("Bai bao dang mo ta cach nhan dien binh luan xuc pham.", "CLEAN", "keyword_context"),
    ("Cau do la vi du trong lop hoc ve ngon tu thu ghet.", "CLEAN", "keyword_context"),
    ("Toi trich dan cau noi doc hai nay de canh bao nguoi doc.", "CLEAN", "keyword_context"),
    ("Nha nghien cuu phan tich tu ngu phan biet doi xu.", "CLEAN", "keyword_context"),
    ("May dung la do kem hieu biet.", "OFFENSIVE", "direct_insult"),
    ("No an noi chang ra gi.", "OFFENSIVE", "direct_insult"),
    ("Cai bai viet nay qua vo nghia.", "OFFENSIVE", "direct_insult"),
    ("Dung la mot ke bat lich su.", "OFFENSIVE", "direct_insult"),
    ("May lam viec chan khong chiu duoc.", "OFFENSIVE", "direct_insult"),
    ("Nguoi nay noi chuyen cuc ky vo le.", "OFFENSIVE", "direct_insult"),
    ("Binh luan cua no that thap kem.", "OFFENSIVE", "direct_insult"),
    ("Bon nguoi do nen bi tong ra khoi thanh pho.", "HATE", "group_attack"),
    ("Nhom do khong xung dang duoc song yen on.", "HATE", "group_attack"),
    ("Khong nen cho nhung nguoi do vao noi cong cong.", "HATE", "group_attack"),
    ("Tat ca cai nhom ay phai bi cam cua.", "HATE", "group_attack"),
    ("Nhung ke do chi pha hoai xa hoi.", "HATE", "dehumanization"),
    ("Bon do la vet ban cua cong dong.", "HATE", "dehumanization"),
    ("Nhom nguoi do nhu thu rac thai can quet sach.", "HATE", "dehumanization"),
    ("Khong cho ho hoc tap hay lam viec o day nua.", "HATE", "exclusion"),
    ("Phai loai bo nhom do khoi moi quyen loi.", "HATE", "exclusion"),
    ("Dung cho nhung nguoi do duoc o gan chung ta.", "HATE", "exclusion"),
]


def build_robustness_frame(label2id: Mapping[str, int]) -> tuple["pd.DataFrame", dict]:
    import pandas as pd

    rows = []
    for text, label, _category in ROBUSTNESS_TRAIN_CASES:
        if label not in label2id:
            raise ValueError(f"Unknown robustness label: {label}")
        rows.append({"text": text, "label": int(label2id[label])})
    return pd.DataFrame(rows), summarize_cases(ROBUSTNESS_TRAIN_CASES)


def summarize_cases(cases: list[tuple[str, str, str]]) -> dict:
    category_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    for _text, label, category in cases:
        category_counts[category] += 1
        label_counts[label] += 1
    return {
        "base_examples": len(cases),
        "label_counts": dict(label_counts),
        "category_counts": dict(category_counts),
    }


def normalized_case_texts(cases: list[tuple[str, str, str]]) -> set[str]:
    return {" ".join(text.lower().split()) for text, _label, _category in cases}
