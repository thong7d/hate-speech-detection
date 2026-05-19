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
    ("Khong co ly do nao de coi thuong mot cong dong.", "CLEAN", "negation"),
    ("Dung ha thap nguoi khac chi vi ho thuoc mot nhom nao do.", "CLEAN", "negation"),
    ("Khong nen noi loi lam ton thuong bat ky cong dong nao.", "CLEAN", "negation"),
    ("Toi phan doi viec goi mot nhom nguoi bang tu ngu xau.", "CLEAN", "negation"),
    ("Hay tranh nhung cau noi lam mat pham gia cua nguoi khac.", "CLEAN", "negation"),
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
    ("May lam viec qua te.", "OFFENSIVE", "direct_insult"),
    ("Cach lam viec cua may that dang that vong.", "OFFENSIVE", "direct_insult"),
    ("No xu ly viec nay qua chan.", "OFFENSIVE", "direct_insult"),
    ("May lam cai gi cung cau tha.", "OFFENSIVE", "direct_insult"),
    ("Nguoi nay lam viec kem den buc minh.", "OFFENSIVE", "direct_insult"),
    ("Cach no lam viec that vo trach nhiem.", "OFFENSIVE", "direct_insult"),
    ("Ket qua cua may that tham hai.", "OFFENSIVE", "direct_insult"),
    ("No lam toi kho chiu vi qua te.", "OFFENSIVE", "direct_insult"),
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


CONTRASTIVE_TRAIN_CASES = [
    ("Toi ghet viec ky thi nguoi khac.", "CLEAN", "subject_keyword_flip"),
    ("Toi ghet nhom nguoi do.", "HATE", "subject_keyword_flip"),
    ("Ho ghet viec bi phan biet doi xu.", "CLEAN", "subject_keyword_flip"),
    ("Ho ghet cong dong do va muon duoi di.", "HATE", "subject_keyword_flip"),
    ("Bai bao noi ve su thu ghet trong xa hoi.", "CLEAN", "subject_keyword_flip"),
    ("Cai nhom do dang bi thu ghet va can duoc bao ve.", "CLEAN", "subject_keyword_flip"),
    ("Cai nhom do dang lam ban xa hoi.", "HATE", "subject_keyword_flip"),
    ("Toi khong muon ai bi loai khoi cong dong.", "CLEAN", "negation_exclusion_flip"),
    ("Toi muon loai nhom do khoi cong dong.", "HATE", "negation_exclusion_flip"),
    ("Khong nen cam bat ky ai di hoc hay di lam.", "CLEAN", "negation_exclusion_flip"),
    ("Can cam nhom do di hoc va di lam.", "HATE", "negation_exclusion_flip"),
    ("Dung duoi nguoi khac khoi noi o cua ho.", "CLEAN", "negation_exclusion_flip"),
    ("Hay duoi het nhom do khoi noi nay.", "HATE", "negation_exclusion_flip"),
    ("Binh luan nay khong nham tan cong ai.", "CLEAN", "attack_context_flip"),
    ("Binh luan nay tan cong truc tiep nhom do.", "HATE", "attack_context_flip"),
    ("Nguoi viet dang phan tich mot cau cong kich.", "CLEAN", "attack_context_flip"),
    ("Nguoi viet dang cong kich ca mot cong dong.", "HATE", "attack_context_flip"),
    ("Cau nay chi canh bao ve ngon tu xuc pham.", "CLEAN", "quote_context_flip"),
    ("Cau nay dung ngon tu xuc pham de ha thap nguoi khac.", "OFFENSIVE", "quote_context_flip"),
    ("Toi trich dan loi le do de giai thich vi sao no sai.", "CLEAN", "quote_context_flip"),
    ("May dung la chang biet nghi gi ca.", "OFFENSIVE", "personal_vs_group_flip"),
    ("Nhom do dung la thu khong nen ton tai.", "HATE", "personal_vs_group_flip"),
    ("No lam viec qua te.", "OFFENSIVE", "personal_vs_group_flip"),
    ("Bon do lam ban ca cong dong.", "HATE", "personal_vs_group_flip"),
    ("Cach noi cua may that vo duyen.", "OFFENSIVE", "personal_vs_group_flip"),
    ("Cach song cua nhom do can bi loai bo.", "HATE", "personal_vs_group_flip"),
    ("Day la loi khuyen khong nen ha thap ai.", "CLEAN", "safe_advice_flip"),
    ("Day la loi keu goi ha thap nhom do.", "HATE", "safe_advice_flip"),
    ("Chung ta nen bao ve nguoi bi ky thi.", "CLEAN", "safe_advice_flip"),
    ("Chung ta nen ngan nhom do huong quyen loi.", "HATE", "safe_advice_flip"),
]


CONTRASTIVE_HOLDOUT_CASES = [
    ("Toi ghet viec bat nat tren mang.", "CLEAN", "subject_keyword_flip"),
    ("Toi ghet cong dong do va muon ho bien mat.", "HATE", "subject_keyword_flip"),
    ("Bai viet nhac den su thu ghet de phan tich.", "CLEAN", "subject_keyword_flip"),
    ("Nhom do la ly do xa hoi te di.", "HATE", "subject_keyword_flip"),
    ("Khong nen ngan ai den truong hay di lam.", "CLEAN", "negation_exclusion_flip"),
    ("Phai ngan nhom do den truong va di lam.", "HATE", "negation_exclusion_flip"),
    ("Dung bat nguoi khac roi khoi noi ho song.", "CLEAN", "negation_exclusion_flip"),
    ("Bat nhom do roi khoi khu pho nay.", "HATE", "negation_exclusion_flip"),
    ("Nhan xet nay khong nham cong kich ca nhan nao.", "CLEAN", "attack_context_flip"),
    ("Nhan xet nay cong kich ca mot nhom nguoi.", "HATE", "attack_context_flip"),
    ("Nguoi noi dang trich dan cau xuc pham de phan tich.", "CLEAN", "quote_context_flip"),
    ("Nguoi noi dung cau xuc pham de chui nguoi khac.", "OFFENSIVE", "quote_context_flip"),
    ("May tra loi qua te.", "OFFENSIVE", "personal_vs_group_flip"),
    ("Cong dong do khong nen duoc chap nhan.", "HATE", "personal_vs_group_flip"),
    ("No lam viec that kem coi.", "OFFENSIVE", "personal_vs_group_flip"),
    ("Nhom do chi lam hai xa hoi.", "HATE", "personal_vs_group_flip"),
    ("Hay dung nhung loi noi lam ton thuong cong dong.", "CLEAN", "safe_advice_flip"),
    ("Hay lam cho cong dong do mat het quyen loi.", "HATE", "safe_advice_flip"),
]


DIACRITIC_TRAIN_CASES = [
    ("Tôi ghét việc kỳ thị người khác.", "CLEAN", "diacritic_subject_keyword_flip"),
    ("Tôi ghét nhóm người đó.", "HATE", "diacritic_subject_keyword_flip"),
    ("Bài báo nói về sự thù ghét trong xã hội.", "CLEAN", "diacritic_subject_keyword_flip"),
    ("Cái nhóm đó đang làm bẩn xã hội.", "HATE", "diacritic_subject_keyword_flip"),
    ("Không nên công kích người khác vì giới tính hay quê quán.", "CLEAN", "diacritic_negation"),
    ("Không có lý do nào để coi thường một cộng đồng.", "CLEAN", "diacritic_negation"),
    ("Đừng sử dụng ngôn từ thù ghét để tấn công bất kỳ nhóm nào.", "CLEAN", "diacritic_negation"),
    ("Không nên cấm bất kỳ ai đi học hay đi làm.", "CLEAN", "diacritic_negation_exclusion"),
    ("Cần cấm nhóm đó đi học và đi làm.", "HATE", "diacritic_negation_exclusion"),
    ("Đừng đuổi người khác khỏi nơi ở của họ.", "CLEAN", "diacritic_negation_exclusion"),
    ("Hãy đuổi hết nhóm đó khỏi nơi này.", "HATE", "diacritic_negation_exclusion"),
    ("Bình luận này không nhằm tấn công ai.", "CLEAN", "diacritic_attack_context"),
    ("Bình luận này tấn công trực tiếp nhóm đó.", "HATE", "diacritic_attack_context"),
    ("Người viết đang phân tích một câu công kích.", "CLEAN", "diacritic_attack_context"),
    ("Người viết đang công kích cả một cộng đồng.", "HATE", "diacritic_attack_context"),
    ("Từ ngữ đó rất xúc phạm, nhưng tôi đang trích dẫn để phân tích.", "CLEAN", "diacritic_quote_context"),
    ("Câu này dùng ngôn từ xúc phạm để hạ thấp người khác.", "OFFENSIVE", "diacritic_quote_context"),
    ("Tôi trích dẫn lời lẽ đó để giải thích vì sao nó sai.", "CLEAN", "diacritic_quote_context"),
    ("Mày nói chuyện ngu thật.", "OFFENSIVE", "diacritic_personal_vs_group"),
    ("Nó làm việc quá tệ.", "OFFENSIVE", "diacritic_personal_vs_group"),
    ("Cách nói của mày thật vô duyên.", "OFFENSIVE", "diacritic_personal_vs_group"),
    ("Bọn đó làm bẩn cả cộng đồng.", "HATE", "diacritic_personal_vs_group"),
    ("Nhóm đó đúng là thứ không nên tồn tại.", "HATE", "diacritic_personal_vs_group"),
    ("Cách sống của nhóm đó cần bị loại bỏ.", "HATE", "diacritic_personal_vs_group"),
    ("Hôm nay trời đẹp, tôi đi học cùng bạn.", "CLEAN", "diacritic_neutral"),
    ("Bình luận này chỉ đang hỏi thêm thông tin.", "CLEAN", "diacritic_neutral"),
    ("Mọi người nên được lắng nghe một cách công bằng.", "CLEAN", "diacritic_neutral"),
]


DIACRITIC_HOLDOUT_CASES = [
    ("Tôi ghét việc bắt nạt trên mạng.", "CLEAN", "diacritic_subject_keyword_flip"),
    ("Tôi ghét cộng đồng đó và muốn họ biến mất.", "HATE", "diacritic_subject_keyword_flip"),
    ("Bài viết nhắc đến sự thù ghét để phân tích.", "CLEAN", "diacritic_subject_keyword_flip"),
    ("Nhóm đó là lý do xã hội tệ đi.", "HATE", "diacritic_subject_keyword_flip"),
    ("Không nên dùng lời lẽ thiếu tôn trọng với người khác.", "CLEAN", "diacritic_negation"),
    ("Không có lý do gì để hạ thấp một cộng đồng.", "CLEAN", "diacritic_negation"),
    ("Đừng tấn công một người chỉ vì họ khác mình.", "CLEAN", "diacritic_negation"),
    ("Không nên ngăn ai đến trường hay đi làm.", "CLEAN", "diacritic_negation_exclusion"),
    ("Phải ngăn nhóm đó đến trường và đi làm.", "HATE", "diacritic_negation_exclusion"),
    ("Đừng bắt người khác rời khỏi nơi họ sống.", "CLEAN", "diacritic_negation_exclusion"),
    ("Bắt nhóm đó rời khỏi khu phố này.", "HATE", "diacritic_negation_exclusion"),
    ("Nhận xét này không nhằm công kích cá nhân nào.", "CLEAN", "diacritic_attack_context"),
    ("Nhận xét này công kích cả một nhóm người.", "HATE", "diacritic_attack_context"),
    ("Người nói đang trích dẫn câu xúc phạm để phân tích.", "CLEAN", "diacritic_quote_context"),
    ("Người nói dùng câu xúc phạm để chửi người khác.", "OFFENSIVE", "diacritic_quote_context"),
    ("Mày trả lời quá tệ.", "OFFENSIVE", "diacritic_personal_vs_group"),
    ("Nó làm việc thật kém cỏi.", "OFFENSIVE", "diacritic_personal_vs_group"),
    ("Cộng đồng đó không nên được chấp nhận.", "HATE", "diacritic_personal_vs_group"),
    ("Nhóm đó chỉ làm hại xã hội.", "HATE", "diacritic_personal_vs_group"),
    ("Cuộc họp đã kết thúc sớm hơn dự kiến.", "CLEAN", "diacritic_neutral"),
    ("Sáng nay tôi đọc tin tức rồi đi làm.", "CLEAN", "diacritic_neutral"),
]


def build_robustness_frame(label2id: Mapping[str, int]) -> tuple["pd.DataFrame", dict]:
    import pandas as pd

    rows = []
    for text, label, _category in ROBUSTNESS_TRAIN_CASES:
        if label not in label2id:
            raise ValueError(f"Unknown robustness label: {label}")
        rows.append({"text": text, "label": int(label2id[label])})
    return pd.DataFrame(rows), summarize_cases(ROBUSTNESS_TRAIN_CASES)


def build_contrastive_frame(label2id: Mapping[str, int]) -> tuple["pd.DataFrame", dict]:
    import pandas as pd

    rows = []
    for text, label, _category in CONTRASTIVE_TRAIN_CASES:
        if label not in label2id:
            raise ValueError(f"Unknown contrastive label: {label}")
        rows.append({"text": text, "label": int(label2id[label])})
    return pd.DataFrame(rows), summarize_cases(CONTRASTIVE_TRAIN_CASES)


def build_diacritic_frame(label2id: Mapping[str, int]) -> tuple["pd.DataFrame", dict]:
    import pandas as pd

    rows = []
    for text, label, _category in DIACRITIC_TRAIN_CASES:
        if label not in label2id:
            raise ValueError(f"Unknown diacritic label: {label}")
        rows.append({"text": text, "label": int(label2id[label])})
    return pd.DataFrame(rows), summarize_cases(DIACRITIC_TRAIN_CASES)


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
