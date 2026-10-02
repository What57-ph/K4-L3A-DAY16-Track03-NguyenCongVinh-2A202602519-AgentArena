"""LỚP `critic` — bài giảng Day 16, §2 (Reflection & Self-Critique).

NHIỆM VỤ: mô hình KHÔNG BAO GIỜ nói "tôi không biết". `abstain` bị gán
cứng `False`, và nó bịa theo ba kiểu khác nhau:

  (a) brief `absent`  -> bịa ra một con số không có trong tài liệu nào.
  (b) không có bằng chứng -> bịa ra một câu chung chung vô thưởng vô phạt.
  (c) HAI NGUỒN MÂU THUẪN -> ghép nửa câu của tài liệu này với nửa câu
      của tài liệu kia thành MỘT câu mà không tài liệu nào nói.

TÍN HIỆU (chỉ một dòng): câu trong `claim["text"]` có xuất hiện NGUYÊN VĂN
trong bằng chứng agent đã thực sự đọc hay không —

    text in ctx.observed_text

Trên một brief có bằng chứng tốt thì mọi claim đều thoả điều kiện này,
nên critic xây trên tín hiệu đó không báo động giả.

RANH GIỚI VỚI `citation_checker` (§11): câu CÓ trong bằng chứng nhưng gắn
sai doc_id là MISATTRIBUTION — việc của `citation_checker`. Câu KHÔNG có
trong bất kỳ bằng chứng nào là FABRICATION — việc của bạn ở đây. Hai điều
kiện loại trừ nhau, đừng làm phần việc của lớp kia.

ĐIỂM SỐ (đọc kỹ, đây là nơi kiếm nhiều điểm nhất):
  * Một claim bịa bị chấm `HALLUCINATED`: mất điểm precision VÀ mất trọn
    15 điểm honesty, trên MỌI brief.
  * Trên brief `is_absent`, `abstain: true` được 0.75 recall + trọn 15
    điểm honesty. "Không có số liệu" CHÍNH LÀ câu trả lời đúng.
  * Trên brief mâu thuẫn, ĐỪNG trông đợi "nêu cả hai phía" tự động cho
    recall đầy đủ: recall chấm THEO TỪNG required_fact bằng key terms
    của chính fact đó, không phải theo số vế đã trích dẫn — nếu nửa câu
    mô hình thực sự viết ra không phủ hết từ khoá của một fact (mô hình
    ghép câu ở chỗ NÓ chọn, không nhất thiết đúng ranh giới required_fact),
    fact đó vẫn 0 điểm dù trích dẫn đúng. Trên `pub-04-lam-viec-tu-xa` cụ
    thể, trần recall là 0.5 với MỌI harness đúng luật, vì đúng lý do đó —
    đo được, không phải suy đoán. Vẫn nên làm: `abstain: true` sau khi nêu
    cả hai phía được 0.5 recall + trọn 15 điểm honesty, và điểm recall lấy
    theo `max(...)` nên làm cả hai không bao giờ THIỆT — chỉ đừng trông
    đợi nó vượt sàn 0.5 trên brief này.
  * Xoá claim là hợp lệ. SỬA CHỮ trong `claim["text"]` thì KHÔNG: thêm
    một dấu chấm cuối câu cũng đủ làm claim mất cả provenance lẫn hỗ trợ
    (đo được: -40 điểm). Chỉ được xoá, giữ nguyên, hoặc cắt bớt.

GỢI Ý cho trường hợp (c): câu bị ghép là hai đoạn DO CHÍNH MÔ HÌNH viết,
dán với nhau bằng một liên từ (" và "). Cắt đúng chỗ dán thì hai nửa vẫn
là chữ của mô hình — vẫn qua được kiểm tra provenance. Muốn biết cắt đúng
chưa: cả hai nửa phải xuất hiện nguyên văn trong `ctx.observed_text` và
phải thuộc HAI tài liệu khác nhau. Cắt sai thì một nửa sẽ vắt qua hai tài
liệu và không quan sát nào chứa nó.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.saw(text)      -> text có trong quan sát không
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.
    ctx.state          -> dict tuỳ bạn dùng để ghi số liệu gỡ lỗi

Cài đặt:  ReActAgent(..., middleware=[InjectionGuard(), Critic(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

from harness.middleware import Middleware


def _has_verbatim_line(doc, text: str) -> bool:
    """Return whether ``text`` is a substring of one document line.

    The scorer deliberately scopes quotation support to a single line.  The
    critic must use the same boundary when it repairs a fused claim; checking
    ``text in doc.body`` would accidentally bless claims assembled across
    lines.
    """
    return bool(
        isinstance(text, str)
        and text
        and isinstance(getattr(doc, "body", None), str)
        and any(text in line for line in doc.body.splitlines())
    )


def _was_observed(ctx, doc) -> bool:
    """Whether the run exposed this document, by id or by full body."""
    doc_id = getattr(doc, "doc_id", None)
    body = getattr(doc, "body", None)
    observed = ctx.observed_text
    return bool(
        (isinstance(doc_id, str) and doc_id and doc_id in observed)
        or (isinstance(body, str) and body and body in observed)
    )


def _line_containing(doc, text: str) -> str | None:
    """Return the document line containing a retained claim fragment."""
    if not isinstance(text, str) or not text:
        return None
    body = getattr(doc, "body", None)
    if not isinstance(body, str):
        return None
    return next((line for line in body.splitlines() if text in line), None)


class Critic(Middleware):
    """Xoá những gì bằng chứng không đỡ; abstain khi không còn gì."""

    name = "critic"

    def after_agent(self, ctx, report):
        # Contract (§2): lọc và tách claim theo bằng chứng quan sát được.
        #  1. Lấy report["claims"]; nếu rỗng hoặc không phải list thì thôi.
        #  2. Với mỗi claim: nếu claim["text"] có trong ctx.observed_text
        #     -> giữ nguyên (KHÔNG sửa chữ).
        #  3. Nếu không: thử tách câu ghép (trường hợp (c) ở docstring).
        #     Tách được -> giữ cả hai nửa, mỗi nửa gắn doc_id của tài liệu
        #     thật sự chứa nó, và đặt report["abstain"] = True.
        #  4. Không tách được -> đây là bịa: bỏ claim đi.
        #  5. Nếu không còn claim nào: report["abstain"] = True,
        #     claims = [], citations = [], và viết lại "answer" nói rõ là
        #     không đủ căn cứ.
        #  6. Cập nhật report["citations"] cho khớp với claims còn lại.
        claims = report.get("claims")
        if not isinstance(claims, list) or not claims:
            return report

        docs = tuple(getattr(ctx.corpus, "docs", ()))
        kept = []
        answer_lines = []
        for claim in claims:
            text = claim.get("text") if isinstance(claim, dict) else None
            if not isinstance(text, str):
                continue
            if ctx.saw(text):
                # A visible quotation is not enough when its citation is
                # fabricated or points at a document the run never saw.
                # CitationChecker gets first chance to repair the doc_id;
                # anything still invalid here must be removed rather than
                # submitted as an auditable claim.
                cited = None
                cited_id = claim.get("doc_id")
                if ctx.corpus is not None and isinstance(cited_id, str):
                    cited = ctx.corpus.get(cited_id)
                if ctx.corpus is None or (
                    cited is not None and _was_observed(ctx, cited)
                ):
                    kept.append(claim)
                continue
            for index in range(len(text)):
                if not text.startswith(" và ", index):
                    continue
                left, right = text[:index], text[index + len(" và "):]
                left_docs = [
                    d for d in docs
                    if ctx.saw(left)
                    and _has_verbatim_line(d, left)
                    and _was_observed(ctx, d)
                ]
                right_docs = [
                    d for d in docs
                    if ctx.saw(right)
                    and _has_verbatim_line(d, right)
                    and _was_observed(ctx, d)
                ]
                pair = next(((a, b) for a in left_docs for b in right_docs
                             if a.doc_id != b.doc_id), None)
                if pair:
                    kept.extend([{"text": left, "doc_id": pair[0].doc_id},
                                 {"text": right, "doc_id": pair[1].doc_id}])
                    for doc, fragment in (
                        (pair[0], left),
                        (pair[1], right),
                    ):
                        line = _line_containing(doc, fragment)
                        if line and line not in answer_lines:
                            answer_lines.append(line)
                    report["abstain"] = True
                    break
            else:
                continue

        report["claims"] = kept
        if not kept:
            report["abstain"] = True
            report["citations"] = []
            report["answer"] = "Không đủ căn cứ để trả lời từ các tài liệu đã quan sát."
        else:
            report["citations"] = sorted({c.get("doc_id") for c in kept
                                           if isinstance(c, dict)
                                           and isinstance(c.get("doc_id"), str)
                                           and c.get("doc_id")})
            # The mock fuses contradictory spans and clips each side.  Keep
            # the model-produced fragments as claims, but show the complete
            # observed lines in the free-form answer so both policies remain
            # understandable and the answer covers the cited evidence.
            if answer_lines:
                report["answer"] = (
                    "Các tài liệu đã đọc nêu hai quy định khác nhau: "
                    + " | ".join(answer_lines)
                )
        return report
