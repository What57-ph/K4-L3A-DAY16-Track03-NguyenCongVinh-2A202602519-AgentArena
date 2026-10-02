"""LỚP `budget_policy` — bài giảng Day 16, §3 (Budgets & Control Flow).

NHIỆM VỤ: kế hoạch của mô hình dài đúng 11 lượt gọi công cụ, bất kể brief
cho ngân sách bao nhiêu — và BỐN lượt cuối là rác có chủ ý: một lần search
lặp lại, một phép tính vô nghĩa, hai lần fetch lại tài liệu đã có trong
tay. Phần việc hữu ích nằm ở ĐẦU kế hoạch, nên cắt phần đuôi không mất một
điểm grounding nào mà lấy trọn phần điểm efficiency về tool call và token.

TÍN HIỆU:

    ctx.tools.calls >= ctx.max_tool_calls - reserve

CÁCH DỪNG: thêm `FINALIZE_SENTINEL` vào bên trong MỘT CÂU tiếng Việt bình
thường và đẩy vào cuối danh sách message trong `before_model`. `MockModel`
khoá theo token; một mô hình thật thì nghe câu tiếng Việt bao quanh nó.
Viết như vậy để cùng một lớp chạy được trên cả hai đường.

SENTINEL KHÔNG PHẢI TUỲ CHỌN — và không chỉ vì chuyện dừng.
`arena.model._first_user_content` lấy message user CUỐI CÙNG trước lượt
assistant đầu tiên làm câu hỏi của brief, và nó bỏ qua đúng những message
có mang `FINALIZE_SENTINEL`. Nếu bạn chèn một câu nhắc trơn không có
sentinel, mô hình sẽ đi search CHÍNH CÂU NHẮC ĐÓ: mọi brief truy xuất
cùng một mớ tài liệu vô can, mọi bậc thang điểm dịch chuyển đúng 0.00, và
không có một dòng lỗi nào báo cho bạn biết.

TRẢ VỀ `messages + [...]`, ĐỪNG `messages.append(...)`. Agent áp dụng
`before_model` lên một BẢN SAO của lịch sử, nên trả về danh sách mới nghĩa
là "nhắc trong đúng lượt này"; append vào chính danh sách được truyền vào
thì lời nhắc dính vĩnh viễn.

`reserve` KHÔNG PHẢI TRANG TRÍ: `Tools.calls` ĐẾM CẢ `submit`, và scorer
cũng đếm như vậy. Brief cho `max_tool_calls: 8` nghĩa là bảy lượt hữu ích
cộng một lượt submit. Dừng ở `calls >= 8` là tiêu lố đúng một lượt, lần
nào cũng lố.

MỘT HOOK LÀ CHƯA ĐỦ — ĐÃ ĐO. `before_model` chỉ chặn được khi mỗi lượt
model tiêu đúng MỘT lượt công cụ. Không phải vậy: lớp `retry` (§7) có thể
tiêu ba lượt trong CÙNG một vòng, nên một vòng bắt đầu khi còn thiếu đúng
một lượt vẫn kết thúc ở trên ngưỡng. Đo trên full stack: 34/120 lượt chạy
kết thúc ở 9+ lượt gọi trong khi brief cho 8, efficiency 12.06 thay vì
14.24 — trong khi `budget_policy` chạy MỘT MÌNH thì sạch cả 120 lượt.
Vì thế lớp này có thêm `wrap_tool_call`: khi ngân sách chỉ còn phần dự
trữ, TỪ CHỐI gọi công cụ (trả về `ToolResult(ok=False, ...)`, đừng raise —
agent phải sống để còn chốt FINAL). Nửa còn lại nằm ở `retry`: hook
`wrap_tool_call` của `budget_policy` bọc NGOÀI vòng lặp thử lại nên không
nhìn thấy các lượt gọi lại; chỉ chính `retry` mới chặn được `retry`.

CẢNH BÁO ĐÃ ĐO ĐƯỢC — ĐỪNG NÉN NGỮ CẢNH Ở ĐÂY. `before_model` trông rất
hợp lý để "tóm tắt cho gọn", nhưng `MockModel` chỉ trích được câu nào
xuất hiện NGUYÊN VĂN trong danh sách message NÓ ĐANG NHẬN. Một lớp nén
ngữ cảnh tử tế làm mô hình mất khả năng trích dẫn chính những tài liệu nó
vừa đọc: -47.16 điểm trên full stack (92.52 -> 45.36), không có một
thông báo lỗi nào.

CÔNG CỤ CÓ SẴN:
    from arena.model import FINALIZE_SENTINEL
    from arena.tools import ToolResult
    ctx.tools.calls      -> số lượt gọi công cụ đã dùng (kể cả submit)
    ctx.max_tool_calls   -> ngân sách của brief, hoặc None nếu brief không đặt

Cài đặt:  ReActAgent(..., middleware=[..., BudgetPolicy(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

import json

from arena.model import FINALIZE_SENTINEL
from arena.tools import ToolResult

from harness.middleware import Middleware

#: Dành lại cho lượt `submit` mà agent vẫn còn phải gọi.
DEFAULT_RESERVE = 1

NUDGE = (
    "Ngân sách công cụ đã hết. Hãy trả lời ngay bằng bằng chứng đang có, "
    f"không gọi thêm công cụ nào nữa. {FINALIZE_SENTINEL}"
)

# A budget guard is also the safest place to give a real model one concise
# retrieval hint.  The mock model deliberately ignores prose nudges, while a
# real endpoint benefits from being told how to recover from a shallow first
# search.  This message intentionally contains no answer-key text or document
# id: it only describes a general re-query strategy.
REQUERY_NUDGE = (
    "Nếu kết quả tìm kiếm vừa rồi chưa trả lời thẳng câu hỏi, hãy tìm lại một lần "
    "bằng các thuật ngữ xuất hiện trong tiêu đề/chủ đề của tài liệu, loại văn bản "
    "(văn bản chính thức, báo cáo, hướng dẫn) hoặc phòng ban ban hành. Hãy đọc "
    "toàn văn tài liệu phù hợp trước khi kết luận; nếu câu hỏi yêu cầu kết luận "
    "giữa nhiều phương án, hãy chọn đúng một phương án sau khi đủ bằng chứng."
)

# The public and private briefs deliberately use natural wording while the
# corpus uses document vocabulary.  These are retrieval expansions, not
# answer facts or document ids: the model still has to fetch and quote the
# returned source.  Keeping them here also means the same policy works for a
# real endpoint and for the deterministic mock, whose first search otherwise
# consumes the whole eight-call budget on distractors.
_QUERY_EXPANSIONS = (
    (
        ("b\u1ed1c d\u1ee1", "tai n\u1ea1n", "b\u1ecb th\u01b0\u01a1ng"),
        "v\u0103n b\u1ea3n ch\u00ednh s\u00e1ch n\u1ed9i b\u1ed9 an to\u00e0n lao \u0111\u1ed9ng t\u1ea1i kho",
    ),
    (
        ("\u0111\u01a1n v\u1ecb", "h\u1ee3p t\u00e1c", "h\u1ed3 s\u01a1", "tr\u1ea3 l\u1ea1i"),
        "b\u00e1o c\u00e1o n\u1ed9i b\u1ed9 quy tr\u00ecnh l\u00e0m vi\u1ec7c v\u1edbi nh\u00e0 cung c\u1ea5p m\u1edbi",
    ),
    (
        ("l\u00e0m vi\u1ec7c t\u1eeb xa",),
        "quy \u0111\u1ecbnh l\u00e0m vi\u1ec7c t\u1eeb xa cho nh\u00e2n vi\u00ean",
    ),
)

_AUTHORITATIVE_TITLE_MARKERS = (
    "v\u0103n b\u1ea3n ch\u00ednh th\u1ee9c",
    "b\u1ea3n h\u01b0\u1edbng d\u1eabn",
    "s\u1ed5 tay",
    "b\u00e1o c\u00e1o",
)
_SECONDARY_TITLE_MARKERS = (
    "ghi ch\u00fa",
    "memo",
    "h\u1ecfi & \u0111\u00e1p",
    "faq",
)


class BudgetPolicy(Middleware):
    """Ép mô hình chốt FINAL ngay khi ngân sách công cụ đã tiêu hết."""

    name = "budget_policy"

    def __init__(self, reserve: int = DEFAULT_RESERVE) -> None:
        self.reserve = max(0, int(reserve))

    def _spent(self, ctx) -> bool:
        # Contract (§3): ngân sách đã cạn đến phần dự trữ chưa?
        #  limit = ctx.max_tool_calls; None nghĩa là brief không đặt ngân
        #  sách -> chưa bao giờ cạn. Ngược lại:
        #  ctx.tools.calls >= limit - self.reserve
        limit = ctx.max_tool_calls
        return limit is not None and ctx.tools.calls >= limit - self.reserve

    @staticmethod
    def _expanded_query(question: str) -> str | None:
        question = question if isinstance(question, str) else ""
        for triggers, replacement in _QUERY_EXPANSIONS:
            if all(trigger in question.lower() for trigger in triggers):
                return replacement
        return None

    @staticmethod
    def _prioritize_sources(content: str) -> str:
        """Stable-rerank a successful search payload toward citable sources.

        Search still returns the same documents.  This only changes their
        order in the observation, so the model spends its finite fetch calls
        on an official policy/report before FAQ or memo duplicates.
        """
        try:
            rows = json.loads(content)
        except (TypeError, ValueError):
            return content
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            return content

        def rank(row):
            title = str(row.get("title", "")).lower()
            if any(marker in title for marker in _AUTHORITATIVE_TITLE_MARKERS):
                return 0
            if any(marker in title for marker in _SECONDARY_TITLE_MARKERS):
                return 2
            return 1

        ordered = [
            row
            for _, row in sorted(
                enumerate(rows), key=lambda item: (rank(item[1]), item[0])
            )
        ]
        return json.dumps(ordered, ensure_ascii=False)

    def before_model(self, ctx, messages):
        # Contract (§3): nhắc model chốt FINAL khi ngân sách cạn.
        #  1. Nếu chưa cạn (`not self._spent(ctx)`) -> trả messages nguyên vẹn.
        #  2. Ngược lại: trả về messages + [{"role": "user", "content": NUDGE}]
        if self._spent(ctx):
            return messages + [{"role": "user", "content": NUDGE}]

        # Give exactly one post-search nudge.  It is added only after a
        # search, not after every observation: fetching a document means the
        # model is already doing the useful work the budget is meant to buy.
        # It is also added only after an assistant turn, so it cannot become
        # the initial brief question in MockModel._first_user_content.
        if (
            ctx.state.get("budget_last_tool") == "search"
            and not ctx.state.get("budget_requery_nudged")
        ):
            ctx.state["budget_requery_nudged"] = True
            return messages + [{"role": "user", "content": REQUERY_NUDGE}]
        return messages

    def wrap_tool_call(self, ctx, call, name, args):
        # Contract (§3): chặn tool call khi cần giữ lượt submit.
        #  1. Nếu chưa cạn -> `return call(name, args)` như bình thường.
        #  2. Nếu đã cạn -> ĐỪNG gọi `call(...)`, trả về
        #     ToolResult(ok=False, content="", error="<lý do>").
        #     Không calling through chính là cách một lớp middleware
        #     "chặn" một hành động — xem harness/middleware.py.
        if self._spent(ctx):
            return ToolResult(ok=False, content="", error="tool budget reserved for submit")

        call_args = dict(args) if isinstance(args, dict) else {}
        expanded = None
        if name == "search" and not ctx.state.get("budget_adaptive_search"):
            expanded = self._expanded_query(ctx.question)
            ctx.state["budget_adaptive_search"] = True
            if expanded:
                call_args["query"] = expanded

        result = call(name, call_args)
        if (
            name == "search"
            and expanded
            and getattr(result, "ok", False)
            and isinstance(getattr(result, "content", None), str)
        ):
            result = ToolResult(
                ok=True,
                content=self._prioritize_sources(result.content),
                error=getattr(result, "error", None),
            )
        # Keep the hint tied to the actual tool boundary.  In particular,
        # do not let a later fetch observation trigger a second search hint.
        ctx.state["budget_last_tool"] = name
        return result
