"""分叉树 branch-next API"""
import gc
import time

from backend.models.model_manager import inference_lock
from backend.platform.oom import exit_if_oom
from backend.core.branch_next import expand_branch_next, BRANCH_NEXT_TOP_K_MAX
from backend.core.completion_generator import PromptTooLongError
from backend.api.analyze import LOCK_WAIT_TIMEOUT
from backend.platform.access_log import get_client_ip, log_request
from backend.platform.source_page import ALLOWED_SOURCE_PAGES, normalize_source_page


def branch_next(branch_next_request):
    prefix = branch_next_request.get("prefix")
    model = branch_next_request.get("model")
    source_page = branch_next_request.get("source_page")
    top_k = branch_next_request.get("top_k")

    if prefix is None or prefix == "":
        return {"success": False, "message": "Missing required field: prefix"}, 400
    if not isinstance(prefix, str):
        return {"success": False, "message": "prefix must be a string"}, 400

    if model is None:
        return {"success": False, "message": "Missing required field: model"}, 400
    if model not in ("base", "instruct"):
        return {"success": False, "message": 'model must be "base" or "instruct"'}, 400

    if source_page is None or source_page == "":
        return {"success": False, "message": "Missing required field: source_page"}, 400
    normalized_source_page = normalize_source_page(source_page)
    if normalized_source_page is None:
        allowed = ", ".join(sorted(ALLOWED_SOURCE_PAGES))
        return {"success": False, "message": f"source_page must be one of: {allowed}"}, 400
    source_page = normalized_source_page

    if top_k is not None:
        if not isinstance(top_k, int):
            return {"success": False, "message": "top_k must be an integer"}, 400
        if top_k < 1:
            return {"success": False, "message": f"top_k must be >= 1"}, 400
        if top_k > BRANCH_NEXT_TOP_K_MAX:
            # clamp 而不是报错（设计 D3）
            top_k = BRANCH_NEXT_TOP_K_MAX

    client_ip = get_client_ip()
    start_time = time.perf_counter()
    log_request("📥 branch_next 请求", f"model={model!r}, source_page={source_page!r}, prefix_chars={len(prefix)}", client_ip)

    lock_acquired = inference_lock.acquire(timeout=LOCK_WAIT_TIMEOUT)
    if not lock_acquired:
        return {"success": False, "message": f"Queue wait exceeded {LOCK_WAIT_TIMEOUT} seconds; server is busy, please try again later."}, 503

    kwargs = {"model": model}
    if top_k is not None:
        kwargs["top_k"] = top_k

    try:
        result = expand_branch_next(prefix, **kwargs)
    except PromptTooLongError as e:
        return {"success": False, "message": str(e)}, 400
    except ValueError as e:
        return {"success": False, "message": str(e)}, 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit_if_oom(e, defer_seconds=1)
        return {"success": False, "message": str(e)}, 500
    finally:
        inference_lock.release()
        gc.collect()

    elapsed = time.perf_counter() - start_time
    print(
        f"\t📤 API branch_next response: "
        f"prefix_tokens={result.get('prefix_tokens')}, "
        f"candidates={len(result.get('candidates', []))}, "
        f"response_time={elapsed:.4f}s"
    )
    return {"success": True, **result}, 200
