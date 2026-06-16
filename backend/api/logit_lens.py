"""Logit Lens API"""
import gc
import time

from backend.models.model_manager import inference_lock
from backend.platform.oom import exit_if_oom
from backend.core.logit_lens import analyze_logit_lens
from backend.api.analyze import LOCK_WAIT_TIMEOUT
from backend.platform.access_log import get_client_ip, log_prediction_attribute_request
from backend.platform.source_page import ALLOWED_SOURCE_PAGES, normalize_source_page


def logit_lens(logit_lens_request):
    context = logit_lens_request.get("context")
    target_prediction = logit_lens_request.get("target_prediction")
    target_token_id = logit_lens_request.get("target_token_id")
    model = logit_lens_request.get("model")
    source_page = logit_lens_request.get("source_page")
    flow_id = logit_lens_request.get("flow_id")
    flow_step = logit_lens_request.get("flow_step")

    if context is None or context == "":
        return {"success": False, "message": "Missing required field: context"}, 400
    if not isinstance(context, str):
        return {"success": False, "message": "context must be a string"}, 400
    if target_prediction is not None and not isinstance(target_prediction, str):
        return {"success": False, "message": "target_prediction must be a string"}, 400
    if target_prediction == "":
        return {"success": False, "message": "target_prediction must not be empty"}, 400
    if target_token_id is not None and not isinstance(target_token_id, int):
        return {"success": False, "message": "target_token_id must be an integer"}, 400
    if target_token_id is not None and target_token_id < 0:
        return {"success": False, "message": "target_token_id must be >= 0"}, 400
    if target_prediction is not None and target_token_id is not None:
        return {"success": False, "message": "target_prediction and target_token_id are mutually exclusive"}, 400

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

    if flow_id is not None and not isinstance(flow_id, str):
        return {"success": False, "message": "flow_id must be a string"}, 400
    if flow_id == "":
        return {"success": False, "message": "flow_id must not be empty"}, 400
    if flow_step is not None and not isinstance(flow_step, int):
        return {"success": False, "message": "flow_step must be an integer"}, 400
    if flow_step is not None and flow_step < 0:
        return {"success": False, "message": "flow_step must be >= 0"}, 400

    is_causal_flow = source_page == "causal_flow"
    if is_causal_flow:
        if flow_id is None:
            return {"success": False, "message": "Missing required field: flow_id for causal flow"}, 400
        if flow_step is None:
            return {"success": False, "message": "Missing required field: flow_step for causal flow"}, 400
    elif flow_id is not None or flow_step is not None:
        return {"success": False, "message": "flow_id/flow_step are only allowed when source_page is causal_flow"}, 400

    client_ip = get_client_ip()
    start_time = time.perf_counter()
    request_id = log_prediction_attribute_request(
        context=context,
        target_prediction=target_prediction,
        target_token_id=target_token_id,
        model=model,
        source_page=source_page,
        flow_id=flow_id,
        flow_step=flow_step,
        client_ip=client_ip,
    )

    lock_acquired = inference_lock.acquire(timeout=LOCK_WAIT_TIMEOUT)
    if not lock_acquired:
        return {"success": False, "message": f"Queue wait exceeded {LOCK_WAIT_TIMEOUT} seconds; server is busy, please try again later."}, 503

    try:
        result = analyze_logit_lens(context, target_prediction, model=model, target_token_id=target_token_id)
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
        f"\t📤 API logit_lens response: req_id={request_id}, "
        f"target={result.get('target_token')!r}, n_layers={result.get('n_layers')}, "
        f"response_time={elapsed:.4f}s"
    )
    return {"success": True, **result}, 200
