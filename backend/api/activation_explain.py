"""Activation Explainer API (Tiny-NLA)"""
import gc
import time

import torch

from backend.platform.oom import exit_if_oom
from backend.api.analyze import LOCK_WAIT_TIMEOUT
from backend.platform.access_log import get_client_ip, log_prediction_attribute_request
from backend.platform.source_page import ALLOWED_SOURCE_PAGES, normalize_source_page
from backend.core.tiny_nla import TinyNLAEngine, tiny_nla_lock, TINY_NLA_LOCK_TIMEOUT


def activation_explain(activation_explain_request):
    model = activation_explain_request.get("model")
    source_page = activation_explain_request.get("source_page")
    text = activation_explain_request.get("text")
    token_index = activation_explain_request.get("token_index")
    vector = activation_explain_request.get("vector")

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

    has_text = isinstance(text, str) and text.strip() != ""
    has_vector = isinstance(vector, list) and len(vector) > 0
    if not has_text and not has_vector:
        return {"success": False, "message": "Missing required field: text or vector must be provided"}, 400

    if token_index is not None and not isinstance(token_index, int):
        return {"success": False, "message": "token_index must be an integer"}, 400
    if token_index is not None and token_index < 0:
        return {"success": False, "message": "token_index must be >= 0"}, 400
    if has_text and token_index is None:
        return {"success": False, "message": "token_index is required when text is provided"}, 400

    client_ip = get_client_ip()
    start_time = time.perf_counter()
    request_id = log_prediction_attribute_request(
        context=text if has_text else str(vector)[:200],
        target_prediction=None,
        target_token_id=token_index,
        model=model,
        source_page=source_page,
        flow_id=None,
        flow_step=None,
        client_ip=client_ip,
    )

    lock_acquired = tiny_nla_lock.acquire(timeout=TINY_NLA_LOCK_TIMEOUT)
    if not lock_acquired:
        return {"success": False, "message": f"Tiny-NLA queue wait exceeded {TINY_NLA_LOCK_TIMEOUT} seconds; please try again later."}, 503

    try:
        engine = TinyNLAEngine()

        if has_vector:
            activation = torch.tensor(vector, dtype=torch.float32)
            if activation.shape[0] != 1024:
                return {"success": False, "message": f"vector dimension must be 1024, got {activation.shape[0]}"}, 400
        else:
            activation = engine.extract_activation(text, token_index)

        explanation = engine.explain(activation)
        roundtrip_cosine = engine.reconstruct_cosine(activation, explanation)

        result = {
            "concept": "",
            "explanation": explanation,
            "roundtrip_cosine": round(roundtrip_cosine, 4),
            "vector_dim": 1024,
            "note": "",
        }
    except ValueError as e:
        return {"success": False, "message": str(e)}, 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit_if_oom(e, defer_seconds=1)
        return {"success": False, "message": str(e)}, 500
    finally:
        tiny_nla_lock.release()
        gc.collect()

    elapsed = time.perf_counter() - start_time
    print(
        f"\t📤 API activation_explain response: req_id={request_id}, "
        f"concept={result.get('concept')!r}, roundtrip={result.get('roundtrip_cosine')}, "
        f"response_time={elapsed:.4f}s"
    )
    return {"success": True, **result}, 200
