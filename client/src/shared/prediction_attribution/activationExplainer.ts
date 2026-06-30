import type { TextAnalysisAPI } from '../api/GLTR_API';
import { tr } from '../lang/i18n-lite';

export interface ActivationExplainerDomIds {
    panelId: string;
    loadingId: string;
    resultId: string;
    errorId: string;
    explanationId: string;
    cosineId: string;
}

export async function runActivationExplain(
    api: TextAnalysisAPI,
    model: string,
    sourcePage: string,
    context: string,
    targetToken: string,
    domIds: ActivationExplainerDomIds
): Promise<void> {
    const panel = document.getElementById(domIds.panelId);
    const loading = document.getElementById(domIds.loadingId);
    const result = document.getElementById(domIds.resultId);
    const errEl = document.getElementById(domIds.errorId);
    if (!panel || !loading || !result || !errEl) return;

    panel.style.display = 'block';
    loading.style.display = '';
    result.style.display = 'none';
    errEl.style.display = 'none';

    try {
        const tok = await api.tokenize(context, model);
        const spans = tok?.spans ?? [];
        let tokenIndex = -1;
        for (let i = 0; i < spans.length; i++) {
            if (spans[i].raw === targetToken) { tokenIndex = i; break; }
        }
        if (tokenIndex < 0) {
            tokenIndex = spans.length - 1;
        }

        const ae = await api.explainActivation(model, sourcePage, context, tokenIndex);
        if (!ae.success) {
            errEl.textContent = ae.message || tr('Activation explain failed');
            errEl.style.display = '';
            return;
        }

        const cosine = ae.roundtrip_cosine ?? 0;
        let cls = 'cosine-low';
        if (cosine >= 0.70) cls = 'cosine-excellent';
        else if (cosine >= 0.60) cls = 'cosine-good';
        else if (cosine >= 0.50) cls = 'cosine-fair';
        const cosineHtml = `<span class="ae-cosine ${cls}">${(cosine * 100).toFixed(1)}%</span>`;

        const explanationEl = document.getElementById(domIds.explanationId);
        const cosineEl = document.getElementById(domIds.cosineId);
        if (explanationEl) explanationEl.textContent = ae.explanation ?? '';
        if (cosineEl) cosineEl.innerHTML = cosineHtml;
        result.style.display = '';
    } catch (err: unknown) {
        errEl.textContent = (err instanceof Error ? err.message : String(err));
        errEl.style.display = '';
    } finally {
        loading.style.display = 'none';
    }
}
