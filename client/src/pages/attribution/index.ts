import * as d3 from 'd3';
import '../../shared/core/d3-polyfill';
import '../../css/pages/attribution.scss';

import { initThemeManager } from '../../shared/ui/theme';
import { initLanguageManager } from '../../shared/ui/language';
import { initI18n, tr, trf } from '../../shared/lang/i18n-lite';
import { AdminManager } from '../../shared/cross/adminManager';
import { SettingsMenuManager } from '../../shared/cross/settingsMenuManager';
import { initChatPanelLayout } from '../../shared/ui/chat_panel_layout';
import { PANEL_SPLIT_STORAGE_KEY_ATTRIBUTION } from '../../shared/cross/panelSplitStorage';
import { TextInputController } from '../../shared/controllers/textInputController';
import { initializeCommonApp } from '../../shared/bootstrap';
import { registerPageBusy } from '../../shared/core/activitySession';
import { showAlertDialog } from '../../shared/ui/dialog';
import URLHandler from '../../shared/core/URLHandler';
import { initCachedHistoryQueryDropdown, type CachedHistorySelectContext } from '../../shared/cross/cachedHistoryUi';
import {
    DEFAULT_CONTENT_URL_PARAM,
    readContentUrlParam,
    replaceContentUrlParam,
    runContentUrlHydrate,
} from '../../shared/cross/contentUrl';
import { initQueryHistoryDropdown, saveHistory } from '../../shared/cross/queryHistory';
import { createToast } from '../../shared/ui/toast';
import { translateApiErrorMessage } from '../../shared/core/errorUtils';
import { createAttributionInspector } from '../../shared/prediction_attribution/inspector/attributionInspector';
import type { AttributionDisplayOptions } from '../../shared/prediction_attribution/core/attributionDisplayModel';
import {
    getCachedEntryByContentKey,
    listCachedHistoryRows,
    removeCachedEntryByContentKey,
    takeSuccessfulAttributionFromCache,
    touchCachedEntryByContentKey,
    type AttributionApiResponse,
    type AttributionCachedEntry,
    type PredictionAttributeModelVariant,
} from '../../shared/prediction_attribution/core/attributionResultCache';
import { loadPredictionAttributeWithCache, fetchAblationAttribute } from '../../shared/prediction_attribution/core/predictionAttributeClient';
import { readStoredEffectiveExcludePromptPatternsText } from '../../shared/prediction_attribution/core/attributionExcludePromptPatternsStorage';
import { entryKey } from '../../shared/prediction_attribution/core/attributionResultCache';
import { bindExcludePromptPatternsUi } from '../../shared/prediction_attribution/core/excludePromptPatternsUi';
import { syncDraftCommittedButtonPair } from '../../shared/cross/syncDraftCommittedButtonPair';
import { lsReadEnum, lsWriteString } from '../../shared/storage/localStorageHelpers';

d3.selectAll('.loadersmall').style('display', 'none');

initI18n();

const showToast = createToast('#toast').show;

const CONTEXT_HISTORY_KEY = 'info_radar_attribution_context_history';
const TARGET_HISTORY_KEY = 'info_radar_attribution_target_history';
const ATTRIBUTION_MODEL_VARIANT_STORAGE_KEY = 'info_radar_attribution_model_variant';
const ATTRIBUTION_METHOD_STORAGE_KEY = 'info_radar_attribution_method';

export type AttributionMethod = 'gradient' | 'ablation' | 'both';

function readStoredAttributionPageModelVariant(): PredictionAttributeModelVariant {
    return lsReadEnum(ATTRIBUTION_MODEL_VARIANT_STORAGE_KEY, ['base', 'instruct'] as const, 'instruct');
}

function readStoredAttributionMethod(): AttributionMethod {
    return lsReadEnum(ATTRIBUTION_METHOD_STORAGE_KEY, ['gradient', 'ablation', 'both'] as const, 'gradient');
}

const apiPrefix = URLHandler.parameters['api'] || '';
const bodyElement = d3.select('body').node() as Element;
const { eventHandler, totalSurprisalFormat, api } = initializeCommonApp(apiPrefix, bodyElement);
/** 与 {@link TextAnalysisAPI} 一致：`?api=` 非空时用其作为基址，否则 `''`，URL 为 `/api/...`（相对当前站点根路径） */
const apiBaseForRequests = apiPrefix === '' ? '' : String(apiPrefix);

const adminManager = AdminManager.getInstance();
api.setAdminToken(adminManager.isInAdminMode() ? adminManager.getAdminToken() : null);

// --- DOM 引用 ---
const contextField = d3.select('#context_text');
const contextCountValue = d3.select('#context_count_value');
const clearContextBtn = d3.select('#clear_context_btn');
const pasteContextBtn = d3.select('#paste_context_btn');
const contextHistoryBtn = document.getElementById('context_history_btn');

const targetField = d3.select('#target_text');
const targetCountValue = d3.select('#target_count_value');
const clearTargetBtn = d3.select('#clear_target_btn');
const pasteTargetBtn = d3.select('#paste_target_btn');
const targetHistoryBtn = document.getElementById('target_history_btn');

const analyzeBtn = d3.select('#analyze_btn');
const modelVariantSelect = document.getElementById('attribution_model_variant') as HTMLSelectElement | null;
const methodSelect = document.getElementById('attribution_method') as HTMLSelectElement | null;
const forceRetryBtn = d3.select('#force_retry_btn');
const loaderSmall = d3.select('.loadersmall');
const resultInfoEl = d3.select('#attribution_result_info');
const useMappingCheckbox = document.getElementById('attribution_use_mapping') as HTMLInputElement | null;
const maxScoreRange = document.getElementById('attribution_max_score_range') as HTMLInputElement | null;
const maxScoreValueEl = document.getElementById('attribution_max_score_value');
if (modelVariantSelect) {
    modelVariantSelect.value = readStoredAttributionPageModelVariant();
}
if (methodSelect) {
    methodSelect.value = readStoredAttributionMethod();
}

function currentAttributionModelVariant(): PredictionAttributeModelVariant {
    const v = modelVariantSelect?.value;
    return v === 'base' || v === 'instruct' ? v : 'instruct';
}

function currentAttributionMethod(): AttributionMethod {
    const v = methodSelect?.value;
    return v === 'gradient' || v === 'ablation' || v === 'both' ? v : 'gradient';
}

modelVariantSelect?.addEventListener('change', () => {
    lsWriteString(ATTRIBUTION_MODEL_VARIANT_STORAGE_KEY, currentAttributionModelVariant());
});

methodSelect?.addEventListener('change', () => {
    lsWriteString(ATTRIBUTION_METHOD_STORAGE_KEY, currentAttributionMethod());
    syncMethodDisplay();
});

// --- TextInputController ---
new TextInputController({
    textField: contextField,
    textCountValue: contextCountValue,
    clearBtn: clearContextBtn,
    submitBtn: analyzeBtn,
    saveBtn: d3.select(null),
    pasteBtn: pasteContextBtn,
    totalSurprisalFormat,
    showAlertDialog,
});

new TextInputController({
    textField: targetField,
    textCountValue: targetCountValue,
    clearBtn: clearTargetBtn,
    submitBtn: analyzeBtn,
    saveBtn: d3.select(null),
    pasteBtn: pasteTargetBtn,
    totalSurprisalFormat,
    showAlertDialog,
});

const gradientResultsRoot = d3.select('#results');
const ablationResultsRoot = d3.select('#ablation_results');

const gradientInspector = createAttributionInspector({
    resultsRoot: gradientResultsRoot,
    eventHandler,
    debugParentId: 'attribution_debug_container',
});

const ablationInspector = createAttributionInspector({
    resultsRoot: ablationResultsRoot,
    eventHandler,
    debugParentId: 'ablation_debug_container',
});

function syncMethodDisplay(): void {
    const method = currentAttributionMethod();
    const gradientPanel = document.querySelector('.attribution-gradient-panel') as HTMLElement | null;
    const ablationPanel = document.querySelector('.attribution-ablation-panel') as HTMLElement | null;
    const bothLegend = document.getElementById('attribution_both_legend');
    if (gradientPanel) gradientPanel.style.display = method === 'ablation' ? 'none' : 'block';
    if (ablationPanel) ablationPanel.style.display = method === 'gradient' ? 'none' : 'block';
    if (bothLegend) bothLegend.style.display = method === 'both' ? 'flex' : 'none';
}

function readAttributionDisplayOptions(): AttributionDisplayOptions {
    return {
        colorRangeMax: readAttributionColorRangeMax(),
        excludePromptPatternsText: readStoredEffectiveExcludePromptPatternsText(),
    };
}

// --- 分析按钮状态管理（草稿 vs 已提交：右侧展示对应 lastCommittedInputs）---
let analyzeInFlight = false;
/** 当前右侧已展示的归因所对应的输入；null 表示尚未成功应用过任何结果 */
let lastCommittedInputs: { context: string; target: string } | null = null;

function syncAnalyzeButtonState(): void {
    const context = (contextField.node() as HTMLTextAreaElement | null)?.value ?? '';
    const target = (targetField.node() as HTMLTextAreaElement | null)?.value ?? '';
    const idleInputsReady = context.length > 0 && target.length > 0;
    const hasUncommittedDraft =
        lastCommittedInputs === null ||
        context !== lastCommittedInputs.context ||
        target !== lastCommittedInputs.target;
    syncDraftCommittedButtonPair({
        primaryBtn: analyzeBtn,
        forceRetryBtn,
        inFlight: analyzeInFlight,
        primaryInFlightMode: 'freeze',
        primaryIdleLabel: tr('Analyze attribution'),
        idleInputsReady,
        hasUncommittedDraft,
    });
}

function setAnalyzeLoading(loading: boolean): void {
    analyzeInFlight = loading;
    loaderSmall.style('display', loading ? null : 'none');
    syncAnalyzeButtonState();
}

registerPageBusy(() => analyzeInFlight);

// input 事件同步按钮状态
[contextField, targetField].forEach((field) => {
    (field.node() as HTMLTextAreaElement | null)?.addEventListener('input', syncAnalyzeButtonState);
});
syncAnalyzeButtonState();

function syncMaxScoreRangeUiEnabled(): void {
    const on = !!useMappingCheckbox?.checked;
    if (maxScoreRange) maxScoreRange.disabled = !on;
}

function updateMaxScoreValueLabel(): void {
    if (!maxScoreRange || !maxScoreValueEl) return;
    const v = Number(maxScoreRange.value);
    maxScoreValueEl.textContent = Number.isFinite(v) ? v.toFixed(2) : '—';
}

syncMaxScoreRangeUiEnabled();
updateMaxScoreValueLabel();

useMappingCheckbox?.addEventListener('change', () => {
    syncMaxScoreRangeUiEnabled();
    reapplyAttributionColorsIfPossible();
});

maxScoreRange?.addEventListener('input', () => {
    updateMaxScoreValueLabel();
    reapplyAttributionColorsIfPossible();
});

/** 勾选「使用映射」且 x∈(0,1]：将已归一化到 [0,1] 的分数中，[0,x] 线性映射到 [0,1] 用于染色，>x 视为 1。未勾选则不设置 colorScores。x=1 时与未勾选等价（恒等染色）。 */
function readAttributionColorRangeMax(): number | null {
    if (!useMappingCheckbox?.checked) return null;
    if (!maxScoreRange) return null;
    const n = Number(maxScoreRange.value);
    if (!Number.isFinite(n) || n <= 0 || n > 1) return null;
    return n;
}

function renderGradientResult(context: string, response: AttributionApiResponse): void {
    gradientInspector.apply(context, response, readAttributionDisplayOptions(), false);
}

function renderAblationResult(context: string, response: AttributionApiResponse): void {
    ablationInspector.apply(context, response, readAttributionDisplayOptions(), true);
}

function renderAttributionResult(context: string, gradientResponse: AttributionApiResponse, ablationResponse?: AttributionApiResponse): void {
    const method = currentAttributionMethod();
    if (method === 'gradient' || method === 'both') {
        renderGradientResult(context, gradientResponse);
    }
    if (method === 'ablation' || method === 'both') {
        renderAblationResult(context, ablationResponse ?? gradientResponse);
    }
    updateResultInfo(gradientResponse, ablationResponse);

    // Subword notice: warn if target_token differs from what user typed
    const userTarget = (targetField.node() as HTMLTextAreaElement | null)?.value?.trim() ?? '';
    const trackedToken = gradientResponse.target_token ?? '';
    let notice = document.getElementById('attribution_subword_notice');
    if (!notice) {
        notice = document.createElement('div');
        notice.id = 'attribution_subword_notice';
        notice.style.cssText = 'font-size:9pt;color:#B45309;margin:4px 0;padding:4px 8px;background:rgba(254,243,199,0.5);border-radius:4px;border:1px solid rgba(245,158,11,0.3)';
        const container = document.querySelector('.attribution-method-panels');
        container?.parentElement?.insertBefore(notice, container);
    }
    if (trackedToken && userTarget && trackedToken.trim() !== userTarget.trim()) {
        notice.style.display = '';
        notice.textContent = `实际归因 token：「${trackedToken}」（"${userTarget}" 被 tokenizer 切成子词，归因针对第一个子词）`;
    } else {
        notice.style.display = 'none';
    }
    lastCommittedInputs = {
        context: (contextField.node() as HTMLTextAreaElement | null)?.value ?? '',
        target: (targetField.node() as HTMLTextAreaElement | null)?.value ?? '',
    };
    syncAnalyzeButtonState();
    if (method === 'both' && ablationResponse) {
        updateConsistencyReadout(gradientResponse, ablationResponse);
    } else {
        clearConsistencyReadout();
    }
}

/** Analyze 成功或 Cached history 恢复：contentUrlKey 须来自 save / MRU / `?content=` hydrate */
function applyAttributionResponse(
    context: string,
    gradientResponse: AttributionApiResponse,
    contentUrlKey: string,
    ablationResponse?: AttributionApiResponse
): void {
    renderAttributionResult(context, gradientResponse, ablationResponse);
    replaceContentUrlParam(contentUrlKey, DEFAULT_CONTENT_URL_PARAM, 'attribution');
}

function reapplyAttributionColorsIfPossible(): void {
    gradientInspector.reapply(readAttributionDisplayOptions(), false);
    ablationInspector.reapply(readAttributionDisplayOptions(), true);
}

bindExcludePromptPatternsUi({
    textInput: document.getElementById('attribution_exclude_prompt_patterns') as HTMLTextAreaElement | null,
    enableCheckbox: document.getElementById('attribution_exclude_prompt_patterns_enable') as HTMLInputElement | null,
    onEffectiveChange: reapplyAttributionColorsIfPossible,
});

function updateResultInfo(gradientResponse: AttributionApiResponse, ablationResponse?: AttributionApiResponse): void {
    const method = currentAttributionMethod();
    const gn = gradientResponse.token_attribution?.length ?? 0;
    const gmodel = gradientResponse.model ?? '–';
    let text = `${trf('{count} tokens', { count: gn })}\n${tr('model')}: ${gmodel}`;
    if (method === 'both' && ablationResponse) {
        const an = ablationResponse.token_attribution?.length ?? 0;
        const amodel = ablationResponse.model ?? '–';
        text += `\n${tr('Ablation')}: ${trf('{count} tokens', { count: an })} / ${tr('model')}: ${amodel}`;
    }
    resultInfoEl.classed('is-hidden', false).text(text);
}

function updateConsistencyReadout(gradientResponse: AttributionApiResponse, ablationResponse: AttributionApiResponse): void {
    const el = document.getElementById('attribution_consistency_readout');
    if (!el) return;
    const gTokens = gradientResponse.token_attribution ?? [];
    const aTokens = ablationResponse.token_attribution ?? [];
    if (gTokens.length === 0 || aTokens.length === 0 || gTokens.length !== aTokens.length) {
        el.textContent = '';
        return;
    }
    const spearman = computeSpearmanCorrelation(
        gTokens.map((t) => t.score),
        aTokens.map((t) => t.score)
    );
    el.textContent = `${tr('Spearman')} ρ ≈ ${spearman.toFixed(3)}`;
}

function clearConsistencyReadout(): void {
    const el = document.getElementById('attribution_consistency_readout');
    if (el) el.textContent = '';
}

function computeSpearmanCorrelation(x: number[], y: number[]): number {
    if (x.length !== y.length || x.length === 0) return NaN;
    const n = x.length;
    const rank = (arr: number[]): number[] => {
        const sorted = arr.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
        const ranks = new Array(n);
        for (let i = 0; i < n; ) {
            let j = i;
            while (j < n && sorted[j].v === sorted[i].v) j++;
            const avgRank = (i + 1 + j) / 2;
            for (let k = i; k < j; k++) ranks[sorted[k].i] = avgRank;
            i = j;
        }
        return ranks;
    };
    const rx = rank(x);
    const ry = rank(y);
    const mx = rx.reduce((a, b) => a + b, 0) / n;
    const my = ry.reduce((a, b) => a + b, 0) / n;
    let num = 0, denX = 0, denY = 0;
    for (let i = 0; i < n; i++) {
        const dx = rx[i] - mx;
        const dy = ry[i] - my;
        num += dx * dy;
        denX += dx * dx;
        denY += dy * dy;
    }
    if (denX === 0 || denY === 0) return 0;
    return num / Math.sqrt(denX * denY);
}

// --- 主分析逻辑 ---
async function runAnalyze(options?: { forceRefresh?: boolean }): Promise<void> {
    const context = (contextField.node() as HTMLTextAreaElement | null)?.value ?? '';
    const target = (targetField.node() as HTMLTextAreaElement | null)?.value ?? '';
    if (analyzeInFlight || !context || !target) return;

    const forceRefresh = options?.forceRefresh === true;
    const method = currentAttributionMethod();

    if (!forceRefresh) {
        const hit = await takeSuccessfulAttributionFromCache(context, target);
        if (hit) {
            if (method === 'ablation') {
                // ablation 模式没有独立缓存，回退到请求
            } else {
                applyAttributionResponse(context, hit.response, hit.contentKey);
                saveHistory(context, CONTEXT_HISTORY_KEY);
                saveHistory(target, TARGET_HISTORY_KEY);
                return;
            }
        }
    }

    setAnalyzeLoading(true);
    let abortController: AbortController | null = null;
    try {
        if (method === 'gradient') {
            const { response: json, contentKey } = await loadPredictionAttributeWithCache({
                apiBaseForRequests,
                context,
                targetPrediction: target,
                model: currentAttributionModelVariant(),
                sourcePage: 'attribution',
                forceRefresh,
            });
            applyAttributionResponse(context, json, contentKey);
        } else if (method === 'ablation') {
            abortController = new AbortController();
            const json = await fetchAblationAttribute(
                apiBaseForRequests,
                context,
                target,
                currentAttributionModelVariant(),
                'attribution',
            );
            applyAttributionResponse(context, json, entryKey(context, target));
        } else {
            // Both 模式：并行请求
            abortController = new AbortController();
            const [gradientRes, ablationRes] = await Promise.all([
                loadPredictionAttributeWithCache({
                    apiBaseForRequests,
                    context,
                    targetPrediction: target,
                    model: currentAttributionModelVariant(),
                    sourcePage: 'attribution',
                    forceRefresh,
                }),
                fetchAblationAttribute(
                    apiBaseForRequests,
                    context,
                    target,
                    currentAttributionModelVariant(),
                    'attribution',
                ),
            ]);
            applyAttributionResponse(context, gradientRes.response, gradientRes.contentKey, ablationRes);
        }
        saveHistory(context, CONTEXT_HISTORY_KEY);
        saveHistory(target, TARGET_HISTORY_KEY);
    } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        showAlertDialog(tr('Context Attribution'), translateApiErrorMessage(msg));
    } finally {
        setAnalyzeLoading(false);
        abortController = null;
    }
}

analyzeBtn.on('click', () => void runAnalyze());
forceRetryBtn.on('click', () => void runAnalyze({ forceRefresh: true }));

// Enter 键（Ctrl/Cmd + Enter）提交
(contextField.node() as HTMLTextAreaElement | null)?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) void runAnalyze();
});

// --- 历史下拉 ---
const contextTextarea = contextField.node() as HTMLTextAreaElement | null;
const targetTextarea = targetField.node() as HTMLTextAreaElement | null;

initQueryHistoryDropdown({
    input: contextTextarea,
    dropdownId: 'context_history_dropdown',
    storageKey: CONTEXT_HISTORY_KEY,
    openDropdownOnFocusInput: false,
    filterHistoryByInput: false,
    onSelect: syncAnalyzeButtonState,
    historyButton: contextHistoryBtn,
    applyHistoryOnHover: true,
});

initQueryHistoryDropdown({
    input: targetTextarea,
    dropdownId: 'target_history_dropdown',
    storageKey: TARGET_HISTORY_KEY,
    openDropdownOnFocusInput: false,
    filterHistoryByInput: false,
    onSelect: syncAnalyzeButtonState,
    historyButton: targetHistoryBtn,
    applyHistoryOnHover: true,
});

async function restoreAttributionFromCachedEntry(
    entry: AttributionCachedEntry,
    options: { shouldTouch: boolean; ctx?: CachedHistorySelectContext; contentKey: string }
): Promise<void> {
    try {
        contextField.property('value', entry.context);
        targetField.property('value', entry.targetPrediction);
        contextTextarea?.dispatchEvent(new Event('input', { bubbles: true }));
        targetTextarea?.dispatchEvent(new Event('input', { bubbles: true }));
        syncAnalyzeButtonState();
        applyAttributionResponse(entry.context, entry.response, options.contentKey);
        if (options.shouldTouch && options.ctx) {
            await touchCachedEntryByContentKey(options.contentKey);
            await options.ctx.refreshList();
        }
    } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        showToast(translateApiErrorMessage(msg), 'error');
    }
}

// --- Cached history ---
const cachedHistoryBtn = document.getElementById('attribution_cached_history_btn');
void initCachedHistoryQueryDropdown({
    dropdownId: 'attribution_cached_history_dropdown',
    historyButton: cachedHistoryBtn,
    clickOutsideRoot: document.getElementById('attribution_cached_history_dropdown'),
    listMru: listCachedHistoryRows,
    onSelectEntry: async (contentKey, shouldTouch, ctx) => {
        const entry = await getCachedEntryByContentKey(contentKey);
        if (!entry) {
            showToast(tr('Cached result not found'), 'error');
            return;
        }
        await restoreAttributionFromCachedEntry(entry, {
            shouldTouch: Boolean(shouldTouch),
            ctx,
            contentKey,
        });
    },
    onRemove: removeCachedEntryByContentKey,
    onPromote: touchCachedEntryByContentKey,
});

void runContentUrlHydrate({
    readRaw: readContentUrlParam,
    fetchEntry: getCachedEntryByContentKey,
    apply: async (entry, rawContentKey) => {
        await restoreAttributionFromCachedEntry(entry, { shouldTouch: false, contentKey: rawContentKey });
    },
    onMissing: async () => {
        showToast(tr('Cached result not found (link may be expired)'), 'error');
        replaceContentUrlParam(null, DEFAULT_CONTENT_URL_PARAM, 'attribution');
    },
    onApplyError: (e: unknown) => {
        const msg = e instanceof Error ? e.message : String(e);
        showToast(translateApiErrorMessage(msg), 'error');
        replaceContentUrlParam(null, DEFAULT_CONTENT_URL_PARAM, 'attribution');
    },
});

initChatPanelLayout({ storageKey: PANEL_SPLIT_STORAGE_KEY_ATTRIBUTION });

const themeManager = initThemeManager(
    {
        onThemeChange: () => {
            reapplyAttributionColorsIfPossible();
        },
    },
    '#theme_dropdown'
);

const languageManager = initLanguageManager({}, '#language_dropdown');

void new SettingsMenuManager(
    '#settings_btn',
    '#settings_menu',
    '#admin_mode_btn',
    adminManager,
    api,
    undefined,
    undefined,
    themeManager,
    languageManager,
    'common'
);

syncMethodDisplay();

// ---- Examples ----
type AttributionExample = { label: string; context: string; target: string };

const ATTRIBUTION_EXAMPLES: AttributionExample[] = [
    {
        label: tr('Factual recall'),
        context: 'The capital of China is',
        target: 'Beijing',
    },
    {
        label: tr('Sentiment'),
        context: 'The movie was absolutely terrible and I hated every minute of it. My overall rating is',
        target: ' negative',
    },
    {
        label: tr('Coreference'),
        context: 'The nurse said that she would come back later. The pronoun "she" refers to the',
        target: ' nurse',
    },
];

function initAttributionExamples(): void {
    const section = document.getElementById('attribution_examples');
    if (!section) return;

    const desc = document.createElement('div');
    desc.className = 'page-examples-desc';
    desc.innerHTML = `<strong>${tr('Attribution')}</strong>: ${tr('which input tokens drove this prediction? Gradient scores each context token by how much changing it would shift the output probability.')}`;
    section.appendChild(desc);

    const label = document.createElement('div');
    label.className = 'page-examples-label';
    label.textContent = tr('Try an example:');
    section.appendChild(label);

    const btns = document.createElement('div');
    btns.className = 'page-examples-buttons';

    for (const ex of ATTRIBUTION_EXAMPLES) {
        const btn = document.createElement('button');
        btn.className = 'page-example-btn';
        btn.type = 'button';
        btn.title = `${tr('Context')}: ${ex.context}\n${tr('Target prediction')}: ${ex.target}`;
        btn.textContent = ex.label;
        btn.addEventListener('click', () => {
            const ctx = contextField.node() as HTMLTextAreaElement | null;
            const tgt = targetField.node() as HTMLTextAreaElement | null;
            if (ctx) { ctx.value = ex.context; ctx.dispatchEvent(new Event('input', { bubbles: true })); }
            if (tgt) { tgt.value = ex.target; tgt.dispatchEvent(new Event('input', { bubbles: true })); }
        });
        btns.appendChild(btn);
    }
    section.appendChild(btns);
}

initAttributionExamples();
