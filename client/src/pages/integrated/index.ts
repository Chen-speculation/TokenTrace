import * as d3 from 'd3';
import '../../shared/core/d3-polyfill';
import '../../css/pages/integrated.scss';

import { initThemeManager } from '../../shared/ui/theme';
import { initLanguageManager } from '../../shared/ui/language';
import { initI18n, tr } from '../../shared/lang/i18n-lite';
import { AdminManager } from '../../shared/cross/adminManager';
import { SettingsMenuManager } from '../../shared/cross/settingsMenuManager';
import { initializeCommonApp } from '../../shared/bootstrap';
import { showAlertDialog } from '../../shared/ui/dialog';
import URLHandler from '../../shared/core/URLHandler';
import { createToast } from '../../shared/ui/toast';
import { createCompletionOptionsRow } from '../../shared/cross/completionOptionsRow';
import { DEFAULT_MAX_NEW_TOKENS, parseMaxNewTokens } from '../../shared/cross/maxNewTokensConfig';
import { initTypewriterPlaceholders } from '../../shared/ui/typewriterPlaceholder';

import {
    initGenAttributeDagView,
    DAG_COMPACTNESS_DEFAULT,
    type GenAttributeDagHandle,
} from '../../shared/prediction_attribution/causal_flow/genAttributeDagView';
import {
    startTokenGenAttribution,
    type TokenGenAttributionHandle,
    type TokenGenStep,
} from '../../shared/prediction_attribution/causal_flow/tokenGenAttributionRunner';
import { extractPromptTokenSpans, type PromptTokenSpan } from '../../shared/prediction_attribution/causal_flow/genAttributeDagPreprocess';
import { fetchTokenize } from '../../shared/prediction_attribution/core/predictionAttributeClient';
import { postCompletionsPrompt } from '../../shared/api/completionsClient';
import { completionFinishReasonLabel } from '../../shared/cross/generationEndReasonLabel';

import { createAttributionInspector, type AttributionInspectorApi } from '../../shared/prediction_attribution/inspector/attributionInspector';
import { runActivationExplain } from '../../shared/prediction_attribution/activationExplainer';

import type { LogitLensResult, BranchNextResult } from '../../shared/api/GLTR_API';
import type { AttributionDisplayOptions } from '../../shared/prediction_attribution/core/attributionDisplayModel';

// ===== Init =====
d3.selectAll('.loadersmall').style('display', 'none');
initI18n();
const showToast = createToast('#toast').show;

const apiPrefix = URLHandler.parameters['api'] || '';
const apiBaseForRequests = apiPrefix === '' ? '' : String(apiPrefix);
const bodyElement = d3.select('body').node() as Element;
const { eventHandler, api } = initializeCommonApp(apiPrefix, bodyElement);

const adminManager = AdminManager.getInstance();
api.setAdminToken(adminManager.isInAdminMode() ? adminManager.getAdminToken() : null);

const themeManager = initThemeManager({}, '#theme_dropdown');
const languageManager = initLanguageManager({}, '#language_dropdown');
void new SettingsMenuManager(
    '#settings_btn', '#settings_menu', '#admin_mode_btn',
    adminManager, api, undefined, undefined,
    themeManager, languageManager, 'common',
);

document.documentElement.lang = 'zh-CN';

// ===== Completion options (model select + max tokens) =====
const completionOptions = createCompletionOptionsRow({
    isSkipChatTemplate: () => true,
    metricModel: d3.select('#integrated_metric_model'),
    alertDialogTitle: tr('Integrated Lab'),
    onStateChange: () => syncSubmitButtonState(),
    adminMode: () => adminManager.isInAdminMode(),
    modelVariantStorageKey: 'integrated_model_variant',
    maxNewTokensStorageKey: 'integrated_max_new_tokens',
});

const {
    currentModelVariant,
    currentMaxTokens,
} = completionOptions;

// ===== State =====
type PanelType = 'highlight' | 'attribution' | 'logit' | 'branch' | 'activation';

let runnerHandle: TokenGenAttributionHandle | null = null;
let dagHandle: GenAttributeDagHandle;
let allSteps: TokenGenStep[] = [];
let inFlight = false;
let genAbort: AbortController | null = null;
let currentPanel: PanelType = 'highlight';
let lastSelectedId: string | null = null;
let selectedStep: TokenGenStep | null = null;
let attributionInspector: AttributionInspectorApi | null = null;

// ===== DOM refs =====
const promptTextarea = document.getElementById('integrated_prompt_text') as HTMLTextAreaElement;
const submitBtn = document.getElementById('integrated_submit_btn') as HTMLButtonElement;
const charCountEl = document.getElementById('integrated_text_count_value') as HTMLSpanElement;
const clearBtn = document.getElementById('integrated_clear_btn') as HTMLButtonElement;
const completeReasonEl = d3.select('#integrated_complete_reason');
const dagResultsEl = document.getElementById('integrated_dag_results')!;

// ===== Typewriter placeholders =====
initTypewriterPlaceholders({
    selector: '#integrated_prompt_text[data-typewriter]',
    placeholders: [
        'Type a prompt to explore...',
        'Try a question, a poem, or a code snippet...',
        'Watch the causal flow DAG unfold...',
        'Click any token to inspect with six lenses...',
    ],
});

// ===== DAG init =====
dagHandle = initGenAttributeDagView(d3.select('#integrated_dag_results'), {
    onDagRefresh: () => {
        checkSelectedNodeChanged();
    },
    dagCompactness: DAG_COMPACTNESS_DEFAULT,
    onFullscreenError: (msg) => showToast(msg, 'error'),
    getEffectiveExcludePromptPatternsText: () => '',
    getEffectiveExcludeGeneratedPatternsText: () => '',
    getEffectiveDeletePromptPatternsText: () => '',
});

// ===== Panel tabs =====
const tabButtons = document.querySelectorAll<HTMLElement>('.integrated-tab-btn');
const panels: Record<PanelType, HTMLElement> = {
    highlight: document.getElementById('integrated_highlight_panel')!,
    attribution: document.getElementById('integrated_attribution_panel')!,
    logit: document.getElementById('integrated_logit_panel')!,
    branch: document.getElementById('integrated_branch_panel')!,
    activation: document.getElementById('integrated_activation_panel')!,
};
const placeholderEl = document.getElementById('integrated_panel_placeholder')!;

tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        tabButtons.forEach(b => b.classList.remove('is-active'));
        btn.classList.add('is-active');
        const panel = btn.dataset.panel as PanelType;
        currentPanel = panel;
        Object.values(panels).forEach(p => p.hidden = true);
        panels[panel].hidden = false;
        if (selectedStep) {
            void updateCurrentPanel(selectedStep);
        }
    });
});

panels.attribution.hidden = true;
panels.logit.hidden = true;
panels.branch.hidden = true;
panels.activation.hidden = true;

// ===== Submit button =====
function syncSubmitButtonState(): void {
    const text = promptTextarea.value.trim();
    const ready = text.length > 0 && !inFlight;
    submitBtn.disabled = !ready;
    submitBtn.classList.toggle('inactive', !ready);
    submitBtn.textContent = inFlight ? tr('Stop') : tr('Start');
}

promptTextarea.addEventListener('input', () => {
    charCountEl.textContent = String(promptTextarea.value.length);
    syncSubmitButtonState();
});

clearBtn.addEventListener('click', () => {
    promptTextarea.value = '';
    charCountEl.textContent = '0';
    syncSubmitButtonState();
});

promptTextarea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        void runGeneration();
    }
});

syncSubmitButtonState();

// ===== Generation =====
function setGenLoading(loading: boolean): void {
    inFlight = loading;
    d3.selectAll('.loadersmall').style('display', loading ? 'inline-block' : 'none');
    syncSubmitButtonState();
}

async function runGeneration(): Promise<void> {
    if (inFlight) {
        genAbort?.abort();
        runnerHandle?.abort();
        return;
    }

    const prompt = promptTextarea.value.trim();
    if (!prompt) return;

    genAbort = new AbortController();
    const { signal } = genAbort;
    setGenLoading(true);
    completeReasonEl.text('');
    allSteps = [];
    runnerHandle = null;
    selectedStep = null;
    lastSelectedId = null;
    placeholderEl.style.display = '';

    dagHandle.reset();
    const emptyEl = document.getElementById('integrated_empty');
    if (emptyEl) emptyEl.style.display = '';

    try {
        const model = currentModelVariant();
        const maxTokens = currentMaxTokens() ?? DEFAULT_MAX_NEW_TOKENS;

        const messages = [{ role: 'user' as const, content: prompt }];
        const assembled = await postCompletionsPrompt(
            { model, messages },
            { signal },
        );
        const initialContext = assembled.prompt_used;

        runnerHandle = startTokenGenAttribution({
            initialContext,
            apiPrefix: apiBaseForRequests,
            model,
            maxTokens,
            flowId: `integrated_${Date.now()}`,
            onStep(step, stepIndex) {
                allSteps.push(step);
                if (stepIndex === 0) {
                    if (!dagHandle.hasPromptSpans()) {
                        dagHandle.setPromptTokenSpans(extractPromptTokenSpans(step), step.context);
                    }
                    dagHandle.fitViewportToContent();
                    const el = document.getElementById('integrated_empty');
                    if (el) el.style.display = 'none';
                }
                dagHandle.update(step);
            },
            onComplete(reason) {
                setGenLoading(false);
                completeReasonEl.text(completionFinishReasonLabel(reason));
            },
            onError(err) {
                showToast(err.message, 'error');
                setGenLoading(false);
            },
        });
    } catch (err: unknown) {
        if (err && typeof err === 'object' && 'name' in err && (err as { name: string }).name === 'AbortError') {
            setGenLoading(false);
            return;
        }
        const msg = err instanceof Error ? err.message : String(err);
        showAlertDialog(tr('Integrated Lab'), msg);
        setGenLoading(false);
    }
}

submitBtn.addEventListener('click', () => void runGeneration());

// ===== Node selection -> panel update =====
function checkSelectedNodeChanged(): void {
    const currentId = dagHandle.getSelectedNodeId();
    if (currentId === lastSelectedId) return;
    lastSelectedId = currentId;
    void onSelectedNodeChanged(currentId);
}

async function onSelectedNodeChanged(nodeId: string | null): Promise<void> {
    if (!nodeId) {
        placeholderEl.style.display = '';
        return;
    }

    const step = findStepByNodeId(nodeId);
    if (!step) return;

    selectedStep = step;
    placeholderEl.style.display = 'none';
    await updateCurrentPanel(step);
}

function findStepByNodeId(nodeId: string): TokenGenStep | null {
    for (const step of allSteps) {
        const tokenStart = step.context.length;
        const tokenEnd = step.context.length + step.token.length;
        if (nodeId === `${tokenStart}_${tokenEnd}`) {
            return step;
        }
    }
    return null;
}

async function updateCurrentPanel(step: TokenGenStep): Promise<void> {
    switch (currentPanel) {
        case 'highlight':
            await updateHighlightPanel(step);
            break;
        case 'attribution':
            updateAttributionPanel(step);
            break;
        case 'logit':
            await updateLogitPanel(step);
            break;
        case 'branch':
            await updateBranchPanel(step);
            break;
        case 'activation':
            await updateActivationPanel(step);
            break;
    }
}

// ===== Panel: Info Highlight =====
async function updateHighlightPanel(step: TokenGenStep): Promise<void> {
    const surface = document.getElementById('integrated_highlight_surface')!;
    renderHighlightFromAttribution(surface, step);
}

function renderHighlightFromAttribution(surface: HTMLElement, step: TokenGenStep): void {
    surface.innerHTML = '';
    const tokens = step.response.token_attribution ?? [];
    const maxScore = Math.max(...tokens.map(t => Math.abs(t.score)), 0.001);

    const container = document.createElement('div');
    container.style.cssText = 'font-size: 13px; line-height: 1.8; word-break: break-all; padding: 8px 0;';

    for (const tok of tokens) {
        const span = document.createElement('span');
        const intensity = Math.abs(tok.score) / maxScore;
        const alpha = 0.15 + intensity * 0.55;
        span.textContent = tok.raw;
        span.style.cssText = `background: rgba(255, 71, 64, ${alpha}); padding: 1px 2px; border-radius: 0;`;
        span.title = `${tok.raw}: ${tok.score.toFixed(4)}`;
        container.appendChild(span);
    }
    surface.appendChild(container);
}

// ===== Panel: Context Attribution =====
function updateAttributionPanel(step: TokenGenStep): void {
    if (!attributionInspector) {
        attributionInspector = createAttributionInspector({
            resultsRoot: d3.select('#integrated_attribution_surface'),
            eventHandler,
            tooltipHideRoot: d3.select('#integrated_attribution_panel'),
        });
    }
    const displayOptions: AttributionDisplayOptions = {
        colorRangeMax: null,
        excludePromptPatternsText: '',
    };
    attributionInspector.apply(step.context, step.response, displayOptions, false);
}

// ===== Panel: Logit Lens (simplified) =====
async function updateLogitPanel(step: TokenGenStep): Promise<void> {
    const heatmapEl = document.getElementById('integrated_logit_heatmap')!;
    const trajEl = document.getElementById('integrated_logit_trajectory')!;
    heatmapEl.innerHTML = '';
    trajEl.innerHTML = '';

    const loading = document.createElement('div');
    loading.className = 'integrated-panel-loading';
    loading.textContent = 'Loading...';
    heatmapEl.appendChild(loading);

    try {
        const ll: LogitLensResult = await api.logitLens(
            step.context, step.token, currentModelVariant(), 'integrated',
        );
        if (!ll.success || !ll.layers) {
            heatmapEl.innerHTML = `<div class="ae-error">${ll.message || 'Failed'}</div>`;
            return;
        }
        renderLogitLensMini(ll, heatmapEl, trajEl);
    } catch (err) {
        heatmapEl.innerHTML = `<div class="ae-error">${err instanceof Error ? err.message : String(err)}</div>`;
    }
}

function renderLogitLensMini(ll: LogitLensResult, heatmapEl: HTMLElement, trajEl: HTMLElement): void {
    const layers = ll.layers ?? [];
    const svgNs = 'http://www.w3.org/2000/svg';
    const w = heatmapEl.clientWidth || 360;
    const margin = { top: 10, right: 10, bottom: 20, left: 30 };
    const cw = w - margin.left - margin.right;
    const ch = 120;
    const svg = document.createElementNS(svgNs, 'svg');
    svg.setAttribute('width', String(w));
    svg.setAttribute('height', String(ch + margin.top + margin.bottom));

    const g = document.createElementNS(svgNs, 'g');
    g.setAttribute('transform', `translate(${margin.left},${margin.top})`);

    const probs = layers.map(l => l.target_prob ?? 0);
    const maxY = Math.max(...probs, 0.1);
    const xScale = (i: number) => (layers.length > 1 ? (i / (layers.length - 1)) * cw : 0);
    const yScale = (v: number) => ch - (v / maxY) * ch;

    let pathD = '';
    layers.forEach((l, i) => {
        const x = xScale(i);
        const y = yScale(l.target_prob ?? 0);
        pathD += (i === 0 ? 'M' : 'L') + ` ${x} ${y} `;
    });

    const path = document.createElementNS(svgNs, 'path');
    path.setAttribute('d', pathD);
    path.setAttribute('stroke', 'var(--accent-color)');
    path.setAttribute('stroke-width', '1.5');
    path.setAttribute('fill', 'none');
    g.appendChild(path);

    layers.forEach((l, i) => {
        const x = xScale(i);
        const y = yScale(l.target_prob ?? 0);
        const circle = document.createElementNS(svgNs, 'circle');
        circle.setAttribute('cx', String(x));
        circle.setAttribute('cy', String(y));
        circle.setAttribute('r', '3');
        circle.setAttribute('fill', 'var(--accent-color)');
        g.appendChild(circle);
    });

    svg.appendChild(g);
    heatmapEl.innerHTML = '';
    heatmapEl.appendChild(svg);

    // Top-k table for last few layers
    const table = document.createElement('div');
    table.style.cssText = 'font-size: 11px; margin-top: 8px;';
    const lastLayer = layers[layers.length - 1];
    if (lastLayer?.topk_tokens) {
        table.innerHTML = `<div style="font-family:var(--font-icon);font-size:0.6875rem;text-transform:uppercase;color:var(--text-muted);margin-bottom:4px">Layer ${lastLayer.layer} top-5</div>`;
        const tokList = document.createElement('div');
        tokList.style.cssText = 'display:flex;flex-wrap:wrap;gap:4px;';
        lastLayer.topk_tokens.slice(0, 5).forEach((tok, i) => {
            const prob = lastLayer.topk_probs?.[i] ?? 0;
            const span = document.createElement('span');
            span.textContent = `${tok} ${(prob * 100).toFixed(1)}%`;
            span.style.cssText = `padding:2px 6px;background:var(--bg-hover-light);font-family:var(--font-icon);font-size:0.6875rem;`;
            tokList.appendChild(span);
        });
        table.appendChild(tokList);
    }
    trajEl.innerHTML = '';
    trajEl.appendChild(table);
}

// ===== Panel: Branch Tree (simplified) =====
async function updateBranchPanel(step: TokenGenStep): Promise<void> {
    const surface = document.getElementById('integrated_branch_surface')!;
    surface.innerHTML = '';

    const loading = document.createElement('div');
    loading.className = 'integrated-panel-loading';
    loading.textContent = 'Loading...';
    surface.appendChild(loading);

    try {
        const prefix = step.context + step.token;
        const res: BranchNextResult = await api.branchNext(
            prefix, currentModelVariant(), 'integrated', 5,
        );
        if (!res.success || !res.candidates) {
            surface.innerHTML = `<div class="ae-error">${res.message || 'Failed'}</div>`;
            return;
        }
        renderBranchTreeMini(res.candidates, surface);
    } catch (err) {
        surface.innerHTML = `<div class="ae-error">${err instanceof Error ? err.message : String(err)}</div>`;
    }
}

function renderBranchTreeMini(candidates: { token: string; prob: number }[], surface: HTMLElement): void {
    surface.innerHTML = '';
    const list = document.createElement('div');
    list.style.cssText = 'display:flex;flex-direction:column;gap:6px;';

    const maxProb = Math.max(...candidates.map(c => c.prob), 0.001);
    candidates.forEach((c, i) => {
        const row = document.createElement('div');
        row.style.cssText = 'display:flex;align-items:center;gap:8px;padding:4px 0;';

        const num = document.createElement('span');
        num.textContent = String(i + 1).padStart(2, '0');
        num.style.cssText = 'font-family:var(--font-icon);font-size:0.6875rem;color:var(--text-muted);min-width:20px;';

        const bar = document.createElement('div');
        bar.style.cssText = 'flex:1;height:20px;position:relative;border:1px solid var(--border-color);';
        const fill = document.createElement('div');
        fill.style.cssText = `position:absolute;top:0;left:0;bottom:0;width:${(c.prob / maxProb * 100).toFixed(1)}%;background:color-mix(in srgb, var(--accent-color) 30%, transparent);transition:width 0.3s ease;`;
        bar.appendChild(fill);

        const tokText = document.createElement('span');
        tokText.textContent = c.token;
        tokText.style.cssText = 'font-size:13px;padding:0 8px;position:relative;z-index:1;';

        const probText = document.createElement('span');
        probText.textContent = `${(c.prob * 100).toFixed(1)}%`;
        probText.style.cssText = 'font-family:var(--font-icon);font-size:0.6875rem;color:var(--text-muted);min-width:50px;text-align:right;';

        row.append(num, bar, tokText, probText);
        list.appendChild(row);
    });

    surface.appendChild(list);
}

// ===== Panel: Activation Explainer =====
async function updateActivationPanel(step: TokenGenStep): Promise<void> {
    await runActivationExplain(
        api,
        currentModelVariant(),
        'integrated',
        step.context,
        step.token,
        {
            panelId: 'integrated_activation_panel',
            loadingId: 'integrated_ae_loading',
            resultId: 'integrated_ae_result',
            errorId: 'integrated_ae_error',
            explanationId: 'integrated_ae_explanation',
            cosineId: 'integrated_ae_cosine',
        },
    );
}

// ===== Resizer (left) =====
const resizerLeft = document.getElementById('resizer_left')!;
const resizerRight = document.getElementById('resizer_right')!;
const frame = document.querySelector('.integrated-frame') as HTMLElement;

function initResizer(resizer: HTMLElement, isLeft: boolean): void {
    let dragging = false;
    resizer.addEventListener('mousedown', (e) => {
        e.preventDefault();
        dragging = true;
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
    });
    document.addEventListener('mousemove', (e) => {
        if (!dragging || !frame) return;
        const cols = getComputedStyle(frame).gridTemplateColumns.split(' ');
        if (isLeft) {
            const newWidth = Math.max(200, Math.min(500, e.clientX));
            cols[0] = `${newWidth}px`;
        } else {
            const rect = frame.getBoundingClientRect();
            const newWidth = Math.max(250, Math.min(600, rect.right - e.clientX));
            cols[4] = `${newWidth}px`;
        }
        frame.style.gridTemplateColumns = cols.join(' ');
    });
    document.addEventListener('mouseup', () => {
        if (dragging) {
            dragging = false;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }
    });
}

initResizer(resizerLeft, true);
initResizer(resizerRight, false);
