import * as d3 from 'd3';
import '../../shared/core/d3-polyfill';
import '../../css/pages/logit_lens.scss';

import { initThemeManager } from '../../shared/ui/theme';
import { initLanguageManager } from '../../shared/ui/language';
import { initI18n, tr } from '../../shared/lang/i18n-lite';
import { AdminManager } from '../../shared/cross/adminManager';
import { SettingsMenuManager } from '../../shared/cross/settingsMenuManager';
import { initChatPanelLayout } from '../../shared/ui/chat_panel_layout';
import { PANEL_SPLIT_STORAGE_KEY_ATTRIBUTION } from '../../shared/cross/panelSplitStorage';
import { TextInputController } from '../../shared/controllers/textInputController';
import { initializeCommonApp } from '../../shared/bootstrap';
import { registerPageBusy } from '../../shared/core/activitySession';
import { showAlertDialog } from '../../shared/ui/dialog';
import URLHandler from '../../shared/core/URLHandler';
import { createToast } from '../../shared/ui/toast';
import { translateApiErrorMessage } from '../../shared/core/errorUtils';
import type { LogitLensResult, LogitLensLayer, ActivationExplainResult } from '../../shared/api/GLTR_API';
import { runActivationExplain } from '../../shared/prediction_attribution/activationExplainer';
import { lsReadEnum, lsWriteString } from '../../shared/storage/localStorageHelpers';

d3.selectAll('.loadersmall').style('display', 'none');

initI18n();

const showToast = createToast('#toast').show;

const LOGIT_LENS_MODEL_VARIANT_STORAGE_KEY = 'info_radar_logit_lens_model_variant';

export type LogitLensModelVariant = 'base' | 'instruct';

function readStoredLogitLensModelVariant(): LogitLensModelVariant {
    return lsReadEnum(LOGIT_LENS_MODEL_VARIANT_STORAGE_KEY, ['base', 'instruct'] as const, 'instruct');
}

const apiPrefix = URLHandler.parameters['api'] || '';
const bodyElement = d3.select('body').node() as Element;
const { eventHandler, totalSurprisalFormat, api } = initializeCommonApp(apiPrefix, bodyElement);
const apiBaseForRequests = apiPrefix === '' ? '' : String(apiPrefix);

const adminManager = AdminManager.getInstance();
api.setAdminToken(adminManager.isInAdminMode() ? adminManager.getAdminToken() : null);

// --- DOM 引用 ---
const contextField = d3.select('#context_text');
const contextCountValue = d3.select('#context_count_value');
const clearContextBtn = d3.select('#clear_context_btn');
const pasteContextBtn = d3.select('#paste_context_btn');

const targetField = d3.select('#target_text');
const targetCountValue = d3.select('#target_count_value');
const clearTargetBtn = d3.select('#clear_target_btn');
const pasteTargetBtn = d3.select('#paste_target_btn');

const analyzeBtn = d3.select('#analyze_btn');
const modelVariantSelect = document.getElementById('logit_lens_model_variant') as HTMLSelectElement | null;
const loaderSmall = d3.select('.loadersmall');
const resultInfoEl = d3.select('#logit_lens_result_info');

if (modelVariantSelect) {
    modelVariantSelect.value = readStoredLogitLensModelVariant();
}

function currentModelVariant(): LogitLensModelVariant {
    const v = modelVariantSelect?.value;
    return v === 'base' || v === 'instruct' ? v : 'instruct';
}

modelVariantSelect?.addEventListener('change', () => {
    lsWriteString(LOGIT_LENS_MODEL_VARIANT_STORAGE_KEY, currentModelVariant());
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

// --- 分析按钮状态管理 ---
let analyzeInFlight = false;
let lastCommittedInputs: { context: string; target: string } | null = null;

function syncAnalyzeButtonState(): void {
    const context = (contextField.node() as HTMLTextAreaElement | null)?.value ?? '';
    const target = (targetField.node() as HTMLTextAreaElement | null)?.value ?? '';
    const idleInputsReady = context.length > 0 && target.length > 0;
    const hasUncommittedDraft =
        lastCommittedInputs === null ||
        context !== lastCommittedInputs.context ||
        target !== lastCommittedInputs.target;
    const btn = analyzeBtn.node() as HTMLButtonElement | null;
    if (!btn) return;
    btn.disabled = !idleInputsReady || analyzeInFlight;
    btn.textContent = analyzeInFlight ? tr('Analyzing...') : tr('Analyze');
    btn.classList.toggle('inactive', !idleInputsReady || analyzeInFlight);
    if (hasUncommittedDraft && !analyzeInFlight) {
        btn.classList.add('has-draft');
    } else {
        btn.classList.remove('has-draft');
    }
}

function setAnalyzeLoading(loading: boolean): void {
    analyzeInFlight = loading;
    loaderSmall.style('display', loading ? null : 'none');
    syncAnalyzeButtonState();
}

registerPageBusy(() => analyzeInFlight);

[contextField, targetField].forEach((field) => {
    (field.node() as HTMLTextAreaElement | null)?.addEventListener('input', syncAnalyzeButtonState);
});
syncAnalyzeButtonState();

// --- Logit Lens 渲染 ---
let _llResponse: LogitLensResult | null = null;
let _hoveredLayer: LogitLensLayer | null = null;

function renderLogitLens(response: LogitLensResult): void {
    const panel = document.getElementById('logit_lens_panel');
    if (!panel) return;
    panel.style.display = 'block';
    _llResponse = response;
    _hoveredLayer = response.layers?.[response.layers.length - 1] ?? null;

    // Show subword notice if target_token differs from what user typed
    const userTarget = (targetField.node() as HTMLTextAreaElement | null)?.value?.trim() ?? '';
    const trackedToken = response.target_token ?? '';
    let noticeEl = document.getElementById('ll_subword_notice');
    if (!noticeEl) {
        noticeEl = document.createElement('div');
        noticeEl.id = 'll_subword_notice';
        noticeEl.style.cssText = 'font-size:9pt;color:#B45309;margin-bottom:6px;padding:4px 8px;background:rgba(254,243,199,0.5);border-radius:4px;border:1px solid rgba(245,158,11,0.3)';
        panel.insertBefore(noticeEl, panel.firstChild);
    }
    const isSubword = trackedToken && userTarget && trackedToken.trim() !== userTarget.trim();
    if (isSubword) {
        noticeEl.style.display = '';
        noticeEl.textContent = `实际追踪 token：「${trackedToken}」（"${userTarget}" 被 tokenizer 切成子词，Logit Lens 追踪第一个子词的概率曲线）`;
    } else {
        noticeEl.style.display = 'none';
    }

    renderLayerChart(response);
    renderLayerCard();
}

function renderLayerChart(response: LogitLensResult): void {
    const container = document.getElementById('logit_lens_layer_heatmap');
    if (!container || !response.layers) return;

    const layers = response.layers;
    const target = response.target_token ?? '';

    // Find eureka layer index (first where target is top-1)
    let eurekaIdx: number | null = null;
    for (let i = 0; i < layers.length; i++) {
        if (layers[i].topk_tokens[0]?.toLowerCase() === target.toLowerCase()) {
            eurekaIdx = i;
            break;
        }
    }

    const w = container.clientWidth || 560;
    const h = 160;
    const pad = { top: 15, right: 15, bottom: 26, left: 42 };
    const cw = w - pad.left - pad.right;
    const ch = h - pad.top - pad.bottom;
    const n = layers.length;

    // --- Auto-scale Y axis to [0, maxProb] so low-prob curves are still visible ---
    const maxProb = Math.max(...layers.map(l => l.target_prob), 0.01);
    const xOf = (i: number) => pad.left + (i / Math.max(n - 1, 1)) * cw;
    const yOf = (p: number) => pad.top + ch - (p / maxProb) * ch;

    const pathD = layers.map((l, i) => `${i === 0 ? 'M' : 'L'}${xOf(i).toFixed(1)},${yOf(l.target_prob).toFixed(1)}`).join(' ');

    // Y-axis: 4 ticks scaled to maxProb
    const gridLines = [0, 0.33, 0.67, 1].map(v => {
        const prob = v * maxProb;
        const y = yOf(prob).toFixed(1);
        const label = prob >= 0.01 ? `${(prob * 100).toFixed(0)}%` : prob > 0 ? `${(prob * 100).toFixed(1)}%` : '0%';
        return `<line x1="${pad.left}" y1="${y}" x2="${w - pad.right}" y2="${y}" stroke="#ddd" stroke-width="0.8" stroke-dasharray="3,3"/>
                <text x="${pad.left - 4}" y="${(parseFloat(y) + 3).toFixed(1)}" text-anchor="end" font-size="9" fill="#999">${label}</text>`;
    }).join('');

    const xLabels = layers.map((l, i) => {
        if (i % 4 !== 0 && i !== n - 1) return '';
        return `<text x="${xOf(i).toFixed(1)}" y="${h - 6}" text-anchor="middle" font-size="9" fill="#999">L${l.layer}</text>`;
    }).join('');

    const eurekaLine = eurekaIdx !== null
        ? `<line x1="${xOf(eurekaIdx).toFixed(1)}" y1="${pad.top}" x2="${xOf(eurekaIdx).toFixed(1)}" y2="${h - pad.bottom}" stroke="#B45309" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.6"/>
           <text x="${(xOf(eurekaIdx) + 4).toFixed(1)}" y="${(pad.top + 10).toFixed(1)}" font-size="9" fill="#B45309">Eureka L${layers[eurekaIdx].layer}</text>`
        : '';

    const dots = layers.map((l, i) => {
        const isEureka = i === eurekaIdx;
        const r = isEureka ? 5 : 3;
        const fill = isEureka ? '#D97706' : '#ff4740';
        return `<circle data-layer-idx="${i}" cx="${xOf(i).toFixed(1)}" cy="${yOf(l.target_prob).toFixed(1)}" r="${r}" fill="${fill}" stroke="none" style="cursor:pointer" opacity="0.85"/>`;
    }).join('');

    const finalPct = (response.final_target_prob * 100).toFixed(1);
    const axisNote = `<text x="${w - pad.right}" y="${h - 6}" text-anchor="end" font-size="9" fill="#999">最终: ${finalPct}%</text>`;

    container.innerHTML = `<svg width="${w}" height="${h}" style="display:block;width:100%;overflow:visible">
        ${gridLines}
        ${xLabels}
        ${axisNote}
        ${eurekaLine}
        <path d="${pathD}" fill="none" stroke="#ff4740" stroke-width="2"/>
        ${dots}
    </svg>`;

    // Bind interactions
    container.querySelectorAll<SVGCircleElement>('circle[data-layer-idx]').forEach(el => {
        const activate = () => {
            const idx = parseInt(el.getAttribute('data-layer-idx') ?? '0', 10);
            _hoveredLayer = layers[idx] ?? null;
            renderLayerCard();
            container.querySelectorAll<SVGCircleElement>('circle[data-layer-idx]').forEach(c => {
                const ci = parseInt(c.getAttribute('data-layer-idx') ?? '0', 10);
                c.setAttribute('r', c === el ? '7' : (ci === eurekaIdx ? '5' : '3'));
                c.setAttribute('stroke', c === el ? '#fff' : 'none');
                c.setAttribute('stroke-width', '2');
            });
        };
        el.addEventListener('click', activate);
        el.addEventListener('mouseenter', activate);
    });

    // Also render the key-layers heatmap below
    renderKeyLayersHeatmap(layers, target, eurekaIdx);
}

/** 精选关键层的 top-k 热力表：直观展示"词在哪层出现" */
function renderKeyLayersHeatmap(layers: LogitLensLayer[], target: string, eurekaIdx: number | null): void {
    // Write into logit_lens_target_trajectory; layer card appends as child later
    const container = document.getElementById('logit_lens_target_trajectory');
    if (!container) return;

    // Pick representative layers: embedding, ~1/3, ~2/3, eureka±1, last
    const n = layers.length;
    const picks = new Set<number>([0, Math.floor(n * 0.33), Math.floor(n * 0.67), n - 1]);
    if (eurekaIdx !== null) {
        if (eurekaIdx > 0) picks.add(eurekaIdx - 1);
        picks.add(eurekaIdx);
        if (eurekaIdx < n - 1) picks.add(eurekaIdx + 1);
    }
    const selected = [...picks].sort((a, b) => a - b).map(i => layers[i]);

    const k = 4;

    const rows = selected.map(layer => {
        const isEureka = eurekaIdx !== null && layer.layer === layers[eurekaIdx]?.layer;
        const labelBg = isEureka ? 'background:#FEF3C7;color:#B45309;font-weight:700' : 'color:#888';
        const layerLabel = layer.is_embedding ? 'L0 emb' : `L${layer.layer}`;
        const cells = layer.topk_tokens.slice(0, k).map((tok, i) => {
            const prob = layer.topk_probs[i] ?? 0;
            const isTarget = tok.toLowerCase() === target.toLowerCase();
            const alpha = Math.min(0.15 + prob * 2, 1);
            const bg = isTarget ? `rgba(180,83,9,${alpha})` : `rgba(255,71,64,${alpha})`;
            const textColor = isTarget ? (alpha > 0.5 ? '#fff' : '#B45309') : (alpha > 0.6 ? '#fff' : 'inherit');
            const border = isTarget ? '1px solid #B45309' : '1px solid transparent';
            return `<td style="padding:3px 5px;text-align:center;background:${bg};color:${textColor};border:${border};border-radius:3px;white-space:nowrap;font-size:11px;font-family:monospace" title="${tok}: ${(prob*100).toFixed(2)}%">${tok}</td>`;
        }).join('');
        return `<tr>
            <th style="${labelBg};padding:3px 8px 3px 4px;font-size:10px;white-space:nowrap;text-align:right">${layerLabel}${isEureka ? ' ⭐' : ''}</th>
            ${cells}
        </tr>`;
    }).join('');

    const headers = `<th style="font-size:9px;color:#aaa;padding:2px 4px">层</th>` +
        Array.from({ length: k }, (_, i) => `<th style="font-size:9px;color:#aaa;padding:2px 6px">#${i+1}</th>`).join('');

    // Clear container and write heatmap; leave room for layer card below
    container.innerHTML = `
        <div style="margin-top:10px;font-size:9pt;color:#999;margin-bottom:4px">
            抽样关键层 top-${k} 候选词（首层/1/3处/2/3处/尾层 + ⭐Eureka 层前后）<br>
            <span style="color:#B45309">橙色边框 = 目标词；颜色越深 = 该词概率越高；⭐ = 目标词首次跃至第 1 位</span>
        </div>
        <div style="overflow-x:auto">
            <table style="border-collapse:separate;border-spacing:2px 2px;min-width:320px">
                <thead><tr>${headers}</tr></thead>
                <tbody>${rows}</tbody>
            </table>
        </div>`;
}

function renderLayerCard(): void {
    const container = document.getElementById('logit_lens_target_trajectory');
    if (!container) return;
    const layer = _hoveredLayer;
    const target = _llResponse?.target_token ?? '';
    if (!layer) return;

    const layerLabel = layer.is_embedding ? 'L0: Embedding' : `Layer ${layer.layer}`;
    const targetIdx = layer.topk_tokens.findIndex(t => t.toLowerCase() === target.toLowerCase());
    const isTop1 = targetIdx === 0;
    const rankLabel = isTop1
        ? `<span class="ll-rank-badge ll-rank-top">目标词排第 #1 ✓</span>`
        : targetIdx >= 0
            ? `<span class="ll-rank-badge ll-rank-other">目标词排第 #${targetIdx + 1}</span>`
            : `<span class="ll-rank-badge ll-rank-other">目标词不在前列</span>`;

    // Scale bars to layer's max prob (not 0-100%), so even low-prob layers show useful bars
    const maxP = Math.max(...layer.topk_probs.slice(0, 4), 0.001);
    const bars = layer.topk_tokens.slice(0, 4).map((tok, i) => {
        const prob = layer.topk_probs[i] ?? 0;
        const isTarget = tok.toLowerCase() === target.toLowerCase();
        const barColor = isTarget ? '#B45309' : '#aaa';
        const pct = (prob * 100).toFixed(2);
        const barWidth = (prob / maxP * 100).toFixed(1);
        return `<div class="ll-bar-row">
            <div class="ll-bar-label ${isTarget ? 'll-bar-label--target' : ''}">#${i + 1}&nbsp;<code>${tok}</code></div>
            <div class="ll-bar-track"><div class="ll-bar-fill" style="width:${barWidth}%;background:${barColor}"></div></div>
            <div class="ll-bar-pct">${pct}%</div>
        </div>`;
    }).join('');

    // Only inject the card section if heatmap is NOT already rendered there
    // (We use a separate card div to avoid overwriting the heatmap)
    let card = document.getElementById('ll_layer_card_inner');
    if (!card) {
        card = document.createElement('div');
        card.id = 'll_layer_card_inner';
        container.appendChild(card);
    }
    card.innerHTML = `<div class="ll-layer-card">
        <div class="ll-layer-card-header">
            <code class="ll-layer-code">${layerLabel}</code>
            <span class="ll-layer-desc">点选折线点可查看该层 top-k</span>
            ${rankLabel}
        </div>
        <div class="ll-bars">${bars}</div>
    </div>`;
}

// --- Activation Explainer ---
async function runActivationExplainAe(context: string, targetToken: string): Promise<void> {
    await runActivationExplain(api, currentModelVariant(), 'logit_lens', context, targetToken, {
        panelId: 'ae_panel',
        loadingId: 'ae_loading',
        resultId: 'ae_result',
        errorId: 'ae_error',
        explanationId: 'ae_explanation',
        cosineId: 'ae_cosine',
    });
}

// --- 主分析逻辑 ---
async function runAnalyze(): Promise<void> {
    const context = (contextField.node() as HTMLTextAreaElement | null)?.value ?? '';
    const target = (targetField.node() as HTMLTextAreaElement | null)?.value ?? '';
    if (analyzeInFlight || !context || !target) return;

    setAnalyzeLoading(true);
    try {
        const ll = await api.logitLens(context, target, currentModelVariant(), 'attribution');
        if (ll.success) {
            renderLogitLens(ll);
            lastCommittedInputs = { context, target };
            const info = `${tr('model')}: ${ll.model ?? '–'}\n${tr('Target token')}: ${ll.target_token ?? '–'}\n${tr('Final prob')}: ${(ll.final_target_prob * 100).toFixed(1)}%\n${tr('Layers')}: ${ll.n_layers ?? '–'}`;
            resultInfoEl.classed('is-hidden', false).text(info);

            // 自动触发 Activation Explainer
            if (context && ll.target_token) {
                void runActivationExplainAe(context, ll.target_token);
            }
        } else {
            showAlertDialog(tr('Logit Lens'), ll.message || tr('Analysis failed'));
        }
    } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        showAlertDialog(tr('Logit Lens'), translateApiErrorMessage(msg));
    } finally {
        setAnalyzeLoading(false);
    }
}

analyzeBtn.on('click', () => void runAnalyze());

(contextField.node() as HTMLTextAreaElement | null)?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) void runAnalyze();
});

// --- 初始化布局 ---
initChatPanelLayout({ storageKey: PANEL_SPLIT_STORAGE_KEY_ATTRIBUTION });

const themeManager = initThemeManager(
    { onThemeChange: () => { /* no-op for logit lens */ } },
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

// ---- Examples ----
type LogitLensExample = { label: string; context: string; target: string };

const LOGIT_LENS_EXAMPLES: LogitLensExample[] = [
    {
        label: '固态水',
        context: 'water freezes and turns into solid',
        target: ' ice',
    },
    {
        label: '太阳东升西落',
        context: 'The sun rises in the east and sets in the',
        target: ' west',
    },
    {
        label: '狗是忠实的',
        context: 'A dog is known as a loyal',
        target: ' pet',
    },
];

function initLogitLensExamples(): void {
    const section = document.getElementById('logit_lens_examples');
    if (!section) return;

    const desc = document.createElement('div');
    desc.className = 'page-examples-desc';
    desc.innerHTML = `<strong>${tr('Logit Lens')}</strong>: ${tr('applies the final output layer to each intermediate layer, revealing how the model\'s prediction takes shape across depth — early layers guess, later layers converge.')}
        <br><span style="color:var(--text-muted);font-size:8pt">💡 适合<b>续写文本</b>（如"水结成固态"→"ice"），而非问答。目标词若被切成子词，实际追踪首个子词的概率曲线，分析后会有提示。</span>`;
    section.appendChild(desc);

    const label = document.createElement('div');
    label.className = 'page-examples-label';
    label.textContent = tr('Try an example:');
    section.appendChild(label);

    const btns = document.createElement('div');
    btns.className = 'page-examples-buttons';

    for (const ex of LOGIT_LENS_EXAMPLES) {
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

initLogitLensExamples();
