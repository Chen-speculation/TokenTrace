import * as d3 from 'd3';
import '../../shared/core/d3-polyfill';
import '../../css/pages/branch_tree.scss';

import { initThemeManager } from '../../shared/ui/theme';
import { initLanguageManager } from '../../shared/ui/language';
import { initI18n, tr } from '../../shared/lang/i18n-lite';
import { AdminManager } from '../../shared/cross/adminManager';
import { SettingsMenuManager } from '../../shared/cross/settingsMenuManager';
import { initChatPanelLayout } from '../../shared/ui/chat_panel_layout';
import { PANEL_SPLIT_STORAGE_KEY_GEN_ATTRIBUTE } from '../../shared/cross/panelSplitStorage';
import { TextInputController } from '../../shared/controllers/textInputController';
import { initializeCommonApp } from '../../shared/bootstrap';
import { registerPageBusy } from '../../shared/core/activitySession';
import { showAlertDialog } from '../../shared/ui/dialog';
import URLHandler from '../../shared/core/URLHandler';
import { createToast } from '../../shared/ui/toast';
import { translateApiErrorMessage } from '../../shared/core/errorUtils';
import type { PredictionAttributeModelVariant } from '../../shared/prediction_attribution/core/attributionResultCache';
import type { BranchNextCandidate } from '../../shared/api/GLTR_API';

d3.selectAll('.loadersmall').style('display', 'none');

initI18n();

const showToast = createToast('#toast').show;

const BRANCH_TREE_MAX_DEPTH = 12;
const BRANCH_TREE_MAX_WIDTH = 5;
const BRANCH_TREE_MAX_NODES = 200;

const apiPrefix = URLHandler.parameters['api'] || '';
const bodyElement = d3.select('body').node() as Element;
const { eventHandler, totalSurprisalFormat, api } = initializeCommonApp(apiPrefix, bodyElement);

const adminManager = AdminManager.getInstance();
api.setAdminToken(adminManager.isInAdminMode() ? adminManager.getAdminToken() : null);

// --- DOM 引用 ---
const rawTextField = d3.select('#branch_tree_raw_text');
const rawTextCountValue = d3.select('#branch_tree_raw_text_count_value');
const clearRawBtn = d3.select('#branch_tree_clear_raw_btn');
const pasteRawBtn = d3.select('#branch_tree_paste_raw_btn');
const submitBtn = d3.select('#branch_tree_submit_btn');
const loaderSmall = d3.select('.loadersmall');

// --- TextInputController ---
new TextInputController({
    textField: rawTextField,
    textCountValue: rawTextCountValue,
    clearBtn: clearRawBtn,
    submitBtn,
    saveBtn: d3.select(null),
    pasteBtn: pasteRawBtn,
    totalSurprisalFormat,
    showAlertDialog,
});

// --- Branch Tree ---
type BranchTreeNode = {
    id: string;
    prefix: string;
    candidateToken: string;
    prob: number;
    depth: number;
    parentId: string | null;
    children: BranchTreeNode[];
    expanded: boolean;
    candidates?: BranchNextCandidate[];
    isContextFull?: boolean;
};

let branchTreeRoot: BranchTreeNode | null = null;
let branchTreeNodeMap = new Map<string, BranchTreeNode>();
let branchTreeAbortController: AbortController | null = null;

function genBranchNodeId(): string {
    return `bt-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 5)}`;
}

function countBranchTreeNodes(node: BranchTreeNode): number {
    let count = 1;
    for (const child of node.children) {
        count += countBranchTreeNodes(child);
    }
    return count;
}

async function expandBranchNode(node: BranchTreeNode): Promise<void> {
    if (node.expanded || node.isContextFull) return;
    const totalNodes = branchTreeRoot ? countBranchTreeNodes(branchTreeRoot) : 0;
    if (totalNodes >= BRANCH_TREE_MAX_NODES) {
        showToast(tr('Branch tree max nodes reached'), 'error');
        return;
    }
    if (node.depth >= BRANCH_TREE_MAX_DEPTH) {
        showToast(tr('Branch tree max depth reached'), 'error');
        return;
    }

    branchTreeAbortController = new AbortController();
    setSubmitLoading(true);
    try {
        const variant: PredictionAttributeModelVariant = 'base';
        const res = await api.branchNext(
            node.prefix,
            variant,
            'causal_flow',
            BRANCH_TREE_MAX_WIDTH,
            branchTreeAbortController.signal
        );
        if (!res.success || !res.candidates) {
            showToast(res.message || tr('Request failed'), 'error');
            return;
        }
        node.candidates = res.candidates.slice(0, BRANCH_TREE_MAX_WIDTH);
        node.isContextFull = res.is_context_full ?? false;
        node.expanded = true;
        for (const cand of node.candidates) {
            const child: BranchTreeNode = {
                id: genBranchNodeId(),
                prefix: node.prefix + cand.token,
                candidateToken: cand.token,
                prob: cand.prob,
                depth: node.depth + 1,
                parentId: node.id,
                children: [],
                expanded: false,
            };
            node.children.push(child);
            branchTreeNodeMap.set(child.id, child);
        }
        renderBranchTree();
    } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        if (msg !== 'AbortError') showToast(translateApiErrorMessage(msg), 'error');
    } finally {
        branchTreeAbortController = null;
        setSubmitLoading(false);
    }
}

function renderBranchTree(): void {
    const surface = document.getElementById('branch_tree_surface');
    if (!surface || !branchTreeRoot) return;
    surface.innerHTML = '';
    const svgNs = 'http://www.w3.org/2000/svg';
    const w = surface.clientWidth || 800;
    const h = Math.max(400, countBranchTreeNodes(branchTreeRoot) * 40);
    const svg = document.createElementNS(svgNs, 'svg');
    svg.setAttribute('width', String(w));
    svg.setAttribute('height', String(h));
    svg.style.display = 'block';
    svg.style.width = '100%';

    // BFS layout
    const levels: BranchTreeNode[][] = [];
    const queue: [BranchTreeNode, number][] = [[branchTreeRoot, 0]];
    while (queue.length) {
        const [n, d] = queue.shift()!;
        if (!levels[d]) levels[d] = [];
        levels[d].push(n);
        for (const c of n.children) queue.push([c, d + 1]);
    }

    const rowH = h / (levels.length || 1);
    const nodePositions = new Map<string, { x: number; y: number }>();

    for (let d = 0; d < levels.length; d++) {
        const row = levels[d];
        const colW = w / (row.length + 1);
        for (let i = 0; i < row.length; i++) {
            const n = row[i];
            const x = colW * (i + 1);
            const y = rowH * d + rowH * 0.5;
            nodePositions.set(n.id, { x, y });
        }
    }

    // Draw links
    for (const [, node] of branchTreeNodeMap) {
        if (!node.parentId) continue;
        const parent = branchTreeNodeMap.get(node.parentId);
        if (!parent) continue;
        const pPos = nodePositions.get(parent.id);
        const cPos = nodePositions.get(node.id);
        if (!pPos || !cPos) continue;
        const line = document.createElementNS(svgNs, 'line');
        line.setAttribute('x1', String(pPos.x));
        line.setAttribute('y1', String(pPos.y));
        line.setAttribute('x2', String(cPos.x));
        line.setAttribute('y2', String(cPos.y));
        line.setAttribute('stroke', 'var(--border-color)');
        line.setAttribute('stroke-width', '1.5');
        svg.appendChild(line);
    }

    // Draw nodes
    for (const [, node] of branchTreeNodeMap) {
        const pos = nodePositions.get(node.id);
        if (!pos) continue;
        const g = document.createElementNS(svgNs, 'g');
        g.style.cursor = node.expanded || node.isContextFull ? 'default' : 'pointer';
        g.addEventListener('click', () => {
            if (!node.expanded && !node.isContextFull) {
                void expandBranchNode(node);
            }
        });

        const intensity = Math.min(1, (node.prob || 0) * 3);
        const color = `rgba(255, 71, 64, ${0.2 + intensity * 0.6})`;

        const circle = document.createElementNS(svgNs, 'circle');
        circle.setAttribute('cx', String(pos.x));
        circle.setAttribute('cy', String(pos.y));
        circle.setAttribute('r', '18');
        circle.setAttribute('fill', color);
        circle.setAttribute('stroke', 'var(--border-color)');
        circle.setAttribute('stroke-width', '1.5');
        g.appendChild(circle);

        const text = document.createElementNS(svgNs, 'text');
        text.setAttribute('x', String(pos.x));
        text.setAttribute('y', String(pos.y + 4));
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('font-size', '11');
        text.setAttribute('fill', 'var(--text-color)');
        text.textContent = node.candidateToken || node.prefix.slice(0, 8);
        g.appendChild(text);

        if (node.prob !== undefined) {
            const probText = document.createElementNS(svgNs, 'text');
            probText.setAttribute('x', String(pos.x));
            probText.setAttribute('y', String(pos.y + 28));
            probText.setAttribute('text-anchor', 'middle');
            probText.setAttribute('font-size', '9');
            probText.setAttribute('fill', 'var(--text-muted)');
            probText.textContent = `${(node.prob * 100).toFixed(1)}%`;
            g.appendChild(probText);
        }

        svg.appendChild(g);
    }

    // Root node
    if (branchTreeRoot) {
        const pos = nodePositions.get(branchTreeRoot.id);
        if (pos) {
            const g = document.createElementNS(svgNs, 'g');
            g.style.cursor = branchTreeRoot.expanded ? 'default' : 'pointer';
            g.addEventListener('click', () => {
                if (!branchTreeRoot!.expanded) void expandBranchNode(branchTreeRoot!);
            });
            const rect = document.createElementNS(svgNs, 'rect');
            rect.setAttribute('x', String(pos.x - 30));
            rect.setAttribute('y', String(pos.y - 14));
            rect.setAttribute('width', '60');
            rect.setAttribute('height', '28');
            rect.setAttribute('rx', '4');
            rect.setAttribute('fill', 'none');
            rect.setAttribute('stroke', '#aaa');
            rect.setAttribute('stroke-width', '1.5');
            g.appendChild(rect);
            const text = document.createElementNS(svgNs, 'text');
            text.setAttribute('x', String(pos.x));
            text.setAttribute('y', String(pos.y + 4));
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('font-size', '11');
            text.setAttribute('fill', 'var(--text-color)');
            text.textContent = branchTreeRoot.prefix.slice(0, 12) || 'root';
            g.appendChild(text);
            svg.appendChild(g);
        }
    }

    surface.appendChild(svg);
}

function initBranchTreeFromRawInput(): void {
    const rawText = (document.getElementById('branch_tree_raw_text') as HTMLTextAreaElement | null)?.value ?? '';
    if (!rawText.trim()) {
        showToast(tr('Please enter a prefix'), 'error');
        return;
    }
    branchTreeRoot = {
        id: genBranchNodeId(),
        prefix: rawText,
        candidateToken: '',
        prob: 1,
        depth: 0,
        parentId: null,
        children: [],
        expanded: false,
    };
    branchTreeNodeMap.clear();
    branchTreeNodeMap.set(branchTreeRoot.id, branchTreeRoot);
    void expandBranchNode(branchTreeRoot);
}

// --- 按钮状态管理 ---
let submitInFlight = false;

function syncSubmitButtonState(): void {
    const rawText = (rawTextField.node() as HTMLTextAreaElement | null)?.value ?? '';
    const btn = submitBtn.node() as HTMLButtonElement | null;
    if (!btn) return;
    btn.disabled = !rawText.trim() || submitInFlight;
    btn.classList.toggle('inactive', !rawText.trim() || submitInFlight);
    btn.textContent = submitInFlight ? tr('Loading...') : tr('Start');
}

function setSubmitLoading(loading: boolean): void {
    submitInFlight = loading;
    loaderSmall.style('display', loading ? null : 'none');
    syncSubmitButtonState();
}

registerPageBusy(() => submitInFlight);

(rawTextField.node() as HTMLTextAreaElement | null)?.addEventListener('input', syncSubmitButtonState);
syncSubmitButtonState();

submitBtn.on('click', () => {
    if (submitInFlight) {
        branchTreeAbortController?.abort();
        setSubmitLoading(false);
        return;
    }
    void initBranchTreeFromRawInput();
});

// Enter 键（Ctrl/Cmd + Enter）提交
(rawTextField.node() as HTMLTextAreaElement | null)?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        if (!submitInFlight) void initBranchTreeFromRawInput();
    }
});

// --- 初始化布局 ---
initChatPanelLayout({ storageKey: PANEL_SPLIT_STORAGE_KEY_GEN_ATTRIBUTE });

const themeManager = initThemeManager(
    { onThemeChange: () => { /* no-op for branch tree */ } },
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
const BRANCH_TREE_EXAMPLES = [
    { label: tr('Story opening'), prefix: '从前，在一个遥远的地方，' },
    { label: tr('Question'), prefix: '人工智能的本质是' },
    { label: tr('Code'), prefix: 'def fibonacci(n):' },
];

function initBranchTreeExamples(): void {
    const section = document.getElementById('branch_tree_examples');
    if (!section) return;

    const desc = document.createElement('div');
    desc.className = 'page-examples-desc';
    desc.innerHTML = `<strong>${tr('Branch Tree')}</strong>: ${tr('shows the model\'s top-k next-token candidates at each step as an interactive tree — revealing how probability mass spreads across possible continuations.')}`;
    section.appendChild(desc);

    const label = document.createElement('div');
    label.className = 'page-examples-label';
    label.textContent = tr('Try an example:');
    section.appendChild(label);

    const btns = document.createElement('div');
    btns.className = 'page-examples-buttons';

    for (const ex of BRANCH_TREE_EXAMPLES) {
        const btn = document.createElement('button');
        btn.className = 'page-example-btn';
        btn.type = 'button';
        btn.title = ex.prefix;
        btn.textContent = ex.label;
        btn.addEventListener('click', () => {
            const textarea = document.getElementById('branch_tree_raw_text') as HTMLTextAreaElement | null;
            if (textarea) {
                textarea.value = ex.prefix;
                textarea.dispatchEvent(new Event('input', { bubbles: true }));
            }
        });
        btns.appendChild(btn);
    }
    section.appendChild(btns);
}

initBranchTreeExamples();
