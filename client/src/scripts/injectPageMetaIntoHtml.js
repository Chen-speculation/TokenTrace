/**
 * 构建时根据 content/page-meta.json 向 HTML 写入英文标题、副标题、浏览器标题与导航锚文本，
 * 便于爬虫与不执行 JS 的环境读取。页内可见文案带 data-i18n，运行时由 initI18n 按英文 key 翻译；
 * 浏览器标题在构建时由本脚本写入：主标题与副标题与 page-meta 一致；副标题若以 "-" 开头（工具页 tagline）则与标题之间只加一个空格；否则中间加 " - "（如首页主副标题）。
 * 中文环境由 <title data-i18n> 在 initI18n 翻译整串 key。
 */

/**
 * @param {string} s
 * @returns {string}
 */
function escapeHtmlText(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

/**
 * 浏览器标题与链接 title：title 与 subtitle 拼接（仅 trim 首尾空白）。
 * 副标题已以 "-" 开头时不额外插入横线，避免 "Title - - tagline"；否则插入 " - "。
 *
 * @param {{ title: string, subtitle: string }} meta
 */
function documentTitleEn(meta) {
    const title = String(meta.title ?? '').trim();
    const subtitle = String(meta.subtitle ?? '').trim();
    if (!title) return subtitle;
    if (!subtitle) return title;
    const joiner = subtitle.startsWith('-') ? ' ' : ' - ';
    return title + joiner + subtitle;
}

/**
 * 将带 `data-page-*` 的成对标签内容替换为纯文本（英文，已转义）。
 * @param {string} html
 * @param {string} attrToken 如 data-page-title
 * @param {string} text
 */
function injectDataPageBlock(html, attrToken, text) {
    const esc = escapeHtmlText(text);
    const re = new RegExp(
        `<([a-z][a-z0-9]*)([^>]*\\b${attrToken}\\b[^>]*)>([\\s\\S]*?)<\\/\\1>`,
        'gi'
    );
    return html.replace(re, (_m, tag, attrs) => `<${tag}${attrs}>${esc}</${tag}>`);
}

/**
 * @param {string} html
 * @param {string} pageKey
 * @param {{ pages: Record<string, { title: string, subtitle: string, href?: string, heartline?: string, formula?: string }>, navPageKeys: string[] }} doc
 * @returns {string}
 */
function injectPageMeta(html, pageKey, doc) {
    const meta = doc.pages[pageKey];
    if (!meta) {
        throw new Error(`injectPageMeta: unknown pageKey "${pageKey}"`);
    }

    const dt = documentTitleEn(meta);
    html = html.replace(/<title[^>]*>[^<]*<\/title>/i, `<title data-i18n>${escapeHtmlText(dt)}</title>`);

    html = injectDataPageBlock(html, 'data-page-title', meta.title);
    html = injectDataPageBlock(html, 'data-page-subtitle', meta.subtitle);

    const heartlineElRe = /<([a-z][a-z0-9]*)([^>]*\bdata-page-heartline\b[^>]*)>([\s\S]*?)<\/\1>/gi;
    if (meta.heartline) {
        html = html.replace(heartlineElRe, (_m, tag, attrs) => `<${tag}${attrs}>${escapeHtmlText(meta.heartline)}</${tag}>`);
    } else {
        html = html.replace(heartlineElRe, '');
    }

    const formulaElRe = /<([a-z][a-z0-9]*)([^>]*\bdata-page-formula\b[^>]*)>([\s\S]*?)<\/\1>/gi;
    if (meta.formula) {
        html = html.replace(formulaElRe, (_m, tag, attrs) => `<${tag}${attrs}>${escapeHtmlText(meta.formula)}</${tag}>`);
    } else {
        html = html.replace(formulaElRe, '');
    }

    if (pageKey === 'home' && Array.isArray(doc.navPageKeys)) {
        // Swiss Style module grid: match any element with data-nav-page attribute
        for (const navKey of doc.navPageKeys) {
            const navMeta = doc.pages[navKey];
            if (!navMeta) {
                throw new Error(`injectPageMeta: navPageKeys references missing page "${navKey}"`);
            }
            const navTitle = documentTitleEn(navMeta);

            // Build module content for new Swiss Style grid
            const moduleNum = String(doc.navPageKeys.indexOf(navKey) + 1).padStart(2, '0');
            const tagMap = {
                causalFlow: '归因',
                analysis: '分析',
                attribution: '归因',
                logitLens: '解码',
                branchTree: '分支',
                chat: '对话',
            };
            const tagText = tagMap[navKey] || '';

            const moduleName = `<div class="nav-landing-module-name" data-i18n="title">${escapeHtmlText(navMeta.title)}</div>`;
            const moduleDesc = `<div class="nav-landing-module-desc" data-i18n="subtitle">${escapeHtmlText(navMeta.subtitle)}</div>`;
            const moduleTag = tagText ? `<span class="nav-landing-module-tag">${escapeHtmlText(tagText)}</span>` : '';

            // All modules: <a> with data-nav-page, filled at build time with SEO-friendly text
            const re = new RegExp(
                `(<a\\b[^>]*\\bdata-nav-page=["']?${navKey}["']?[^>]*>)([\\s\\S]*?)(<\\/a>)`,
                'i'
            );
            const m = html.match(re);
            if (!m) {
                throw new Error(`injectPageMeta: missing element with data-nav-page="${navKey}" in home HTML`);
            }

            let openTag = m[1];
            const closeTag = m[3];

            // Update href if present in page-meta
            if (navMeta.href && /\bhref\s*=/.test(openTag)) {
                openTag = openTag.replace(
                    /\bhref\s*=\s*("[^"]*"|'[^']*')/i,
                    `href="${escapeHtmlText(navMeta.href)}"`
                );
            }

            // Update or add title
            if (/\btitle\s*=/.test(openTag)) {
                openTag = openTag.replace(/\btitle\s*=\s*("[^"]*"|'[^']*')/i, `title="${escapeHtmlText(navTitle)}"`);
            } else if (openTag.endsWith('>')) {
                openTag = openTag.slice(0, -1) + ` title="${escapeHtmlText(navTitle)}">`;
            }

            const demoMap = {
                causalFlow: `<div class="card-demo card-demo--dag">
                    <svg viewBox="0 0 220 80" class="demo-dag-svg">
                        <defs>
                            <marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                                <path d="M0,0 L6,3 L0,6 Z" fill="var(--demo-muted)"/>
                            </marker>
                        </defs>
                        <line x1="36" y1="40" x2="74" y2="40" stroke="var(--demo-muted)" stroke-width="1" marker-end="url(#arr)"/>
                        <line x1="116" y1="40" x2="154" y2="40" stroke="var(--demo-muted)" stroke-width="1" marker-end="url(#arr)"/>
                        <line x1="36" y1="40" x2="114" y2="20" stroke="var(--demo-accent)" stroke-width="1.5" marker-end="url(#arr)" class="demo-dag-edge"/>
                        <rect x="2" y="28" width="34" height="22" rx="3" fill="var(--demo-node-bg)" stroke="var(--demo-border)"/>
                        <text x="19" y="42" text-anchor="middle" class="demo-node-text">The</text>
                        <rect x="76" y="28" width="38" height="22" rx="3" fill="var(--demo-node-bg)" stroke="var(--demo-border)"/>
                        <text x="95" y="42" text-anchor="middle" class="demo-node-text">capital</text>
                        <rect x="116" y="8" width="30" height="22" rx="3" fill="var(--demo-accent-bg)" stroke="var(--demo-accent)" stroke-width="1.5"/>
                        <text x="131" y="22" text-anchor="middle" class="demo-node-text demo-node-text--accent">of</text>
                        <rect x="156" y="28" width="60" height="22" rx="3" fill="var(--demo-node-bg)" stroke="var(--demo-border)"/>
                        <text x="186" y="42" text-anchor="middle" class="demo-node-text">→ Paris</text>
                    </svg>
                </div>`,
                analysis: `<div class="card-demo card-demo--highlight">
                    <span class="demo-tok" style="--a:0.08">The </span><span class="demo-tok" style="--a:0.12">quick </span><span class="demo-tok demo-tok--hi" style="--a:0.72">brown </span><span class="demo-tok demo-tok--hi" style="--a:0.85">fox </span><span class="demo-tok" style="--a:0.1">jumps </span><span class="demo-tok" style="--a:0.15">over </span><span class="demo-tok demo-tok--hi" style="--a:0.55">lazy </span><span class="demo-tok" style="--a:0.09">dog</span>
                </div>`,
                attribution: `<div class="card-demo card-demo--attribution">
                    <div class="demo-attr-ctx"><span class="demo-attr-tok" style="--s:0.9">France</span> <span class="demo-attr-tok" style="--s:0.4">'s</span> <span class="demo-attr-tok" style="--s:0.6">capital</span> <span class="demo-attr-tok" style="--s:0.15">is</span></div>
                    <div class="demo-attr-target">→ <em>Paris</em></div>
                </div>`,
                logitLens: `<div class="card-demo card-demo--logit">
                    <svg viewBox="0 0 200 64" class="demo-logit-svg">
                        <polyline points="0,58 20,57 40,55 60,50 80,40 100,25 120,12 140,7 160,5 180,4 200,4" fill="none" stroke="var(--demo-accent)" stroke-width="2" class="demo-logit-line"/>
                        <circle cx="120" cy="12" r="4" fill="var(--demo-accent)" class="demo-logit-eureka"/>
                        <text x="124" y="10" font-size="7" fill="var(--demo-accent)" class="demo-logit-label">Eureka L12</text>
                        <line x1="120" y1="0" x2="120" y2="60" stroke="var(--demo-accent)" stroke-width="1" stroke-dasharray="3,2" opacity="0.4"/>
                    </svg>
                </div>`,
                branchTree: `<div class="card-demo card-demo--tree">
                    <svg viewBox="0 0 200 80" class="demo-tree-svg">
                        <rect x="82" y="4" width="36" height="20" rx="3" fill="var(--demo-node-bg)" stroke="var(--demo-border)"/>
                        <text x="100" y="18" text-anchor="middle" class="demo-node-text">Once</text>
                        <g class="demo-tree-branch">
                            <line x1="100" y1="24" x2="40" y2="50" stroke="var(--demo-muted)" stroke-width="1"/>
                            <rect x="14" y="50" width="52" height="20" rx="3" fill="var(--demo-accent-bg)" stroke="var(--demo-accent)" stroke-width="1.5"/>
                            <text x="40" y="64" text-anchor="middle" class="demo-node-text demo-node-text--accent"> upon</text>
                            <text x="40" y="74" text-anchor="middle" class="demo-node-text" style="font-size:7px">56%</text>
                        </g>
                        <g class="demo-tree-branch">
                            <line x1="100" y1="24" x2="100" y2="50" stroke="var(--demo-muted)" stroke-width="1"/>
                            <rect x="74" y="50" width="52" height="20" rx="3" fill="var(--demo-node-bg)" stroke="var(--demo-border)"/>
                            <text x="100" y="64" text-anchor="middle" class="demo-node-text"> more</text>
                        </g>
                        <g class="demo-tree-branch">
                            <line x1="100" y1="24" x2="160" y2="50" stroke="var(--demo-muted)" stroke-width="1"/>
                            <rect x="134" y="50" width="52" height="20" rx="3" fill="var(--demo-node-bg)" stroke="var(--demo-border)"/>
                            <text x="160" y="64" text-anchor="middle" class="demo-node-text"> in</text>
                        </g>
                    </svg>
                </div>`,
                chat: `<div class="card-demo card-demo--chat">
                    <div class="demo-chat-row demo-chat-row--prompt">What is 1+1?</div>
                    <div class="demo-chat-row demo-chat-row--reply"><span class="demo-chat-tok" style="--d:0s">1+1</span><span class="demo-chat-tok" style="--d:.15s"> equals</span><span class="demo-chat-tok demo-chat-tok--hi" style="--d:.3s"> 2</span><span class="demo-chat-tok" style="--d:.45s">.</span></div>
                </div>`,
            };
            const demoHtml = demoMap[navKey] || '';

            const inner = `<div class="nav-landing-module-num">${moduleNum}</div>` + moduleName + moduleDesc + moduleTag + demoHtml;
            html = html.replace(re, `${openTag}${inner}${closeTag}`);
        }
    }

    return html;
}

module.exports = { injectPageMeta, escapeHtmlText, documentTitleEn };
