const CHAR_DELAY = 75;
const IDLE_DELAY = 2200;

const DEFAULT_PLACEHOLDERS = [
    'Ask anything...',
    "What's on your mind?",
    'How can I help you?',
    'What would you like to know?',
];

interface TypewriterState {
    intervalId: number | null;
    timeoutId: number | null;
    placeholderIndex: number;
    displayedText: string;
    isTyping: boolean;
    active: boolean;
}

const states = new WeakMap<HTMLTextAreaElement | HTMLInputElement, TypewriterState>();

function clearTimers(state: TypewriterState): void {
    if (state.intervalId !== null) {
        clearInterval(state.intervalId);
        state.intervalId = null;
    }
    if (state.timeoutId !== null) {
        clearTimeout(state.timeoutId);
        state.timeoutId = null;
    }
}

function startTyping(
    el: HTMLTextAreaElement | HTMLInputElement,
    state: TypewriterState,
    placeholders: string[]
): void {
    clearTimers(state);
    const current = placeholders[state.placeholderIndex % placeholders.length];
    if (!current) {
        state.displayedText = '';
        state.isTyping = false;
        applyPlaceholder(el, state);
        return;
    }

    const chars = Array.from(current);
    state.displayedText = '';
    state.isTyping = true;
    applyPlaceholder(el, state);

    let charIndex = 0;
    state.intervalId = window.setInterval(() => {
        if (charIndex < chars.length) {
            state.displayedText = chars.slice(0, charIndex + 1).join('');
            applyPlaceholder(el, state);
            charIndex++;
        } else {
            if (state.intervalId !== null) {
                clearInterval(state.intervalId);
                state.intervalId = null;
            }
            state.isTyping = false;
            applyPlaceholder(el, state);

            state.timeoutId = window.setTimeout(() => {
                state.placeholderIndex = (state.placeholderIndex + 1) % placeholders.length;
                startTyping(el, state, placeholders);
            }, IDLE_DELAY);
        }
    }, CHAR_DELAY);
}

function applyPlaceholder(el: HTMLTextAreaElement | HTMLInputElement, state: TypewriterState): void {
    if (!state.active) return;
    if (el.value !== '') return;
    if (document.activeElement === el) return;
    const cursor = state.isTyping ? '|' : '';
    el.setAttribute('data-typewriter-placeholder', state.displayedText + cursor);
    el.placeholder = state.displayedText + cursor;
}

function stopTyping(el: HTMLTextAreaElement | HTMLInputElement, state: TypewriterState): void {
    state.active = false;
    clearTimers(state);
    el.placeholder = el.getAttribute('data-original-placeholder') || '';
    el.removeAttribute('data-typewriter-placeholder');
}

export function initTypewriterPlaceholders(
    options: {
        selector?: string;
        placeholders?: string[];
    } = {}
): void {
    const selector = options.selector ?? 'textarea[data-typewriter], input[data-typewriter]';
    const placeholders = options.placeholders ?? DEFAULT_PLACEHOLDERS;
    const elements = document.querySelectorAll<HTMLTextAreaElement | HTMLInputElement>(selector);

    elements.forEach((el) => {
        const existing = states.get(el);
        if (existing) {
            clearTimers(existing);
        }

        const originalPlaceholder = el.placeholder || '';
        el.setAttribute('data-original-placeholder', originalPlaceholder);

        const state: TypewriterState = {
            intervalId: null,
            timeoutId: null,
            placeholderIndex: 0,
            displayedText: '',
            isTyping: true,
            active: true,
        };
        states.set(el, state);

        const onFocus = () => {
            const s = states.get(el);
            if (!s) return;
            clearTimers(s);
            el.placeholder = originalPlaceholder;
        };
        const onBlur = () => {
            const s = states.get(el);
            if (!s || !s.active) return;
            if (el.value === '') {
                startTyping(el, s, placeholders);
            }
        };
        const onInput = () => {
            if (el.value !== '') {
                const s = states.get(el);
                if (s) clearTimers(s);
            }
        };

        el.addEventListener('focus', onFocus);
        el.addEventListener('blur', onBlur);
        el.addEventListener('input', onInput);

        if (el.value === '' && document.activeElement !== el) {
            startTyping(el, state, placeholders);
        }
    });
}
