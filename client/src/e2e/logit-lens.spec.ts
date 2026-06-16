import { test, expect } from '@playwright/test';

async function dismissDevServerOverlay(page: any): Promise<void> {
    await page.evaluate(() => {
        const overlay = document.getElementById('webpack-dev-server-client-overlay');
        if (overlay) overlay.remove();
    });
}

test.describe('Logit Lens', () => {
    test('page loads and renders input form', async ({ page }) => {
        await page.goto('/logit_lens.html');
        await dismissDevServerOverlay(page);
        await expect(page.locator('text=Context')).toBeVisible();
        await expect(page.locator('text=Target prediction')).toBeVisible();
        await expect(page.locator('#analyze_btn')).toBeVisible();
    });

    test('analyze button disabled when inputs are empty', async ({ page }) => {
        await page.goto('/logit_lens.html');
        await dismissDevServerOverlay(page);
        const btn = page.locator('#analyze_btn');
        await expect(btn).toBeDisabled();
    });

    test('analyze button enabled after filling inputs', async ({ page }) => {
        await page.goto('/logit_lens.html');
        await dismissDevServerOverlay(page);
        await page.fill('#context_text', '中国的首都是');
        await page.fill('#target_text', '北京');
        const btn = page.locator('#analyze_btn');
        await expect(btn).toBeEnabled();
    });

    test('clicking analyze shows Logit Lens panel', async ({ page }) => {
        await page.goto('/logit_lens.html');
        await page.fill('#context_text', 'The capital of China is');
        await page.fill('#target_text', 'Beijing');
        await dismissDevServerOverlay(page);
        await page.evaluate(() => {
            const btn = document.getElementById('analyze_btn');
            if (btn) btn.click();
        });
        // Wait for panel to appear
        await page.waitForSelector('#logit_lens_panel', { state: 'visible', timeout: 30_000 });
        await expect(page.locator('#logit_lens_panel')).toBeVisible();
    });

    test('heatmap table renders with layer rows', async ({ page }) => {
        await page.goto('/logit_lens.html');
        await page.fill('#context_text', 'Hello world');
        await page.fill('#target_text', '!');
        await dismissDevServerOverlay(page);
        await page.evaluate(() => {
            const btn = document.getElementById('analyze_btn');
            if (btn) btn.click();
        });
        await page.waitForSelector('#logit_lens_layer_heatmap table', { timeout: 30_000 });
        const rows = page.locator('#logit_lens_layer_heatmap table tbody tr');
        await expect(rows).toHaveCount(29, { timeout: 30_000 });
    });

    test('trajectory SVG renders', async ({ page }) => {
        await page.goto('/logit_lens.html');
        await page.fill('#context_text', 'The capital of China is');
        await page.fill('#target_text', 'Beijing');
        await dismissDevServerOverlay(page);
        await page.evaluate(() => {
            const btn = document.getElementById('analyze_btn');
            if (btn) btn.click();
        });
        await page.waitForSelector('#logit_lens_target_trajectory svg', { timeout: 30_000 });
        await expect(page.locator('#logit_lens_target_trajectory svg')).toBeVisible();
    });
});
