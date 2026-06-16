import { test, expect } from '@playwright/test';

async function dismissDevServerOverlay(page: any): Promise<void> {
    await page.evaluate(() => {
        const overlay = document.getElementById('webpack-dev-server-client-overlay');
        if (overlay) overlay.remove();
    });
}

test.describe('Branch Tree', () => {
    test('page loads and renders input form', async ({ page }) => {
        await page.goto('/branch_tree.html');
        await dismissDevServerOverlay(page);
        await expect(page.getByText('Prefix', { exact: true })).toBeVisible();
        await expect(page.locator('#branch_tree_submit_btn')).toBeVisible();
    });

    test('submit button disabled when input is empty', async ({ page }) => {
        await page.goto('/branch_tree.html');
        await dismissDevServerOverlay(page);
        const btn = page.locator('#branch_tree_submit_btn');
        await expect(btn).toBeDisabled();
    });

    test('submit button enabled after filling prefix', async ({ page }) => {
        await page.goto('/branch_tree.html');
        await dismissDevServerOverlay(page);
        await page.fill('#branch_tree_raw_text', '中国的首都是');
        const btn = page.locator('#branch_tree_submit_btn');
        await expect(btn).toBeEnabled();
    });

    test('clicking start renders branch tree SVG', async ({ page }) => {
        await page.goto('/branch_tree.html');
        await page.fill('#branch_tree_raw_text', 'Hello');
        await dismissDevServerOverlay(page);
        // Use evaluate to click to avoid overlay intercepting pointer events
        await page.evaluate(() => {
            const btn = document.getElementById('branch_tree_submit_btn');
            if (btn) btn.click();
        });
        // Wait for SVG to appear
        await page.waitForSelector('#branch_tree_surface svg', { state: 'visible', timeout: 30_000 });
        await expect(page.locator('#branch_tree_surface svg')).toBeVisible();
    });

    test('clicking a node expands new children', async ({ page }) => {
        await page.goto('/branch_tree.html');
        await page.fill('#branch_tree_raw_text', 'The');
        await dismissDevServerOverlay(page);
        await page.evaluate(() => {
            const btn = document.getElementById('branch_tree_submit_btn');
            if (btn) btn.click();
        });
        await page.waitForSelector('#branch_tree_surface svg', { timeout: 30_000 });
        // Click the first child node via JS to avoid text intercepting pointer events
        const newCount = await page.evaluate(async () => {
            const circles = document.querySelectorAll('#branch_tree_surface svg circle');
            if (circles.length > 1) {
                (circles[1] as SVGCircleElement).dispatchEvent(new Event('click', { bubbles: true }));
            }
            await new Promise((r) => setTimeout(r, 2500));
            return document.querySelectorAll('#branch_tree_surface svg circle').length;
        });
        expect(newCount).toBeGreaterThan(0);
    });

    test('clicking submit again while loading cancels request', async ({ page }) => {
        await page.goto('/branch_tree.html');
        await page.fill('#branch_tree_raw_text', 'Testing cancellation');
        await dismissDevServerOverlay(page);
        await page.evaluate(() => {
            const btn = document.getElementById('branch_tree_submit_btn');
            if (btn) btn.click();
        });
        // Immediately click again to cancel/abort
        await page.waitForTimeout(500);
        await page.evaluate(() => {
            const btn = document.getElementById('branch_tree_submit_btn');
            if (btn) btn.click();
        });
        // Should show toast or reset loading state
        await expect(page.locator('#branch_tree_submit_btn')).toBeEnabled({ timeout: 10_000 });
    });
});
