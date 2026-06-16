/**
 * 极简导航首页：Swiss Style 模块网格
 * 主题与语言与 analysis 等页通过 localStorage 一致。
 */
import '../../css/pages/home.scss';

import { initThemeManager, type Theme } from '../../shared/ui/theme';
import { initLanguageManager } from '../../shared/ui/language';
import { getCurrentLanguage, initI18n } from '../../shared/lang/i18n-lite';
import { AdminManager } from '../../shared/cross/adminManager';
import { SettingsMenuManager } from '../../shared/cross/settingsMenuManager';
import { initializeCommonApp } from '../../shared/bootstrap';
import URLHandler from '../../shared/core/URLHandler';

initI18n();

const apiPrefix = URLHandler.parameters['api'] || '';
const { api } = initializeCommonApp(apiPrefix);
const adminManager = AdminManager.getInstance();
api.setAdminToken(adminManager.isInAdminMode() ? adminManager.getAdminToken() : null);

const themeManager = initThemeManager({}, '#theme_dropdown');
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

document.documentElement.lang = getCurrentLanguage() === 'zh' ? 'zh-CN' : 'en';
