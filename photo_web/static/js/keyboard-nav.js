/**
 * AutoAlbum Keyboard Navigation
 * 键盘快捷键导航
 */

class KeyboardNav {
  constructor(options = {}) {
    this.options = {
      // 是否启用全局快捷键
      enabled: options.enabled !== false,
      // 是否显示快捷键提示
      showHints: options.showHints || false,
      // 自定义快捷键映射
      shortcuts: {
        // 导航
        home: 'g',
        gallery: 'h',
        browse: 'b',
        persons: 'p',
        search: 'f',
        // 灯箱
        lightboxNext: 'ArrowRight',
        lightboxPrev: 'ArrowLeft',
        lightboxClose: 'Escape',
        lightboxZoomIn: '+',
        lightboxZoomOut: '-',
        lightboxFavorite: 'f',
        lightboxInfo: 'i',
        // 其他
        help: '?',
      },
      onShortcut: options.onShortcut || null,
    };

    this.hintsElement = null;
    this.init();
  }

  init() {
    if (!this.options.enabled) return;

    document.addEventListener('keydown', (e) => this.handleKeydown(e));

    if (this.options.showHints) {
      this.createHintsElement();
    }
  }

  handleKeydown(e) {
    // 忽略表单内的按键
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
      // 但 ESC 仍然有效
      if (e.key === 'Escape') {
        e.target.blur();
      }
      return;
    }

    const key = e.key;
    const shortcuts = this.options.shortcuts;

    // 检查是否匹配快捷键
    for (const [action, shortcut] of Object.entries(shortcuts)) {
      if (key === shortcut || (key.toLowerCase() === shortcut.toLowerCase())) {
        // 检查是否需要修饰键
        if (shortcut === key || shortcut.toLowerCase() === key.toLowerCase()) {
          e.preventDefault();
          this.executeAction(action);
          break;
        }
      }
    }
  }

  executeAction(action) {
    // 灯箱状态优先
    const lightboxEl = document.querySelector('.lightbox.active');
    if (lightboxEl && window.lightbox?.isOpen) {
      switch (action) {
        case 'lightboxNext':
          window.lightbox.next();
          break;
        case 'lightboxPrev':
          window.lightbox.prev();
          break;
        case 'lightboxClose':
          window.lightbox.close();
          break;
        case 'lightboxZoomIn':
          window.lightbox.zoomIn();
          break;
        case 'lightboxZoomOut':
          window.lightbox.zoomOut();
          break;
        case 'lightboxFavorite':
          window.lightbox.toggleFavorite();
          break;
        case 'lightboxInfo':
          window.lightbox.toggleInfoPanel();
          break;
      }
      return;
    }

    // 全局导航
    switch (action) {
      case 'home':
        window.location.href = '/';
        break;
      case 'gallery':
        window.location.href = '/gallery';
        break;
      case 'browse':
        window.location.href = '/browse';
        break;
      case 'persons':
        window.location.href = '/persons';
        break;
      case 'search':
        const searchInput = document.querySelector('input[name="q"]');
        if (searchInput) {
          searchInput.focus();
        } else {
          window.location.href = '/search';
        }
        break;
      case 'help':
        this.toggleHints();
        break;
    }

    // 自定义回调
    if (this.options.onShortcut) {
      this.options.onShortcut(action);
    }
  }

  createHintsElement() {
    const hints = `
      <div class="keyboard-hints" id="keyboard-hints" style="display: none;">
        <div class="keyboard-hints-header">
          <span>键盘快捷键</span>
          <button class="keyboard-hints-close">&times;</button>
        </div>
        <div class="keyboard-hints-content">
          <div class="hint-group">
            <h4>导航</h4>
            <div class="hint-item">
              <kbd>G</kbd> 首页
            </div>
            <div class="hint-item">
              <kbd>H</kbd> 画廊
            </div>
            <div class="hint-item">
              <kbd>B</kbd> 浏览
            </div>
            <div class="hint-item">
              <kbd>P</kbd> 人物
            </div>
            <div class="hint-item">
              <kbd>F</kbd> 搜索
            </div>
          </div>
          <div class="hint-group">
            <h4>灯箱</h4>
            <div class="hint-item">
              <kbd>←</kbd> / <kbd>→</kbd> 上一张/下一张
            </div>
            <div class="hint-item">
              <kbd>Esc</kbd> 关闭
            </div>
            <div class="hint-item">
              <kbd>+</kbd> / <kbd>-</kbd> 放大/缩小
            </div>
            <div class="hint-item">
              <kbd>F</kbd> 收藏
            </div>
            <div class="hint-item">
              <kbd>I</kbd> 信息
            </div>
          </div>
          <div class="hint-group">
            <h4>其他</h4>
            <div class="hint-item">
              <kbd>?</kbd> 显示快捷键
            </div>
          </div>
        </div>
      </div>
    `;

    document.body.insertAdjacentHTML('beforeend', hints);

    this.hintsElement = document.getElementById('keyboard-hints');
    this.hintsElement.querySelector('.keyboard-hints-close').addEventListener('click', () => {
      this.hideHints();
    });

    // 点击背景关闭
    this.hintsElement.addEventListener('click', (e) => {
      if (e.target === this.hintsElement) {
        this.hideHints();
      }
    });
  }

  toggleHints() {
    if (this.hintsElement) {
      const isVisible = this.hintsElement.style.display === 'block';
      this.hintsElement.style.display = isVisible ? 'none' : 'block';
    }
  }

  showHints() {
    if (this.hintsElement) {
      this.hintsElement.style.display = 'block';
    }
  }

  hideHints() {
    if (this.hintsElement) {
      this.hintsElement.style.display = 'none';
    }
  }
}

// 创建全局实例
const keyboardNav = new KeyboardNav({
  enabled: true,
  showHints: false,
});

// 导出
window.KeyboardNav = KeyboardNav;
window.keyboardNav = keyboardNav;
