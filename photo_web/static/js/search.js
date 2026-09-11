/**
 * AutoAlbum Search
 * 即时搜索与过滤
 */

class SearchManager {
  constructor(options = {}) {
    this.options = {
      debounceDelay: options.debounceDelay || 300,
      minLength: options.minLength || 2,
      searchUrl: options.searchUrl || '/api/search',
      onResults: options.onResults || null,
      onNoResults: options.onNoResults || null,
    };

    this.searchTimer = null;
    this.lastQuery = '';
    this.init();
  }

  init() {
    // 绑定搜索框事件
    this.bindSearchBoxes();

    // 绑定过滤芯片事件
    this.bindFilterChips();

    // 绑定快捷搜索（按 / 聚焦搜索框）
    this.bindQuickSearch();
  }

  bindSearchBoxes() {
    const searchBoxes = document.querySelectorAll('input[data-search="true"], input[name="q"]');
    searchBoxes.forEach((input) => {
      input.addEventListener('input', (e) => {
        const query = e.target.value.trim();
        this.onSearchInput(query, input);
      });

      // 支持按 Enter 提交
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          this.submitSearch(input.value.trim());
        }
      });
    });
  }

  bindFilterChips() {
    const filterChips = document.querySelectorAll('.filter-chip');
    filterChips.forEach((chip) => {
      chip.addEventListener('click', (e) => {
        e.preventDefault();
        const value = chip.dataset.value;
        const type = chip.dataset.type;

        if (chip.classList.contains('active')) {
          this.removeFilter(type, value);
        } else {
          this.addFilter(type, value);
        }
      });
    });

    // 清除所有过滤
    const clearAll = document.querySelector('.filter-chips-clear');
    if (clearAll) {
      clearAll.addEventListener('click', () => {
        this.clearAllFilters();
      });
    }
  }

  bindQuickSearch() {
    document.addEventListener('keydown', (e) => {
      // 按 / 聚焦搜索框
      if (e.key === '/' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
        e.preventDefault();
        const searchInput = document.querySelector('input[name="q"]');
        if (searchInput) {
          searchInput.focus();
        }
      }
    });
  }

  onSearchInput(query, input) {
    // 防抖
    clearTimeout(this.searchTimer);

    // 如果长度不够，清除结果
    if (query.length < this.options.minLength) {
      this.hideSuggestions();
      return;
    }

    this.searchTimer = setTimeout(() => {
      this.search(query, input);
    }, this.options.debounceDelay);
  }

  async search(query, input) {
    if (query === this.lastQuery) return;
    this.lastQuery = query;

    try {
      const response = await fetch(`${this.options.searchUrl}?q=${encodeURIComponent(query)}`);
      const results = await response.json();

      this.showSuggestions(results, input);

      if (this.options.onResults) {
        this.options.onResults(results);
      }
    } catch (error) {
      console.error('Search error:', error);
    }
  }

  showSuggestions(results, input) {
    // 移除已有的建议下拉框
    this.hideSuggestions();

    if (!results || results.length === 0) {
      if (this.options.onNoResults) {
        this.options.onNoResults();
      }
      return;
    }

    const dropdown = document.createElement('div');
    dropdown.className = 'search-suggestions';
    dropdown.innerHTML = `
      <div class="search-suggestions-header">
        <span>${results.length} 个结果</span>
        <button class="search-suggestions-close">&times;</button>
      </div>
      <div class="search-suggestions-list">
        ${results.map((item) => `
          <a href="/image/${item.id}" class="suggestion-item">
            ${item.thumbnail ? `<img src="${item.thumbnail}" alt="">` : ''}
            <div class="suggestion-content">
              ${item.ocr_text ? `<div class="suggestion-ocr">${this.truncate(item.ocr_text, 50)}</div>` : ''}
              ${item.category ? `<div class="suggestion-category">${item.category}</div>` : ''}
              ${item.captured_at ? `<div class="suggestion-date">${item.captured_at}</div>` : ''}
            </div>
          </a>
        `).join('')}
      </div>
    `;

    // 关闭按钮
    dropdown.querySelector('.search-suggestions-close').addEventListener('click', () => {
      this.hideSuggestions();
    });

    // 定位在输入框下方
    input.parentNode.appendChild(dropdown);

    // 点击外部关闭
    document.addEventListener('click', this.handleOutsideClick.bind(this));
  }

  hideSuggestions() {
    const suggestions = document.querySelectorAll('.search-suggestions');
    suggestions.forEach((el) => el.remove());
  }

  handleOutsideClick(e) {
    if (!e.target.closest('.search-suggestions') && !e.target.closest('input[name="q"]')) {
      this.hideSuggestions();
      document.removeEventListener('click', this.handleOutsideClick.bind(this));
    }
  }

  submitSearch(query) {
    if (query) {
      window.location.href = `/search?q=${encodeURIComponent(query)}`;
    }
  }

  // ========== 过滤芯片管理 ==========

  addFilter(type, value) {
    const chip = document.querySelector(`.filter-chip[data-type="${type}"][data-value="${value}"]`);
    if (chip) {
      chip.classList.add('active');
    }

    // 更新 URL 参数
    this.updateUrlFilter(type, value);

    // 触发自定义事件
    document.dispatchEvent(new CustomEvent('filter-add', {
      detail: { type, value }
    }));
  }

  removeFilter(type, value) {
    const chip = document.querySelector(`.filter-chip[data-type="${type}"][data-value="${value}"]`);
    if (chip) {
      chip.classList.remove('active');
    }

    // 更新 URL 参数
    this.removeUrlFilter(type, value);

    // 触发自定义事件
    document.dispatchEvent(new CustomEvent('filter-remove', {
      detail: { type, value }
    }));
  }

  clearAllFilters() {
    const activeChips = document.querySelectorAll('.filter-chip.active');
    activeChips.forEach((chip) => {
      chip.classList.remove('active');
    });

    // 清除 URL 参数
    window.history.pushState({}, '', window.location.pathname);

    // 触发自定义事件
    document.dispatchEvent(new CustomEvent('filters-clear'));
  }

  updateUrlFilter(type, value) {
    const url = new URL(window.location.href);
    url.searchParams.set(type, value);
    window.history.pushState({}, '', url);
  }

  removeUrlFilter(type, value) {
    const url = new URL(window.location.href);
    url.searchParams.delete(type, value);
    window.history.pushState({}, '', url);
  }

  // ========== 工具函数 ==========

  truncate(str, len) {
    if (!str) return '';
    if (str.length <= len) return str;
    return str.substring(0, len) + '...';
  }

  // ========== 即时搜索高亮 ==========

  highlightMatches(text, query) {
    if (!text || !query) return text;

    const regex = new RegExp(`(${this.escapeRegex(query)})`, 'gi');
    return text.replace(regex, '<mark>$1</mark>');
  }

  escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
}

// 创建全局实例
const searchManager = new SearchManager({
  debounceDelay: 300,
  minLength: 2,
});

// 导出
window.SearchManager = SearchManager;
window.searchManager = searchManager;
