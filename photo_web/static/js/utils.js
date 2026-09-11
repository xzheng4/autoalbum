/**
 * AutoAlbum Utils
 * 工具函数库
 */

const Utils = {
  /**
   * 防抖函数
   */
  debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  },

  /**
   * 节流函数
   */
  throttle(func, limit) {
    let inThrottle;
    return function(...args) {
      if (!inThrottle) {
        func.apply(this, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  },

  /**
   * 格式化日期
   */
  formatDate(date, format = 'YYYY-MM-DD') {
    if (!date) return '';

    const d = new Date(date);
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    const hours = String(d.getHours()).padStart(2, '0');
    const minutes = String(d.getMinutes()).padStart(2, '0');
    const seconds = String(d.getSeconds()).padStart(2, '0');

    return format
      .replace('YYYY', year)
      .replace('MM', month)
      .replace('DD', day)
      .replace('HH', hours)
      .replace('mm', minutes)
      .replace('ss', seconds);
  },

  /**
   * 格式化相对时间
   */
  formatRelativeTime(date) {
    if (!date) return '';

    const now = new Date();
    const then = new Date(date);
    const diffMs = now - then;
    const diffSecs = Math.floor(diffMs / 1000);
    const diffMins = Math.floor(diffSecs / 60);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);
    const diffYears = now.getFullYear() - then.getFullYear();

    if (diffSecs < 60) {
      return '刚刚';
    } else if (diffMins < 60) {
      return `${diffMins} 分钟前`;
    } else if (diffHours < 24) {
      return `${diffHours} 小时前`;
    } else if (diffDays < 30) {
      return `${diffDays} 天前`;
    } else if (diffYears === 0) {
      return this.formatDate(date, 'MM-DD');
    } else {
      return this.formatDate(date, 'YYYY-MM-DD');
    }
  },

  /**
   * 格式化文件大小
   */
  formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  },

  /**
   * 格式化像素尺寸
   */
  formatDimensions(width, height) {
    if (!width || !height) return '';
    return `${width} × ${height}`;
  },

  /**
   * 截断文本
   */
  truncate(str, length = 100, suffix = '...') {
    if (!str) return '';
    if (str.length <= length) return str;
    return str.substring(0, length) + suffix;
  },

  /**
   * 转义 HTML
   */
  escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  },

  /**
   * 生成唯一 ID
   */
  generateId() {
    return 'id_' + Math.random().toString(36).substr(2, 9);
  },

  /**
   * 检查元素是否在视口内
   */
  isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
      rect.top >= 0 &&
      rect.left >= 0 &&
      rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
      rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
  },

  /**
   * 平滑滚动到元素
   */
  scrollToElement(element, offset = 0) {
    if (!element) return;

    const top = element.getBoundingClientRect().top + window.pageYOffset - offset;

    window.scrollTo({
      top,
      behavior: 'smooth',
    });
  },

  /**
   * 复制文本到剪贴板
   */
  async copyToClipboard(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (err) {
      // 回退方法
      const textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      try {
        document.execCommand('copy');
        return true;
      } catch {
        return false;
      } finally {
        document.body.removeChild(textarea);
      }
    }
  },

  /**
   * 下载文件
   */
  downloadFile(url, filename) {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || 'download';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  },

  /**
   * 分享
   */
  async share(options = {}) {
    if (navigator.share) {
      try {
        await navigator.share({
          title: options.title || '分享',
          text: options.text || '',
          url: options.url || window.location.href,
        });
        return true;
      } catch (err) {
        if (err.name !== 'AbortError') {
          console.error('Share failed:', err);
        }
        return false;
      }
    } else {
      // 回退：复制链接
      await this.copyToClipboard(options.url || window.location.href);
      alert('链接已复制到剪贴板');
      return true;
    }
  },

  /**
   * 懒加载图片
   */
  lazyLoadImages(container) {
    const images = container.querySelectorAll('img[data-src]:not([src])');

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const img = entry.target;
          img.src = img.dataset.src;
          img.classList.add('loaded');
          observer.unobserve(img);
        }
      });
    }, {
      rootMargin: '50px 0px',
    });

    images.forEach((img) => observer.observe(img));
  },

  /**
   * 预加载图片
   */
  preloadImages(urls) {
    urls.forEach((url) => {
      const img = new Image();
      img.src = url;
    });
  },

  /**
   * 获取 URL 参数
   */
  getUrlParam(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
  },

  /**
   * 设置 URL 参数
   */
  setUrlParam(name, value) {
    const url = new URL(window.location.href);
    url.searchParams.set(name, value);
    window.history.pushState({}, '', url);
  },

  /**
   * 删除 URL 参数
   */
  removeUrlParam(name) {
    const url = new URL(window.location.href);
    url.searchParams.delete(name);
    window.history.pushState({}, '', url);
  },

  /**
   * 本地存储封装
   */
  storage: {
    get(key) {
      try {
        const item = localStorage.getItem(key);
        return item ? JSON.parse(item) : null;
      } catch (e) {
        return null;
      }
    },
    set(key, value) {
      try {
        localStorage.setItem(key, JSON.stringify(value));
        return true;
      } catch (e) {
        return false;
      }
    },
    remove(key) {
      try {
        localStorage.removeItem(key);
        return true;
      } catch (e) {
        return false;
      }
    },
    clear() {
      localStorage.clear();
    },
  },

  /**
   * 事件总线
   */
  bus: {
    events: {},
    on(event, callback) {
      if (!this.events[event]) {
        this.events[event] = [];
      }
      this.events[event].push(callback);
    },
    off(event, callback) {
      if (!this.events[event]) return;
      const index = this.events[event].indexOf(callback);
      if (index > -1) {
        this.events[event].splice(index, 1);
      }
    },
    emit(event, data) {
      if (!this.events[event]) return;
      this.events[event].forEach((callback) => callback(data));
    },
  },
};

// 导出
window.Utils = Utils;

// 快捷方法
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => document.querySelectorAll(selector);

// 导出快捷方法
window.$ = $;
window.$$ = $$;
