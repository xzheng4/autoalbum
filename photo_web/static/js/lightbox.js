/**
 * AutoAlbum Lightbox
 * 灯箱查看器 - 全屏图片浏览
 */

class Lightbox {
  constructor(options = {}) {
    this.options = {
      showCounter: options.showCounter !== false,
      showThumbnails: options.showThumbnails || false,
      showInfoPanel: options.showInfoPanel !== false,
      loop: options.loop !== false,
      autoHideToolbar: options.autoHideToolbar !== false,
      toolbarTimeout: options.toolbarTimeout || 3000,
      onOpen: options.onOpen || null,
      onClose: options.onClose || null,
      onChange: options.onChange || null,
    };

    this.currentIndex = 0;
    this.images = [];
    this.isOpen = false;
    this.isLoading = false;
    this.toolbarTimer = null;
    this.infoPanelOpen = false;

    this.element = null;
    this.init();
  }

  init() {
    this.render();
    this.bindEvents();
  }

  render() {
    // 创建灯箱 DOM
    const template = `
      <div class="lightbox" role="dialog" aria-modal="true">
        <div class="lightbox-backdrop"></div>

        <div class="lightbox-content">
          <!-- 顶部工具栏 -->
          <div class="lightbox-toolbar">
            <div class="lightbox-toolbar-left">
              ${this.options.showCounter ? '<span class="lightbox-counter">1 / 1</span>' : ''}
            </div>
            <div class="lightbox-toolbar-right">
              <button class="lightbox-close" aria-label="关闭">
                <i class="bi bi-x-lg"></i>
              </button>
            </div>
          </div>

          <!-- 图片容器 -->
          <div class="lightbox-image-container">
            <div class="lightbox-image-wrapper">
              <img class="lightbox-image" src="" alt="">
            </div>

            <!-- 导航按钮 -->
            <button class="lightbox-nav lightbox-prev" aria-label="上一张">
              <i class="bi bi-chevron-left"></i>
            </button>
            <button class="lightbox-nav lightbox-next" aria-label="下一张">
              <i class="bi bi-chevron-right"></i>
            </button>
          </div>

          <!-- 底部工具栏 -->
          <div class="lightbox-footer">
            <div class="lightbox-actions">
              <button class="lightbox-action-btn btn-favorite" aria-label="收藏">
                <i class="bi bi-heart"></i>
              </button>
              <button class="lightbox-action-btn btn-info" aria-label="信息">
                <i class="bi bi-info-circle"></i>
              </button>
              <button class="lightbox-action-btn btn-download" aria-label="下载">
                <i class="bi bi-download"></i>
              </button>
              <button class="lightbox-action-btn btn-share" aria-label="分享">
                <i class="bi bi-share"></i>
              </button>
            </div>
            <div class="lightbox-info">
              <div class="lightbox-caption"></div>
              <div class="lightbox-meta"></div>
            </div>
            <div class="lightbox-actions">
              <button class="lightbox-action-btn btn-zoom-in" aria-label="放大">
                <i class="bi bi-plus-lg"></i>
              </button>
              <button class="lightbox-action-btn btn-zoom-out" aria-label="缩小">
                <i class="bi bi-dash-lg"></i>
              </button>
            </div>
          </div>

          <!-- 缩略图导航 -->
          ${this.options.showThumbnails ? '<div class="lightbox-thumbnails"></div>' : ''}

          <!-- 信息面板 -->
          <div class="lightbox-info-panel">
            <div class="lightbox-info-panel-header">
              <span class="lightbox-info-panel-title">详情</span>
              <button class="lightbox-info-panel-close">
                <i class="bi bi-x-lg"></i>
              </button>
            </div>
            <div class="lightbox-info-panel-content">
              <!-- 动态内容 -->
            </div>
          </div>

          <!-- 加载状态 -->
          <div class="lightbox-loading" style="display: none;">
            <div class="lightbox-loading-spinner"></div>
            <div class="lightbox-loading-text">加载中...</div>
          </div>

          <!-- 错误状态 -->
          <div class="lightbox-error" style="display: none;">
            <div class="lightbox-error-icon">⚠️</div>
            <div class="lightbox-error-text">图片加载失败</div>
            <div class="lightbox-error-subtext">请检查图片链接是否有效</div>
          </div>
        </div>
      </div>
    `;

    document.body.insertAdjacentHTML('beforeend', template);
    this.element = document.querySelector('.lightbox');

    // 缓存元素引用
    this.elements = {
      backdrop: this.element.querySelector('.lightbox-backdrop'),
      content: this.element.querySelector('.lightbox-content'),
      toolbar: this.element.querySelector('.lightbox-toolbar'),
      footer: this.element.querySelector('.lightbox-footer'),
      image: this.element.querySelector('.lightbox-image'),
      imageWrapper: this.element.querySelector('.lightbox-image-wrapper'),
      prev: this.element.querySelector('.lightbox-prev'),
      next: this.element.querySelector('.lightbox-next'),
      close: this.element.querySelector('.lightbox-close'),
      counter: this.element.querySelector('.lightbox-counter'),
      caption: this.element.querySelector('.lightbox-caption'),
      meta: this.element.querySelector('.lightbox-meta'),
      thumbnails: this.element.querySelector('.lightbox-thumbnails'),
      infoPanel: this.element.querySelector('.lightbox-info-panel'),
      infoPanelContent: this.element.querySelector('.lightbox-info-panel-content'),
      infoPanelClose: this.element.querySelector('.lightbox-info-panel-close'),
      loading: this.element.querySelector('.lightbox-loading'),
      error: this.element.querySelector('.lightbox-error'),
      btnFavorite: this.element.querySelector('.btn-favorite'),
      btnInfo: this.element.querySelector('.btn-info'),
      btnDownload: this.element.querySelector('.btn-download'),
      btnShare: this.element.querySelector('.btn-share'),
      btnZoomIn: this.element.querySelector('.btn-zoom-in'),
      btnZoomOut: this.element.querySelector('.btn-zoom-out'),
    };
  }

  bindEvents() {
    // 关闭
    this.elements.backdrop.addEventListener('click', () => this.close());
    this.elements.close.addEventListener('click', () => this.close());
    this.elements.infoPanelClose.addEventListener('click', () => this.toggleInfoPanel(false));

    // 导航
    this.elements.prev.addEventListener('click', (e) => {
      e.stopPropagation();
      this.prev();
    });
    this.elements.next.addEventListener('click', (e) => {
      e.stopPropagation();
      this.next();
    });

    // 动作按钮
    this.elements.btnFavorite.addEventListener('click', (e) => {
      e.stopPropagation();
      this.toggleFavorite();
    });
    this.elements.btnInfo.addEventListener('click', (e) => {
      e.stopPropagation();
      this.toggleInfoPanel();
    });
    this.elements.btnDownload.addEventListener('click', (e) => {
      e.stopPropagation();
      this.download();
    });
    this.elements.btnShare.addEventListener('click', (e) => {
      e.stopPropagation();
      this.share();
    });
    this.elements.btnZoomIn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.zoomIn();
    });
    this.elements.btnZoomOut.addEventListener('click', (e) => {
      e.stopPropagation();
      this.zoomOut();
    });

    // 键盘事件
    document.addEventListener('keydown', (e) => this.handleKeydown(e));

    // 触摸事件（用于移动端手势）
    this.setupTouch();

    // 鼠标滚轮缩放
    this.elements.content.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (e.deltaY < 0) {
        this.zoomIn();
      } else {
        this.zoomOut();
      }
    }, { passive: false });

    // 自动隐藏工具栏
    if (this.options.autoHideToolbar) {
      this.elements.content.addEventListener('mousemove', () => {
        this.showToolbar();
        this.resetToolbarTimeout();
      });
    }
  }

  handleKeydown(e) {
    if (!this.isOpen) return;

    switch (e.key) {
      case 'Escape':
        this.close();
        break;
      case 'ArrowLeft':
        this.prev();
        break;
      case 'ArrowRight':
        this.next();
        break;
      case '+':
      case '=':
        this.zoomIn();
        break;
      case '-':
        this.zoomOut();
        break;
      case 'f':
        this.toggleFavorite();
        break;
      case 'i':
        this.toggleInfoPanel();
        break;
    }
  }

  setupTouch() {
    let touchStartX = 0;
    let touchStartY = 0;
    let lastTouchX = 0;
    let lastTouchY = 0;

    this.elements.content.addEventListener('touchstart', (e) => {
      touchStartX = e.touches[0].clientX;
      touchStartY = e.touches[0].clientY;
      lastTouchX = touchStartX;
      lastTouchY = touchStartY;
    }, { passive: true });

    this.elements.content.addEventListener('touchmove', (e) => {
      const touchX = e.touches[0].clientX;
      const touchY = e.touches[0].clientY;
      const deltaX = touchX - lastTouchX;
      const deltaY = touchY - lastTouchY;

      // 处理拖拽平移（当图片放大时）
      if (this.currentScale > 1) {
        e.preventDefault();
        // 这里可以添加平移逻辑
      }

      lastTouchX = touchX;
      lastTouchY = touchY;
    }, { passive: false });

    this.elements.content.addEventListener('touchend', (e) => {
      const touchEndX = e.changedTouches[0].clientX;
      const deltaX = touchEndX - touchStartX;

      // 左右滑动切换图片
      if (Math.abs(deltaX) > 100) {
        if (deltaX > 0) {
          this.prev();
        } else {
          this.next();
        }
      }
    }, { passive: true });

    // 双指缩放
    let initialPinchDistance = null;
    let initialScale = 1;

    this.elements.content.addEventListener('touchstart', (e) => {
      if (e.touches.length === 2) {
        initialPinchDistance = this.getPinchDistance(e.touches);
        initialScale = this.currentScale || 1;
      }
    }, { passive: true });

    this.elements.content.addEventListener('touchmove', (e) => {
      if (e.touches.length === 2 && initialPinchDistance !== null) {
        e.preventDefault();
        const currentDistance = this.getPinchDistance(e.touches);
        const scale = initialScale * (currentDistance / initialPinchDistance);
        this.setScale(Math.min(Math.max(scale, 1), 4));
      }
    }, { passive: false });
  }

  getPinchDistance(touches) {
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    return Math.sqrt(dx * dx + dy * dy);
  }

  // ========== 公共方法 ==========

  open(index = 0, images = []) {
    if (images.length > 0) {
      this.images = images;
    }

    if (this.images.length === 0) {
      console.error('No images to display');
      return;
    }

    this.currentIndex = Math.max(0, Math.min(index, this.images.length - 1));
    this.isOpen = true;
    this.currentScale = 1;

    // 显示灯箱
    this.element.classList.add('active');
    document.body.style.overflow = 'hidden';

    // 加载图片
    this.loadImage(this.currentIndex);

    // 更新 UI
    this.updateCounter();
    this.updateNavigation();
    this.updateActions();

    if (this.options.onOpen) {
      this.options.onOpen(this.currentIndex, this.images[this.currentIndex]);
    }
  }

  close() {
    if (!this.isOpen) return;

    this.isOpen = false;
    this.element.classList.remove('active');
    document.body.style.overflow = '';

    // 清空图片
    this.elements.image.src = '';

    if (this.options.onClose) {
      this.options.onClose(this.currentIndex, this.images[this.currentIndex]);
    }
  }

  prev() {
    if (this.images.length === 0) return;

    if (this.currentIndex === 0) {
      if (this.options.loop) {
        this.currentIndex = this.images.length - 1;
      } else {
        return;
      }
    } else {
      this.currentIndex--;
    }

    this.loadImage(this.currentIndex);
    this.updateCounter();
    this.updateNavigation();
    this.updateActions();

    if (this.options.onChange) {
      this.options.onChange(this.currentIndex, this.images[this.currentIndex]);
    }
  }

  next() {
    if (this.images.length === 0) return;

    if (this.currentIndex === this.images.length - 1) {
      if (this.options.loop) {
        this.currentIndex = 0;
      } else {
        return;
      }
    } else {
      this.currentIndex++;
    }

    this.loadImage(this.currentIndex);
    this.updateCounter();
    this.updateNavigation();
    this.updateActions();

    if (this.options.onChange) {
      this.options.onChange(this.currentIndex, this.images[this.currentIndex]);
    }
  }

  // ========== 图片加载 ==========

  loadImage(index) {
    const image = this.images[index];
    if (!image) return;

    this.isLoading = true;
    this.showLoading();

    const img = new Image();
    img.onload = () => {
      this.elements.image.src = image.src || image.url;
      this.elements.image.alt = image.alt || image.caption || '';
      this.hideLoading();
      this.isLoading = false;

      // 更新信息
      if (image.caption) {
        this.elements.caption.textContent = image.caption;
      }
      if (image.meta) {
        this.elements.meta.textContent = image.meta;
      }
    };

    img.onerror = () => {
      this.showLoading();
      this.elements.error.style.display = 'flex';
      this.isLoading = false;
    };

    img.src = image.src || image.url;
  }

  showLoading() {
    this.elements.loading.style.display = 'flex';
    this.elements.error.style.display = 'none';
  }

  hideLoading() {
    this.elements.loading.style.display = 'none';
  }

  // ========== UI 更新 ==========

  updateCounter() {
    if (this.elements.counter) {
      this.elements.counter.textContent = `${this.currentIndex + 1} / ${this.images.length}`;
    }
  }

  updateNavigation() {
    // 更新导航按钮状态
    if (!this.options.loop) {
      this.elements.prev.style.opacity = this.currentIndex === 0 ? '0.3' : '1';
      this.elements.next.style.opacity = this.currentIndex === this.images.length - 1 ? '0.3' : '1';
    }
  }

  updateActions() {
    // 更新收藏按钮状态
    const image = this.images[this.currentIndex];
    if (image && image.isFavorite) {
      this.elements.btnFavorite.classList.add('active');
      this.elements.btnFavorite.innerHTML = '<i class="bi bi-heart-fill"></i>';
    } else {
      this.elements.btnFavorite.classList.remove('active');
      this.elements.btnFavorite.innerHTML = '<i class="bi bi-heart"></i>';
    }
  }

  // ========== 工具栏 ==========

  showToolbar() {
    this.elements.toolbar.style.opacity = '1';
    this.elements.footer.style.opacity = '1';
  }

  hideToolbar() {
    this.elements.toolbar.style.opacity = '0';
    this.elements.footer.style.opacity = '0';
  }

  resetToolbarTimeout() {
    if (this.toolbarTimer) {
      clearTimeout(this.toolbarTimer);
    }
    this.toolbarTimer = setTimeout(() => {
      this.hideToolbar();
    }, this.options.toolbarTimeout);
  }

  // ========== 信息面板 ==========

  toggleInfoPanel(force) {
    this.infoPanelOpen = force !== undefined ? force : !this.infoPanelOpen;

    if (this.infoPanelOpen) {
      this.elements.infoPanel.classList.add('active');
      this.elements.btnInfo.classList.add('active');
      this.renderInfoPanel();
    } else {
      this.elements.infoPanel.classList.remove('active');
      this.elements.btnInfo.classList.remove('active');
    }
  }

  renderInfoPanel() {
    const image = this.images[this.currentIndex];
    if (!image) return;

    let html = '';

    // OCR 文字
    if (image.ocrText) {
      html += `
        <div class="info-section">
          <div class="info-section-title"><i class="bi bi-text-paragraph"></i> OCR 文字</div>
          <div class="info-section-content">
            <div class="ocr-text">${this.escapeHtml(image.ocrText)}</div>
          </div>
        </div>
      `;
    }

    // 场景描述
    if (image.sceneDescription) {
      html += `
        <div class="info-section">
          <div class="info-section-title"><i class="bi bi-image"></i> 场景描述</div>
          <div class="info-section-content">${image.sceneDescription}</div>
        </div>
      `;
    }

    // 分类标签
    if (image.category || image.objects || image.mood) {
      html += `
        <div class="info-section">
          <div class="info-section-title"><i class="bi bi-tags"></i> 标签</div>
          <div class="info-section-content">
            <div class="tag-list">
      `;

      if (image.category) {
        html += `<span class="tag-item">${this.escapeHtml(image.category)}</span>`;
      }
      if (image.mood) {
        html += `<span class="tag-item"><i class="bi bi-heart"></i> ${this.escapeHtml(image.mood)}</span>`;
      }
      if (image.objects) {
        image.objects.forEach(obj => {
          html += `<span class="tag-item">${this.escapeHtml(obj)}</span>`;
        });
      }

      html += `</div></div></div>`;
    }

    // 人物
    if (image.persons && image.persons.length > 0) {
      html += `
        <div class="info-section">
          <div class="info-section-title"><i class="bi bi-people"></i> 人物</div>
          <div class="info-section-content">
            <div class="person-list">
      `;

      image.persons.forEach(person => {
        html += `
          <div class="person-item">
            ${person.avatar ? `<img src="${person.avatar}" class="person-avatar" alt="${person.name}">` : ''}
            <div>
              <div class="person-name">${this.escapeHtml(person.name)}</div>
              ${person.confidence ? `<div class="person-confidence">${Math.round(person.confidence * 100)}% 匹配</div>` : ''}
            </div>
          </div>
        `;
      });

      html += `</div></div></div>`;
    }

    // EXIF 数据
    if (image.exif) {
      html += `
        <div class="info-section">
          <div class="info-section-title"><i class="bi bi-camera"></i> 拍摄参数</div>
          <div class="info-section-content">
            <div class="exif-grid">
      `;

      const exifMap = {
        make: '品牌',
        model: '型号',
        iso: 'ISO',
        aperture: '光圈',
        shutterSpeed: '快门',
        focalLength: '焦距',
        capturedAt: '拍摄时间',
      };

      Object.entries(exifMap).forEach(([key, label]) => {
        if (image.exif[key]) {
          html += `
            <div class="exif-item">
              <div class="exif-label">${label}</div>
              <div class="exif-value">${this.escapeHtml(String(image.exif[key]))}</div>
            </div>
          `;
        }
      });

      html += `</div></div></div>`;
    }

    // 文件信息
    if (image.filePath || image.fileSize || image.dimensions) {
      html += `
        <div class="info-section">
          <div class="info-section-title"><i class="bi bi-file-earmark"></i> 文件信息</div>
          <div class="info-section-content">
            <div class="exif-grid">
      `;

      if (image.filePath) {
        html += `
          <div class="exif-item" style="grid-column: span 2;">
            <div class="exif-label">路径</div>
            <div class="exif-value" style="font-size: 0.75rem; word-break: break-all;">${this.escapeHtml(image.filePath)}</div>
          </div>
        `;
      }
      if (image.dimensions) {
        html += `
          <div class="exif-item">
            <div class="exif-label">尺寸</div>
            <div class="exif-value">${image.dimensions.width} x ${image.dimensions.height}</div>
          </div>
        `;
      }
      if (image.fileSize) {
        html += `
          <div class="exif-item">
            <div class="exif-label">大小</div>
            <div class="exif-value">${this.formatFileSize(image.fileSize)}</div>
          </div>
        `;
      }

      html += `</div></div></div>`;
    }

    if (!html) {
      html = '<div class="empty-state"><p class="text-muted">暂无详细信息</p></div>';
    }

    this.elements.infoPanelContent.innerHTML = html;
  }

  // ========== 缩放 ==========

  setScale(scale) {
    this.currentScale = Math.max(1, Math.min(scale, 4));
    this.elements.imageWrapper.style.transform = `scale(${this.currentScale})`;

    if (this.currentScale > 1) {
      this.elements.imageWrapper.classList.add('zoomed');
    } else {
      this.elements.imageWrapper.classList.remove('zoomed');
    }
  }

  zoomIn() {
    this.setScale((this.currentScale || 1) + 0.25);
  }

  zoomOut() {
    this.setScale((this.currentScale || 1) - 0.25);
  }

  resetZoom() {
    this.setScale(1);
  }

  // ========== 动作 ==========

  toggleFavorite() {
    const image = this.images[this.currentIndex];
    if (image) {
      image.isFavorite = !image.isFavorite;
      this.updateActions();

      // 触发自定义事件
      this.element.dispatchEvent(new CustomEvent('favorite', {
        detail: { index: this.currentIndex, image, isFavorite: image.isFavorite }
      }));
    }
  }

  download() {
    const image = this.images[this.currentIndex];
    if (image && (image.src || image.url)) {
      const a = document.createElement('a');
      a.href = image.src || image.url;
      a.download = image.downloadName || `photo_${this.currentIndex + 1}.jpg`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  }

  share() {
    const image = this.images[this.currentIndex];
    if (image && navigator.share) {
      navigator.share({
        title: image.title || '分享照片',
        text: image.caption || '',
        url: image.src || image.url || window.location.href,
      });
    }
  }

  // ========== 工具函数 ==========

  escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1024 / 1024).toFixed(2) + ' MB';
  }

  // ========== 销毁 ==========

  destroy() {
    this.element?.remove();
    this.element = null;
    this.images = [];
    this.isOpen = false;
  }
}

// 创建全局实例
const lightbox = new Lightbox({
  showCounter: true,
  showThumbnails: false,
  showInfoPanel: true,
  loop: true,
  autoHideToolbar: true,
  toolbarTimeout: 3000,
});

// 导出
window.Lightbox = Lightbox;
window.lightbox = lightbox;
