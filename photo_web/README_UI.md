# AutoAlbum 前端设计

## 设计理念

照片优先的现代化相册界面（Apple Photos / Immich 风格）：

- **暗色默认** — 照片在深色背景上更突出，侧边栏底部可切换亮色
- **单一暖色点缀** — 琥珀金（`#f5a524`）作为唯一强调色，色彩让位给照片
- **方形照片网格** — 等距方形瓦片 + 3px 间隙，悬停显示日期遮罩
- **左侧边栏导航** — 桌面端固定侧栏，移动端收起为汉堡菜单

## 文件结构

```
static/
├── css/
│   ├── app.css        # 设计系统（令牌 / 布局 / 组件 / 响应式）
│   └── lightbox.css   # 灯箱样式
└── js/
    ├── lightbox.js    # 灯箱查看器（全局 window.lightbox）
    ├── search.js      # 搜索建议（绑定 input[name="q"]）
    ├── keyboard-nav.js # 键盘导航
    └── utils.js       # 工具函数

templates/
├── base.html          # 布局骨架：侧边栏 + 顶栏搜索 + 主题切换
├── index.html         # 首页：问候 + 统计 + 成员 + 最近照片
├── gallery.html       # 画廊：密度可调的方格 + 分页
├── browse.html        # 浏览：左侧锚点导航 + 时间/人物/设备/地点
├── persons.html       # 人物：圆形头像卡片
├── person.html        # 单人照片墙
├── search.html        # 搜索：主视觉搜索框 + 结果网格
├── image_detail.html  # 详情：黑色舞台 + AI 分析 + EXIF
├── images.html        # 筛选结果列表
├── category.html      # 分类照片
└── error.html         # 错误页
```

## 主题

CSS 变量定义在 `app.css` 的 `:root`（暗色）和 `[data-theme="light"]`（亮色）。
`base.html` 内联脚本在页面加载前从 `localStorage`（键 `aa-theme`）读取主题，避免闪烁。

## 核心组件类

| 类 | 用途 |
|---|---|
| `.pgrid` / `.ptile` | 方形照片网格 / 瓦片（`.compact` `.wide` 调整密度） |
| `.ptile-meta` / `.ptile-badge` | 瓦片悬停遮罩 / 角标 |
| `.stat` | 统计卡片 |
| `.pcircle` / `.pcard` | 人物圆形头像 / 人物卡片 |
| `.panel` | 内容面板（`.panel-head` `.panel-body`） |
| `.chip` / `.badge` | 标签芯片 / 徽章 |
| `.btn` (`.btn-accent` `.btn-outline` `.btn-ghost`) | 按钮 |
| `.rail` / `.rail-link` | 浏览页锚点导航 |
| `.detail` / `.stage` / `.kv` | 详情页布局 / 照片舞台 / 键值表 |

## 灯箱

`lightbox.js` 自动生成 DOM 并创建全局实例：

```js
lightbox.open(index, [
  { id: 1, src: '...', caption: '...', url: '/image/1/full' }
]);
```
