(() => {
  'use strict';

  const STORAGE_KEY = 'krasow-terminal-size';
  const LEGACY_HEIGHT_KEY = 'krasow-terminal-height';

  class TerminalResizer {
    constructor(element, handle, options = {}) {
      this.element = element;
      this.handle = handle;
      this.minimumWidth = options.minimumWidth ?? 480;
      this.minimumHeight = options.minimumHeight ?? 320;
      this.viewportMargin = options.viewportMargin ?? 48;
      this.mobileQuery = window.matchMedia(options.mobileQuery ?? '(max-width: 560px)');
      this.drag = null;

      this.onPointerDown = this.onPointerDown.bind(this);
      this.onPointerMove = this.onPointerMove.bind(this);
      this.onPointerEnd = this.onPointerEnd.bind(this);
      this.onViewportResize = this.onViewportResize.bind(this);
    }

    start() {
      if (!this.element || !this.handle) return;
      this.restore();
      this.handle.addEventListener('pointerdown', this.onPointerDown);
      this.handle.addEventListener('pointermove', this.onPointerMove);
      this.handle.addEventListener('pointerup', this.onPointerEnd);
      this.handle.addEventListener('pointercancel', this.onPointerEnd);
      window.addEventListener('resize', this.onViewportResize);
    }

    bounds() {
      return {
        maximumWidth: Math.max(this.minimumWidth, window.innerWidth - this.viewportMargin),
        maximumHeight: Math.max(this.minimumHeight, window.innerHeight - this.viewportMargin),
      };
    }

    resize(width, height) {
      const { maximumWidth, maximumHeight } = this.bounds();
      const boundedWidth = Math.min(maximumWidth, Math.max(this.minimumWidth, width));
      const boundedHeight = Math.min(maximumHeight, Math.max(this.minimumHeight, height));
      this.element.style.width = `${boundedWidth}px`;
      this.element.style.height = `${boundedHeight}px`;
    }

    onPointerDown(event) {
      if (this.mobileQuery.matches) return;
      event.preventDefault();
      const rect = this.element.getBoundingClientRect();
      this.drag = {
        pointerId: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        width: rect.width,
        height: rect.height,
      };
      this.handle.classList.add('dragging');
      this.handle.setPointerCapture(event.pointerId);
    }

    onPointerMove(event) {
      if (!this.drag || event.pointerId !== this.drag.pointerId) return;
      this.resize(
        this.drag.width + event.clientX - this.drag.x,
        this.drag.height + event.clientY - this.drag.y,
      );
    }

    onPointerEnd(event) {
      if (!this.drag || event.pointerId !== this.drag.pointerId) return;
      this.drag = null;
      this.handle.classList.remove('dragging');
      this.persist();
    }

    onViewportResize() {
      if (this.mobileQuery.matches) return;
      const rect = this.element.getBoundingClientRect();
      this.resize(rect.width, rect.height);
    }

    restore() {
      if (this.mobileQuery.matches) return;
      try {
        const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
        if (Number.isFinite(saved?.width) && Number.isFinite(saved?.height)) {
          this.resize(saved.width, saved.height);
          return;
        }

        const legacyHeight = Number.parseFloat(localStorage.getItem(LEGACY_HEIGHT_KEY));
        if (Number.isFinite(legacyHeight)) {
          this.resize(this.element.getBoundingClientRect().width, legacyHeight);
        }
      } catch (error) {
        // Storage may be unavailable or malformed.
      }
    }

    persist() {
      try {
        const { width, height } = this.element.getBoundingClientRect();
        localStorage.setItem(STORAGE_KEY, JSON.stringify({ width, height }));
      } catch (error) {
        // Storage may be unavailable in private or restricted browser contexts.
      }
    }
  }

  window.KrasowTerminalResize = { TerminalResizer };
})();
