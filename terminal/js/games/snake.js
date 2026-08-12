(() => {
  'use strict';

  const HIGH_SCORE_KEY = 'krasow-terminal-snake-high-score';

  class SnakeGame {
    constructor(
      output,
      onEnd,
      width = 24,
      height = 12,
      cellWidth = 1,
      horizontalDelay = 45,
      verticalDelay = 120,
    ) {
      this.output = output;
      this.onEnd = onEnd;
      this.width = width;
      this.height = height;
      this.cellWidth = cellWidth;
      this.horizontalDelay = horizontalDelay;
      this.verticalDelay = verticalDelay;
      this.highScore = this.loadHighScore();
      this.reset();
    }

    loadHighScore() {
      try {
        const value = Number.parseInt(localStorage.getItem(HIGH_SCORE_KEY), 10);
        return Number.isSafeInteger(value) && value >= 0 ? value : 0;
      } catch (error) {
        return 0;
      }
    }

    saveHighScore() {
      try {
        localStorage.setItem(HIGH_SCORE_KEY, String(this.highScore));
      } catch (error) {
        // Storage may be unavailable in private or restricted browser contexts.
      }
    }

    reset() {
      const x = Math.floor(this.width / 2);
      const y = Math.floor(this.height / 2);
      this.snake = [
        { x, y },
        { x: x - 1, y },
        { x: x - 2, y },
      ];
      this.direction = { x: 1, y: 0 };
      this.nextDirection = this.direction;
      this.score = 0;
      this.food = this.placeFood();
      this.timer = null;
      this.ended = false;
    }

    start() {
      this.render();
      this.schedule();
    }

    schedule() {
      const delay = this.direction.x ? this.horizontalDelay : this.verticalDelay;
      this.timer = setTimeout(() => this.step(), delay);
    }

    handleKey(event) {
      event.preventDefault();
      if (this.ended) {
        if (event.key === 'Enter' || event.key === ' ') {
          this.reset();
          this.start();
        } else if (event.key === 'Escape' || event.key.toLowerCase() === 'q') {
          this.stop();
        }
        return;
      }
      const directions = {
        ArrowUp: { x: 0, y: -1 },
        w: { x: 0, y: -1 },
        ArrowDown: { x: 0, y: 1 },
        s: { x: 0, y: 1 },
        ArrowLeft: { x: -1, y: 0 },
        a: { x: -1, y: 0 },
        ArrowRight: { x: 1, y: 0 },
        d: { x: 1, y: 0 },
      };
      if (event.key === 'Escape' || event.key.toLowerCase() === 'q') {
        this.stop();
        return;
      }
      const next = directions[event.key.toLowerCase()] ?? directions[event.key];
      if (next && (next.x !== -this.direction.x || next.y !== -this.direction.y)) {
        this.nextDirection = next;
      }
    }

    step() {
      this.direction = this.nextDirection;
      const head = {
        x: (this.snake[0].x + this.direction.x + this.width) % this.width,
        y: (this.snake[0].y + this.direction.y + this.height) % this.height,
      };
      const eating = head.x === this.food.x && head.y === this.food.y;
      const body = eating ? this.snake : this.snake.slice(0, -1);
      const hitSelf = body.some(({ x, y }) => x === head.x && y === head.y);
      if (hitSelf) {
        clearTimeout(this.timer);
        this.timer = null;
        this.ended = true;
        this.render();
        return;
      }

      this.snake.unshift(head);
      if (eating) {
        this.score += 1;
        if (this.score > this.highScore) {
          this.highScore = this.score;
          this.saveHighScore();
        }
        this.food = this.placeFood();
      } else {
        this.snake.pop();
      }
      this.render();
      this.schedule();
    }

    placeFood() {
      const open = [];
      for (let y = 0; y < this.height; y += 1) {
        for (let x = 0; x < this.width; x += 1) {
          if (!this.snake.some((part) => part.x === x && part.y === y)) open.push({ x, y });
        }
      }
      return open[Math.floor(Math.random() * open.length)];
    }

    render() {
      const cells = Array.from({ length: this.height }, () => Array(this.width).fill(' '));
      cells[this.food.y][this.food.x] = '*';
      this.snake.slice(1).forEach(({ x, y }) => {
        cells[y][x] = 'o';
      });
      cells[this.snake[0].y][this.snake[0].x] = '@';
      const drawCell = (cell) =>
        cell === ' '
          ? ' '.repeat(this.cellWidth)
          : cell.padStart(Math.ceil(this.cellWidth / 2), ' ').padEnd(this.cellWidth, ' ');
      const border = `+${'-'.repeat(this.width * this.cellWidth)}+`;
      this.output.textContent = [
        `snake · score ${this.score} · high score ${this.highScore}`,
        border,
        ...cells.map((row) => `|${row.map(drawCell).join('')}|`),
        border,
        this.ended
          ? 'game over · Enter/Space to replay · Esc/Q to quit'
          : 'arrows/WASD to move · Esc/Q to quit',
      ].join('\n');
    }

    stop() {
      clearTimeout(this.timer);
      this.timer = null;
      this.onEnd(this.score);
    }
  }

  class SnakeApp {
    constructor(terminal) {
      this.terminal = terminal;
      this.game = null;
    }

    run(args) {
      return window.TerminalApp.exact(args, 0, () => this.start());
    }

    start() {
      const output = document.createElement('pre');
      output.className = 'ln snake-game';
      this.terminal.append(output);
      const style = getComputedStyle(output);
      const fontSize = parseFloat(style.fontSize) || 14;
      const lineHeight = parseFloat(style.lineHeight) || fontSize * 1.75;
      const charWidth = fontSize * 0.62;
      const cellWidth = Math.max(1, Math.round(lineHeight / charWidth));
      const maxBoardPx = 720;
      const availWidth = Math.min(this.terminal.ui.log.clientWidth, maxBoardPx);
      const width = Math.max(
        10,
        Math.min(60, Math.floor(availWidth / (charWidth * cellWidth)) - 2),
      );
      const height = Math.max(
        12,
        Math.min(30, Math.floor(this.terminal.ui.log.clientHeight / lineHeight) - 5),
      );

      this.game = new SnakeGame(
        output,
        (score) => {
          this.game = null;
          this.terminal.write(`snake: quit · score ${score}`, 'hint');
          this.terminal.ui.input.focus();
        },
        width,
        height,
        cellWidth,
        Math.round((120 * charWidth * cellWidth) / lineHeight),
        120,
      );
      this.game.start();
      this.terminal.ui.log.scrollTop = this.terminal.ui.log.scrollHeight;
    }

    handleKey(event) {
      if (!this.game) return false;
      this.game.handleKey(event);
      return true;
    }
  }

  window.SnakeApp = SnakeApp;
})();
