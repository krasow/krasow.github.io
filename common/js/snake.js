(() => {
  'use strict';

  class SnakeGame {
    constructor(output, onEnd, width = 24, height = 12) {
      this.output = output;
      this.onEnd = onEnd;
      this.width = width;
      this.height = height;
      this.snake = [{ x: 12, y: 6 }, { x: 11, y: 6 }, { x: 10, y: 6 }];
      this.direction = { x: 1, y: 0 };
      this.nextDirection = this.direction;
      this.score = 0;
      this.food = this.placeFood();
      this.timer = null;
    }

    start() {
      this.render();
      this.timer = setInterval(() => this.step(), 120);
    }

    handleKey(event) {
      event.preventDefault();
      const directions = {
        ArrowUp: { x: 0, y: -1 }, w: { x: 0, y: -1 },
        ArrowDown: { x: 0, y: 1 }, s: { x: 0, y: 1 },
        ArrowLeft: { x: -1, y: 0 }, a: { x: -1, y: 0 },
        ArrowRight: { x: 1, y: 0 }, d: { x: 1, y: 0 },
      };
      if (event.key === 'Escape' || event.key.toLowerCase() === 'q') {
        this.stop('quit');
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
        x: this.snake[0].x + this.direction.x,
        y: this.snake[0].y + this.direction.y,
      };
      const hitWall = head.x < 0 || head.x >= this.width || head.y < 0 || head.y >= this.height;
      const hitSelf = this.snake.some(({ x, y }) => x === head.x && y === head.y);
      if (hitWall || hitSelf) {
        this.stop('game over');
        return;
      }

      this.snake.unshift(head);
      if (head.x === this.food.x && head.y === this.food.y) {
        this.score += 1;
        this.food = this.placeFood();
      } else {
        this.snake.pop();
      }
      this.render();
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
      cells[this.food.y][this.food.x] = '◆';
      this.snake.slice(1).forEach(({ x, y }) => { cells[y][x] = '■'; });
      cells[this.snake[0].y][this.snake[0].x] = '●';
      const border = `+${'-'.repeat(this.width)}+`;
      this.output.textContent = [
        `snake · score ${this.score}`,
        border,
        ...cells.map((row) => `|${row.join('')}|`),
        border,
        'arrows/WASD to move · Esc/Q to quit',
      ].join('\n');
    }

    stop(reason) {
      clearInterval(this.timer);
      this.timer = null;
      this.onEnd(reason, this.score);
    }
  }

  window.KrasowSnake = { SnakeGame };
})();
