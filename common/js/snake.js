(() => {
  'use strict';

  class SnakeGame {
    constructor(output, onEnd, width = 24, height = 12) {
      this.output = output;
      this.onEnd = onEnd;
      this.width = width;
      this.height = height;
      this.reset();
    }

    reset() {
      const x = Math.floor(this.width / 2);
      const y = Math.floor(this.height / 2);
      this.snake = [{ x, y }, { x: x - 1, y }, { x: x - 2, y }];
      this.direction = { x: 1, y: 0 };
      this.nextDirection = this.direction;
      this.score = 0;
      this.food = this.placeFood();
      this.timer = null;
      this.ended = false;
    }

    start() {
      this.render();
      this.timer = setInterval(() => this.step(), 120);
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
        ArrowUp: { x: 0, y: -1 }, w: { x: 0, y: -1 },
        ArrowDown: { x: 0, y: 1 }, s: { x: 0, y: 1 },
        ArrowLeft: { x: -1, y: 0 }, a: { x: -1, y: 0 },
        ArrowRight: { x: 1, y: 0 }, d: { x: 1, y: 0 },
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
        clearInterval(this.timer);
        this.timer = null;
        this.ended = true;
        this.render();
        return;
      }

      this.snake.unshift(head);
      if (eating) {
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
      cells[this.food.y][this.food.x] = '*';
      this.snake.slice(1).forEach(({ x, y }) => { cells[y][x] = 'o'; });
      cells[this.snake[0].y][this.snake[0].x] = '@';
      const border = `+${'-'.repeat(this.width)}+`;
      this.output.textContent = [
        `snake · score ${this.score}`,
        border,
        ...cells.map((row) => `|${row.join('')}|`),
        border,
        this.ended
          ? 'game over · Enter/Space to replay · Esc/Q to quit'
          : 'arrows/WASD to move · Esc/Q to quit',
      ].join('\n');
    }

    stop() {
      clearInterval(this.timer);
      this.timer = null;
      this.onEnd(this.score);
    }
  }

  window.KrasowSnake = { SnakeGame };
})();
