// Shared engine entry point for non-browser hosts. The browser keeps loading
// these scripts directly so the same implementation serves both environments.
import './app.js';
import './apps/cat.js';
import './apps/cd.js';
import './apps/chat.js';
import './apps/clear.js';
import './apps/copy.js';
import './apps/cowsay.js';
import './apps/download.js';
import './apps/find.js';
import './apps/grep.js';
import './apps/help.js';
import './apps/ls.js';
import './apps/open.js';
import './apps/reset.js';
import './apps/rm.js';
import './apps/show.js';
import './apps/theme.js';
import './apps/tree.js';
import './apps/wc.js';
import './games/snake.js';
import './commands.js';
import './filesystem.js';
import './autocomplete.js';
