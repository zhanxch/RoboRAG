const fs = require('fs');
const path = require('path');
const katex = require(path.join(__dirname, '..', 'assets', 'katex', 'katex.min.js'));
const items = JSON.parse(fs.readFileSync(0, 'utf8'));
const result = items.map(({tex, display}) => {
  try {
    return {html: katex.renderToString(tex, {displayMode: display, throwOnError: true, strict: 'ignore', output:'htmlAndMathml'}), error: null};
  } catch (err) {
    return {html: null, error: String(err)};
  }
});
process.stdout.write(JSON.stringify(result));
