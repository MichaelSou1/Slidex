const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ channel: process.platform === 'darwin' ? 'chrome' : undefined });
  const page = await browser.newPage();
  const file = process.argv[2];
  await page.goto(`file://${file}`);
  const before = await page.evaluate(() => {
    const s = window.getComputedStyle(document.body);
    return { width: s.width, height: s.height, scrollW: document.body.scrollWidth, scrollH: document.body.scrollHeight, innerW: window.innerWidth, innerH: window.innerHeight };
  });
  console.log('BEFORE addStyleTag:', before);
  await page.addStyleTag({ content: '*, *::before, *::after { box-sizing: border-box !important; }' });
  const after = await page.evaluate(() => {
    const s = window.getComputedStyle(document.body);
    return { width: s.width, height: s.height, scrollW: document.body.scrollWidth, scrollH: document.body.scrollHeight };
  });
  console.log('AFTER addStyleTag:', after);
  await browser.close();
})();
